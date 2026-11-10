"""Direct-audio conversation: user speech goes straight into Gemma 4.

Instead of the two-step pipeline (audio -> Gemma STT -> text -> Gemma chat),
one multimodal call does hearing, thinking, and tool calling at once — the
same idea as reachy_mini_conversation_app's direct-audio mode. Measured on an
M-series Mac this roughly halves speech-end -> reply latency (one ~0.8-1.7 s
call replaces a 0.7-2.7 s STT call plus a ~0.8 s chat call).

What still needs text (per-person memory, fact extraction, name learning) is
served by a background transcription AFTER the reply — off the critical path,
exactly like reachy's remote_stt. Deliberately NOT folded into the reply call:
asking gemma4 to also emit a word-for-word transcript alongside reply + tool
calls degrades all three (tried Aug 2026 — transcripts got sloppy and tool
turns sometimes lost their prose), which is why reachy split it too.

The reply carries its language as a leading ISO tag ("[sv] Hej!") so the TTS
voice can be routed without a separate detection step; tools are fetched once
from a service MCP server (candytron_mcp) and executed over short-lived SSE
sessions, mirroring ``ServiceHost``.
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
import time
from typing import Callable, Optional

import numpy as np
import soundfile as sf

from sentences import SentenceStream, split_sentences

logger = logging.getLogger("direct_llm")

# System prompt: the face-agent identity + the service role, with the rules the
# direct path needs (language tag, tools, brevity). Mirrors llm.SYSTEM_PROMPT.
import prompts as _prompts

# The rules text lives in face/prompts.py and can be edited live (launcher's
# "Prompts..." window); this is the default, kept for reference and tests.
DIRECT_SYSTEM = _prompts.RULES

# Positions the vision system reports, e.g. "A2 : Riesen-kola, brun ...".
_SCENE_POS = re.compile(r"^\s*([A-Z]\d)\s*:\s*(.+?)\s*$", re.MULTILINE)

# The robot claiming it did something physical. If a turn ends with one of
# these and NO tool was called, the arm never moved and the visitor is being
# told a lie -- the single worst failure mode at a stand.
# Offers and descriptions ("I can show you HOW I move candy", "shall I move
# it?") are not claims. Treating them as one triggers a retry that tells the
# model it failed to act -- which can push it into moving candy nobody asked
# for, the exact failure this guard exists to prevent.
_NOT_A_CLAIM = re.compile(
    r"\b(hur jag|om du vill|vill du att jag|ska jag|kan jag visa|"
    r"how i|if you (want|like)|shall i|would you like|do you want me)\b",
    re.IGNORECASE)

_ACTION_CLAIM = re.compile(
    r"\b("
    # sv -- "Här är din Marianne" was missed at the demo: any "här är/kommer"
    r"jag (flyttar|tar|ger|hämtar|lägger|placerar|skickar)"
    r"|här (får|har) du|här (är|kommer) (din|ditt|dina|en|ett|två|tre)|varsågod|nu (flyttar|ger|tar|hämtar) jag"
    r"|jag har (flyttat|gett|tagit|hämtat|lagt)"
    # en
    r"|i (moved|gave|placed|took|put|fetched|handed)|i'?m (moving|giving|getting|taking|fetching|handing|bringing)"
    r"|here (you go|it is|you are|comes)|here'?s (your|a|an|another|one)|here (is|are) (your|a|an|another|one|two)"
    r"|there you go|moving it|i'?ll (move|give|get|take|fetch|hand|bring)|i will (move|give|get|take|fetch|hand|bring)|i(?: have|'ve) (moved|given|placed|put|handed)"
    # "I can hand you a Dumle" in answer to "can I have one?" is a non-move
    # dressed as a yes (8/12 plain-prompt requests moved before this).
    r"|i can (hand|give|get|bring|fetch) you|jag kan (ge|hämta|räcka|ta) dig"
    # de / fr / es / it
    r"|ich (bewege|gebe|nehme|hole|bringe)|hier (ist|kommt) (dein|deine|ein|eine)|hier hast du|bitte schön"
    r"|je (déplace|donne|prends|t'apporte)|je (vous|te|t') ?(donne|apporte|passe)|voici|voilà"
    r"|aquí (tienes|está|lo tienes)|te (doy|traigo)|toma"
    r"|ecco|ti (do|porto)|ho (spostato|messo|dato|preso)"
    r"|j'ai (déplacé|donné|mis|pris)|he (movido|puesto|dado|traído)|ich habe (dir|ihnen)"
    r")\b", re.IGNORECASE)

# gemma4 sometimes WRITES its tool call as text instead of making it, e.g.
#   [toolcall] movebetween{dst:<|"|>O0<|"|>,src:<|"|>A1<|"|>}
#   <|tool_call>call:move_between{dst:<|"|>O0<|"|>,...}<tool_call|>
# (seen at the demo with the expressive prompt, which teaches "[tag]" syntax).
# The arm then never moved and the voice read the gibberish out. Such text is
# never spoken, and the call in it is executed as if it had been made.
# Braces never occur in speech, so "name{...}" is always a written-out call --
# with or without a [toolcall] / <|tool_call> prefix (the bare form
# "movebetween{dst: O0 ,src: C1 }" and "showdemomove{}" were read out loud).
_TOOL_TEXT = re.compile(r"\[tool_?call\]|<\|tool_call>|<tool_call\|>|<\|\"\|>|\w+\s*\{[^{}]*\}|[{}]",
                        re.IGNORECASE)
_TOOL_TEXT_CALL = re.compile(r"(?:\[tool_?call\]|<\|tool_call>|\bcall:)?\s*(?:call:)?\s*([a-z_]+)\s*\{([^{}]*)\}",
                             re.IGNORECASE)
# Any known tool written as text is executed as if it had been called --
# except ones that act at random or do nothing, which are just dropped.
_NOT_RECOVERED = {"show_demo_move", "default_action"}
_TOOL_TEXT_ARG = re.compile(r"(\w+)\s*:\s*(?:<\|\"\|>(.*?)<\|\"\|>|\"([^\"]*)\"|([^,}\s]+))")


def _text_tool_calls(content: str, tool_names: list) -> list:
    """Tool calls the model wrote as text, as Ollama-style call dicts."""
    norm = {n.replace("_", "").lower(): n for n in tool_names}
    calls = []
    for m in _TOOL_TEXT_CALL.finditer(content or ""):
        name = norm.get(m.group(1).replace("_", "").lower())
        if not name or name in _NOT_RECOVERED:
            continue
        args = {}
        for a in _TOOL_TEXT_ARG.finditer(m.group(2)):
            args[a.group(1)] = next(v for v in a.groups()[1:] if v is not None)
        calls.append({"function": {"name": name, "arguments": args}})
    return calls


def _strip_tool_text(content: str) -> str:
    """The prose around written-out tool calls, without them. The real answer
    often comes AFTER the calls ("[toolcall] ...{...} [en] Here is a Dumle")."""
    text = _TOOL_TEXT_CALL.sub(" ", content or "")
    text = re.sub(r"<\|tool_call>|<tool_call\|>|\[tool_?call\]|<\|\"\|>|[{}]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    # a language tag left in the middle moves to the front, where it is read
    m = re.match(r"^(.*?)\s*(\[[a-z]{2}\])\s*(.*)$", text)
    if m and not m.group(1):
        return text
    if m and not _TOOL_TEXT.search(m.group(1)) and len(m.group(1)) < 3:
        return f"{m.group(2)} {m.group(3)}".strip()
    return text


# What the server answers when the arm really started (or queued) a move.
_MOVE_OK = re.compile(r"^(Started moving|Queued|Successfully demonstrated)", re.IGNORECASE)
# Generic, for any service: a tool call counts as DONE unless its result says
# it was refused, failed, or did nothing.
_NOT_DONE = re.compile(r"^\s*(refused|failed|error|did nothing|tool \S+ failed|unknown)",
                       re.IGNORECASE)
_NO_OP_TOOLS = {"default_action"}


def parse_scene_positions(augmentation: Optional[str]) -> dict:
    """{position: description} from the vision system's scene text."""
    if not augmentation:
        return {}
    return {m.group(1): m.group(2) for m in _SCENE_POS.finditer(augmentation)}


_LANG_TAG = re.compile(r"^\s*\[([a-z]{2}(?:-[a-z]{2})?)\]\s*", re.IGNORECASE)
_MAX_TOOL_ROUNDS = 3
# Keep gemma4 in memory for the whole demo. Ollama's default unloads it after
# 5 idle minutes, and the first visitor after a quiet spell then waits for a
# reload before the robot says anything.
KEEP_ALIVE = -1
# Utterances shorter than this carry too little language signal to switch on.
SHORT_UTTERANCE_S = 1.0


def _split_lang(text: str, default: str) -> tuple:
    """('sv', 'Hej!') from '[sv] Hej!'; the tag is never spoken."""
    m = _LANG_TAG.match(text or "")
    if not m:
        return default, (text or "").strip()
    return m.group(1).lower(), text[m.end():].strip()


def _is_claim(text: str) -> bool:
    """The robot saying it did something physical (see _ACTION_CLAIM)."""
    return bool(_ACTION_CLAIM.search(text)) and not _NOT_A_CLAIM.search(text)


def _candy_aliases(scene: dict) -> dict:
    """{candy name: {ways it is said}} for the candies on the table. The name
    is the head of the vision system's description ("Riesen-kola", "Dumle");
    its first word is an alias too unless another candy shares it
    ("Nötkräm päron" / "Nötkräm original")."""
    heads = sorted({desc.split(",")[0].strip().lower() for desc in scene.values()} - {""})
    # "licorice (lakrits)": the Swedish name in brackets is a name too.
    extra = {h: set(re.findall(r"\(([^)]+)\)", h)) for h in heads}
    first = {}
    for h in heads:
        w = re.split(r"[\s\-]+", h)[0]
        first.setdefault(w, []).append(h)
    out = {}
    for h in heads:
        w = re.split(r"[\s\-]+", h)[0]
        bare = re.sub(r"\s*\([^)]*\)", "", h).strip()
        out[h] = ({h, bare} | {x.strip() for x in extra[h]}
                  | ({w} if len(w) >= 4 and len(first[w]) == 1 else set()))
    return out


def _claimed_candies(text: str, scene: dict) -> int:
    """How many different candies the CLAIM sentences of *text* hand over."""
    return len(_claimed_kinds(text, scene))


def _claimed_kinds(text: str, scene: dict) -> set:
    """The candies ON THE TABLE that the claim sentences of *text* name."""
    aliases = _candy_aliases(scene)
    if not aliases:
        return set()
    seen = set()
    for sentence in split_sentences(text or ""):
        if not _is_claim(sentence):
            continue
        low = sentence.lower()
        seen.update(h for h, names in aliases.items()
                    if any(re.search(r"\b" + re.escape(n) + r"\b", low) for n in names))
    return seen


def _history_messages(history) -> list:
    """Earlier turns for the model, INCLUDING the arm calls they made. With
    text only, the model saw itself answer "Kan jag få en Dumle?" with "Här är
    din Dumle!" and nothing else, learned that handing out candy means saying
    so, and stopped calling the tool (0/4 requests moved in a test)."""
    out = []
    for m in history or []:
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            out.append({"role": "assistant", "content": str(m.get("content") or ""),
                        "tool_calls": m["tool_calls"]})
        elif role == "tool":
            out.append({"role": "tool", "tool_name": m.get("tool_name", ""),
                        "content": str(m.get("content") or "")})
        elif role in ("user", "assistant") and m.get("content"):
            out.append({"role": role, "content": str(m["content"])})
    while out and out[0]["role"] != "user":   # never start mid-turn
        out.pop(0)
    return out


def _claimed_kinds_any(text: str, scene: dict) -> set:
    """Candies on the table named anywhere in *text* (for a claim found by the
    judge, whose wording the phrase list did not know)."""
    aliases = _candy_aliases(scene)
    low = (text or "").lower()
    return {h for h, names in aliases.items()
            if any(re.search(r"\b" + re.escape(n) + r"\b", low) for n in names)}


def _has_words(text: str) -> bool:
    """True if there is something to say once the language tag is removed."""
    return bool(_split_lang(text or "", "")[1])


def _to_wav_bytes(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio, dtype=np.float32).reshape(-1),
             sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class DirectAudioLLM:
    """One-call audio conversation via an audio-capable Ollama model.

    ``respond()`` is the hot path. ``transcriber`` (a ``Gemma4Transcriber``)
    is exposed for the caller's background transcription of the same
    utterance for memory purposes.
    """

    def __init__(self, *,
                 model: str = "gemma4:latest",
                 host: str = "http://localhost:11434",
                 agent_name: str = "Face Agent",
                 service_prompt: Optional[str] = None,
                 augmentation_provider: Optional[Callable[[str], Optional[str]]] = None,
                 tools_url: Optional[str] = None,
                 sample_rate: int = 16000,
                 temperature: float = 0.2):
        self.last_tool_calls: list = []
        self.rejected_tool_calls: list = []
        import ollama
        self._client = ollama.Client(host=host)
        self._model = model
        # Low temperature: tool calling on audio turns is otherwise flaky —
        # the same request sometimes calls move_between, sometimes just says
        # it did (measured Aug 2026).
        self._options = {"temperature": temperature}
        self._agent_name = agent_name
        self._service_prompt = service_prompt
        self._augmentation_provider = augmentation_provider
        self._tools_url = tools_url
        self._sample_rate = sample_rate
        self._tools: list = []
        if tools_url:
            self._tools = self._load_tools(tools_url)

        from gemma_stt import Gemma4Transcriber
        self.transcriber = Gemma4Transcriber(model=model, host=host,
                                             sample_rate=sample_rate)

    # --- MCP tools over short-lived SSE sessions (like ServiceHost) ---

    def _load_tools(self, url: str) -> list:
        """Fetch the service's MCP tools once and map them to Ollama schema."""
        async def _list():
            from fastmcp import Client
            from fastmcp.client.transports import SSETransport
            async with Client(transport=SSETransport(url)) as client:
                return await client.list_tools()
        try:
            tools = asyncio.run(_list())
        except Exception as e:
            logger.warning(f"direct: could not load tools from {url}: {e}")
            return []
        mapped = [{
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or t.name,
                "parameters": t.inputSchema,
            },
        } for t in tools]
        logger.info(f"direct: {len(mapped)} tool(s) from {url}: "
                    f"{[t['function']['name'] for t in mapped]}")
        return mapped

    def _call_tool(self, name: str, args: dict) -> str:
        async def _call():
            from fastmcp import Client
            from fastmcp.client.transports import SSETransport
            async with Client(transport=SSETransport(self._tools_url)) as client:
                res = await client.call_tool(name, args or {})
                parts = getattr(res, "content", None) or []
                return "".join(getattr(c, "text", "") for c in parts)
        try:
            return asyncio.run(_call()) or "ok"
        except Exception as e:
            logger.warning(f"direct: tool {name}({args}) failed: {e}")
            return f"Tool {name} failed: {e}"

    _JUDGE = ("An assistant that acts in the real world ONLY through these tools:\n"
              "{tools}\n\n"
              "It said to the user:\n\"{reply}\"\n\n"
              "Does that say or imply that the assistant itself has just done, or is "
              "right now doing, something that needs one of these tools -- e.g. moving "
              "or handing over something, switching something on or off, changing a "
              "setting? Offers (\"would you like...\", \"I can...\"), questions and "
              "descriptions of what it CAN do are NO. Answer with one word: YES or NO.")

    def _tool_summary(self) -> str:
        lines = []
        for t in self._tools or []:
            fn = t.get("function", {})
            if fn.get("name") in _NO_OP_TOOLS:
                continue
            desc = (fn.get("description") or "").strip().splitlines()
            lines.append(f"- {fn.get('name')}: {desc[0][:120] if desc else ''}")
        return "\n".join(lines) or "- (none)"

    def _judge_claims_action(self, reply: str) -> bool:
        """Language-independent check: does *reply* claim a physical action?"""
        text = _split_lang(_strip_tool_text(reply), "")[1]
        if not text:
            return False
        t0 = time.time()
        try:
            out = self._client.chat(model=self._model, messages=[{"role": "user", "content":
                                    self._JUDGE.format(tools=self._tool_summary(),
                                                       reply=text[:400])}],
                                    options={"temperature": 0, "num_predict": 3}, think=False,
                                    stream=False, keep_alive=KEEP_ALIVE)["message"]["content"]
        except Exception as e:  # noqa: BLE001
            logger.warning(f"direct: action judge failed: {e}")
            return False
        verdict = (out or "").strip().upper().startswith(("YES", "JA"))
        logger.info(f"direct: action judge {'YES' if verdict else 'NO'} in "
                    f"{time.time() - t0:.2f}s for {text[:60]!r}")
        if verdict:
            logger.warning(f"direct: judge says the reply claims an action: {text[:70]!r}")
        return verdict

    _NOTHING_MOVED = {
        "sv": "[sv] Förlåt, det gick inte. Vilket godis vill du ha?",
        "en": "[en] Sorry, that didn't work. Which candy would you like?",
        "de": "[de] Entschuldigung, das hat nicht geklappt. Welche Süßigkeit möchtest du?",
        "fr": "[fr] Désolé, ça n'a pas marché. Quel bonbon veux-tu ?",
        "es": "[es] Lo siento, no ha funcionado. ¿Qué caramelo quieres?",
        "it": "[it] Scusa, non ha funzionato. Quale caramella vuoi?",
    }

    def _honest_reply(self, messages: list, language: str) -> str:
        """A truthful one-liner after a claimed hand-out that did not happen:
        say so, and if the candy is not on the table, what is."""
        messages = messages + [{"role": "user", "content":
            "That was NOT done: no tool ran, so nothing happened. Do not say it "
            "was done. In ONE short sentence in the person's language, starting "
            "with its ISO code in brackets, say so honestly. If what they asked "
            "for is not available, say that and name what is."}]
        try:
            text = (self._chat(messages, None, None, language).get("content") or "").strip()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"direct: honest reply failed: {e}")
            text = ""
        if not _has_words(text) or _is_claim(text) or _TOOL_TEXT.search(text):
            text = self._NOTHING_MOVED.get(language, self._NOTHING_MOVED["en"])
        return text

    def _recover_text_calls(self, content: str, scene: dict) -> list:
        """Execute what the model wrote as a tool call instead of making one.
        At most one move per kind of candy: written out, it tends to list
        every Dumle on the table for "a Dumle"."""
        calls = _text_tool_calls(content, [t["function"]["name"] for t in self._tools])
        kept, kinds = [], set()
        for c in calls:
            fn = c["function"]
            if fn["name"] == "move_between":
                src = str(fn["arguments"].get("src", "")).upper().strip()
                kind = (scene.get(src) or src).split(",")[0].strip().lower()
                if kind in kinds:
                    continue
                kinds.add(kind)
            kept.append(c)
        logger.warning(f"direct: model wrote {len(calls)} tool call(s) as text; "
                       f"executing {len(kept)}: {[c['function'] for c in kept]}")
        return kept

    @staticmethod
    def _apply_move(name: str, args: dict, result: str, scene: dict) -> None:
        """Update this turn's scene copy after a move the arm accepted, so a
        second move in the same turn is checked against the board as it will
        be (B3 -> O0 after A1 -> B3 is fine; A1 -> O0 twice is not)."""
        if name != "move_between" or not _MOVE_OK.match(result or ""):
            return
        src = str(args.get("src", "")).upper().strip()
        dst = str(args.get("dst", "")).upper().strip()
        desc = scene.pop(src, None)
        if desc is not None and dst and dst != "O0":
            scene[dst] = desc

    def _reject_bad_move(self, name: str, args: dict, scene: dict) -> Optional[str]:
        """Reason to refuse this call, or None to let it through.

        The server happily reports "Successfully moved from B1 to O0" for an
        empty square -- the arm grips air and the visitor gets nothing while
        the robot says it worked. Only the caller knows what the vision system
        actually sees, so the check belongs here.
        """
        if name != "move_between" or not scene:
            return None
        src = str(args.get("src", "")).upper().strip()
        dst = str(args.get("dst", "")).upper().strip()
        known = ", ".join(f"{p} ({d.split(',')[0]})" for p, d in sorted(scene.items()))
        if not src or not dst:
            return f"Missing src or dst. Candy is at: {known}."
        if src == dst:
            return f"src and dst are both {src}; pick a different destination."
        if src not in scene:
            return (f"Position {src} is EMPTY -- there is no candy there, so "
                    f"nothing was moved. Candy is at: {known}. "
                    f"Call move_between again with one of those as src.")
        return None

    def _chat(self, messages: list, tools: Optional[list], say, default_lang: str):
        """One model call; returns its message. With *say* set it streams and
        hands every sentence to ``say(sentence, language)`` as soon as it is
        written, the leading language tag stripped."""
        if say is None:
            return self._client.chat(model=self._model, messages=messages,
                                     tools=tools, options=self._options,
                                     think=False, stream=False,
                                     keep_alive=KEEP_ALIVE)["message"]
        import ollama
        content, calls = "", []
        lang, fed = None, 0   # language once the tag is read; chars fed so far
        cutter = SentenceStream()
        for part in self._client.chat(model=self._model, messages=messages,
                                      tools=tools, options=self._options,
                                      think=False, stream=True,
                                      keep_alive=KEEP_ALIVE):
            msg = part["message"]
            calls += msg.get("tool_calls") or []
            content += msg.get("content") or ""
            if lang is None:
                head = content.lstrip()
                if head.startswith("[") and "]" not in head and len(head) < 10:
                    continue   # the language tag is still being written
                lang, body = _split_lang(content, default_lang)
                fed = len(content) - len(body)
            for sentence in cutter.feed(content[fed:]):
                if not _TOOL_TEXT.search(sentence):
                    say(sentence, lang)
            fed = len(content)
        for sentence in cutter.flush():
            if not _TOOL_TEXT.search(sentence):
                say(sentence, lang or default_lang)
        return ollama.Message(role="assistant", content=content,
                              tool_calls=calls or None)

    # --- Hot path ---

    _LANG_NAMES = {"en": "English", "sv": "Swedish", "de": "German",
                   "fr": "French", "es": "Spanish", "it": "Italian"}

    def warm(self) -> None:
        """Load the model now (and keep it loaded), so the first visitor does
        not pay for it."""
        try:
            self._client.chat(model=self._model, messages=[{"role": "user", "content": "hi"}],
                              options={"num_predict": 1}, think=False, stream=False,
                              keep_alive=KEEP_ALIVE)
        except Exception as e:
            logger.warning(f"direct: warm-up failed: {e}")

    def _system(self, service_prompt, augmentation, language_hint, context,
                lang_rule: Optional[str] = None) -> str:
        service = ""
        persona = service_prompt if service_prompt is not None else self._service_prompt
        if persona:
            service = f"\nYour service role:\n{persona}\n"
        if augmentation:
            service += (f"\nCurrent state from your service "
                        f"(internal — never read it out verbatim):\n{augmentation}\n")
        if lang_rule is None:
            # No "else use Swedish" fallback here: with one, gemma4 answered
            # clear English and German speech in Swedish. Too-short utterances
            # get their own rule in respond() instead.
            lang_rule = ("- LANGUAGE: reply in the language the person SPEAKS in the "
                         "audio — Swedish, English, German, French, Spanish or "
                         "Italian — judged only from the words you hear. Ignore the "
                         "language of the information above and of earlier turns.\n"
                         "- Never repeat or transcribe what the person said; answer it.\n")
        system = _prompts.fill(_prompts.get("rules"), name=self._agent_name,
                               service=service, lang_rule=lang_rule)
        if context:
            system += f"\nAbout the person you hear:\n{context}\n"
        _prompts.write_last(system)
        return system

    def greet(self, instruction: str, *,
              language: str = "sv",
              service_prompt: Optional[str] = None,
              augmentation: Optional[str] = None,
              history: Optional[list] = None,
              on_sentence: Optional[Callable[[str, str], None]] = None,
              ) -> tuple[str, str]:
        """Text-only greeting turn, streamed sentence by sentence and without
        tools (a greeting never moves the arm), so the first words are heard
        well under a second after contact."""
        lname = self._LANG_NAMES.get(language, "Swedish")
        lang_rule = (f"- Reply in {lname} (tag [{language}]) unless the person "
                     f"spoke another language earlier in the conversation.\n")
        system = self._system(service_prompt, augmentation, language, "", lang_rule)
        messages = [{"role": "system", "content": system}]
        messages += _history_messages(history)
        messages.append({"role": "user", "content": instruction})
        self.last_tool_calls, self.rejected_tool_calls = [], []
        start = time.time()
        msg = self._chat(messages, None, on_sentence, language)
        lang, reply = _split_lang((msg.get("content") or "").strip(), language)
        logger.info(f"direct: greeting in {time.time() - start:.2f}s lang={lang}: {reply}")
        return reply, lang

    def _moves_ok(self) -> int:
        """Moves the arm really started or queued this turn (a refused call,
        e.g. "arm busy", does not count)."""
        return sum(1 for n, _a, r in self.last_tool_calls
                   if n not in _NO_OP_TOOLS and not _NOT_DONE.match(r or ""))

    def respond(self, audio: np.ndarray, *,
                context: str = "",
                language_hint: str = "en",
                service_prompt: Optional[str] = None,
                augmentation: Optional[str] = None,
                history: Optional[list] = None,
                on_sentence: Optional[Callable[[str, str], None]] = None,
                ) -> tuple[str, str]:
        """Audio in, (reply_text, language) out. Blocking; one to a few
        model calls depending on tool use. Transcription for memory is the
        caller's job, in the background (see agent._on_heard_audio).

        With *on_sentence* the model streams and each sentence is handed to
        ``on_sentence(sentence, language)`` the moment it is written, to be
        spoken while the rest is still coming. A sentence claiming a physical
        action is held back until a tool has actually run; whatever the turn
        finally settles on and was not yet handed over follows at the end.
        The returned reply is then exactly what was handed over.

        *service_prompt* / *augmentation* override the constructor's persona
        and the augmentation provider for this turn (clients that already
        fetch them per language pass them in). *history* is a list of prior
        text turns ({"role": "user"|"assistant", "content": ...}) inserted
        before the audio so the model has the conversation context.
        """
        wav = _to_wav_bytes(audio, self._sample_rate)
        aug = augmentation
        if aug is None and self._augmentation_provider:
            aug = self._augmentation_provider(language_hint)
        lang_rule = None
        seconds = len(audio) / float(self._sample_rate or 16000)
        if (seconds < SHORT_UTTERANCE_S and language_hint
                and language_hint in self._LANG_NAMES):
            # A one-word "yeah" / noise blip got Spanish and Italian replies at
            # the fair. Too short to switch on: stay in the conversation's
            # language unless the person explicitly asks for another one.
            lname = self._LANG_NAMES[language_hint]
            lang_rule = (f"- This utterance is very short ({seconds:.1f} s). Reply in "
                         f"{lname} (the conversation's language) unless the person "
                         f"clearly asked you to switch language.\n")
        system = self._system(service_prompt, aug, language_hint, context, lang_rule)

        messages = [{"role": "system", "content": system}]
        messages += _history_messages(history)
        messages.append({"role": "user",
                         "content": "(user speech attached as audio)",
                         "images": [wav]})

        start = time.time()
        # What this turn actually called: [(name, args, result)]. The caller
        # shows it — a demo where the arm silently does nothing looks the
        # same as one where it works, so this must not be INFO-only.
        self.last_tool_calls = []
        self.rejected_tool_calls = []
        scene = parse_scene_positions(aug)
        scene_at_start = dict(scene)   # _apply_move updates `scene` as moves happen
        reply, used_tools = "", False
        texts: list[str] = []   # every piece of prose the model produced
        spoken: list[str] = []  # sentences already handed to on_sentence
        say = None
        if on_sentence is not None:
            def say(sentence: str, lang: str):
                # "Här får du en Plopp!" before the arm has moved could be a
                # lie; the end of the turn decides whether it is said.
                if self._tools and not self._moves_ok() and _is_claim(sentence):
                    return
                spoken.append(sentence)
                on_sentence(sentence, lang)

        for round_no in range(_MAX_TOOL_ROUNDS):
            msg = self._chat(messages, self._tools or None, say, language_hint or "en")
            calls = msg.get("tool_calls") or []
            content = (msg.get("content") or "").strip()
            if not calls and self._tools and _TOOL_TEXT.search(content):
                calls = self._recover_text_calls(content, scene)
                content = _strip_tool_text(content)
                msg = {"role": "assistant", "content": content, "tool_calls": calls}
            if content:
                texts.append(content)
            if not calls:
                reply = content
                break
            used_tools = True
            messages.append(msg)
            for call in calls:
                fn = call["function"]
                logger.info(f"direct: tool call {fn['name']}({dict(fn['arguments'])})")
                args = dict(fn["arguments"])
                refusal = self._reject_bad_move(fn["name"], args, scene)
                if refusal:
                    # Don't send the arm after candy that isn't there. Hand the
                    # model the reason plus the real positions so it can retry
                    # in the next round instead of miming an empty square.
                    logger.warning(f"direct: rejected {fn['name']}({args}): {refusal}")
                    self.rejected_tool_calls.append((fn["name"], args, refusal))
                    messages.append({"role": "tool", "tool_name": fn["name"],
                                     "content": refusal})
                    continue
                result = self._call_tool(fn["name"], args)
                self.last_tool_calls.append((fn["name"], args, result))
                self._apply_move(fn["name"], args, result, scene)
                messages.append({"role": "tool", "tool_name": fn["name"],
                                 "content": result})

        # A tool round can yield no prose at all. Never go silent after
        # moving the arm: fall back to earlier prose, else ask for a short
        # confirmation (text-only, cheap) — the model still has the tool
        # results in its history.
        reply = reply or (texts[-1] if texts else "")

        # Claimed an action without calling the tool: the arm never moved and
        # the visitor is being told it did. Give the model exactly one chance
        # to make good on it, with the tools still attached.
        claimed_kinds = _claimed_kinds(" ".join(texts + [reply]), scene_at_start) if reply else set()
        claimed = len(claimed_kinds)
        moved = self._moves_ok()
        is_claim = bool(reply) and _is_claim(reply)
        if self._tools and reply and not is_claim and moved == 0:
            # The phrase list only knows phrasings it has seen. Ask the model
            # itself -- any language, any wording -- whether the reply says it
            # handed something over. Runs while the reply is already playing.
            is_claim = self._judge_claims_action(reply)
            if is_claim:
                claimed_kinds = _claimed_kinds_any(reply, scene_at_start)
                claimed = len(claimed_kinds)
        if (self._tools and reply and is_claim and moved == 0 and not claimed
                and scene_at_start):
            # It "handed over" something that is not on the table (at the demo:
            # "Here is your Marianne" with no Marianne). A retry then grabbed
            # whatever was there -- a Plopp. Do not retry; say it honestly.
            logger.warning(f"direct: claimed a candy that is not on the table: {reply[:70]!r}"
                           " — no retry, honest answer")
            reply = self._honest_reply(messages + [{"role": "assistant", "content": reply}],
                                       language_hint or "en")
        elif self._tools and reply and is_claim and (moved == 0 or claimed > moved):
            candy_arm = any(t.get("function", {}).get("name") == "move_between"
                            for t in self._tools or [])
            if moved == 0 and not candy_arm:
                # Any service (lights, music, ...): the reply says something was
                # done, but no tool ran.
                logger.warning(f"direct: action claimed without a tool call: {reply[:70]!r}"
                               " — retrying")
                nudge = ("You just told the user you did something, but you called no "
                         "tool (or the tool refused), so NOTHING happened. If you meant "
                         "to do it, call the right tool NOW for exactly that action. "
                         "Then answer in one sentence starting with the ISO language "
                         "code in brackets -- and if you did not call a tool, say "
                         "honestly that it was not done.")
            elif moved == 0:
                logger.warning(f"direct: action claimed without a move: {reply[:70]!r}"
                               " — retrying")
                nudge = ("You just told the person you were moving or handing over "
                         "candy, but no move happened (no tool call, or the arm "
                         "refused it), so nothing happened. Call move_between NOW "
                         "for exactly that action, then confirm in one sentence "
                         "starting with the ISO language code in brackets.")
            else:
                logger.warning(f"direct: {claimed} candies claimed but {moved} moved"
                               f" — retrying for the rest: {reply[:70]!r}")
                nudge = (f"You told the person you are handing over {claimed} "
                         f"candies but only {moved} move(s) happened. Call "
                         f"move_between NOW for each candy that was not moved yet "
                         f"(one call per candy), then confirm in one sentence "
                         f"starting with the ISO language code in brackets. If you "
                         f"only meant to hand over one candy, call no tool and just "
                         f"answer.")
            had_moves = moved
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": nudge})
            for _ in range(2):
                # Not streamed: its words count only if a tool ran (below),
                # so the end of the turn decides what of it is said.
                m2 = self._chat(messages, self._tools or None, None, language_hint or "en")
                calls2 = m2.get("tool_calls") or []
                content2 = (m2.get("content") or "").strip()
                if not calls2 and _TOOL_TEXT.search(content2):
                    calls2 = self._recover_text_calls(content2, scene)
                    content2 = _strip_tool_text(content2)
                    m2 = {"role": "assistant", "content": content2, "tool_calls": calls2}
                if not calls2:
                    if _has_words(content2) and self._moves_ok() > had_moves:
                        reply = content2
                    break
                messages.append(m2)
                for call in calls2:
                    fn = call["function"]
                    args = dict(fn["arguments"])
                    refusal = self._reject_bad_move(fn["name"], args, scene)
                    if not refusal and fn["name"] == "move_between" and claimed_kinds:
                        src = str(args.get("src", "")).upper().strip()
                        kind = (scene_at_start.get(src) or "").split(",")[0].strip().lower()
                        if kind not in claimed_kinds:
                            refusal = (f"Refused: you promised {', '.join(sorted(claimed_kinds))}, "
                                       f"but {src} holds {kind or 'nothing'}. Never hand over a "
                                       f"different candy than the one promised.")
                    if refusal:
                        self.rejected_tool_calls.append((fn["name"], args, refusal))
                        messages.append({"role": "tool", "tool_name": fn["name"],
                                         "content": refusal})
                        continue
                    logger.info(f"direct: recovered tool call {fn['name']}({args})")
                    result = self._call_tool(fn["name"], args)
                    self.last_tool_calls.append((fn["name"], args, result))
                    self._apply_move(fn["name"], args, result, scene)
                    messages.append({"role": "tool", "tool_name": fn["name"],
                                     "content": result})
                # Only take the retry's words if it produced any: after a
                # recovered tool call gemma often answers with the bare
                # language tag ("[sv]"), and the sentence it already said in
                # round 1 is now TRUE, so keeping it beats going silent.
                if _has_words(content2):
                    reply = content2
            if self._moves_ok() == had_moves == 0:
                # Never tell the visitor they got something they did not. At
                # the demo "Here is your Marianne" was said with no Marianne on
                # the table and nothing moved.
                logger.warning("direct: retry made no move — replacing the claim "
                               "with an honest answer")
                reply = self._honest_reply(messages, language_hint or "en")
            elif self._moves_ok() == had_moves:
                logger.warning("direct: retry made no move — the claim stands "
                               "but the arm did not move")

        language, reply = _split_lang(reply, language_hint or "en")

        # Never go silent, especially right after moving the arm: a visitor
        # who gets candy and hears nothing assumes it is broken. Checked on
        # the tag-stripped text, since a bare "[sv]" is not an answer.
        if not reply and self.last_tool_calls:
            messages.append({"role": "user", "content":
                             "Tell the person in ONE short sentence what you "
                             "just did, in their spoken language, starting "
                             "with its ISO code in brackets."})
            msg = self._chat(messages, None, say, language)
            language, reply = _split_lang((msg.get("content") or "").strip(), language)
        if on_sentence is not None:
            # What the turn settled on but was not said yet: held claims that
            # turned out true (or stand after a failed retry), a fallback to
            # earlier prose. Then report what was actually said.
            for sentence in split_sentences(reply):
                if sentence not in spoken:
                    spoken.append(sentence)
                    on_sentence(sentence, language)
            reply = " ".join(spoken)
        elapsed = time.time() - start

        logger.info(f"direct: reply in {elapsed:.2f}s lang={language}: {reply}")
        return reply, language
