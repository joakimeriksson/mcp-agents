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
from a service MCP server and executed over short-lived SSE sessions,
mirroring ``ServiceHost``.

Nothing here knows about a particular service. What a service's tools mean --
which positions exist, which call hands something over, what to say when
nothing happened -- comes from its ``ServicePolicy`` (face/service_policy.py;
CandyTron's is candytron_mcp/client_policy.py).
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
from service_policy import GENERIC, ServicePolicy

logger = logging.getLogger("direct_llm")

# System prompt: the face-agent identity + the service role, with the rules the
# direct path needs (language tag, tools, brevity). Mirrors llm.SYSTEM_PROMPT.
import prompts as _prompts

# The rules text lives in face/prompts.py and can be edited live (launcher's
# "Prompts..." window); this is the default, kept for reference and tests.
DIRECT_SYSTEM = _prompts.RULES

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
# Also "movebetween(src='B2', dst='O0')" -- parentheses, but only for KNOWN
# tool names (ordinary speech has parentheses too). The names are the tools
# actually loaded; set_known_tools() rebuilds the patterns.
_TOOL_TEXT = _TOOL_TEXT_CALL = None


def _name_pattern(name: str) -> str:
    """'move_between' -> 'move_?between' (gemma4 drops the underscores)."""
    return "_?".join(re.escape(p) for p in name.split("_"))


def set_known_tools(names) -> None:
    """(Re)build the written-call patterns for these tool names."""
    global _TOOL_TEXT, _TOOL_TEXT_CALL, _JUNK_TAG
    alts = "|".join(sorted({_name_pattern(n) for n in names if n}, key=len, reverse=True))
    known = rf"(?:{alts})\s*\([^()]*\)" if alts else r"(?!x)x"
    _TOOL_TEXT = re.compile(r"\[tool_?call\]|<\|tool_call>|<tool_call\|>|<\|\"\|>|\w+\s*\{[^{}]*\}|[{}]|" + known,
                            re.IGNORECASE)
    _TOOL_TEXT_CALL = re.compile(r"(?:\[tool_?call\]|<\|tool_call>|\bcall:)?\s*(?:call:)?\s*"
                                 r"(?:([a-z_]+)\s*\{([^{}]*)\}|"
                                 rf"({alts or '(?!x)x'})\s*\(([^()]*)\))",
                                 re.IGNORECASE)
    _JUNK_TAG = re.compile(rf"\[\s*(?:tool[^\]]*|{alts or '(?!x)x'}|[a-z]{{2}})\s*\]", re.IGNORECASE)


# Until tools are loaded: the local express tool plus the classic names, so
# a service-less client still never reads a written call aloud.
set_known_tools(["express", "default_action"])
_TOOL_TEXT_ARG = re.compile(r"(\w+)\s*[:=]\s*(?:<\|\"\|>(.*?)<\|\"\|>|\"([^\"]*)\"|'([^']*)'|([^,})\s]+))")


# Optional local tool: the model sets the voice style with a real function
# call instead of writing "[glad]" tags inline (inline [tag] markup is what
# made gemma4 start writing its arm calls as text). Never sent to the MCP
# server; it only prefixes the matching style tag to the next spoken sentence.
EXPRESS_TAGS = {"laugh": "[skrattar]", "happy": "[glad]",
                "whisper": "[viskar]", "thinking": "[tänker]"}
EXPRESS_TOOL = {
    "type": "function",
    "function": {
        "name": "express",
        "description": ("Optional: set HOW your spoken reply sounds. Call at most once, "
                        "together with any other tool call, never instead of one. "
                        "laugh = after something actually funny; happy = excited; "
                        "whisper = telling a secret; thinking = thinking out loud."),
        "parameters": {"type": "object",
                       "properties": {"mood": {"type": "string",
                                               "enum": list(EXPRESS_TAGS)}},
                       "required": ["mood"]},
    },
}


# No Kokoro voice for these: a style tag routes the sentence to VoxCPM2 on
# the voice server, which speaks them properly (a Swedish Kokoro voice read
# Dutch as broken Swedish).
VOX_LANGS = ("nl", "pt", "da")
VOX_NEUTRAL_TAG = "[speaking naturally]"


def _express_tag(call) -> str:
    args = call["function"].get("arguments") or {}
    return EXPRESS_TAGS.get(str(args.get("mood", "")).lower().strip(), "")


_MOOD_WORDS = {"laugh": "laugh", "laughing": "laugh", "skrattar": "laugh",
               "happy": "happy", "glad": "happy", "excited": "happy",
               "whisper": "whisper", "viskar": "whisper",
               "thinking": "thinking", "tänker": "thinking", "think": "thinking"}
# express() written as text: "express(mood: laugh)", "[express:happy]", "express{mood:happy}"
_EXPRESS_TEXT = re.compile(r"\[?\s*express\s*[(:{\[]\s*(?:mood\s*[:=]\s*)?[\"']?(\w+)[\"']?\s*[)}\]]?\]?",
                           re.IGNORECASE)
# _JUNK_TAG (built by set_known_tools above): other markup gemma4 leaks next to
# tool calls -- "[tool: ]", "[defaultaction]", "[movebetween]", a mid-text "[en]".
_BRACKET_MOOD = re.compile(r"\[\s*(\w+)\s*\]")


def _clean_spoken(sentence: str) -> tuple:
    """(text safe to speak, mood or '') -- for express-tool mode, where the
    model sometimes writes the mood (or other tool syntax) instead of calling it."""
    mood = ""
    m = _EXPRESS_TEXT.search(sentence)
    if m:
        mood = _MOOD_WORDS.get(m.group(1).lower(), "")
        sentence = _EXPRESS_TEXT.sub(" ", sentence)
    def _bm(mm):
        nonlocal mood
        w = _MOOD_WORDS.get(mm.group(1).lower())
        if w:
            mood = mood or w
            return " "
        return mm.group(0)
    sentence = _BRACKET_MOOD.sub(_bm, sentence)
    sentence = _JUNK_TAG.sub(" ", sentence)
    sentence = _EMPTY_TAG.sub(" ", sentence)
    sentence = re.sub(r"\*+([^*]+)\*+", r"\1", sentence).replace("*", "")
    sentence = re.sub(r"\s+", " ", sentence).strip()
    sentence = re.sub(r"^[!?.,;:\s]+", "", sentence)
    if mood:   # "express(mood: laugh) en Why did..." leaves a bare language code
        sentence = re.sub(r"^(?:sv|en|de|fr|es|it)\b[:\s]*", "", sentence)
    return sentence, mood


def _text_tool_calls(content: str, tool_names: list, not_recovered=frozenset()) -> list:
    """Tool calls the model wrote as text, as Ollama-style call dicts."""
    norm = {n.replace("_", "").lower(): n for n in tool_names}
    calls = []
    for m in _TOOL_TEXT_CALL.finditer(content or ""):
        raw_name, raw_args = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        name = norm.get(raw_name.replace("_", "").lower())
        if not name or name in not_recovered:
            continue
        args = {}
        for a in _TOOL_TEXT_ARG.finditer(raw_args):
            args[a.group(1)] = next(v for v in a.groups()[1:] if v is not None)
        calls.append({"function": {"name": name, "arguments": args}})
    return calls


def has_tool_text(content: str) -> bool:
    """True if *content* contains a tool call written as text. (A function,
    not the pattern: the pattern is rebuilt when tools are loaded.)"""
    return bool(_TOOL_TEXT.search(content or ""))


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


# Generic, for any service: a tool call counts as DONE unless its result says
# it was refused, failed, or did nothing.
_NOT_DONE = re.compile(r"^\s*(refused|failed|error|did nothing|tool \S+ failed|unknown)",
                       re.IGNORECASE)


_LANG_TAG = re.compile(r"^\s*\[([a-z]{2}(?:-[a-z]{2})?)\]\s*", re.IGNORECASE)
_MAX_TOOL_ROUNDS = 3
# Keep gemma4 in memory for the whole demo. Ollama's default unloads it after
# 5 idle minutes, and the first visitor after a quiet spell then waits for a
# reload before the robot says anything.
KEEP_ALIVE = -1
# Utterances shorter than this carry too little language signal to switch on.
SHORT_UTTERANCE_S = 1.0


_LEAD_TAGS = re.compile(r"^\s*(?:\[[^\[\]]{1,24}\]\s*)+")


_EMPTY_TAG = re.compile(r"\s*\[\s*\]\s*")


def _split_lang(text: str, default: str) -> tuple:
    """('sv', 'Hej!') from '[sv] Hej!'; the tag is never spoken. The language
    tag may come after a mood tag ("[glad] [en] Here...") -- the mood tag is
    kept, only the language tag is removed. An empty tag ("[ ] Varsågod")
    -- gemma4 sometimes leaves one -- is dropped."""
    text = _EMPTY_TAG.sub(" ", text or "").strip()
    m = _LANG_TAG.match(text)
    if m:
        return m.group(1).lower(), text[m.end():].strip()
    lead = _LEAD_TAGS.match(text)
    if lead:
        tags = re.findall(r"\[([^\[\]]{1,24})\]", lead.group(0))
        for t in tags:
            if re.fullmatch(r"[a-z]{2}(?:-[a-z]{2})?", t.strip(), re.IGNORECASE):
                rest = [f"[{x}]" for x in tags if x is not t]
                body = (" ".join(rest) + " " + text[lead.end():]).strip()
                return t.strip().lower(), body
    return default, text.strip()


def _is_claim(text: str) -> bool:
    """The robot saying it did something physical (see _ACTION_CLAIM)."""
    return bool(_ACTION_CLAIM.search(text)) and not _NOT_A_CLAIM.search(text)


def _claim_sentences(text: str) -> list:
    """The sentences of *text* that claim an action (see _ACTION_CLAIM)."""
    return [x for x in split_sentences(text or "") if _is_claim(x)]


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

    # Defaults (also for tests that build the object with __new__).
    policy: ServicePolicy = GENERIC
    express_tool = False   # offer the local express(mood) tool
    lang_id = None         # lang_id.LanguageID: Whisper decides the reply language
    last_lid = (None, 0.0)
    last_mood = ""         # style tag the model chose this turn
    # "gemma4": the audio goes into gemma4. "whisper": Whisper (MLX, GPU)
    # transcribes it and gemma4 answers the TEXT.
    ear = "gemma4"
    last_heard = None      # lang_id.Heard of this turn (ear == "whisper")
    # Languages the VOICE can speak: Kokoro's six + what VoxCPM2 speaks well
    # (tested Nov 2026 by synthesizing and transcribing back: nl/pt/da
    # good, fi garbled). Anything else gets a reply in English.
    speakable = ("en", "sv", "de", "fr", "es", "it") + VOX_LANGS
    service_rules = ""     # the service's get_service_rules text

    def __init__(self, *,
                 model: str = "gemma4:latest",
                 host: str = "http://localhost:11434",
                 agent_name: str = "Face Agent",
                 service_prompt: Optional[str] = None,
                 augmentation_provider: Optional[Callable[[str], Optional[str]]] = None,
                 tools_url: Optional[str] = None,
                 policy: Optional[ServicePolicy] = None,
                 sample_rate: int = 16000,
                 temperature: float = 0.2):
        self.last_tool_calls: list = []
        self.rejected_tool_calls: list = []
        # What the service's policy found worth remembering this turn
        # ("was given a Dumle"); the client keeps it in its visit memory.
        self.last_notes: list = []
        import ollama
        self._client = ollama.Client(host=host)
        self._model = model
        # Low temperature: tool calling on audio turns is otherwise flaky —
        # the same request sometimes calls the tool, sometimes just says it
        # did (measured Aug 2026).
        self._options = {"temperature": temperature}
        self._agent_name = agent_name
        self._service_prompt = service_prompt
        self._augmentation_provider = augmentation_provider
        self._tools_url = tools_url
        self._sample_rate = sample_rate
        self._tools: list = []
        self.policy = policy or GENERIC
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
        names = [t["function"]["name"] for t in mapped]
        set_known_tools(names + ["express", "default_action"])
        logger.info(f"direct: {len(mapped)} tool(s) from {url}: {names}")
        return mapped

    def _record(self, name: str, args: dict, result: str, state: dict) -> None:
        """A call the service ran: keep it, let the policy note what is
        worth remembering about the visitor (on the state the call acted
        on), then let it update the state."""
        self.last_tool_calls.append((name, args, result))
        try:
            note = self.policy.remember(name, args, result, state)
        except Exception as e:  # noqa: BLE001 - a policy bug must not break the turn
            logger.warning(f"direct: policy.remember failed: {e}")
            note = None
        if note:
            self.last_notes.append(note)
        self.policy.apply(name, args, result, state)

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

    # With the person's words (Whisper ear): catches AGREEMENTS -- "Can I have
    # a Plopp?" -> "Of course, you can have a Plopp!" with no move. The
    # reply-only judge above missed 6/6 of those (Nov 2026, gemma4); this one
    # got 17/18 over request/answer/offer/question pairs, and its one miss
    # was the safe kind (a demo request judged NO, never a question judged YES).
    _JUDGE_REQUEST = ("An assistant acts in the real world ONLY through these tools:\n"
                      "{tools}\n\n"
                      "The person said:\n\"{request}\"\n\n"
                      "The assistant replied:\n\"{reply}\"\n\n"
                      "Answer YES only if BOTH are true: (1) the person asked for a specific "
                      "thing to be done for them NOW (\"can I have X\", \"give me X\", "
                      "\"I want X\"), and (2) the reply agrees to do it or says it is done "
                      "or being done. Answer NO if the person asked a question about what "
                      "exists or what the assistant can do, or if the reply only answers, "
                      "offers (\"would you like one?\"), asks back, or refuses. Answer with "
                      "one word: YES or NO.")

    def _tool_summary(self) -> str:
        lines = []
        for t in self._tools or []:
            fn = t.get("function", {})
            if fn.get("name") in self.policy.noop_tools:
                continue
            desc = (fn.get("description") or "").strip().splitlines()
            lines.append(f"- {fn.get('name')}: {desc[0][:120] if desc else ''}")
        return "\n".join(lines) or "- (none)"

    def _judge_claims_action(self, reply: str, request: Optional[str] = None) -> bool:
        """Language-independent check: does *reply* claim (or agree to) an
        action that needs a tool? With *request* (what the person said) it also
        catches a "yes, you can have one" that never called the tool."""
        text = _split_lang(_strip_tool_text(reply), "")[1]
        if not text:
            return False
        if request:
            prompt = self._JUDGE_REQUEST.format(tools=self._tool_summary(),
                                                request=request[:300], reply=text[:400])
        else:
            prompt = self._JUDGE.format(tools=self._tool_summary(), reply=text[:400])
        t0 = time.time()
        try:
            out = self._client.chat(model=self._model, messages=[{"role": "user", "content":
                                    prompt}],
                                    options={"temperature": 0, "num_predict": 3}, think=False,
                                    stream=False, keep_alive=KEEP_ALIVE)["message"]["content"]
        except Exception as e:  # noqa: BLE001
            logger.warning(f"direct: action judge failed: {e}")
            return False
        verdict = (out or "").strip().upper().startswith(("YES", "JA"))
        logger.info(f"direct: action judge{' (with request)' if request else ''} "
                    f"{'YES' if verdict else 'NO'} in "
                    f"{time.time() - t0:.2f}s for {text[:60]!r}")
        if verdict:
            logger.warning(f"direct: judge says the reply claims an action: {text[:70]!r}")
        return verdict

    _NOTHING_DONE = {
        "sv": "[sv] Förlåt, det gick inte.",
        "en": "[en] Sorry, that didn't work.",
        "de": "[de] Entschuldigung, das hat nicht geklappt.",
        "fr": "[fr] Désolé, ça n'a pas marché.",
        "es": "[es] Lo siento, no ha funcionado.",
        "it": "[it] Scusa, non ha funzionato.",
    }

    def _honest_reply(self, messages: list, language: str) -> str:
        """A truthful one-liner after a claimed action that did not happen:
        say so, and if what they asked for is not available, what is."""
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
            text = (self.policy.fallback_reply(language)
                    or self._NOTHING_DONE.get(language, self._NOTHING_DONE["en"]))
        return text

    def _recover_text_calls(self, content: str, scene: dict) -> list:
        """Execute what the model wrote as a tool call instead of making one."""
        calls = _text_tool_calls(content, [t["function"]["name"] for t in self._tools],
                                 self.policy.not_recovered)
        kept = self.policy.dedupe_text_calls(calls, scene)
        logger.warning(f"direct: model wrote {len(calls)} tool call(s) as text; "
                       f"executing {len(kept)}: {[c['function'] for c in kept]}")
        return kept

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
            for c in msg.get("tool_calls") or []:
                if c["function"]["name"] == "express":
                    self.last_mood = self.last_mood or _express_tag(c)
            calls += msg.get("tool_calls") or []
            content += msg.get("content") or ""
            if lang is None:
                head = content.lstrip()
                if head.startswith("[") and len(head) < 40:
                    # the leading tags are still being written: wait until the
                    # text after them has started ("[glad] [en] Here")
                    lead = _LEAD_TAGS.match(head)
                    if (lead is None or lead.end() >= len(head)
                            or head[lead.end():].lstrip().startswith("[")):
                        continue
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
                               service=service, lang_rule=lang_rule,
                               service_rules=(self.service_rules or "").rstrip() + "\n"
                               if self.service_rules else "")
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
              context: str = "",
              ) -> tuple[str, str]:
        """Text-only greeting turn, streamed sentence by sentence and without
        tools (a greeting never moves the arm), so the first words are heard
        well under a second after contact. ``context``: what we remember
        about the person (e.g. the visit memory's "Earlier: ..." line)."""
        lname = self._LANG_NAMES.get(language, "Swedish")
        lang_rule = (f"- Reply in {lname} (tag [{language}]) unless the person "
                     f"spoke another language earlier in the conversation.\n")
        system = self._system(service_prompt, augmentation, language, context, lang_rule)
        messages = [{"role": "system", "content": system}]
        messages += _history_messages(history)
        messages.append({"role": "user", "content": instruction})
        self.last_tool_calls, self.rejected_tool_calls, self.last_notes = [], [], []
        start = time.time()
        msg = self._chat(messages, None, on_sentence, language)
        lang, reply = _split_lang((msg.get("content") or "").strip(), language)
        logger.info(f"direct: greeting in {time.time() - start:.2f}s lang={lang}: {reply}")
        return reply, lang

    def _styled(self, sentence: str, spoken: list, lang: str = "") -> str:
        """The express() style tag goes on the FIRST spoken sentence only;
        what is recorded (spoken / history) stays untagged. A language with
        no Kokoro voice gets a neutral tag on every sentence (-> VoxCPM2)."""
        if self.last_mood and len(spoken) == 1:
            return f"{self.last_mood} {sentence}"
        if lang in VOX_LANGS and not sentence.lstrip().startswith("["):
            return f"{VOX_NEUTRAL_TAG} {sentence}"
        return sentence

    def _lang_order(self, code: str) -> str:
        """The LANGUAGE rule for a detected spoken language."""
        from lang_id import language_name
        tail = self.policy.language_tail()
        if code not in self.speakable:
            heard = language_name(code)
            return (f"- LANGUAGE: the person is speaking {heard}, which your voice cannot "
                    f"speak. Reply in simple English, starting with [en].{tail}")
        name = language_name(code)
        return (f"- LANGUAGE: the person is speaking {name}. Reply in {name}, starting "
                f"with [{code}] -- whatever language the earlier turns or the "
                f"information above are in.{tail}")

    def _moves_ok(self) -> int:
        """Actions really started or queued this turn (a refused call, e.g.
        "arm busy", does not count)."""
        pol = self.policy
        return sum(1 for n, _a, r in self.last_tool_calls
                   if n not in pol.noop_tools
                   and (not pol.action_tools or n in pol.action_tools)
                   and not _NOT_DONE.match(r or ""))

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
        self.last_lid = (None, 0.0)
        self.last_heard = None
        heard = None
        if self.ear == "whisper" and self.lang_id is not None and self.lang_id.available:
            from lang_id import language_name
            heard = self.lang_id.transcribe(
                audio, self.policy.hint_words(self.policy.parse_state(aug)))
            fixed = self.policy.normalize_heard(heard.text, self.policy.parse_state(aug)) \
                if heard.text else heard.text
            if fixed != heard.text:
                logger.info(f"direct: heard {heard.text!r} -> {fixed!r} (service vocabulary)")
                heard = heard._replace(text=fixed)
            self.last_heard = heard
            if not heard.text:
                # Noise, a cough, crowd babble: say nothing rather than answer it.
                logger.info(f"direct: whisper heard nothing usable ({heard.reason})")
                self.last_tool_calls, self.rejected_tool_calls, self.last_notes = [], [], []
                return "", language_hint or "en"
            code = heard.language
            prob = 1.0
            if code and code != language_hint:
                # A SWITCH needs a confident language id (~0.12 s, only on a
                # switch): accented speech gave Dutch -> "Norwegian" at 0.37
                # and Finnish -> "Swedish" at 0.45. Unsure: gemma4 decides
                # from the text itself.
                code2, prob = self.lang_id.detect(audio)
                if code2 != code:
                    prob = min(prob, 0.5)
            self.last_lid = (code, prob)
            # Stickiness: a 1-2 word utterance only switches language when
            # Whisper agrees with the conversation anyway.
            if code and prob >= 0.6 and (len(heard.text.split()) >= 3 or code == language_hint):
                lang_rule = self._lang_order(code)
            logger.info(f"direct: whisper heard [{code}] {heard.text!r}")
        elif self.lang_id is not None:
            # gemma4's ear guesses short Swedish as English; Whisper's language
            # head does not (~0.12 s on the GPU). A confident answer becomes an
            # ORDER in the prompt. Unsure (noise, a mumble): leave it to gemma4.
            from lang_id import language_name
            code, prob = self.lang_id.detect(audio)
            self.last_lid = (code, prob)
            if code and prob >= 0.6 and (seconds >= SHORT_UTTERANCE_S or prob >= 0.9
                                         or code == language_hint):
                lang_rule = self._lang_order(code)
            logger.info(f"direct: lang-id {code} p={prob:.2f} -> "
                        f"{'ordered' if lang_rule and code and code in lang_rule else 'model decides'}")
        system = self._system(service_prompt, aug, language_hint, context, lang_rule)
        if heard is not None:
            system = system.replace("their speech is attached\nas audio",
                                    "their speech reaches you as text from speech recognition")

        messages = [{"role": "system", "content": system}]
        messages += _history_messages(history)
        if heard is not None:
            messages.append({"role": "user", "content":
                             f'The person said (speech recognition, may contain small '
                             f'errors): "{heard.text}"'})
        else:
            messages.append({"role": "user",
                             "content": "(user speech attached as audio)",
                             "images": [wav]})

        start = time.time()
        # What this turn actually called: [(name, args, result)]. The caller
        # shows it — a demo where the arm silently does nothing looks the
        # same as one where it works, so this must not be INFO-only.
        self.last_tool_calls = []
        self.rejected_tool_calls = []
        self.last_notes = []
        scene = self.policy.parse_state(aug)
        scene_at_start = dict(scene)   # policy.apply updates `scene` as calls happen
        reply, used_tools = "", False
        texts: list[str] = []   # every piece of prose the model produced
        spoken: list[str] = []  # sentences already handed to on_sentence
        say = None
        if on_sentence is not None:
            def say(sentence: str, lang: str):
                if self.express_tool:
                    sentence, mood = _clean_spoken(sentence)
                    if mood and not self.last_mood and not spoken:
                        self.last_mood = EXPRESS_TAGS[mood]
                    if not _has_words(sentence):
                        return
                # "Här får du en Plopp!" before the arm has moved could be a
                # lie; the end of the turn decides whether it is said.
                if self._tools and not self._moves_ok() and _is_claim(sentence):
                    return
                spoken.append(sentence)
                on_sentence(self._styled(sentence, spoken, lang), lang)

        self.last_mood = ""
        tools = list(self._tools or [])
        if self.express_tool:
            tools.append(EXPRESS_TOOL)
        for round_no in range(_MAX_TOOL_ROUNDS):
            msg = self._chat(messages, tools or None, say, language_hint or "en")
            calls = msg.get("tool_calls") or []
            content = (msg.get("content") or "").strip()
            if not calls and self._tools and _TOOL_TEXT.search(content):
                calls = self._recover_text_calls(content, scene)
                content = _strip_tool_text(content)
                msg = {"role": "assistant", "content": content, "tool_calls": calls}
            # express() is local: it only styles the voice, never costs a round.
            moods = [c for c in calls if c["function"]["name"] == "express"]
            if moods:
                self.last_mood = self.last_mood or _express_tag(moods[0])
                calls = [c for c in calls if c["function"]["name"] != "express"]
                msg = {"role": "assistant", "content": content, "tool_calls": calls or None}
                if not calls and not content:
                    # Only a mood, no words yet: ask again, without the tool.
                    tools = [t for t in tools if t is not EXPRESS_TOOL]
                    continue
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
                refusal = self.policy.reject(fn["name"], args, scene)
                if refusal:
                    # e.g. the arm sent after candy that isn't there. Hand the
                    # model the reason (the policy adds the real state) so it
                    # can retry in the next round.
                    logger.warning(f"direct: rejected {fn['name']}({args}): {refusal}")
                    self.rejected_tool_calls.append((fn["name"], args, refusal))
                    messages.append({"role": "tool", "tool_name": fn["name"],
                                     "content": refusal})
                    continue
                result = self._call_tool(fn["name"], args)
                self._record(fn["name"], args, result, scene)
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
        claimed_kinds = (self.policy.claimed_items(_claim_sentences(" ".join(texts + [reply])),
                                                   scene_at_start) if reply else set())
        claimed = len(claimed_kinds)
        moved = self._moves_ok()
        is_claim = bool(reply) and _is_claim(reply)
        if self._tools and reply and not is_claim and moved == 0:
            # The phrase list only knows phrasings it has seen. Ask the model
            # itself -- any language, any wording -- whether the reply says it
            # handed something over. Runs while the reply is already playing.
            # With a transcript it also sees the request, so an agreement
            # ("Of course, you can have a Plopp!") counts too.
            is_claim = self._judge_claims_action(reply, heard.text if heard is not None else None)
            if is_claim:
                claimed_kinds = self.policy.claimed_items([reply], scene_at_start)
                claimed = len(claimed_kinds)
                if not claimed and scene_at_start:
                    # The judge's YES names nothing the service has: too weak
                    # to replace a possibly fine answer with "that did not
                    # work" (the phrase list's own claims still are).
                    logger.info("direct: judge YES but nothing in the service state "
                                f"is named -- leaving the reply: {reply[:60]!r}")
                    is_claim = False
        if (self._tools and reply and is_claim and moved == 0 and not claimed
                and scene_at_start):
            # It claimed acting on something the service does not have (at the demo:
            # "Here is your Marianne" with no Marianne). A retry then grabbed
            # whatever was there -- a Plopp. Do not retry; say it honestly.
            logger.warning(f"direct: claimed something not in the service state: {reply[:70]!r}"
                           " — no retry, honest answer")
            reply = self._honest_reply(messages + [{"role": "assistant", "content": reply}],
                                       language_hint or "en")
        elif self._tools and reply and is_claim and (moved == 0 or claimed > moved):
            generic = ("You just told the user you did something, but you called no "
                       "tool (or the tool refused), so NOTHING happened. If you meant "
                       "to do it, call the right tool NOW for exactly that action. "
                       "Then answer in one sentence starting with the ISO language "
                       "code in brackets -- and if you did not call a tool, say "
                       "honestly that it was not done.")
            if moved == 0:
                logger.warning(f"direct: action claimed without a tool call: {reply[:70]!r}"
                               " — retrying")
                nudge = self.policy.nudge_no_action() or generic
            else:
                logger.warning(f"direct: {claimed} items claimed but {moved} done"
                               f" — retrying for the rest: {reply[:70]!r}")
                nudge = self.policy.nudge_partial(claimed, moved) or generic
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
                    refusal = (self.policy.reject(fn["name"], args, scene)
                               or self.policy.retry_refusal(fn["name"], args, claimed_kinds,
                                                            scene_at_start))
                    if refusal:
                        self.rejected_tool_calls.append((fn["name"], args, refusal))
                        messages.append({"role": "tool", "tool_name": fn["name"],
                                         "content": refusal})
                        continue
                    logger.info(f"direct: recovered tool call {fn['name']}({args})")
                    result = self._call_tool(fn["name"], args)
                    self._record(fn["name"], args, result, scene)
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

        raw_reply_for_lang = reply or ""
        language, reply = _split_lang(reply, language_hint or "en")
        if not re.search(r"\[[a-z]{2}\]", raw_reply_for_lang, re.IGNORECASE) and _has_words(reply):
            # No language tag at all ("Hello there, ..." in a Swedish
            # conversation): judge from the words, or the conversation never
            # switches and the fillers stay Swedish while it talks English.
            try:
                from reply_language import reply_language
                detected, _conf = reply_language(re.sub(r"\[[^\]]*\]", " ", reply),
                                                 language, self._LANG_NAMES.keys())
                language = detected or language
            except Exception as e:  # noqa: BLE001
                logger.debug(f"direct: language detection failed: {e}")
        if self.express_tool:
            reply = _clean_spoken(reply)[0]

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
                    on_sentence(self._styled(sentence, spoken, language), language)
            reply = " ".join(spoken)
        elapsed = time.time() - start

        logger.info(f"direct: reply in {elapsed:.2f}s lang={language}: {reply}")
        return reply, language
