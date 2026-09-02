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

logger = logging.getLogger("direct_llm")

# System prompt: the face-agent identity + the service role, with the rules the
# direct path needs (language tag, tools, brevity). Mirrors llm.SYSTEM_PROMPT.
DIRECT_SYSTEM = """\
You are {name}, a camera-based assistant that can see, hear, and speak.
You are listening to a person through a microphone; their speech is attached
as audio. You remember people you've met.
{service}
Rules:
- Begin your reply with the ISO language code of the SPOKEN language in
  square brackets, e.g. [sv] or [en], then reply in that same language.
{lang_rule}- Reply in 1-2 short sentences. No markdown or emojis.
- ACT, do not narrate. If the person asks for candy, or asks you to move,
  fetch, hand out or give anything, you MUST call the tool in THIS turn
  before you answer. Never say "I can", "I could", "let me" or "I will" —
  those are forbidden; call the tool and then say what you DID.
  To hand candy to the person, move it to position O0.
- But only act on what you actually HEARD. If the audio was unclear, or you
  cannot tell WHICH candy they mean, ask them to say it again — do NOT call a
  tool and do NOT move a random candy. A wrong move is worse than a question.
"""

# Positions the vision system reports, e.g. "A2 : Riesen-kola, brun ...".
_SCENE_POS = re.compile(r"^\s*([A-Z]\d)\s*:\s*(.+?)\s*$", re.MULTILINE)

# The robot claiming it did something physical. If a turn ends with one of
# these and NO tool was called, the arm never moved and the visitor is being
# told a lie -- the single worst failure mode at a stand.
_ACTION_CLAIM = re.compile(
    r"\b("
    r"jag (flyttar|tar|ger|hämtar|lägger|placerar|skickar)"
    r"|här (får|har) du|varsågod|nu (flyttar|ger|tar) jag|jag har (flyttat|gett|tagit)"
    r"|i (moved|gave|placed|took|put)|i'?m (moving|giving|getting|taking)"
    r"|here (you go|it is)|moving it|i'?ll (move|give|get|take)"
    r"|ich (bewege|gebe|nehme)|je (déplace|donne|prends)"
    r")\b", re.IGNORECASE)


def parse_scene_positions(augmentation: Optional[str]) -> dict:
    """{position: description} from the vision system's scene text."""
    if not augmentation:
        return {}
    return {m.group(1): m.group(2) for m in _SCENE_POS.finditer(augmentation)}


_LANG_TAG = re.compile(r"^\s*\[([a-z]{2}(?:-[a-z]{2})?)\]\s*", re.IGNORECASE)
_MAX_TOOL_ROUNDS = 3


def _split_lang(text: str, default: str) -> tuple:
    """('sv', 'Hej!') from '[sv] Hej!'; the tag is never spoken."""
    m = _LANG_TAG.match(text or "")
    if not m:
        return default, (text or "").strip()
    return m.group(1).lower(), text[m.end():].strip()


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

    # --- Hot path ---

    _LANG_NAMES = {"en": "English", "sv": "Swedish", "de": "German",
                   "fr": "French", "es": "Spanish", "it": "Italian"}

    def respond(self, audio: np.ndarray, *,
                context: str = "",
                language_hint: str = "en",
                service_prompt: Optional[str] = None,
                augmentation: Optional[str] = None,
                history: Optional[list] = None) -> tuple[str, str]:
        """Audio in, (reply_text, language) out. Blocking; one to a few
        model calls depending on tool use. Transcription for memory is the
        caller's job, in the background (see agent._on_heard_audio).

        *service_prompt* / *augmentation* override the constructor's persona
        and the augmentation provider for this turn (clients that already
        fetch them per language pass them in). *history* is a list of prior
        text turns ({"role": "user"|"assistant", "content": ...}) inserted
        before the audio so the model has the conversation context.
        """
        wav = _to_wav_bytes(audio, self._sample_rate)

        service = ""
        persona = service_prompt if service_prompt is not None else self._service_prompt
        if persona:
            service = f"\nYour service role:\n{persona}\n"
        aug = augmentation
        if aug is None and self._augmentation_provider:
            aug = self._augmentation_provider(language_hint)
        if aug:
            service += (f"\nCurrent state from your service "
                        f"(internal — never read it out verbatim):\n{aug}\n")
        lang_rule = ""
        if language_hint and language_hint in self._LANG_NAMES:
            lname = self._LANG_NAMES[language_hint]
            lang_rule = (f"- Reply in the language the person actually spoke. "
                         f"Only if the utterance is too short to tell, use "
                         f"{lname}, the conversation's language so far.\n")
        system = DIRECT_SYSTEM.format(name=self._agent_name, service=service,
                                      lang_rule=lang_rule)
        if context:
            system += f"\nAbout the person you hear:\n{context}\n"

        messages = [{"role": "system", "content": system}]
        for m in history or []:
            if m.get("role") in ("user", "assistant") and m.get("content"):
                messages.append({"role": m["role"], "content": str(m["content"])})
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
        reply, used_tools = "", False
        texts: list[str] = []   # every piece of prose the model produced

        for round_no in range(_MAX_TOOL_ROUNDS):
            resp = self._client.chat(model=self._model, messages=messages,
                                     tools=self._tools or None,
                                     options=self._options,
                                     think=False, stream=False)
            msg = resp["message"]
            calls = msg.get("tool_calls") or []
            content = (msg.get("content") or "").strip()
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
        if (self._tools and not self.last_tool_calls
                and reply and _ACTION_CLAIM.search(reply)):
            logger.warning(f"direct: action claimed without a tool call: {reply[:70]!r}"
                           " — retrying")
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content":
                             "You just told the person you were moving or "
                             "handing over candy, but you did not call the "
                             "tool, so nothing happened. Call the tool NOW "
                             "for exactly that action, then confirm in one "
                             "sentence starting with the ISO language code "
                             "in brackets."})
            for _ in range(2):
                resp = self._client.chat(model=self._model, messages=messages,
                                         tools=self._tools or None,
                                         options=self._options,
                                         think=False, stream=False)
                m2 = resp["message"]
                calls2 = m2.get("tool_calls") or []
                content2 = (m2.get("content") or "").strip()
                if not calls2:
                    if _has_words(content2) and self.last_tool_calls:
                        reply = content2
                    break
                messages.append(m2)
                for call in calls2:
                    fn = call["function"]
                    args = dict(fn["arguments"])
                    refusal = self._reject_bad_move(fn["name"], args, scene)
                    if refusal:
                        self.rejected_tool_calls.append((fn["name"], args, refusal))
                        messages.append({"role": "tool", "tool_name": fn["name"],
                                         "content": refusal})
                        continue
                    logger.info(f"direct: recovered tool call {fn['name']}({args})")
                    result = self._call_tool(fn["name"], args)
                    self.last_tool_calls.append((fn["name"], args, result))
                    messages.append({"role": "tool", "tool_name": fn["name"],
                                     "content": result})
                # Only take the retry's words if it produced any: after a
                # recovered tool call gemma often answers with the bare
                # language tag ("[sv]"), and the sentence it already said in
                # round 1 is now TRUE, so keeping it beats going silent.
                if _has_words(content2):
                    reply = content2
            if not self.last_tool_calls:
                logger.warning("direct: still no tool call — the claim stands "
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
            resp = self._client.chat(model=self._model, messages=messages,
                                     options=self._options,
                                     think=False, stream=False)
            language, reply = _split_lang(
                (resp["message"].get("content") or "").strip(), language)
        elapsed = time.time() - start

        logger.info(f"direct: reply in {elapsed:.2f}s lang={language}: {reply}")
        return reply, language
