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
"""

_LANG_TAG = re.compile(r"^\s*\[([a-z]{2}(?:-[a-z]{2})?)\]\s*", re.IGNORECASE)
_MAX_TOOL_ROUNDS = 3


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
                result = self._call_tool(fn["name"], args)
                self.last_tool_calls.append((fn["name"], args, result))
                messages.append({"role": "tool", "tool_name": fn["name"],
                                 "content": result})

        # A tool round can yield no prose at all. Never go silent after
        # moving the arm: fall back to earlier prose, else ask for a short
        # confirmation (text-only, cheap) — the model still has the tool
        # results in its history.
        reply = reply or (texts[-1] if texts else "")
        if not reply and used_tools:
            messages.append({"role": "user", "content":
                             "Briefly tell the user what you just did, in "
                             "their spoken language, starting with its ISO "
                             "code in brackets. 1 sentence."})
            resp = self._client.chat(model=self._model, messages=messages,
                                     options=self._options,
                                     think=False, stream=False)
            reply = (resp["message"].get("content") or "").strip()
        elapsed = time.time() - start

        language = language_hint or "en"
        m = _LANG_TAG.match(reply)
        if m:
            language = m.group(1).lower()
            reply = reply[m.end():].strip()
        logger.info(f"direct: reply in {elapsed:.2f}s lang={language}: {reply}")
        return reply, language
