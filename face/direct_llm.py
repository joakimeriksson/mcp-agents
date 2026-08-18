"""Direct-audio conversation: user speech goes straight into Gemma 4.

Instead of the two-step pipeline (audio -> Gemma STT -> text -> Gemma chat),
one multimodal call does hearing, thinking, and tool calling at once — the
same idea as reachy_mini_conversation_app's direct-audio mode. Measured on an
M-series Mac this roughly halves speech-end -> reply latency (one ~0.8-1.7 s
call replaces a 0.7-2.7 s STT call plus a ~0.8 s chat call).

What still needs text (per-person memory, fact extraction, name learning) is
served by a background transcription AFTER the reply — off the critical path,
exactly like reachy's remote_stt.

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
- Reply in 1-2 short sentences. No markdown or emojis.
- Use your tools when the request calls for them — never claim to have done
  something a tool does without actually calling the tool.
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
                 sample_rate: int = 16000):
        import ollama
        self._client = ollama.Client(host=host)
        self._model = model
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

    def respond(self, audio: np.ndarray, *,
                context: str = "",
                language_hint: str = "en") -> tuple[str, str]:
        """Audio in, (reply_text, language) out. Blocking; one to a few
        model calls depending on tool use."""
        wav = _to_wav_bytes(audio, self._sample_rate)

        service = ""
        if self._service_prompt:
            service = f"\nYour service role:\n{self._service_prompt}\n"
        if self._augmentation_provider:
            aug = self._augmentation_provider(language_hint)
            if aug:
                service += (f"\nCurrent state from your service "
                            f"(internal — never read it out verbatim):\n{aug}\n")
        system = DIRECT_SYSTEM.format(name=self._agent_name, service=service)
        if context:
            system += f"\nAbout the person you hear:\n{context}\n"

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "(user speech attached as audio)",
             "images": [wav]},
        ]

        start = time.time()
        reply = last_content = ""
        for round_no in range(_MAX_TOOL_ROUNDS):
            resp = self._client.chat(model=self._model, messages=messages,
                                     tools=self._tools or None,
                                     think=False, stream=False)
            msg = resp["message"]
            calls = msg.get("tool_calls") or []
            reply = (msg.get("content") or "").strip()
            last_content = reply or last_content
            if not calls:
                break
            messages.append(msg)
            for call in calls:
                fn = call["function"]
                logger.info(f"direct: tool call {fn['name']}({dict(fn['arguments'])})")
                result = self._call_tool(fn["name"], dict(fn["arguments"]))
                messages.append({"role": "tool", "tool_name": fn["name"],
                                 "content": result})
        elapsed = time.time() - start
        # A tool round can end with an empty final message — fall back to the
        # last text the model produced alongside its tool call.
        reply = reply or last_content

        language = language_hint or "en"
        m = _LANG_TAG.match(reply)
        if m:
            language = m.group(1).lower()
            reply = reply[m.end():].strip()
        logger.info(f"direct: reply in {elapsed:.2f}s lang={language}: {reply}")
        return reply, language
