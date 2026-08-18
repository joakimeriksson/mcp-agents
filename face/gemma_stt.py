"""Gemma 4 native-audio STT backend — a drop-in alternative to faster-whisper.

Gemma 4 understands raw audio directly, so there is no separate ASR step: the
16 kHz mono speech captured by VAD is wrapped as a WAV and handed to a
multimodal Gemma model through Ollama's ``images`` field. The model returns the
transcription (and a language guess).

The public surface mirrors the slice of ``faster_whisper.WhisperModel`` that
``voice_input.VoiceInput`` actually uses::

    segments, info = transcriber.transcribe(audio)   # audio: float32 @ 16 kHz
    for seg in segments:        # seg.text / seg.start / seg.end
        ...
    info.language               # ISO 639-1 code (best-effort from Gemma)
    info.language_probability   # 1.0 when Gemma reported a language, else 0.0

so ``VoiceInput`` can swap backends without changing its VAD / event pipeline.
"""

from __future__ import annotations

import io
import json
import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf

logger = logging.getLogger("gemma_stt")

DEFAULT_MODEL = "gemma4:latest"
DEFAULT_HOST = "http://localhost:11434"

# Ask for strict JSON so we can recover both the text and a language guess.
# Gemma is an LLM, so this is reliable; we still parse defensively.
DEFAULT_PROMPT = (
    "Listen to the audio and transcribe the speech exactly, word for word. "
    "Detect the language being spoken. Respond with ONLY a single JSON object "
    'of the form {"language": "<ISO 639-1 code>", "text": "<exact transcription>"} '
    "and nothing else. If there is no intelligible speech, use an empty text."
)


@dataclass(frozen=True)
class Segment:
    """Minimal stand-in for faster-whisper's segment objects."""
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class TranscriptionInfo:
    """Minimal stand-in for faster-whisper's transcription info object."""
    language: str
    language_probability: float


class Gemma4Transcriber:
    """Native-audio speech-to-text via a multimodal Gemma model on Ollama."""

    def __init__(self,
                 model: str = DEFAULT_MODEL,
                 host: str = DEFAULT_HOST,
                 prompt: str = DEFAULT_PROMPT,
                 sample_rate: int = 16000):
        import ollama  # imported here so the dep is only needed for this backend

        self._model = model
        self._prompt = prompt
        self._sample_rate = sample_rate
        self._client = ollama.Client(host=host)

    @property
    def model_name(self) -> str:
        return self._model

    def check(self) -> None:
        """Verify Ollama is up, the model is pulled, AND it can take audio.

        Raises with an actionable message on any failure — in particular, a
        model without the ``audio`` capability (e.g. gemma4:26b) cannot
        transcribe, so we catch that here instead of at the first utterance.
        """
        models = self._client.list().get("models", [])
        names = {m.get("model") or m.get("name") for m in models}
        # Accept an exact match or the bare name (Ollama reports e.g. "gemma4:latest").
        if self._model not in names and f"{self._model}:latest" not in names:
            raise RuntimeError(
                f"Model {self._model!r} not found in Ollama. "
                f"Pull it with: `ollama pull {self._model}`")

        caps = self.capabilities()
        if "audio" not in caps:
            raise RuntimeError(
                f"Model {self._model!r} has no 'audio' capability "
                f"(capabilities: {caps or 'unknown'}) — it cannot transcribe. "
                f"Use an audio-capable model such as 'gemma4:latest'.")

    def capabilities(self) -> list:
        """Return the model's Ollama capabilities (e.g. ['completion','audio',...])."""
        try:
            info = self._client.show(self._model)
            caps = getattr(info, "capabilities", None)
            if caps is None and isinstance(info, dict):
                caps = info.get("capabilities")
            return list(caps or [])
        except Exception as e:
            logger.warning(f"Could not read capabilities for {self._model!r}: {e}")
            return []

    def transcribe(self, audio: np.ndarray, beam_size: int = None
                   ) -> Tuple[List[Segment], TranscriptionInfo]:
        """Transcribe a float32 mono waveform. Signature matches WhisperModel."""
        wav_bytes = self._to_wav_bytes(audio)

        response = self._client.chat(
            model=self._model,
            messages=[{
                "role": "user",
                "content": self._prompt,
                "images": [wav_bytes],
            }],
            think=False,
            stream=False,
        )
        raw = (response.get("message", {}).get("content") or "").strip()
        text, language = self._parse(raw)

        duration = len(audio) / self._sample_rate if len(audio) else 0.0
        segments = [Segment(text=text, start=0.0, end=duration)] if text else []
        info = TranscriptionInfo(
            language=language,
            language_probability=1.0 if language else 0.0,
        )
        return segments, info

    # --- helpers ---------------------------------------------------------

    def _to_wav_bytes(self, audio: np.ndarray) -> bytes:
        buf = io.BytesIO()
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        sf.write(buf, samples, self._sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    @staticmethod
    def _parse(raw: str) -> Tuple[str, str]:
        """Extract (text, language) from Gemma's reply, tolerating extra prose."""
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
                text = str(obj.get("text", "")).strip()
                language = str(obj.get("language", "")).strip().lower()
                # Normalise occasional full names to ISO codes.
                language = _LANG_ALIASES.get(language, language)
                if len(language) > 3:  # not an ISO code we recognise
                    language = ""
                return text, language
            except (json.JSONDecodeError, TypeError, ValueError):
                logger.warning("Gemma STT: JSON parse failed, using raw text")
        # Fallback: treat the whole reply as the transcription.
        return raw.strip(), ""


_LANG_ALIASES = {
    "swedish": "sv", "svenska": "sv",
    "english": "en",
    "norwegian": "no", "danish": "da", "finnish": "fi",
    "german": "de", "french": "fr", "spanish": "es",
}
