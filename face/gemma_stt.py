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

_LANG_NAMES = {
    "en": "English", "sv": "Swedish", "de": "German", "fr": "French",
    "es": "Spanish", "it": "Italian", "no": "Norwegian", "da": "Danish",
    "fi": "Finnish", "nl": "Dutch", "pt": "Portuguese",
}

# Replies that are not transcriptions: the model echoing its own instruction,
# or describing silence in words. Both must become an empty transcript, or
# they reach the LLM as if the person had said them.
_NON_SPEECH = re.compile(
    r"^\W*(no|there is no|there's no)?\s*(intelligible|audible|clear)?\s*"
    r"(speech|audio|sound|voice)\b.*(detected|found|heard|present|audible)?\W*$"
    r"|^\W*(silence|inaudible|unintelligible|\[.*\]|\(.*\))\W*$",
    re.IGNORECASE)


# Fragments of the instruction as the model tends to echo (or translate) it.
_PROMPT_ECHO_MARKERS = (
    "transcribe the speech", "detect the language", "json object", "iso 639",
    "listen to the audio", "word for word",
    "transkrib", "lyssna på ljud", "hör på ljud", "ord för ord",      # sv
    "transcri", "écoute", "mot pour mot",                            # fr
    "escucha el audio", "palabra por palabra",                       # es
    "trascri", "ascolta", "parola per parola",                       # it
    "höre dir", "wort für wort",                                     # de
)


def build_prompt(expected_languages=None, language_hint=None) -> str:
    """The transcription prompt with a language prior.

    *expected_languages*: ISO codes the deployment supports — the model is
    told to choose among them, which stops one-word replies coming back as
    Hindi or Italian. *language_hint*: the conversation's language so far;
    short or unclear utterances are most likely in it.
    """
    prompt = DEFAULT_PROMPT
    if expected_languages:
        names = ", ".join(f"{_LANG_NAMES.get(c, c)} ({c})" for c in expected_languages)
        prompt += f" The speech is in one of these languages: {names}."
    if language_hint:
        # NOTE: measured Aug 2026 — telling gemma4 to *assume* a language makes
        # it over-commit (English mis-heard as that language, hallucinated text
        # on noise). Prefer listing candidates without a hint (see transcribe()).
        name = _LANG_NAMES.get(language_hint, language_hint)
        prompt += (f" The conversation so far has been in {name}, so when the "
                   f"audio is short or unclear assume {name} — but if the words "
                   f"are clearly another listed language, report that language.")
    return prompt


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
                 expected_languages: Optional[List[str]] = None,
                 sample_rate: int = 16000):
        import ollama  # imported here so the dep is only needed for this backend

        self._model = model
        self._prompt = prompt
        self._expected = list(expected_languages) if expected_languages else []
        # Conversation language so far (ISO code); set by the caller between
        # turns. Used as a prior in the prompt, never as a hard override.
        self.language_hint: str = ""
        self._rehear_max_s = 3.0   # utterances up to this long get the pass-2 prior
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

        duration = len(audio) / self._sample_rate if len(audio) else 0.0

        # Pass 1: the plain prompt. Measured (Aug 2026): any language prior in
        # the prompt makes gemma4 over-commit to it and hallucinate on noise,
        # while the plain prompt is English-biased on SHORT Swedish ("Ja tack"
        # -> "Hi there"). So the prior is applied only as a targeted pass 2.
        text, language = self._ask(wav_bytes, self._prompt)
        text, language = self._sanitize(text, language)

        # Pass 2: a short utterance that disagrees with the conversation's
        # language is re-heard with that language as the prior. Long
        # utterances are trusted as heard (that is how a language switch
        # happens); empty ones stay empty (no prior on noise).
        hint = self.language_hint
        if (text and hint and language != hint and duration <= self._rehear_max_s
                and (not self._expected or hint in self._expected)):
            # Neutral re-hear: name the two candidates, don't say which to
            # assume — "assume Swedish" made gemma4 mis-transcribe English
            # into Swedish gibberish; "Swedish or English?" keeps both right.
            candidates = [hint] + ([language] if language and language != hint else [])
            text2, lang2 = self._ask(wav_bytes, build_prompt(candidates, None)
                                     + " Decide the language from the words you hear.")
            text2, lang2 = self._sanitize(text2, lang2)
            if text2:
                logger.info(f"Gemma STT: re-heard {duration:.1f}s utterance with "
                            f"prior {hint}: [{language}] {text!r} -> [{lang2}] {text2!r}")
                text, language = text2, lang2

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

    def _ask(self, wav_bytes: bytes, prompt: str) -> Tuple[str, str]:
        response = self._client.chat(
            model=self._model,
            messages=[{"role": "user", "content": prompt, "images": [wav_bytes]}],
            think=False,
            stream=False,
        )
        raw = (response.get("message", {}).get("content") or "").strip()
        return self._parse(raw)

    def _sanitize(self, text: str, language: str) -> Tuple[str, str]:
        """Drop non-transcriptions and languages outside the expected set."""
        if text:
            lowered = text.lower()
            # Prompt echo: the model repeated (part of) its own instruction.
            if any(chunk in lowered for chunk in _PROMPT_ECHO_MARKERS):
                logger.warning(f"Gemma STT: prompt echo dropped: {text[:60]!r}")
                text = ""
            elif _NON_SPEECH.match(text):
                logger.info(f"Gemma STT: non-speech reply dropped: {text!r}")
                text = ""
        if language and self._expected and language not in self._expected:
            logger.info(f"Gemma STT: language {language!r} outside expected "
                        f"{self._expected} -> unknown")
            language = ""
        return text, language

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
                logger.warning("Gemma STT: JSON parse failed, salvaging fields")
        if raw.lstrip().startswith("{"):
            # Truncated / malformed JSON: salvage the fields by regex rather
            # than passing the JSON fragment on as if it were speech.
            m_text = re.search(r'"text"\s*:\s*"([^"]*)', raw)
            m_lang = re.search(r'"language"\s*:\s*"([^"]*)"', raw)
            language = (m_lang.group(1).strip().lower() if m_lang else "")
            language = _LANG_ALIASES.get(language, language)
            return (m_text.group(1).strip() if m_text else ""), (language if len(language) <= 3 else "")
        # Fallback: treat the whole reply as the transcription.
        return raw.strip(), ""


_LANG_ALIASES = {
    "swedish": "sv", "svenska": "sv", "sw": "sv", "swe": "sv",  # 'sw' is what gemma writes for Swedish
    "english": "en",
    "norwegian": "no", "danish": "da", "finnish": "fi",
    "german": "de", "french": "fr", "spanish": "es",
}
