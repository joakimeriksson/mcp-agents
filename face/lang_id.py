"""Spoken-language identification with Whisper on the Apple GPU (MLX).

gemma4's own ear is English-biased on short Swedish ("Ja tack" -> English), so
the direct-audio turn guessed the reply language wrong ~40% of the time.
Whisper large-v3-turbo's language head gets it right in ~0.12 s (measured
Nov 2026: 14/14 on sv/en/de/fr/es/it incl. "Ja tack" and "Hej!"). The result
goes into the prompt as an order, not a hint.

MLX arrays belong to the thread that made them, so the model is loaded and
run on ONE dedicated worker thread.
"""
from __future__ import annotations

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple, Optional

import numpy as np

logger = logging.getLogger(__name__)

MODEL = os.environ.get("CANDYTRON_LID_MODEL", "mlx-community/whisper-large-v3-turbo")


class LanguageID:
    def __init__(self, model: str = MODEL):
        self._repo = model
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lang-id")
        self._model = None
        self.available = True

    def _load(self):
        if self._model is None:
            from mlx_whisper.load_models import load_model
            self._model = load_model(self._repo)
        return self._model

    def _detect(self, audio: np.ndarray) -> tuple[Optional[str], float]:
        import mlx.core as mx
        from mlx_whisper.audio import N_SAMPLES, log_mel_spectrogram, pad_or_trim
        from mlx_whisper.decoding import detect_language
        m = self._load()
        mel = log_mel_spectrogram(pad_or_trim(mx.array(audio.astype(np.float32)), N_SAMPLES),
                                  n_mels=m.dims.n_mels)
        _tok, probs = detect_language(m, mel[None] if mel.ndim == 2 else mel)
        p = probs[0] if isinstance(probs, list) else probs
        best = max(p, key=p.get)
        return best, float(p[best])

    def warm(self) -> None:
        """Load the model in the background (first call otherwise ~seconds)."""
        def _w():
            try:
                self._detect(np.zeros(16000, np.float32))
                _transcribe_impl(self, np.zeros(16000, np.float32))   # mlx_whisper's own model cache
                logger.info(f"lang-id: {self._repo} ready")
            except Exception as e:  # noqa: BLE001
                self.available = False
                logger.warning(f"lang-id unavailable ({e}); gemma4 guesses the language itself")
        self._pool.submit(_w)

    def detect(self, audio: np.ndarray, timeout: float = 1.5) -> tuple[Optional[str], float]:
        """(language code, probability) for 16 kHz mono audio; (None, 0) on failure."""
        if not self.available or audio is None or len(audio) == 0:
            return None, 0.0
        try:
            return self._pool.submit(self._detect, audio).result(timeout=timeout)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"lang-id failed: {e}")
            return None, 0.0


# Whisper's classic inventions on silence / noise (YouTube outro training data).
_HALLUCINATIONS = re.compile(
    r"tack för att du (tittade|lyssnade)|thanks? (you )?for watching|subtitles? by|"
    r"undertext|textning|amara\.org|prenumerera|subscribe|www\.|\.com",
    re.IGNORECASE)


class Heard(NamedTuple):
    text: str            # "" = nothing usable was heard
    language: Optional[str]
    reason: str          # why it was dropped, or "ok"


def _transcribe_impl(self, audio: np.ndarray, hint_words: str = "") -> Heard:
    import mlx_whisper
    self._load()
    r = mlx_whisper.transcribe(audio.astype(np.float32), path_or_hf_repo=self._repo,
                               condition_on_previous_text=False,
                               initial_prompt=hint_words or None, verbose=None)
    segs = r.get("segments") or []
    text = (r.get("text") or "").strip()
    lang = r.get("language")
    if not text:
        return Heard("", lang, "empty")
    if segs and all(sg.get("no_speech_prob", 0) > 0.6 and sg.get("avg_logprob", 0) < -0.8 for sg in segs):
        return Heard("", lang, "no speech")
    if segs and sum(sg.get("avg_logprob", 0) for sg in segs) / len(segs) < -1.2:
        return Heard("", lang, f"low confidence: {text!r}")
    if _HALLUCINATIONS.search(text):
        return Heard("", lang, f"hallucination: {text!r}")
    return Heard(text, lang, "ok")


def _transcribe(self, audio: np.ndarray, hint_words: str = "", timeout: float = 15.0) -> Heard:
    """Speech -> text on the GPU (~0.16 s for a short utterance)."""
    if not self.available or audio is None or len(audio) == 0:
        return Heard("", None, "unavailable")
    try:
        return self._pool.submit(_transcribe_impl, self, audio, hint_words).result(timeout=timeout)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"whisper transcribe failed: {e}")
        return Heard("", None, f"failed: {e}")


LanguageID.transcribe = _transcribe


def language_name(code: str) -> str:
    try:
        from mlx_whisper.tokenizer import LANGUAGES
        return LANGUAGES.get(code, code).title()
    except Exception:  # noqa: BLE001
        return code
