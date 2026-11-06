"""Which language a reply is actually written in, for picking its TTS voice.

The conversation language (``choose_language`` in the clients) is sticky on
purpose: one stray transcript must not flip the conversation. But that makes
it lag the model's replies during a switch, and the voice used to follow the
label, not the text. Measured on the robot (Oct 2026): Swedish replies labelled
'en' fell through to the server's default Swedish voice, French labelled 'sv'
came out at Greta's 1.3x speed, French labelled 'it' was read by the Italian
voice. So the voice follows the reply text; the conversation language only
breaks ties on fragments ("Ja!", "OK") that carry no signal.

Same approach as kokoro-voice-server's ``_detect_lang``: lingua restricted to
the languages we can speak, with a confidence floor below which the
conversation language wins.
"""

import logging
from typing import Iterable, Optional

logger = logging.getLogger("reply_language")

# Below this the detector is guessing (short fragments land around 0.3-0.4 with
# a tiny margin; real sentences score >= 0.5). Matches the voice server.
MIN_CONFIDENCE = 0.5

_LINGUA_NAMES = {
    "sv": "SWEDISH", "en": "ENGLISH", "fr": "FRENCH", "es": "SPANISH",
    "it": "ITALIAN", "de": "GERMAN", "pt": "PORTUGUESE",
}
_CODES = {name: code for code, name in _LINGUA_NAMES.items()}
_detectors: dict = {}   # frozenset of codes -> lingua detector (or False)


def _detector(codes: frozenset):
    if codes not in _detectors:
        try:
            from lingua import Language, LanguageDetectorBuilder
            langs = [getattr(Language, _LINGUA_NAMES[c]) for c in codes]
            _detectors[codes] = LanguageDetectorBuilder.from_languages(*langs).build()
        except Exception as e:   # lingua missing: keep the old label-based behaviour
            logger.warning(f"lingua unavailable ({e}); TTS voice follows the conversation language")
            _detectors[codes] = False
    return _detectors[codes]


def reply_language(text: str, hint: Optional[str],
                   candidates: Iterable[str]) -> tuple[Optional[str], float]:
    """Return ``(language, confidence)`` for *text* among *candidates*.

    *hint* (the conversation language) is returned, with confidence 0.0, when
    the text is too short or ambiguous to tell, or when detection can't run.
    """
    codes = frozenset(c for c in candidates if c in _LINGUA_NAMES)
    if len(codes) < 2 or not text or not text.strip():
        return hint, 0.0
    det = _detector(codes)
    if not det:
        return hint, 0.0
    ranked = det.compute_language_confidence_values(text)
    if not ranked:
        return hint, 0.0
    best = ranked[0]
    lang, conf = _CODES.get(best.language.name), float(best.value)
    if lang is None or (conf < MIN_CONFIDENCE and hint in codes):
        return hint, conf
    return lang, conf
