"""
Language configuration loader.

Loads languages.toml once at import time and provides accessor functions
for language models, pronunciations, greetings, and goodbyes.

No heavy dependencies — safe to import from any module.
"""

import os
import random

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def _load_config() -> dict:
    path = os.path.join(os.path.dirname(__file__), "languages.toml")
    with open(path, "rb") as f:
        return tomllib.load(f)


_CONFIG = _load_config()
_DEFAULT_LANG = _CONFIG.get("default", "en")
_LANGUAGES = _CONFIG.get("languages", {})


def get_default_language() -> str:
    """Return the default language code (e.g. ``"en"``)."""
    return _DEFAULT_LANG


def get_language_config(lang: str = None) -> dict:
    """Return the config block for a language.

    When *lang* is ``None`` or not found, returns the default language's
    config.
    """
    if lang and lang in _LANGUAGES:
        return _LANGUAGES[lang]
    return _LANGUAGES.get(_DEFAULT_LANG, {})


def get_language_models() -> dict[str, str]:
    """Return ``{lang_code: tts_model_name}`` for all configured languages."""
    return {lang: cfg["tts_model"] for lang, cfg in _LANGUAGES.items()
            if "tts_model" in cfg}


def get_language_tts_servers() -> dict[str, dict[str, str]]:
    """Return ``{lang_code: {"url": ..., "voice": ...}}`` for languages that
    use an HTTP TTS server (OpenAI-compatible /v1/audio/speech) instead of a
    local Piper model. ``voice`` may be absent (server default)."""
    servers = {}
    for lang, cfg in _LANGUAGES.items():
        if "tts_server" in cfg:
            servers[lang] = {"url": cfg["tts_server"]}
            if "tts_voice" in cfg:
                servers[lang]["voice"] = cfg["tts_voice"]
    return servers


def get_language_pronunciations() -> dict[str, dict[str, str]]:
    """Return ``{lang_code: {word: replacement}}`` for all configured languages."""
    return {lang: cfg["pronunciations"] for lang, cfg in _LANGUAGES.items()
            if "pronunciations" in cfg}


def get_goodbye(name: str, language: str = None) -> str:
    """Return a random goodbye phrase for the given language."""
    cfg = get_language_config(language)
    goodbyes = cfg.get("goodbyes", get_language_config().get("goodbyes", []))
    if goodbyes:
        return random.choice(goodbyes).format(name=name)
    return f"Goodbye, {name}!"
