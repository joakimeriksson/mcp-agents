import sys
import tomllib
from pathlib import Path

_DEFAULTS: dict = {
    "llm": {
        "model": "gemma4:latest",
        "base_url": "http://localhost:11434/v1/",
        "api_key": "ollama",
        # Speech straight into the (audio-capable) model: one call for
        # hearing + reasoning + tools; transcript in the background.
        "direct_audio": False,
    },
    "face": {
        "omit_names_and_prefs": False,
        # Greet a focused face even when the face DB has no identity for it.
        # False = the old behaviour: wait for recognition/auto-enrollment.
        "talk_to_unknown": True,
    },
    "voice_id": {
        # Speaker verification via the LOCAL voice server (/v1/audio/speaker).
        # An utterance that clearly is not the focused visitor's voice is
        # dropped instead of answered. Everything stays on this machine; by
        # default nothing is written to disk (voiceprints are biometric data).
        "enabled": True,
        "persist_named": False,      # opt in: keep named people's voices in known_voices/
        "confirm": 0.65,             # cosine >= this: same person (and learn from it)
        "reject": 0.40,              # cosine <  this: someone else -> drop the utterance
        "min_seconds": 2.0,          # shorter clips carry no identity: never decide on them
        "url": "http://127.0.0.1:8880/v1/audio/speaker",
    },
    "audio": {
        # A voice must be this many times louder than the room's own noise
        # floor to count as talking to the robot (1.0 = off). Silero rates
        # crowd chatter as speech, so without this every utterance in a busy
        # hall runs to the 15 s cap and background babble triggers turns.
        "near_field_ratio": 3.0,
    },
    "devices": {
        "microphone": None,
        # Index, or part of a device name ("macbook", "brio"). Names are
        # stable across replugging; indices are not.
        "camera": None,
    },
    "debug": {
        # Audio debug panel in the eye window: VU meters (mic/VAD/silence
        # countdown) + in/out oscilloscope. Widens the window slightly.
        "audio_panel": False,
    },
}


def _warn(msg: str) -> None:
    print(f"config: {msg}", file=sys.stderr)


def load_config(path: str | Path | None = None) -> dict:
    cfg = {k: dict(v) for k, v in _DEFAULTS.items()}
    if path is None:
        path = Path(__file__).parent / "config.toml"
    path = Path(path)
    if not path.exists():
        return cfg
    with open(path, "rb") as f:
        data = tomllib.load(f)
    for section, values in data.items():
        if section not in _DEFAULTS:
            _warn(f"unknown section [{section}] in {path}; ignored")
            continue
        valid = {}
        for key, val in values.items():
            if key in _DEFAULTS[section]:
                valid[key] = val
            else:
                _warn(f"unknown key {section}.{key} in {path}; ignored")
        cfg[section].update(valid)
    return cfg
