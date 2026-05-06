import sys
import tomllib
from pathlib import Path

_DEFAULTS: dict = {
    "llm": {
        "model": "PetrosStav/gemma3-tools:12b",
        "base_url": "http://localhost:11434/v1/",
        "api_key": "ollama",
    },
    "face": {
        "omit_names_and_prefs": False,
    },
    "devices": {
        "microphone": None,
        "camera": None,
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
