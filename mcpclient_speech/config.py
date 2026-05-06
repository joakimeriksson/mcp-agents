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
    "devices": {},
}

def load_config(path: str | Path | None = None) -> dict:
    cfg = {k: dict(v) for k, v in _DEFAULTS.items()}
    if path is None:
        path = Path(__file__).parent / "config.toml"
    if Path(path).exists():
        with open(path, "rb") as f:
            for section, values in tomllib.load(f).items():
                cfg.setdefault(section, {}).update(values)
    return cfg
