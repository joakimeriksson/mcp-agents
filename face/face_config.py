"""
Face subsystem configuration loader.

Loads face_config.toml once at import time. Keep this file tiny — it's
imported by agent.py at startup, so no heavy dependencies.
"""

import os

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def _load_config() -> dict:
    path = os.path.join(os.path.dirname(__file__), "face_config.toml")
    with open(path, "rb") as f:
        return tomllib.load(f)


_CONFIG = _load_config()


def get_tracker_config() -> dict:
    return _CONFIG.get("tracker", {})


# Map [tracker] config keys -> FaceTracker / FaceDatabase constructor kwargs.
# Only keys actually present in the TOML are forwarded, so the dataclass/
# constructor defaults remain the single source of truth for anything unset.
_DB_KEYS = {
    "recognition_tolerance": "tolerance",
    "recognition_k": "recognition_k",
}
_TRACKER_KEYS = {
    "frame_scale": "frame_scale",
    "max_missing_seconds": "max_missing_seconds",
    "focus_min_area_frac": "focus_min_area_frac",
    "focus_dwell_seconds": "focus_dwell_seconds",
    "enroll_min_face_px": "enroll_min_face_px",
    "enroll_min_sharpness": "enroll_min_sharpness",
    "enroll_frontal_tolerance": "enroll_frontal_tolerance",
}


def build_db_kwargs() -> dict:
    """FaceDatabase kwargs from [tracker] config (present keys only)."""
    tc = get_tracker_config()
    return {kw: tc[key] for key, kw in _DB_KEYS.items() if key in tc}


def build_tracker_kwargs() -> dict:
    """FaceTracker kwargs from [tracker] config (present keys only)."""
    tc = get_tracker_config()
    return {kw: tc[key] for key, kw in _TRACKER_KEYS.items() if key in tc}
