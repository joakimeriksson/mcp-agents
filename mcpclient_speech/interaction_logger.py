"""JSONL interaction logger for the face-tracking speech client.

Each session writes a stream of `{"ts": ..., "type": ..., ...}` events to a
single JSONL file. Two output modes are supported via `from_args`:

  * `--log-file PATH` — overwrite a single file every run.
  * `--log-dir DIR`   — create a new `<UTC-timestamp>Z.jsonl` per run inside DIR.

Failures inside `log()` are caught and re-emitted as warnings on the standard
"mcpclient_speech" logger so a broken disk never crashes the application.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone

logger = logging.getLogger("mcpclient_speech")

SCHEMA_VERSION = 1


def _json_default(obj):
    # OpenAI ChatCompletionMessage and related pydantic models
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(mode="json", exclude_none=True)
        except TypeError:
            return model_dump(exclude_none=True)
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:
            pass
    return str(obj)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_now_filename() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f") + "Z"


class InteractionLogger:
    def __init__(self, path: str, *, overwrite: bool):
        self.path = path
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._fp = open(path, "w" if overwrite else "a", encoding="utf-8")
        self._start_monotonic = time.monotonic()
        self._closed = False

    @classmethod
    def from_args(cls, args) -> "InteractionLogger | None":
        log_file = getattr(args, "log_file", None)
        log_dir = getattr(args, "log_dir", None)
        if log_file:
            if not log_file.endswith(".jsonl"):
                log_file = log_file + ".jsonl"
            return cls(log_file, overwrite=True)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            path = os.path.join(log_dir, _utc_now_filename() + ".jsonl")
            return cls(path, overwrite=False)
        return None

    def log(self, event_type: str, **fields) -> None:
        if self._closed:
            return
        try:
            payload = {"ts": _utc_now_iso(), "type": event_type, **fields}
            line = json.dumps(payload, default=_json_default, ensure_ascii=False)
            self._fp.write(line + "\n")
            self._fp.flush()
        except Exception:
            logger.warning("interaction logger: failed to write %s event", event_type, exc_info=True)

    def close(self, reason: str = "normal", **extra) -> None:
        if self._closed:
            return
        try:
            duration = round(time.monotonic() - self._start_monotonic, 3)
            payload = {
                "ts": _utc_now_iso(),
                "type": "session_end",
                "reason": reason,
                "duration_seconds": duration,
                **extra,
            }
            self._fp.write(json.dumps(payload, default=_json_default, ensure_ascii=False) + "\n")
            self._fp.flush()
        except Exception:
            logger.warning("interaction logger: failed to write session_end", exc_info=True)
        finally:
            try:
                self._fp.close()
            except Exception:
                pass
            self._closed = True
