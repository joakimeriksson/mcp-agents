"""Short-term visit memory, shared by the voice clients.

What a robot needs for "welcome back" and "you already got a Dumle": when
someone was here, in which language, and what the SERVICE did for them.
It lives only in memory and only for VISIT_TTL_S, so nothing here can
identify anyone later (long-term memory of people is people_memory.py, and
only for people who told us their name).

The generic layer never knows what a note means: the service policy's
``remember`` hook turns a tool call the service accepted into a note
("was given a Dumle"); this module only stores and times it, and the client
puts ``line()`` into the prompt.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Hashable, Optional

VISIT_TTL_S = 2 * 3600      # forgotten this long after they were last seen
AWAY_S = 60.0               # gone at least this long = they "came back"
MAX_NOTES = 5


def _ago(seconds: float) -> str:
    if seconds < 90:
        return "just now"
    if seconds < 3600:
        return f"{round(seconds / 60)} min ago"
    return f"{seconds / 3600:.1f} h ago"


@dataclass
class Visit:
    first_seen: float
    last_seen: float
    lang: str = ""
    returned_after: Optional[float] = None   # seconds away before this return
    notes: list = field(default_factory=list)  # [(time, text)]


class VisitMemory:
    def __init__(self, ttl: float = VISIT_TTL_S, max_notes: int = MAX_NOTES):
        self.ttl = ttl
        self.max_notes = max_notes
        self._visits: dict[Hashable, Visit] = {}
        self._lock = threading.Lock()

    def _prune(self, now: float):
        for k in [k for k, v in self._visits.items() if now - v.last_seen > self.ttl]:
            del self._visits[k]

    def seen(self, key: Hashable, lang: Optional[str] = None,
             now: Optional[float] = None) -> Visit:
        """They are in front of the robot now (call on face change / each turn)."""
        now = time.time() if now is None else now
        with self._lock:
            self._prune(now)
            v = self._visits.get(key)
            if v is None:
                v = self._visits[key] = Visit(first_seen=now, last_seen=now)
            elif now - v.last_seen >= AWAY_S:
                v.returned_after = now - v.last_seen
            v.last_seen = now
            if lang:
                v.lang = lang
            return v

    def note(self, key: Hashable, text: str, now: Optional[float] = None):
        """Something the service did for them ("was given a Dumle")."""
        if not text:
            return
        now = time.time() if now is None else now
        with self._lock:
            v = self._visits.get(key)
            if v is None:
                v = self._visits[key] = Visit(first_seen=now, last_seen=now)
            v.notes.append((now, text))
            del v.notes[:-self.max_notes]

    def rekey(self, old: Hashable, new: Hashable):
        """A provisional face id became a stable one."""
        with self._lock:
            if old in self._visits and new not in self._visits:
                self._visits[new] = self._visits.pop(old)

    def forget(self, key: Hashable):
        with self._lock:
            self._visits.pop(key, None)

    def clear(self):
        with self._lock:
            self._visits.clear()

    def line(self, key: Hashable, now: Optional[float] = None) -> str:
        """One line for the prompt, or "" when there is nothing worth saying:
        "They came back: first here 40 min ago. Earlier: was given a Dumle (6 min ago)." """
        now = time.time() if now is None else now
        with self._lock:
            v = self._visits.get(key)
            if v is None or now - v.last_seen > self.ttl:
                return ""
            parts = []
            if v.returned_after:
                parts.append(f"They came back: first here {_ago(now - v.first_seen)}.")
            if v.notes:
                parts.append("Earlier: " + "; ".join(
                    f"{text} ({_ago(now - t)})" for t, text in v.notes) + ".")
            return " ".join(parts)
