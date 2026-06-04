"""
People memory module: stores per-person data (dialogues, facts, asked topics).

People are keyed by a stable ``person_id`` (e.g. ``p001``) that never
changes, even on rename. Each person has one JSON file at
``{storage_dir}/{person_id}.json``; the display name lives inside that
JSON and is the single source of truth for what to call the person.

At runtime the active session maps ``track_id -> Person`` via
``identify(track_id, person_id)``.
"""

import json
import os
import re
import time
import threading
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger("people_memory")


# Interview topics: small-talk questions the agent asks during greetings
# to build up a person's profile over time. One topic per greeting; once
# the topic has been asked it goes into ``person.asked_topics`` and we
# move on, regardless of whether the user actually answered. Whatever
# they say in response is captured by ``extract_facts`` like everything
# else — there's no separate slot storage anymore.
INTERVIEW_TOPICS: list[str] = [
    "favourite_candy",
    "favourite_coffee",
    "favourite_colour",
    "hobby",
    "favourite_food",
    "favourite_music",
]


def _is_person_id(name: str) -> bool:
    """Return True if name looks like a person ID (e.g. 'p001')."""
    return bool(re.match(r"^p\d+$", name))


@dataclass
class DialogueEntry:
    """A single exchange in a conversation."""
    timestamp: float          # time.time()
    speaker: str              # "system" or "person"
    text: str
    language: str = ""
    emotion: str = ""


@dataclass
class Person:
    """Everything we know about a person, keyed by track_id at runtime."""
    track_id: int                          # current session track_id
    name: Optional[str] = None             # None until identified
    first_met: float = 0.0
    last_seen: float = 0.0
    last_talked: float = 0.0
    times_seen: int = 0
    emotion: str = ""

    dialogues: list[DialogueEntry] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    asked_topics: list[str] = field(default_factory=list)
    summary: str = ""

    # Persistent ID (e.g. "p001") — stable across renames, matches filename.
    persistent_id: str = ""

    @property
    def is_identified(self) -> bool:
        """True if the person has a real name (not None, not a person ID like 'p001')."""
        return self.name is not None and not _is_person_id(self.name)

    def to_dict(self) -> dict:
        return {
            "persistent_id": self.persistent_id,
            "name": self.name,
            "first_met": self.first_met,
            "last_seen": self.last_seen,
            "last_talked": self.last_talked,
            "times_seen": self.times_seen,
            "emotion": self.emotion,
            "facts": self.facts,
            "asked_topics": self.asked_topics,
            "summary": self.summary,
            "dialogues": [
                {
                    "timestamp": e.timestamp,
                    "speaker": e.speaker,
                    "text": e.text,
                    "language": e.language,
                    "emotion": e.emotion,
                }
                for e in self.dialogues
            ],
        }

    @staticmethod
    def from_dict(d: dict, track_id: int = 0) -> "Person":
        dialogues = [DialogueEntry(**e) for e in d.get("dialogues", [])]
        return Person(
            track_id=track_id,
            persistent_id=d.get("persistent_id", d.get("name", "")),
            name=d.get("name"),
            first_met=d.get("first_met", 0.0),
            last_seen=d.get("last_seen", 0.0),
            last_talked=d.get("last_talked", 0.0),
            times_seen=d.get("times_seen", 0),
            emotion=d.get("emotion", ""),
            dialogues=dialogues,
            facts=d.get("facts", []),
            asked_topics=d.get("asked_topics", []),
            summary=d.get("summary", ""),
        )

    @property
    def last_language(self) -> str:
        """Language from the most recent dialogue entry that has one, or ''."""
        for d in reversed(self.dialogues):
            if d.language:
                return d.language
        return ""

    def missing_topics(self) -> list[str]:
        """Interview topics we haven't asked this person about yet."""
        return [t for t in INTERVIEW_TOPICS if t not in self.asked_topics]

    @property
    def first_met_dt(self) -> Optional[datetime]:
        return datetime.fromtimestamp(self.first_met) if self.first_met else None

    @property
    def last_seen_dt(self) -> Optional[datetime]:
        return datetime.fromtimestamp(self.last_seen) if self.last_seen else None

    @property
    def last_talked_dt(self) -> Optional[datetime]:
        return datetime.fromtimestamp(self.last_talked) if self.last_talked else None

    def time_since_seen(self) -> Optional[float]:
        if not self.last_seen:
            return None
        return time.time() - self.last_seen

    def time_since_talked(self) -> Optional[float]:
        if not self.last_talked:
            return None
        return time.time() - self.last_talked


_PERSON_ID_RE = re.compile(r"^p(\d+)$")


# ---------------------------------------------------------------------------
# Fact normalization and similarity
# ---------------------------------------------------------------------------

_SUBJECT_PRONOUN_RE = re.compile(r"^(the\s+person|he|she|they)\s+", re.IGNORECASE)

# Words ignored in token-set comparisons so that "likes chess" and
# "likes the game of chess" still compare as dupes.
_STOPWORDS = {
    "a", "an", "the", "is", "was", "are", "were", "am", "be", "been", "being",
    "has", "have", "had", "does", "did", "do", "that", "this", "these", "those",
    "with", "and", "or", "of", "in", "on", "to", "for", "at", "by", "as",
    "his", "her", "their", "its", "my", "your",
    "mentioned", "said", "says", "stated",
}


def _strip_subject(fact: str, person_name: Optional[str]) -> str:
    """Normalize a fact to bare-predicate form: strip leading subject.

    "Joakim likes lingonsylt" -> "likes lingonsylt"
    "The person mentioned a hotel" -> "mentioned a hotel"
    "Joakim's spare time activity is AI" -> "spare time activity is AI"
    Also lowercases the first letter and strips a trailing period so that
    variants like "Likes X." and "likes X" collapse.
    """
    f = (fact or "").strip()
    if not f:
        return ""
    if person_name:
        # Possessive first: "Joakim's ..."
        poss = re.compile(rf"^{re.escape(person_name)}'s\s+", re.IGNORECASE)
        f = poss.sub("", f).strip()
        # Plain subject: "Joakim ..."
        plain = re.compile(rf"^{re.escape(person_name)}\s+", re.IGNORECASE)
        f = plain.sub("", f).strip()
    f = _SUBJECT_PRONOUN_RE.sub("", f).strip()
    if not f:
        return ""
    f = f[0].lower() + f[1:]
    return f.rstrip(".").strip()


def _fact_tokens(s: str, extra_stops: Optional[set] = None) -> set:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    stops = _STOPWORDS | (extra_stops or set())
    return {w for w in s.split() if w and w not in stops and len(w) > 1}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class PeopleMemory:
    """In-memory people database with JSON file persistence.

    Each stored person has a stable ``person_id`` (e.g. ``p001``) which
    is the filename and the key in ``_stored``. Display names live inside
    the JSON and can be changed freely without touching the ID.
    """

    _DEFAULT_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "people"
    )

    def __init__(self, storage_dir: str = _DEFAULT_DIR):
        self._dir = storage_dir
        # Active session: track_id -> Person
        self._active: dict[int, Person] = {}
        # Persistent store: person_id -> Person dict (loaded from disk)
        self._stored: dict[str, dict] = {}
        self._lock = threading.Lock()

    def load(self):
        """Load all person files from disk into the persistent store."""
        os.makedirs(self._dir, exist_ok=True)
        count = 0
        for fname in os.listdir(self._dir):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(self._dir, fname)
            pid = fname[:-5]
            try:
                with open(fpath, "r") as f:
                    content = f.read().strip()
                if not content:
                    raise ValueError("empty file")
                data = json.loads(content)
                data["persistent_id"] = pid
                self._stored[pid] = data
                count += 1
            except Exception as e:
                logger.warning(f"Failed to load {fpath}: {e}, creating skeleton record")
                data = {"persistent_id": pid, "name": None}
                self._stored[pid] = data
                count += 1
        logger.info(f"People memory loaded: {count} people from {self._dir}/")

    def next_person_id(self) -> str:
        """Allocate the next free p### ID by scanning stored records."""
        with self._lock:
            max_n = 0
            for pid in self._stored:
                m = _PERSON_ID_RE.match(pid)
                if m:
                    max_n = max(max_n, int(m.group(1)))
        return f"p{max_n + 1:03d}"

    def create_person(self, track_id: int, name: str) -> str:
        """Allocate a new person_id, attach it to the active track, persist.

        Returns the new person_id so callers (e.g. the face tracker) can
        store it alongside the face encoding.
        """
        person_id = self.next_person_id()
        person = self.get_or_create(track_id)
        person.name = name
        person.persistent_id = person_id
        logger.info(f"Created person {person_id!r} ({name}) for track {track_id}")
        self._save(person)
        return person_id

    def register_enrolled(self, track_id: int, person_id: str) -> Person:
        """Attach a freshly enrolled (but unnamed) person_id to a track.

        Called when the face tracker auto-saves a new face and emits
        FACE_ENROLLED. The encoding is already on disk in the face DB;
        here we just create the matching in-memory record so the agent
        can accumulate dialogues/facts keyed by ``person_id`` while the
        name is still unknown. No JSON is written until a name is set
        (``is_identified`` is False without a real name).
        """
        person = self.get_or_create(track_id)
        if person.persistent_id == person_id:
            return person
        person.persistent_id = person_id
        with self._lock:
            self._stored.setdefault(person_id, {
                "persistent_id": person_id,
                "name": None,
            })
        logger.info(f"Registered enrolled face {person_id!r} for track {track_id}")
        return person

    def set_name(self, track_id: int, name: str) -> Optional[str]:
        """Attach a display name to the active person for this track.

        If the track already has a ``persistent_id`` (typical after auto
        enrollment), saves the full record. Otherwise allocates a new
        person_id. Returns the person_id or None on failure.
        """
        person = self.get_or_create(track_id)
        if person.persistent_id:
            person.name = name
            logger.info(
                f"Set name {name!r} on {person.persistent_id} "
                f"(track {track_id})"
            )
            self._save(person)
            return person.persistent_id
        return self.create_person(track_id, name)

    # --- Lookup by track_id ---

    def get(self, track_id: int) -> Optional[Person]:
        """Get person by current session track_id."""
        with self._lock:
            return self._active.get(track_id)

    def get_or_create(self, track_id: int) -> Person:
        """Get or create a person for this track_id."""
        with self._lock:
            if track_id not in self._active:
                now = time.time()
                self._active[track_id] = Person(
                    track_id=track_id,
                    first_met=now,
                    last_seen=now,
                    times_seen=1,
                )
            return self._active[track_id]

    def identify(self, track_id: int, person_id: str):
        """Link a track_id to a stored person by ID.

        Loads the JSON record, uses its stored ``name`` as the display
        name, and restores facts / asked_topics / dialogues. If no JSON
        exists for ``person_id``, treats it as an enrolled-but-unnamed
        person (the face tracker auto-saves encodings before a name is
        learned, so the tracker may report a person_id we've never
        written to disk).
        """
        with self._lock:
            stored = self._stored.get(person_id)
        if not stored:
            self.register_enrolled(track_id, person_id)
            return

        person = self.get_or_create(track_id)
        if person.persistent_id == person_id:
            return  # already identified

        person.persistent_id = person_id
        person.name = stored.get("name") or None
        person.facts = list(stored.get("facts", []))
        person.asked_topics = list(stored.get("asked_topics", []))
        person.summary = stored.get("summary", "")
        person.dialogues = [
            DialogueEntry(**e) for e in stored.get("dialogues", [])
        ]
        person.times_seen = stored.get("times_seen", 0) + 1
        stored_first = stored.get("first_met", 0.0)
        if stored_first and stored_first < person.first_met:
            person.first_met = stored_first
        logger.info(
            f"Identified track {track_id} as {person.name} ({person_id}, "
            f"restored {len(person.dialogues)} dialogues, {len(person.facts)} facts)"
        )
        self._save(person)

    def remove_track(self, track_id: int):
        """Remove a track_id from active session (e.g. face disappeared).
        Persists data if person was identified."""
        with self._lock:
            person = self._active.pop(track_id, None)
        if person and person.is_identified:
            self._save(person)

    # --- Lookup by name or ID (for persistent queries) ---

    def get_by_id(self, person_id: str) -> Optional[Person]:
        """Find a stored person by ID, loading from disk if needed."""
        with self._lock:
            for p in self._active.values():
                if p.persistent_id == person_id:
                    return p
            stored = self._stored.get(person_id)
        if stored:
            return Person.from_dict(stored)
        return None

    def get_by_name(self, name: str) -> Optional[Person]:
        """Find a person by display name (case-insensitive linear scan)."""
        needle = name.lower()
        with self._lock:
            for p in self._active.values():
                if p.name and p.name.lower() == needle:
                    return p
            for stored in self._stored.values():
                if (stored.get("name") or "").lower() == needle:
                    return Person.from_dict(stored)
        return None

    @property
    def active_ids(self) -> list[int]:
        with self._lock:
            return list(self._active.keys())

    @property
    def known_names(self) -> list[str]:
        """All names that have been persisted."""
        with self._lock:
            return [d.get("name", pid) for pid, d in self._stored.items()]

    @property
    def known_person_ids(self) -> list[str]:
        """All stored person IDs."""
        with self._lock:
            return list(self._stored.keys())

    @property
    def active_count(self) -> int:
        return len(self._active)

    # --- Updates (all by track_id) ---

    def update_seen(self, track_id: int, emotion: str = ""):
        """Record that we saw this track."""
        person = self.get_or_create(track_id)
        now = time.time()
        if person.last_seen == 0 or (now - person.last_seen) > 300:
            person.times_seen += 1
        person.last_seen = now
        if emotion:
            person.emotion = emotion

    def add_dialogue(self, track_id: int, speaker: str, text: str,
                     language: str = "", emotion: str = ""):
        """Add a dialogue entry."""
        person = self.get_or_create(track_id)
        person.dialogues.append(DialogueEntry(
            timestamp=time.time(), speaker=speaker, text=text,
            language=language, emotion=emotion,
        ))
        person.last_talked = time.time()
        if person.is_identified:
            self._save(person)

    @staticmethod
    def _fact_similar(a: str, b: str, person_name: Optional[str] = None) -> bool:
        """True if two facts are near-duplicates.

        Uses token-set Jaccard (>= 0.75) plus a subset check for the case
        where one fact is a condensed version of the other. ``person_name``
        is added to the stopword set so that "Joakim likes lingonsylt" and
        "likes lingonsylt" compare as duplicates even if one side still
        has the subject.
        """
        extra = set()
        if person_name:
            extra |= {t for t in person_name.lower().split() if len(t) > 1}
        ta = _fact_tokens(a, extra)
        tb = _fact_tokens(b, extra)
        if not ta or not tb:
            return a.strip().lower().rstrip(".") == b.strip().lower().rstrip(".")
        if _jaccard(ta, tb) >= 0.75:
            return True
        short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
        if len(short) >= 3 and short.issubset(long_):
            return True
        return False

    def add_fact(self, track_id: int, fact: str):
        """Add a fact about a person (strips subject, deduplicates by similarity)."""
        person = self.get_or_create(track_id)
        normalized = _strip_subject(fact, person.name)
        if not normalized:
            return
        for existing in person.facts:
            if self._fact_similar(normalized, existing, person.name):
                logger.debug(f"Duplicate fact skipped for track {track_id}: "
                             f"{normalized!r} ~ {existing!r}")
                return
        person.facts.append(normalized)
        logger.info(f"New fact about track {track_id} ({person.name or '?'}): {normalized}")
        if person.is_identified:
            self._save(person)

    def replace_fact(self, track_id: int, old_fact: str, new_fact: str):
        """Replace an existing fact with an updated version (normalized)."""
        person = self.get_or_create(track_id)
        normalized = _strip_subject(new_fact, person.name)
        if not normalized:
            return
        try:
            idx = person.facts.index(old_fact)
            person.facts[idx] = normalized
            logger.info(f"Replaced fact for track {track_id} ({person.name or '?'}): "
                        f"{old_fact!r} -> {normalized!r}")
        except ValueError:
            # Fall through to add_fact so dedup kicks in
            self.add_fact(track_id, new_fact)
            return
        if person.is_identified:
            self._save(person)

    def mark_topic_asked(self, track_id: int, topic: str):
        """Record that we've asked this person about ``topic``."""
        person = self.get_or_create(track_id)
        if topic not in person.asked_topics:
            person.asked_topics.append(topic)
            logger.info(f"Asked track {track_id} ({person.name or '?'}) about {topic}")
            if person.is_identified:
                self._save(person)

    def update_summary(self, track_id: int, summary: str):
        person = self.get_or_create(track_id)
        person.summary = summary
        if person.is_identified:
            self._save(person)

    # --- Context for LLM ---

    def get_context_for_llm(self, track_id: int, max_dialogues: int = 20) -> str:
        """Format a person's memory as context text for an LLM prompt."""
        person = self.get(track_id)
        if not person:
            return f"Unknown person (track {track_id}), no history."

        lines = []
        if person.name:
            lines.append(f"Person: {person.name}")
        else:
            lines.append(f"Unknown person (track {person.track_id})")

        if person.first_met:
            lines.append(f"First met: {person.first_met_dt.strftime('%Y-%m-%d %H:%M')}")

        ago = person.time_since_seen()
        if ago is not None:
            if ago < 60:
                lines.append(f"Last seen: {ago:.0f} seconds ago")
            elif ago < 3600:
                lines.append(f"Last seen: {ago/60:.0f} minutes ago")
            elif ago < 86400:
                lines.append(f"Last seen: {ago/3600:.1f} hours ago")
            else:
                lines.append(f"Last seen: {ago/86400:.1f} days ago")

        talk_ago = person.time_since_talked()
        if talk_ago is not None:
            if talk_ago < 3600:
                lines.append(f"Last talked: {talk_ago/60:.0f} minutes ago")
            else:
                lines.append(f"Last talked: {talk_ago/3600:.1f} hours ago")

        lines.append(f"Times seen: {person.times_seen}")

        if person.emotion:
            lines.append(f"Current emotion: {person.emotion}")

        if person.facts:
            lines.append(f"Known facts: {'; '.join(person.facts)}")

        if person.summary:
            lines.append(f"Relationship summary: {person.summary}")

        recent = person.dialogues[-max_dialogues:]
        if recent:
            lines.append(f"Recent conversation ({len(recent)} exchanges):")
            for d in recent:
                ts = datetime.fromtimestamp(d.timestamp).strftime("%H:%M")
                who = "Them" if d.speaker == "person" else "Us"
                lang_tag = f" [{d.language}]" if d.language else ""
                emo_tag = f" ({d.emotion})" if d.emotion else ""
                lines.append(f"  [{ts}] {who}{lang_tag}{emo_tag}: {d.text}")

        return "\n".join(lines)

    def get_short_context(self, track_id: int) -> str:
        """One-line summary."""
        person = self.get(track_id)
        if not person:
            return f"Unknown track {track_id}"
        parts = []
        if person.name:
            parts.append(person.name)
        else:
            parts.append(f"track#{person.track_id}")
        if person.times_seen > 1:
            parts.append(f"seen {person.times_seen}x")
        if person.emotion:
            parts.append(person.emotion)
        if person.facts:
            parts.append(person.facts[0])
        return ", ".join(parts)

    # --- Persistence ---

    def _save(self, person: Person):
        """Save a person to disk (only if identified).

        Uses atomic write (temp file + rename) to prevent corruption
        from concurrent saves.
        """
        if not person.is_identified or not person.persistent_id:
            return
        pid = person.persistent_id
        data = person.to_dict()
        with self._lock:
            self._stored[pid] = data
        os.makedirs(self._dir, exist_ok=True)
        fpath = os.path.join(self._dir, f"{pid}.json")
        tmp_path = fpath + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, fpath)

    def save_all(self):
        """Save all identified active people."""
        with self._lock:
            people = [p for p in self._active.values() if p.is_identified]
        for p in people:
            self._save(p)

    def delete(self, person_id: str) -> bool:
        """Remove a person from memory and disk by ID."""
        removed = False
        with self._lock:
            if person_id in self._stored:
                del self._stored[person_id]
                removed = True
            for tid in [t for t, p in self._active.items()
                        if p.persistent_id == person_id]:
                del self._active[tid]
                removed = True
        fpath = os.path.join(self._dir, f"{person_id}.json")
        if os.path.exists(fpath):
            os.remove(fpath)
            removed = True
        return removed

    def rename(self, person_id: str, new_name: str) -> bool:
        """Update the display name of a stored person. ID stays the same."""
        with self._lock:
            stored = self._stored.get(person_id)
            if not stored:
                return False
            stored["name"] = new_name
            for p in self._active.values():
                if p.persistent_id == person_id:
                    p.name = new_name

        fpath = os.path.join(self._dir, f"{person_id}.json")
        with open(fpath, "w") as f:
            json.dump(stored, f, indent=2, ensure_ascii=False)
        logger.info(f"Renamed {person_id} -> {new_name!r}")
        return True


# ---------------------------------------------------------------------------
# Offline cleanup helpers (used by `dedupe` and `compress` CLI commands)
# ---------------------------------------------------------------------------

def _dedupe_person_facts(
    facts: list[str], person_name: Optional[str]
) -> tuple[list[str], list[tuple]]:
    """Rule-based dedupe: strip subject, cluster by similarity, keep longest.

    Returns (new_facts, changes) where changes is a list of tuples:
      ("normalized", original, new)  — subject stripped
      ("dropped",    victim,   kept) — merged into an earlier, longer fact
      ("replaced",   old_kept, new)  — victim was longer; promoted over kept
    """
    changes: list[tuple] = []
    normalized: list[str] = []
    for f in facts:
        stripped = _strip_subject(f, person_name)
        if not stripped:
            changes.append(("dropped", f, ""))
            continue
        if stripped != f:
            changes.append(("normalized", f, stripped))
        normalized.append(stripped)

    kept: list[str] = []
    for f in normalized:
        merged = False
        for i, k in enumerate(kept):
            if PeopleMemory._fact_similar(f, k, person_name):
                if len(f) > len(k):
                    changes.append(("replaced", k, f))
                    kept[i] = f
                else:
                    changes.append(("dropped", f, k))
                merged = True
                break
        if not merged:
            kept.append(f)
    return kept, changes


def _load_face_encodings_by_person(db_dir: str) -> dict:
    """Load ``known_faces/faces.pkl`` and return {person_id: [encoding, ...]}.

    NOTE: this reads the legacy dlib db (``faces.pkl``, 128-d, Euclidean) used by
    the offline ``dedupe``/``similar`` CLIs. The live recognition path now defaults
    to the InsightFace backend, which writes ``faces_arcface.pkl`` (512-d, cosine);
    those CLIs and the L2 thresholds in ``main()`` are not yet updated for it.
    """
    path = os.path.join(db_dir, "faces.pkl")
    if not os.path.exists(path):
        return {}
    import pickle
    with open(path, "rb") as fh:
        db = pickle.load(fh)
    out: dict = {}
    for pid, enc in zip(db.get("person_ids", []), db.get("encodings", [])):
        out.setdefault(pid, []).append(enc)
    return out


def _pair_face_distances(encs_a: list, encs_b: list) -> tuple[float, float]:
    """Return (min, mean) L2 distance between two encoding sets.

    ``(inf, inf)`` if either set is empty.
    """
    if not encs_a or not encs_b:
        return float("inf"), float("inf")
    import numpy as _np
    A = _np.asarray(encs_a)
    B = _np.asarray(encs_b)
    # Pairwise L2: ||a-b|| for every a in A, b in B
    diffs = A[:, None, :] - B[None, :, :]
    d = _np.linalg.norm(diffs, axis=2)
    return float(d.min()), float(d.mean())


def _name_similarity(a: Optional[str], b: Optional[str]) -> float:
    if not a or not b:
        return 0.0
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _facts_similarity(facts_a: list[str], facts_b: list[str]) -> float:
    if not facts_a or not facts_b:
        return 0.0
    toks_a: set = set()
    toks_b: set = set()
    for f in facts_a:
        toks_a |= _fact_tokens(f)
    for f in facts_b:
        toks_b |= _fact_tokens(f)
    return _jaccard(toks_a, toks_b)


def _similarity_verdict(face_min: float, name: float, facts: float) -> Optional[str]:
    """Classify a pair. Returns a short label or None to hide."""
    if face_min < 0.40:
        return "LIKELY SAME"
    if face_min < 0.55 and (name > 0.85 or facts > 0.30):
        return "LIKELY SAME"
    if face_min < 0.55:
        return "maybe"
    if name > 0.90:
        return "name collision"
    if facts > 0.50:
        return "shared facts"
    return None


def _llm_compress_facts(facts: list[str], person_name: Optional[str]) -> list[str]:
    """Semantic compression: send facts to an Ollama LLM and take its merged list.

    Uses OLLAMA_URL (default http://localhost:11434/v1) and COMPRESS_MODEL
    (default qwen3:8b). Returns an empty list on failure so the caller can
    decide not to write.
    """
    from pydantic_ai import Agent as _Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434/v1")
    model_name = os.environ.get("COMPRESS_MODEL", "qwen3:8b")
    provider = OpenAIProvider(base_url=ollama_url, api_key="ollama")
    model = OpenAIChatModel(model_name, provider=provider)
    agent = _Agent(
        model,
        output_type=list[str],
        system_prompt=(
            "You deduplicate and compress a list of facts about one person. "
            "Output ONLY a JSON array of bare-predicate strings (no subject, "
            "no person name, no pronouns like 'he'/'she'/'the person'). "
            "Merge paraphrases. Preserve every unique piece of information. "
            "Do not invent facts. Keep each item concise."
        ),
    )
    subject = person_name or "the person"
    joined = "\n".join(f"- {f}" for f in facts)
    prompt = (
        f"Facts about {subject}:\n{joined}\n\n"
        "Return the deduplicated, canonical list."
    )
    try:
        result = agent.run_sync(prompt)
        out = result.output if isinstance(result.output, list) else []
        return [s.strip() for s in out if isinstance(s, str) and s.strip()]
    except Exception as e:
        logger.error(f"LLM compress failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Standalone: inspect/manage people memory from CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="People memory CLI")
    parser.add_argument("--dir", default="people", help="Storage directory")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="List all known people")
    show_p = sub.add_parser("show", help="Show a person's full memory")
    show_p.add_argument("name")
    context_p = sub.add_parser("context", help="Show LLM context for a person")
    context_p.add_argument("name")
    fact_p = sub.add_parser("add-fact", help="Add a fact about a person")
    fact_p.add_argument("name")
    fact_p.add_argument("fact")
    dedupe_p = sub.add_parser("dedupe",
        help="Rule-based cleanup: strip subject prefixes, merge near-duplicates")
    dedupe_p.add_argument("--person", help="Limit to one person_id (e.g. p001)")
    dedupe_p.add_argument("--apply", action="store_true",
        help="Write changes (without this, dry-run only)")
    compress_p = sub.add_parser("compress",
        help="LLM-based semantic compression via Ollama")
    compress_p.add_argument("person", help="person_id or name")
    compress_p.add_argument("--apply", action="store_true",
        help="Write changes (without this, dry-run only)")
    similar_p = sub.add_parser("similar",
        help="Find people records that may be duplicates of each other")
    similar_p.add_argument("--person",
        help="Compare only this person_id against all others")
    similar_p.add_argument("--db-dir", default="known_faces",
        help="Face encoding database directory (default: known_faces)")
    similar_p.add_argument("--all", action="store_true",
        help="Show every pair, even ones below the verdict thresholds")
    sub.add_parser("shell", help="Interactive shell for poking at people memory")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    mem = PeopleMemory(storage_dir=args.dir)
    mem.load()

    if args.command == "list" or args.command is None:
        names = mem.known_names
        if not names:
            print("No people in memory.")
        else:
            print(f"{len(names)} people:\n")
            for name in sorted(names):
                person = mem.get_by_name(name)
                if person:
                    parts = [name]
                    if person.times_seen > 1:
                        parts.append(f"seen {person.times_seen}x")
                    if person.facts:
                        parts.append(person.facts[0])
                    ago = person.time_since_seen()
                    if ago and ago > 3600:
                        parts.append(f"last {ago/3600:.0f}h ago")
                    print(f"  {', '.join(parts)}")

    elif args.command == "show":
        person = mem.get_by_name(args.name)
        if not person:
            print(f"No person named '{args.name}'")
        else:
            print(json.dumps(person.to_dict(), indent=2, ensure_ascii=False))

    elif args.command == "context":
        person = mem.get_by_name(args.name)
        if not person:
            print(f"No person named '{args.name}'")
        else:
            # Use a fake track_id for display
            person.track_id = -1
            mem._active[-1] = person
            print(mem.get_context_for_llm(-1))

    elif args.command == "add-fact":
        person = mem.get_by_name(args.name)
        if not person:
            print(f"No person named '{args.name}', creating...")
            mem.create_person(0, args.name)
            person = mem.get(0)
        else:
            mem._active[0] = person
        mem.add_fact(0 if person.track_id == 0 else person.track_id, args.fact)
        print(f"Added fact about {args.name}: {args.fact}")

    elif args.command == "dedupe":
        pids = [args.person] if args.person else sorted(mem._stored.keys())
        total_before = total_after = total_changes = 0
        for pid in pids:
            stored = mem._stored.get(pid)
            if not stored:
                print(f"  {pid}: not found")
                continue
            name = stored.get("name")
            facts = list(stored.get("facts", []))
            new_facts, changes = _dedupe_person_facts(facts, name)
            total_before += len(facts)
            total_after += len(new_facts)
            total_changes += len(changes)
            head = f"{pid} ({name or 'unnamed'}): {len(facts)} -> {len(new_facts)}"
            if not changes:
                print(f"{head}  (unchanged)")
                continue
            print(head)
            for kind, a, b in changes:
                if kind == "normalized":
                    print(f"  ~ {a!r}")
                    print(f"    -> {b!r}")
                elif kind == "dropped":
                    print(f"  - drop {a!r}")
                    if b:
                        print(f"    (merged into {b!r})")
                elif kind == "replaced":
                    print(f"  * replaced {a!r}")
                    print(f"    with     {b!r}")
            if args.apply:
                stored["facts"] = new_facts
                fpath = os.path.join(mem._dir, f"{pid}.json")
                tmp = fpath + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(stored, f, indent=2, ensure_ascii=False)
                os.replace(tmp, fpath)
                print("  [written]")
        suffix = "" if args.apply else "  (dry-run; pass --apply to write)"
        print(f"\nTotal: {total_before} -> {total_after} facts, "
              f"{total_changes} changes across {len(pids)} people.{suffix}")

    elif args.command == "compress":
        # Accept either person_id or name
        person = mem.get_by_id(args.person) or mem.get_by_name(args.person)
        if not person:
            print(f"No person found for {args.person!r}")
            return
        pid = person.persistent_id
        name = person.name
        facts = list(person.facts)
        if not facts:
            print(f"{pid} ({name}): no facts to compress")
            return
        print(f"Compressing {len(facts)} facts for {pid} ({name}) via LLM...")
        new_facts = _llm_compress_facts(facts, name)
        if not new_facts:
            print("LLM returned nothing; aborting.")
            return
        # Always run the rule-based dedup over the LLM output too, to
        # catch any residual subjects/whitespace the model left behind.
        new_facts, _ = _dedupe_person_facts(new_facts, name)
        print(f"\nBefore ({len(facts)}):")
        for f in facts:
            print(f"  - {f}")
        print(f"\nAfter ({len(new_facts)}):")
        for f in new_facts:
            print(f"  - {f}")
        if args.apply:
            stored = mem._stored.get(pid, {})
            stored["facts"] = new_facts
            mem._stored[pid] = stored
            fpath = os.path.join(mem._dir, f"{pid}.json")
            tmp = fpath + ".tmp"
            with open(tmp, "w") as f:
                json.dump(stored, f, indent=2, ensure_ascii=False)
            os.replace(tmp, fpath)
            print(f"\n[written to {fpath}]")
        else:
            print("\n(dry-run; pass --apply to write)")

    elif args.command == "similar":
        pids = sorted(mem._stored.keys())
        if not pids:
            print("No people records.")
            return
        encs_by_pid = _load_face_encodings_by_person(args.db_dir)
        if not encs_by_pid:
            print(f"Warning: no face encodings loaded from {args.db_dir}/faces.pkl — "
                  "falling back to name+facts only.")

        def meta(pid):
            s = mem._stored.get(pid, {})
            return s.get("name"), list(s.get("facts", []))

        focus = [args.person] if args.person else pids
        shown = 0
        for i, pid_a in enumerate(focus):
            if pid_a not in mem._stored:
                print(f"  {pid_a}: not found"); continue
            name_a, facts_a = meta(pid_a)
            encs_a = encs_by_pid.get(pid_a, [])
            start = 0 if args.person else i + 1
            for pid_b in pids[start:]:
                if pid_b == pid_a:
                    continue
                name_b, facts_b = meta(pid_b)
                encs_b = encs_by_pid.get(pid_b, [])
                face_min, face_mean = _pair_face_distances(encs_a, encs_b)
                name_sim = _name_similarity(name_a, name_b)
                facts_sim = _facts_similarity(facts_a, facts_b)
                verdict = _similarity_verdict(face_min, name_sim, facts_sim)
                if verdict is None and not args.all:
                    continue
                shown += 1
                face_txt = (f"{face_min:.2f}/{face_mean:.2f}"
                            if face_min != float("inf") else "n/a")
                label = verdict or "below thresholds"
                print(f"{pid_a} ({name_a or '?'})  ~  {pid_b} ({name_b or '?'})")
                print(f"  face min/mean: {face_txt}  "
                      f"name: {name_sim:.2f}  facts: {facts_sim:.2f}  "
                      f"-> {label}")
        if shown == 0:
            print("No suspicious pairs."
                  " Use --all to see every pair regardless of threshold.")
        else:
            print(f"\n{shown} pair(s) shown.")

    elif args.command == "shell":
        from debug_shell import run_shell
        run_shell(mem)


if __name__ == "__main__":
    main()
