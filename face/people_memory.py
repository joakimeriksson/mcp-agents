"""
People memory: what the agent remembers about the people it has met.

Three kinds of memory, written at different times:

* **episodic** -- the raw conversation (``dialogues``), appended as it
  happens and capped at MAX_DIALOGUES lines per person.
* **facts** -- short English statements ("likes sushi"), each with when it
  was first and last heard, how often, and the person's own words it came
  from. They are written once per VISIT (``apply_visit_facts``), not after
  every sentence, and only when the quoted words really occur in what the
  person said: one misheard sentence used to become a permanent fact
  ("talking with a coconut voice"). A fact is *confirmed* when it comes up
  again in a later visit or the person stated it outright; an unconfirmed
  fact is forgotten after FACT_TTL_DAYS.
* **summary** -- one or two sentences, rewritten at the end of each visit.

The prompt gets a small, fixed budget of this (``get_context_for_llm``).

People are keyed by a stable ``person_id`` (``p001``) shared with the face
database; the display name lives inside the record and can change freely.
At runtime the active session maps ``track_id -> Person`` via
``identify(track_id, person_id)``.

Nothing here is service-specific. What a service did for someone ("was
given a Dumle") is short-term visit memory (visit_memory.py, fed by the
service policy's ``remember`` hook); interview topics come from
languages.toml and the service policy.
"""

import json
import os
import re
import tempfile
import time
import threading
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

logger = logging.getLogger("people_memory")

FACT_TTL_DAYS = 30          # an unconfirmed fact is forgotten after this
MAX_DIALOGUES = 60          # conversation lines kept per person
VISIT_GAP_S = 600.0         # a silence this long ends a visit
CONTEXT_CHARS = 700         # what the prompt gets about a person, at most
MAX_CONFIRMED_IN_CONTEXT = 5
MAX_UNSURE_IN_CONTEXT = 2


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
class Fact:
    """One thing we know about a person, with where it came from."""
    text: str                   # bare English predicate: "likes sushi"
    first_seen: float = 0.0
    last_seen: float = 0.0
    count: int = 1              # visits in which it came up
    quote: str = ""             # the person's own words it was taken from
    explicit: bool = False      # they said it outright about themselves

    @property
    def confirmed(self) -> bool:
        return self.explicit or self.count >= 2

    @staticmethod
    def load(raw, now: float) -> Optional["Fact"]:
        """From a stored dict, or a plain string (records written before
        facts had provenance: unconfirmed, their clock starts now)."""
        if isinstance(raw, str):
            return Fact(text=raw, first_seen=now, last_seen=now) if raw.strip() else None
        if isinstance(raw, dict) and str(raw.get("text", "")).strip():
            return Fact(
                text=str(raw["text"]).strip(),
                first_seen=float(raw.get("first_seen") or now),
                last_seen=float(raw.get("last_seen") or now),
                count=max(1, int(raw.get("count") or 1)),
                quote=str(raw.get("quote") or ""),
                explicit=bool(raw.get("explicit", False)),
            )
        return None


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
    facts: list[Fact] = field(default_factory=list)
    asked_topics: list[str] = field(default_factory=list)
    summary: str = ""
    # Dialogue up to this time has been turned into facts/summary.
    consolidated_upto: float = 0.0

    # Persistent ID (e.g. "p001") — stable across renames, matches filename.
    persistent_id: str = ""

    @property
    def is_identified(self) -> bool:
        """True if the person has a real name (not None, not a person ID like 'p001')."""
        return self.name is not None and not _is_person_id(self.name)

    @property
    def fact_texts(self) -> list[str]:
        return [f.text for f in self.facts]

    def to_dict(self) -> dict:
        return {
            "persistent_id": self.persistent_id,
            "name": self.name,
            "first_met": self.first_met,
            "last_seen": self.last_seen,
            "last_talked": self.last_talked,
            "times_seen": self.times_seen,
            "emotion": self.emotion,
            "facts": [asdict(f) for f in self.facts],
            "asked_topics": list(self.asked_topics),
            "summary": self.summary,
            "consolidated_upto": self.consolidated_upto,
            "dialogues": [asdict(e) for e in self.dialogues[-MAX_DIALOGUES:]],
        }

    @staticmethod
    def from_dict(d: dict, track_id: int = 0, now: Optional[float] = None) -> "Person":
        now = time.time() if now is None else now
        p = Person(
            track_id=track_id,
            persistent_id=d.get("persistent_id", d.get("name", "")) or "",
            name=d.get("name"),
            first_met=d.get("first_met", 0.0),
            last_seen=d.get("last_seen", 0.0),
            last_talked=d.get("last_talked", 0.0),
            times_seen=d.get("times_seen", 0),
            emotion=d.get("emotion", ""),
        )
        p._fill_from(d, now)
        return p

    def _fill_from(self, d: dict, now: float):
        """Restore the remembered parts of a stored record."""
        self.facts = [f for f in (Fact.load(x, now) for x in d.get("facts", [])) if f]
        self.asked_topics = list(d.get("asked_topics", []))
        self.summary = d.get("summary", "")
        self.dialogues = [DialogueEntry(**e) for e in d.get("dialogues", [])]
        # Old records have no marker: their dialogue was already mined
        # sentence by sentence, so do not mine it again.
        self.consolidated_upto = d.get(
            "consolidated_upto",
            self.dialogues[-1].timestamp if self.dialogues else 0.0)

    @property
    def last_language(self) -> str:
        """Language from the most recent dialogue entry that has one, or ''."""
        for d in reversed(self.dialogues):
            if d.language:
                return d.language
        return ""

    def missing_topics(self, topics: list[str]) -> list[str]:
        """Of the interview ``topics`` on offer, those not asked yet."""
        return [t for t in topics if t not in self.asked_topics]

    def current_visit(self, now: Optional[float] = None) -> list[DialogueEntry]:
        """The dialogue of the visit going on now: the trailing lines with no
        gap longer than VISIT_GAP_S between them (empty if the last line is
        that old)."""
        now = time.time() if now is None else now
        out: list[DialogueEntry] = []
        t_next = now
        for d in reversed(self.dialogues):
            if t_next - d.timestamp > VISIT_GAP_S:
                break
            out.append(d)
            t_next = d.timestamp
        return out[::-1]

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
# Fact normalization, junk filter, grounding, similarity
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

# Not facts about the person: what they talked about, asked or said (one
# misheard sentence each: "mentioned 123", "talking with a coconut voice"),
# and their name (that is the record's name, set by the name flow).
_JUNK_FACT_RE = re.compile(
    r"^(mention(s|ed)?|talk(s|ed|ing)?|said|says|asked|asks|wants to know|"
    r"(is|was) (talking|asking|saying|speaking)|name is|is called|is named)\b",
    re.IGNORECASE)


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


def is_junk_fact(fact: str, person_name: Optional[str] = None) -> bool:
    """True for things that are not facts about the person: a lone word
    ("silverplate"), what they talked about or asked, their name."""
    f = _strip_subject(fact, person_name)
    return not f or len(f.split()) < 2 or bool(_JUNK_FACT_RE.match(f))


def _words(s: str) -> list[str]:
    return re.findall(r"[^\W_]+", (s or "").lower())


def is_grounded(quote: str, heard: str) -> bool:
    """True if ``quote`` is words the person really said in ``heard``: at
    least two words, and either a verbatim run of what they said or at most
    one word in five missing from it (Whisper and the model disagree on
    small words). A fact whose quote fails this was made up or misread."""
    q = _words(quote)
    if len(q) < 2:
        return False
    h = _words(heard)
    if f" {' '.join(q)} " in f" {' '.join(h)} ":
        return True
    present = set(h)
    missing = sum(1 for w in q if w not in present)
    return missing <= len(q) // 5


# First-person words: a fact "stated outright" must be said about oneself.
# A TV or radio in the room ("... i Strömsunds kommun") is grounded in the
# transcript too, but never says "I".
_FIRST_PERSON = {
    "i", "im", "my", "me", "mine", "we", "our",
    "jag", "mig", "min", "mitt", "mina", "vi", "vår", "vårt", "våra",
    "ich", "mich", "mir", "mein", "meine", "meinen", "meiner", "wir", "unser",
    "je", "j", "moi", "mon", "ma", "mes", "nous", "notre",
    "yo", "mi", "mis", "soy", "estoy", "tengo", "nosotros", "nuestro",
    "io", "mio", "mia", "miei", "sono", "ho", "noi", "nostro",
}


def is_first_person(quote: str) -> bool:
    # English "I" only as a capital: Swedish/Italian "i" means "in".
    return any(w == "I" or (w.lower() in _FIRST_PERSON and w.lower() != "i")
               for w in re.findall(r"[^\W\d_]+", quote or ""))


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


def _merge_facts(into: list[Fact], extra: list[Fact], person_name: Optional[str]) -> list[Fact]:
    """``into`` plus ``extra``; near-duplicates combine their history (counts
    add up, the longer wording wins)."""
    out = [Fact(**asdict(f)) for f in into]
    for f in extra:
        for k in out:
            if PeopleMemory._fact_similar(f.text, k.text, person_name):
                k.count += f.count
                k.first_seen = min(k.first_seen, f.first_seen)
                k.last_seen = max(k.last_seen, f.last_seen)
                k.explicit = k.explicit or f.explicit
                if len(f.text) > len(k.text):
                    k.text = f.text
                k.quote = k.quote or f.quote
                break
        else:
            out.append(Fact(**asdict(f)))
    return out


def _ago(seconds: float) -> str:
    if seconds < 3600:
        return f"{max(1, round(seconds / 60))} min ago"
    if seconds < 86400:
        return f"{round(seconds / 3600)} h ago"
    return f"{round(seconds / 86400)} days ago"


class PeopleMemory:
    """In-memory people database with JSON file persistence.

    Each stored person has a stable ``person_id`` (e.g. ``p001``) which
    is the filename and the key in ``_stored``. Display names live inside
    the JSON and can be changed freely without touching the ID.

    Thread-safe: the camera thread, the conversation and the background
    consolidation all write. ``_lock`` guards the records; ``_io_lock``
    serializes file writes, and a record is snapshotted inside it so an
    older snapshot can never overwrite a newer one.
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
        self._lock = threading.RLock()
        self._io_lock = threading.Lock()

    def load(self):
        """Load all person files from disk into the persistent store."""
        os.makedirs(self._dir, exist_ok=True)
        count = 0
        for fname in os.listdir(self._dir):
            if not fname.endswith(".json") or fname.startswith("."):
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
        with self._lock:
            person_id = self.next_person_id()
            person = self.get_or_create(track_id)
            person.name = name
            person.persistent_id = person_id
            # Reserve the id before the (unlocked) write below.
            self._stored.setdefault(person_id, {"persistent_id": person_id, "name": name})
        logger.info(f"Created person {person_id!r} ({name}) for track {track_id}")
        self._save(person)
        return person_id

    def register_enrolled(self, track_id: int, person_id: str) -> Person:
        """Attach a freshly enrolled (but unnamed) person_id to a track.

        Called when the face tracker auto-saves a new face and emits
        FACE_ENROLLED. The encoding is already on disk in the face DB;
        here we just create the matching in-memory record so the agent
        can accumulate dialogues while the name is still unknown. No JSON
        is written until a name is set (``is_identified`` is False
        without a real name).
        """
        with self._lock:
            person = self.get_or_create(track_id)
            if person.persistent_id == person_id:
                return person
            person.persistent_id = person_id
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
            with self._lock:
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
        name, and restores facts / asked_topics / dialogues (forgetting
        unconfirmed facts that have expired). If no JSON exists for
        ``person_id``, treats it as an enrolled-but-unnamed person (the
        face tracker auto-saves encodings before a name is learned).
        """
        with self._lock:
            stored = self._stored.get(person_id) or None
            if stored is not None:
                person = self.get_or_create(track_id)
                if person.persistent_id == person_id:
                    return  # already identified
                person.persistent_id = person_id
                person.name = stored.get("name") or None
                person._fill_from(stored, time.time())
                person.times_seen = stored.get("times_seen", 0) + 1
                stored_first = stored.get("first_met", 0.0)
                if stored_first and stored_first < person.first_met:
                    person.first_met = stored_first
                expired = self._expire_facts(person)
        if stored is None:
            self.register_enrolled(track_id, person_id)
            return
        logger.info(
            f"Identified track {track_id} as {person.name} ({person_id}, "
            f"restored {len(person.dialogues)} dialogues, {len(person.facts)} facts"
            + (f", forgot {expired} expired" if expired else "") + ")"
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
            return [d.get("name") or pid for pid, d in self._stored.items()]

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
        with self._lock:
            person = self.get_or_create(track_id)
            now = time.time()
            if person.last_seen == 0 or (now - person.last_seen) > 300:
                person.times_seen += 1
            person.last_seen = now
            if emotion:
                person.emotion = emotion

    def add_dialogue(self, track_id: int, speaker: str, text: str,
                     language: str = "", emotion: str = ""):
        """Add a dialogue entry (the oldest lines beyond MAX_DIALOGUES go)."""
        with self._lock:
            person = self.get_or_create(track_id)
            now = time.time()
            person.dialogues.append(DialogueEntry(
                timestamp=now, speaker=speaker, text=text,
                language=language, emotion=emotion,
            ))
            del person.dialogues[:-MAX_DIALOGUES]
            person.last_talked = now
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

    def add_fact(self, track_id: int, fact: str, quote: str = "",
                 explicit: bool = False, now: Optional[float] = None) -> str:
        """Record a fact. Returns "new", "again" (a near-duplicate was heard
        again: its count goes up, which confirms it) or "junk"."""
        now = time.time() if now is None else now
        with self._lock:
            person = self.get_or_create(track_id)
            normalized = _strip_subject(fact, person.name)
            if is_junk_fact(normalized, person.name):
                logger.info(f"Not a fact, skipped: {fact!r}")
                return "junk"
            for existing in person.facts:
                if self._fact_similar(normalized, existing.text, person.name):
                    existing.count += 1
                    existing.last_seen = now
                    existing.explicit = existing.explicit or explicit
                    existing.quote = existing.quote or quote
                    status = "again"
                    logger.info(f"Fact heard again ({existing.count}x) about track "
                                f"{track_id}: {existing.text!r}")
                    break
            else:
                person.facts.append(Fact(text=normalized, first_seen=now, last_seen=now,
                                         quote=quote, explicit=explicit))
                status = "new"
                logger.info(f"New fact about track {track_id} ({person.name or '?'}): "
                            f"{normalized}{' (stated)' if explicit else ''}")
        if person.is_identified:
            self._save(person)
        return status

    def apply_visit_facts(self, track_id: int, items: list, heard: str,
                          summary: Optional[str] = None,
                          now: Optional[float] = None) -> dict:
        """Store what the end-of-visit pass found. ``items`` are dicts
        ``{"fact", "quote", "explicit"}``; ``heard`` is everything the person
        said this visit. A fact is kept only if it is a real fact
        (``is_junk_fact``) and its quote is words they really said
        (``is_grounded``); the same fact twice in one visit counts once. It
        counts as stated outright only if the quote is in the first person.
        The summary is replaced only when the visit taught us something:
        a summary of a visit with no facts is a summary of noise.
        Returns {"kept": [(fact, status)], "dropped": [(fact, reason)]}."""
        kept, dropped, this_visit = [], [], []
        person = self.get(track_id)
        name = person.name if person else None
        for it in items or []:
            if not isinstance(it, dict):
                continue
            fact = str(it.get("fact") or "").strip()
            quote = str(it.get("quote") or "").strip()
            if is_junk_fact(fact, name):
                dropped.append((fact, "not a fact about them"))
            elif not is_grounded(quote, heard):
                dropped.append((fact, f"quote not in what they said: {quote!r}"))
            elif any(self._fact_similar(fact, f, name) for f in this_visit):
                dropped.append((fact, "same fact twice this visit"))
            else:
                this_visit.append(fact)
                explicit = bool(it.get("explicit")) and is_first_person(quote)
                status = self.add_fact(track_id, fact, quote=quote,
                                       explicit=explicit, now=now)
                kept.append((fact, status))
        if kept and summary and summary.strip():
            self.update_summary(track_id, summary.strip()[:300])
        for fact, why in dropped:
            logger.info(f"Visit fact dropped ({why}): {fact!r}")
        return {"kept": kept, "dropped": dropped}

    def pending_visit_lines(self, track_id: int) -> list[DialogueEntry]:
        """Dialogue not yet turned into facts (see ``mark_consolidated``)."""
        with self._lock:
            person = self._active.get(track_id)
            if not person:
                return []
            return [d for d in person.dialogues if d.timestamp > person.consolidated_upto]

    def mark_consolidated(self, track_id: int, upto: float):
        with self._lock:
            person = self._active.get(track_id)
            if not person:
                return
            person.consolidated_upto = max(person.consolidated_upto, upto)
        if person.is_identified:
            self._save(person)

    def _expire_facts(self, person: Person, now: Optional[float] = None) -> int:
        """Forget unconfirmed facts not heard for FACT_TTL_DAYS. Returns how
        many went. Caller holds the lock."""
        cutoff = (time.time() if now is None else now) - FACT_TTL_DAYS * 86400
        before = len(person.facts)
        person.facts = [f for f in person.facts if f.confirmed or f.last_seen >= cutoff]
        return before - len(person.facts)

    def mark_topic_asked(self, track_id: int, topic: str):
        """Record that we've asked this person about ``topic``."""
        with self._lock:
            person = self.get_or_create(track_id)
            if topic in person.asked_topics:
                return
            person.asked_topics.append(topic)
        logger.info(f"Asked track {track_id} ({person.name or '?'}) about {topic}")
        if person.is_identified:
            self._save(person)

    def update_summary(self, track_id: int, summary: str):
        with self._lock:
            person = self.get_or_create(track_id)
            person.summary = summary
        if person.is_identified:
            self._save(person)

    # --- Context for LLM ---

    def get_context_for_llm(self, track_id: int, max_dialogues: int = 4,
                            max_chars: int = CONTEXT_CHARS) -> str:
        """What the prompt gets about a person, in at most ``max_chars``:
        who they are, confirmed facts, last visit's summary, the last lines
        of this visit, then (if room is left) facts heard only once, marked
        as possibly misheard. Junk from older records never gets in."""
        with self._lock:
            person = self.get(track_id)
            if not person:
                return "Unknown person."
            now = time.time()
            name = person.name if person.is_identified else None
            facts = [f for f in person.facts if not is_junk_fact(f.text, name)]
            whole_visit = person.current_visit(now)
            visit = whole_visit[-max_dialogues:] if max_dialogues else []
            summary = person.summary
            lang = person.last_language
            head = [f"Person: {name}" if name else "Person: name not known"]
            if person.times_seen > 1:
                head.append(f"met {person.times_seen} times")
            if lang:
                head.append(f"speaks {lang}")
            n_before = len(person.dialogues) - len(whole_visit)
            if n_before > 0:
                head.append("last talked "
                            + _ago(now - person.dialogues[n_before - 1].timestamp))

        confirmed = sorted((f for f in facts if f.confirmed),
                           key=lambda f: (-f.count, -f.last_seen))[:MAX_CONFIRMED_IN_CONTEXT]
        unsure = sorted((f for f in facts if not f.confirmed),
                        key=lambda f: -f.last_seen)[:MAX_UNSURE_IN_CONTEXT]
        lines = [", ".join(head)]
        if confirmed:
            lines.append("About them: " + "; ".join(f.text for f in confirmed))
        if summary:
            lines.append(f"Earlier: {summary}")
        if visit:
            lines.append("This visit:")
            for d in visit:
                who = "Them" if d.speaker == "person" else "You"
                lines.append(f"  {who}: {d.text}")
        if unsure:
            lines.append("Heard once, may be misheard: " + "; ".join(f.text for f in unsure))

        # Keep within budget: drop the oldest visit lines first, then trim.
        while len("\n".join(lines)) > max_chars and "This visit:" in lines:
            i = lines.index("This visit:")
            if i + 1 < len(lines) and lines[i + 1].startswith("  "):
                del lines[i + 1]
            else:
                del lines[i]
        text = "\n".join(lines)
        return text if len(text) <= max_chars else text[:max_chars - 1] + "…"

    def get_short_context(self, track_id: int) -> str:
        """One-line summary."""
        person = self.get(track_id)
        if not person:
            return f"Unknown track {track_id}"
        parts = [person.name or f"track#{person.track_id}"]
        if person.times_seen > 1:
            parts.append(f"seen {person.times_seen}x")
        if person.emotion:
            parts.append(person.emotion)
        confirmed = [f.text for f in person.facts if f.confirmed]
        if confirmed:
            parts.append(confirmed[0])
        return ", ".join(parts)

    # --- Persistence ---

    def _write(self, pid: str, data: dict):
        """Atomic write: a private temp file, then rename over the record."""
        os.makedirs(self._dir, exist_ok=True)
        fpath = os.path.join(self._dir, f"{pid}.json")
        fd, tmp = tempfile.mkstemp(dir=self._dir, prefix=f".{pid}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, fpath)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _save(self, person: Person):
        """Save a person to disk (only if identified)."""
        if not person.is_identified or not person.persistent_id:
            return
        with self._io_lock:
            with self._lock:
                pid = person.persistent_id
                # Not in the store = forgotten (or merged away) while a
                # background thread still held the Person: do not bring it back.
                if not pid or pid not in self._stored:
                    return
                data = person.to_dict()
                self._stored[pid] = data
            self._write(pid, data)

    def save_all(self):
        """Save all identified active people."""
        with self._lock:
            people = [p for p in self._active.values() if p.is_identified]
        for p in people:
            self._save(p)

    def delete(self, person_id: str) -> bool:
        """Remove a person from memory and disk by ID ("forget me")."""
        removed = False
        with self._io_lock:
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
        if removed:
            logger.info(f"Forgot {person_id}")
        return removed

    forget = delete

    def rename(self, person_id: str, new_name: str) -> bool:
        """Update the display name of a stored person. ID stays the same."""
        with self._io_lock:
            with self._lock:
                stored = self._stored.get(person_id)
                if not stored:
                    return False
                stored["name"] = new_name
                for p in self._active.values():
                    if p.persistent_id == person_id:
                        p.name = new_name
                data = dict(stored)
            self._write(person_id, data)
        logger.info(f"Renamed {person_id} -> {new_name!r}")
        return True

    def merge(self, src: str, dst: str) -> Optional[dict]:
        """Merge record ``src`` into ``dst`` (one person enrolled twice):
        facts combine their history, dialogue interleaves, counts add up;
        ``dst`` keeps its name unless it has none. ``src`` is deleted.
        Returns the merged record, or None if either is missing. (The face
        encodings are merged separately: FaceDatabase.rename_person.)"""
        if src == dst:
            return None
        with self._io_lock:
            with self._lock:
                a, b = self._stored.get(src), self._stored.get(dst)
                if not a or not b:
                    return None
                now = time.time()
                pa, pb = Person.from_dict(a, now=now), Person.from_dict(b, now=now)
                m = pb
                m.persistent_id = dst
                if not m.is_identified and pa.is_identified:
                    m.name = pa.name
                m.facts = _merge_facts(pb.facts, pa.facts, m.name)
                m.dialogues = sorted(pa.dialogues + pb.dialogues,
                                     key=lambda d: d.timestamp)[-MAX_DIALOGUES:]
                m.times_seen = pa.times_seen + pb.times_seen
                m.first_met = min(x for x in (pa.first_met, pb.first_met, now) if x)
                m.last_seen = max(pa.last_seen, pb.last_seen)
                m.last_talked = max(pa.last_talked, pb.last_talked)
                m.asked_topics = list(dict.fromkeys(pb.asked_topics + pa.asked_topics))
                m.summary = pb.summary or pa.summary
                m.consolidated_upto = max(pa.consolidated_upto, pb.consolidated_upto)
                data = m.to_dict()
                self._stored[dst] = data
                del self._stored[src]
                for p in self._active.values():
                    if p.persistent_id in (src, dst):
                        p.persistent_id = dst
                        p.name = m.name
                        p._fill_from(data, now)
            self._write(dst, data)
            src_path = os.path.join(self._dir, f"{src}.json")
            if os.path.exists(src_path):
                os.remove(src_path)
        logger.info(f"Merged {src} into {dst} ({len(m.facts)} facts, "
                    f"{len(m.dialogues)} dialogue lines)")
        return data


# ---------------------------------------------------------------------------
# Offline cleanup helpers (used by the `dedupe`, `similar` and `merge` CLI)
# ---------------------------------------------------------------------------

def _dedupe_person_facts(
    facts: list, person_name: Optional[str]
) -> tuple[list[Fact], list[tuple]]:
    """Rule-based cleanup: strip subject, drop junk, merge near-duplicates
    (their counts add up). Accepts Fact objects, dicts or plain strings.

    Returns (new_facts, changes) where changes is a list of tuples:
      ("normalized", original, new)  — subject stripped
      ("dropped",    victim,   why)  — junk, or merged into ``why``
    """
    now = time.time()
    changes: list[tuple] = []
    cleaned: list[Fact] = []
    for raw in facts:
        f = raw if isinstance(raw, Fact) else Fact.load(raw, now)
        if f is None:
            continue
        stripped = _strip_subject(f.text, person_name)
        if is_junk_fact(stripped, person_name):
            changes.append(("dropped", f.text, "not a fact about them"))
            continue
        if stripped != f.text:
            changes.append(("normalized", f.text, stripped))
        cleaned.append(Fact(**{**asdict(f), "text": stripped}))

    kept: list[Fact] = []
    for f in cleaned:
        match = next((k for k in kept
                      if PeopleMemory._fact_similar(f.text, k.text, person_name)), None)
        if match is None:
            kept.append(f)
            continue
        merged = _merge_facts([match], [f], person_name)[0]
        changes.append(("dropped", f.text if merged.text == match.text else match.text,
                        f"merged into {merged.text!r}"))
        kept[kept.index(match)] = merged
    return kept, changes


def _load_face_encodings_by_person(db_dir: str) -> tuple[dict, str]:
    """Load the face db and return ({person_id: [encoding, ...]}, metric).

    Prefers the InsightFace db (``faces_arcface.pkl``, 512-d, cosine) and falls
    back to the legacy dlib db (``faces.pkl``, 128-d, Euclidean); the returned
    metric tells callers how to compare the encodings. ``({}, "cosine")`` when
    neither db exists.
    """
    import pickle
    for fname, metric in (("faces_arcface.pkl", "cosine"), ("faces.pkl", "euclidean")):
        path = os.path.join(db_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path, "rb") as fh:
            db = pickle.load(fh)
        out: dict = {}
        for pid, enc in zip(db.get("person_ids", []), db.get("encodings", [])):
            out.setdefault(pid, []).append(enc)
        return out, metric
    return {}, "cosine"


def _pair_face_distances(encs_a: list, encs_b: list,
                         metric: str = "cosine") -> tuple[float, float]:
    """Return (min, mean) distance between two encoding sets under ``metric``.

    ``"cosine"`` (``1 - cos``) for L2-normalized ArcFace embeddings, ``"euclidean"``
    for legacy dlib encodings. ``(inf, inf)`` if either set is empty.
    """
    if not encs_a or not encs_b:
        return float("inf"), float("inf")
    import numpy as _np
    A = _np.asarray(encs_a, dtype=float)
    B = _np.asarray(encs_b, dtype=float)
    if metric == "cosine":
        # Normalize defensively, then cosine distance = 1 - A·B for every pair.
        A = A / (_np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        B = B / (_np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
        d = 1.0 - A @ B.T
    else:
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


def _similarity_verdict(face_min: float, name: float, facts: float,
                        metric: str = "cosine") -> Optional[str]:
    """Classify a pair. Returns a short label or None to hide.

    Face thresholds depend on the metric: cosine distance for ArcFace embeddings
    sits on a different scale than dlib's Euclidean, so each gets its own
    "same / maybe" cutoffs.
    """
    same, maybe = (0.30, 0.45) if metric == "cosine" else (0.40, 0.55)
    if face_min < same:
        return "LIKELY SAME"
    if face_min < maybe and (name > 0.85 or facts > 0.30):
        return "LIKELY SAME"
    if face_min < maybe:
        return "maybe"
    if name > 0.90:
        return "name collision"
    if facts > 0.50:
        return "shared facts"
    return None


def _stored_fact_texts(stored: dict) -> list[str]:
    out = []
    for raw in stored.get("facts", []):
        f = Fact.load(raw, 0.0)
        if f:
            out.append(f.text)
    return out


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
        help="Rule-based cleanup: strip subjects, drop non-facts, merge near-duplicates")
    dedupe_p.add_argument("--person", help="Limit to one person_id (e.g. p001)")
    dedupe_p.add_argument("--apply", action="store_true",
        help="Write changes (without this, dry-run only)")
    similar_p = sub.add_parser("similar",
        help="Find people records that may be duplicates of each other")
    similar_p.add_argument("--person",
        help="Compare only this person_id against all others")
    similar_p.add_argument("--db-dir", default="known_faces",
        help="Face encoding database directory (default: known_faces)")
    similar_p.add_argument("--all", action="store_true",
        help="Show every pair, even ones below the verdict thresholds")
    merge_p = sub.add_parser("merge",
        help="Merge one person into another (memory AND face encodings)")
    merge_p.add_argument("src", help="person_id to merge away (e.g. p002)")
    merge_p.add_argument("dst", help="person_id to keep (e.g. p001)")
    merge_p.add_argument("--db-dir", default="known_faces",
        help="Face encoding database directory (default: known_faces)")
    merge_p.add_argument("--apply", action="store_true",
        help="Write changes (without this, dry-run only)")
    forget_p = sub.add_parser("forget",
        help="Delete a person: memory record, face encodings and face crops")
    forget_p.add_argument("person", help="person_id (e.g. p003)")
    forget_p.add_argument("--db-dir", default="known_faces")
    forget_p.add_argument("--apply", action="store_true",
        help="Really delete (without this, dry-run only)")
    sub.add_parser("shell", help="Interactive shell for poking at people memory")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    mem = PeopleMemory(storage_dir=args.dir)
    mem.load()

    if args.command == "list" or args.command is None:
        pids = sorted(mem.known_person_ids)
        if not pids:
            print("No people in memory.")
        else:
            print(f"{len(pids)} people:\n")
            for pid in pids:
                person = mem.get_by_id(pid)
                if not person:
                    continue
                parts = [f"{pid} {person.name or '(no name)'}"]
                if person.times_seen > 1:
                    parts.append(f"seen {person.times_seen}x")
                n_conf = sum(1 for f in person.facts if f.confirmed)
                parts.append(f"{len(person.facts)} facts ({n_conf} confirmed)")
                ago = person.time_since_seen()
                if ago and ago > 3600:
                    parts.append(f"last {_ago(ago)}")
                print(f"  {', '.join(parts)}")

    elif args.command == "show":
        person = mem.get_by_name(args.name) or mem.get_by_id(args.name)
        if not person:
            print(f"No person named '{args.name}'")
        else:
            print(json.dumps(person.to_dict(), indent=2, ensure_ascii=False))

    elif args.command == "context":
        person = mem.get_by_name(args.name) or mem.get_by_id(args.name)
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
        # Typed in by hand: as good as said outright.
        status = mem.add_fact(0 if person.track_id == 0 else person.track_id,
                              args.fact, explicit=True)
        print(f"{status}: {args.fact}")

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
                else:
                    print(f"  - drop {a!r}  ({b})")
            if args.apply:
                stored["facts"] = [asdict(f) for f in new_facts]
                mem._write(pid, stored)
                print("  [written]")
        suffix = "" if args.apply else "  (dry-run; pass --apply to write)"
        print(f"\nTotal: {total_before} -> {total_after} facts, "
              f"{total_changes} changes across {len(pids)} people.{suffix}")

    elif args.command == "similar":
        pids = sorted(mem._stored.keys())
        if not pids:
            print("No people records.")
            return
        encs_by_pid, face_metric = _load_face_encodings_by_person(args.db_dir)
        if not encs_by_pid:
            print(f"Warning: no face encodings found in {args.db_dir} "
                  "(faces_arcface.pkl / faces.pkl) — falling back to name+facts only.")
        else:
            print(f"Comparing faces with {face_metric} distance "
                  f"({len(encs_by_pid)} people).\n")

        def meta(pid):
            s = mem._stored.get(pid, {})
            return s.get("name"), _stored_fact_texts(s)

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
                face_min, face_mean = _pair_face_distances(encs_a, encs_b, face_metric)
                name_sim = _name_similarity(name_a, name_b)
                facts_sim = _facts_similarity(facts_a, facts_b)
                verdict = _similarity_verdict(face_min, name_sim, facts_sim, face_metric)
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
                if verdict == "LIKELY SAME":
                    # Keep the older (lower) id: it is the one other data points to.
                    keep, away = sorted((pid_a, pid_b))
                    print(f"  merge with: python people_memory.py merge {away} {keep} --apply")
        if shown == 0:
            print("No suspicious pairs."
                  " Use --all to see every pair regardless of threshold.")
        else:
            print(f"\n{shown} pair(s) shown.")

    elif args.command == "merge":
        a, b = mem._stored.get(args.src), mem._stored.get(args.dst)
        if not a or not b:
            print(f"Not found: {args.src if not a else args.dst}")
            return
        print(f"Merge {args.src} ({a.get('name') or '?'}) into "
              f"{args.dst} ({b.get('name') or '?'})")
        if not args.apply:
            pa, pb = Person.from_dict(a), Person.from_dict(b)
            facts = _merge_facts(pb.facts, pa.facts, pb.name or pa.name)
            print(f"  facts: {len(pb.facts)} + {len(pa.facts)} -> {len(facts)}")
            print(f"  dialogue lines: {len(pb.dialogues)} + {len(pa.dialogues)} "
                  f"(kept: last {MAX_DIALOGUES})")
            print("\n(dry-run; pass --apply to merge memory and face encodings)")
            return
        from face_tracker import FaceDatabase
        from face_config import build_db_kwargs
        db = FaceDatabase(db_dir=args.db_dir, **build_db_kwargs())
        db.load()
        moved = db.rename_person(args.src, args.dst)
        mem.merge(args.src, args.dst)
        print(f"  merged; {moved} face encodings moved to {args.dst}")

    elif args.command == "forget":
        rec = mem._stored.get(args.person)
        print(f"Forget {args.person} ({(rec or {}).get('name') or '?'}): memory record, "
              f"face encodings and face crops in {args.db_dir}/{args.person}/")
        if not args.apply:
            print("(dry-run; pass --apply to delete)")
            return
        from face_tracker import FaceDatabase
        from face_config import build_db_kwargs
        db = FaceDatabase(db_dir=args.db_dir, **build_db_kwargs())
        db.load()
        n = db.forget_person(args.person)
        mem.forget(args.person)
        print(f"  deleted ({n} face encodings)")

    elif args.command == "shell":
        from debug_shell import run_shell
        run_shell(mem)


if __name__ == "__main__":
    main()
