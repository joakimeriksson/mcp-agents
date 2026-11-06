"""Voice identity: speaker embeddings from the voice server, matched locally.

The audio twin of the face database. The voice server turns an utterance into
a 192-d unit vector (ECAPA-TDNN, ``/v1/audio/speaker``) and keeps nothing; this
module owns everything that identifies a person, on this machine only:

- ``SpeakerClient``  posts an utterance to the local voice server, gets a vector.
- ``VoiceDB``        stores vectors per person id, verifies and identifies.

What it is for, in order of value:
1. Authorship: once the visitor in focus has spoken, later utterances are
   checked against their voice, so a bystander's sentence is dropped rather
   than answered. This is the crowd failure the near-field gate only halves.
2. Continuity: a voice keeps the conversation attached to a person whose face
   turned toward the candy.
3. Recognition of a returning voice.

Privacy, by default: nothing is written to disk. Voiceprints are biometric
data, so the store is ephemeral -- kept in memory for the session and gone at
exit. Persistence is opt-in (``persist_named=True``) and even then only for
people the caller has explicitly marked as named (staff, yourself); provisional
``track:N`` ids are never written. ``purge()`` deletes the file.

Thresholds were calibrated on the ten synthetic Kokoro voices (Sep 2026):
same speaker >= 0.75 on clips over ~2.5 s, unrelated speakers median 0.38;
clips under ~1 s carry no identity (same-speaker median 0.31), which is why
``min_seconds`` gates every decision. Real voices vary more than synthetic
ones, so expect to loosen ``confirm`` slightly after a session in a real room.
"""

from __future__ import annotations

import io
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger("voice_id")

DEFAULT_URL = "http://127.0.0.1:8880/v1/audio/speaker"
PROVISIONAL_PREFIX = "track:"


# ---------------------------------------------------------------------------
# Embeddings from the (local) voice server
# ---------------------------------------------------------------------------

class SpeakerClient:
    """Fetch a speaker embedding for one utterance from the voice server."""

    def __init__(self, url: str = DEFAULT_URL, timeout: float = 2.0):
        self.url = url
        self.timeout = timeout
        self.model: str = ""

    def available(self) -> bool:
        """True if the server is up and started with --speaker."""
        import requests
        try:
            base = self.url.split("/v1/")[0]
            r = requests.get(f"{base}/health", timeout=self.timeout)
            return bool(r.ok and r.json().get("speaker"))
        except Exception:
            return False

    def embed(self, audio: np.ndarray, sample_rate: int = 16000) -> tuple[Optional[np.ndarray], float]:
        """(unit-norm embedding, seconds) or (None, seconds) on any failure.

        Failure is deliberately soft: the caller must fail OPEN (accept the
        utterance) when identity is unavailable -- a down endpoint must not
        mute the robot.
        """
        import requests
        import soundfile as sf

        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        seconds = len(samples) / sample_rate
        buf = io.BytesIO()
        sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
        try:
            r = requests.post(self.url, files={"file": ("u.wav", buf.getvalue(), "audio/wav")},
                              timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.warning(f"speaker embedding failed ({self.url}): {e}")
            return None, seconds
        vec = data.get("embedding") or []
        if not vec:
            return None, seconds
        self.model = data.get("model", "")
        return np.asarray(vec, dtype=np.float32), float(data.get("seconds", seconds))


# ---------------------------------------------------------------------------
# Local voice database
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VoiceMatch:
    person_id: Optional[str]
    similarity: float
    decision: str          # "same" | "unsure" | "different" | "no-decision"
    seconds: float
    reason: str = ""


@dataclass
class _Voice:
    samples: list = field(default_factory=list)   # list[np.ndarray], newest last
    seconds: list = field(default_factory=list)
    updated: float = 0.0

    def centroid(self) -> np.ndarray:
        c = np.mean(np.stack(self.samples), axis=0)
        n = float(np.linalg.norm(c)) or 1.0
        return c / n


class VoiceDB:
    """Per-person voice embeddings with verify / identify / enroll.

    ``confirm``: similarity at or above which an utterance is the same person
    (and the store learns from it). ``reject``: below which it is someone else.
    Between the two the answer is "unsure" -- accepted, but not learned from.
    ``min_seconds``: clips shorter than this get "no-decision" (see module
    docstring). ``identify_threshold`` is stricter than ``confirm`` because
    identification searches all people instead of checking one.
    """

    def __init__(self, path: Optional[str] = None, *, persist_named: bool = False,
                 confirm: float = 0.65, reject: float = 0.40, min_seconds: float = 2.0,
                 identify_threshold: float = 0.70, max_samples: int = 5):
        self.path = path
        self.persist_named = persist_named
        self.confirm = confirm
        self.reject = reject
        self.min_seconds = min_seconds
        self.identify_threshold = identify_threshold
        self.max_samples = max_samples
        self._voices: dict[str, _Voice] = {}
        self._named: set[str] = set()
        self._lock = threading.Lock()
        self.model: str = ""
        if path and persist_named:
            self.load()

    # --- state -------------------------------------------------------------

    def has(self, person_id: str) -> bool:
        return person_id in self._voices

    def people(self) -> list[str]:
        return sorted(self._voices)

    def mark_named(self, person_id: str) -> None:
        """Allow this person to be persisted (only if persist_named is on)."""
        if not person_id.startswith(PROVISIONAL_PREFIX):
            self._named.add(person_id)

    def rename(self, old_id: str, new_id: str) -> None:
        """A provisional ``track:N`` became a real id: carry the voice over."""
        with self._lock:
            v = self._voices.pop(old_id, None)
            if v is None:
                return
            if new_id in self._voices:
                merged = self._voices[new_id]
                merged.samples = (merged.samples + v.samples)[-self.max_samples:]
                merged.seconds = (merged.seconds + v.seconds)[-self.max_samples:]
            else:
                self._voices[new_id] = v

    def forget(self, person_id: str) -> None:
        with self._lock:
            self._voices.pop(person_id, None)
            self._named.discard(person_id)

    # --- decisions ---------------------------------------------------------

    def enroll(self, person_id: str, emb: np.ndarray, seconds: float) -> bool:
        """Add a sample. Refused for clips under ``min_seconds`` -- a short
        clip enrolls noise, and every later check would fail against it."""
        if emb is None or seconds < self.min_seconds:
            return False
        with self._lock:
            v = self._voices.setdefault(person_id, _Voice())
            v.samples = (v.samples + [np.asarray(emb, dtype=np.float32)])[-self.max_samples:]
            v.seconds = (v.seconds + [seconds])[-self.max_samples:]
            v.updated = time.time()
        return True

    def verify(self, person_id: str, emb: Optional[np.ndarray], seconds: float,
               learn: bool = True) -> VoiceMatch:
        """Is this utterance from *person_id*? Fails OPEN: "no-decision" when
        there is nothing to compare, or the clip is too short."""
        if emb is None:
            return VoiceMatch(person_id, 0.0, "no-decision", seconds, "no embedding")
        if seconds < self.min_seconds:
            return VoiceMatch(person_id, 0.0, "no-decision", seconds,
                              f"clip {seconds:.1f}s < {self.min_seconds}s")
        with self._lock:
            v = self._voices.get(person_id)
            if v is None or not v.samples:
                return VoiceMatch(person_id, 0.0, "no-decision", seconds, "no enrolled voice")
            sim = float(np.dot(v.centroid(), emb))
        if sim >= self.confirm:
            if learn:
                self.enroll(person_id, emb, seconds)
            return VoiceMatch(person_id, sim, "same", seconds)
        if sim < self.reject:
            return VoiceMatch(person_id, sim, "different", seconds)
        return VoiceMatch(person_id, sim, "unsure", seconds)

    def identify(self, emb: Optional[np.ndarray], seconds: float) -> Optional[VoiceMatch]:
        """Best-matching known person, or None if nobody clears the bar."""
        if emb is None or seconds < self.min_seconds:
            return None
        with self._lock:
            best, best_sim = None, -1.0
            for pid, v in self._voices.items():
                if not v.samples:
                    continue
                sim = float(np.dot(v.centroid(), emb))
                if sim > best_sim:
                    best, best_sim = pid, sim
        if best is None or best_sim < self.identify_threshold:
            return None
        return VoiceMatch(best, best_sim, "same", seconds)

    # --- persistence (opt-in, named people only) ---------------------------

    def save(self) -> int:
        """Write named people's voices to ``path``. Returns how many were
        written. No-op unless ``persist_named`` -- ephemeral by default."""
        if not (self.path and self.persist_named):
            return 0
        with self._lock:
            people = {
                pid: {"samples": [s.tolist() for s in v.samples],
                      "seconds": v.seconds, "updated": v.updated}
                for pid, v in self._voices.items()
                if pid in self._named and not pid.startswith(PROVISIONAL_PREFIX) and v.samples
            }
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"version": 1, "model": self.model, "people": people}, f)
        os.replace(tmp, self.path)
        logger.info(f"voice db: saved {len(people)} named voice(s) to {self.path}")
        return len(people)

    def load(self) -> int:
        if not (self.path and os.path.exists(self.path)):
            return 0
        with open(self.path) as f:
            data = json.load(f)
        n = 0
        with self._lock:
            for pid, v in data.get("people", {}).items():
                if pid.startswith(PROVISIONAL_PREFIX):
                    continue
                self._voices[pid] = _Voice(
                    samples=[np.asarray(s, dtype=np.float32) for s in v.get("samples", [])],
                    seconds=list(v.get("seconds", [])), updated=float(v.get("updated", 0.0)))
                self._named.add(pid)
                n += 1
        self.model = data.get("model", "")
        logger.info(f"voice db: loaded {n} named voice(s) from {self.path}")
        return n

    def purge(self) -> None:
        """Forget everything, in memory and on disk."""
        with self._lock:
            self._voices.clear()
            self._named.clear()
        if self.path and os.path.exists(self.path):
            os.remove(self.path)
            logger.info(f"voice db: deleted {self.path}")
