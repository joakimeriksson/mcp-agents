"""
Face detection, recognition, and tracking module.

Provides stable face tracking across video frames with:
- Persistent tracking IDs (survive brief occlusions)
- Recognition with hysteresis (no identity flickering)
- Emotion detection
- Face database management
- Typed event system with subscribe/unsubscribe

TrackedFace is pure tracking data (ID, bbox, encoding, emotion).
Identity (``person_id``) is a separate concern managed by FaceTracker.
Display names are resolved externally via ``PeopleMemory``; the tracker
itself only knows stable person IDs.

Can be run standalone:
    python face_tracker.py [--db-dir known_faces] [--camera 0]
"""

import copy
import cv2
import face_recognition
import numpy as np
import os
import pickle
import re
import time
import threading
import logging
import argparse
from enum import Enum, auto
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable, Union

import onnxruntime as ort

from events import EventDispatcher

logger = logging.getLogger("face_tracker")

_SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))

EMOTION_MODEL_DIR = os.path.join(_SOURCE_DIR, "emotion_model")
_KNOWN_FACES_DIR = os.path.join(_SOURCE_DIR, "known_faces")
EMOTION_MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
EMOTION_LABELS = ["neutral", "happy", "surprise", "sad", "angry", "disgust", "fear", "contempt"]


# ---------------------------------------------------------------------------
# Embedding distance helpers
# ---------------------------------------------------------------------------

def _embedding_distances(known, encoding, metric: str) -> np.ndarray:
    """Distance from ``encoding`` to each row of ``known`` under ``metric``.

    ``"euclidean"`` is the legacy dlib metric (L2 on 128-d vectors).
    ``"cosine"`` (``1 - cos``) is correct for L2-normalized ArcFace embeddings;
    for unit vectors the cosine similarity is just the dot product.
    """
    known = np.asarray(known, dtype=np.float64)
    enc = np.asarray(encoding, dtype=np.float64)
    if metric == "cosine":
        return 1.0 - known @ enc
    return np.linalg.norm(known - enc, axis=1)


def _pair_distance(a, b, metric: str) -> float:
    """Distance between two single embeddings under ``metric``."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if metric == "cosine":
        return float(1.0 - np.dot(a, b))
    return float(np.linalg.norm(a - b))


def _pose_yaw_pitch(extra) -> tuple:
    """Extract (yaw, pitch) in degrees from a detection extra, or (0.0, 0.0).

    InsightFace reports pose as ``[pitch, yaw, roll]``; backends without pose
    (dlib) pass ``None``, leaving the face treated as facing the camera.
    """
    if not extra:
        return 0.0, 0.0
    pose = extra.get("pose")
    if pose is None:
        return 0.0, 0.0
    pose = np.asarray(pose, dtype=float).ravel()
    if pose.size >= 2:
        return float(pose[1]), float(pose[0])  # yaw, pitch
    return 0.0, 0.0


def _select_providers(override=None) -> list:
    """Pick onnxruntime execution providers, best-available first.

    Preference CUDA (Nvidia) -> CoreML (Apple Silicon) -> CPU, intersected with
    what this onnxruntime build actually exposes. ``override`` (from config) is
    honoured but still filtered to available providers so a GPU-less machine
    never fails to start.
    """
    available = set(ort.get_available_providers())
    if override:
        chosen = [p for p in override if p in available]
        return chosen or ["CPUExecutionProvider"]
    preferred = ["CUDAExecutionProvider", "CoreMLExecutionProvider",
                 "CPUExecutionProvider"]
    chosen = [p for p in preferred if p in available]
    return chosen or ["CPUExecutionProvider"]


# ---------------------------------------------------------------------------
# Event types and payloads
# ---------------------------------------------------------------------------

class FaceEventType(Enum):
    FACE_APPEARED = auto()
    FACE_DISAPPEARED = auto()
    FACE_OCCLUDED = auto()        # visible -> grace period
    FACE_RECOVERED = auto()       # grace period -> visible
    IDENTITY_CONFIRMED = auto()   # unknown -> named
    IDENTITY_LOST = auto()        # named -> unknown
    IDENTITY_CHANGED = auto()     # name A -> name B
    FACE_LEARNED = auto()         # learn_face() called
    FACE_ENROLLED = auto()        # new unknown face auto-saved with fresh person_id
    FOCUS_CHANGED = auto()        # primary focus switched
    FACE_ENGAGED = auto()         # focused face is looking at the camera ("makes contact")
    FACE_DISENGAGED = auto()      # engaged face looked away / lost focus
    EMOTION_CHANGED = auto()      # emotion label changed


@dataclass(frozen=True)
class FaceAppearedPayload:
    bbox: tuple
    emotion: str
    emotion_confidence: float
    initial_person_id: Optional[str]
    initial_confidence: float


@dataclass(frozen=True)
class FaceDisappearedPayload:
    last_bbox: tuple
    person_id: Optional[str]
    duration_visible: float
    total_frames: int


@dataclass(frozen=True)
class FaceOccludedPayload:
    last_bbox: tuple
    person_id: Optional[str]


@dataclass(frozen=True)
class FaceRecoveredPayload:
    bbox: tuple
    person_id: Optional[str]
    seconds_missing: float


@dataclass(frozen=True)
class IdentityConfirmedPayload:
    person_id: str
    confidence: float
    last_seen_timestamp: Optional[float]


@dataclass(frozen=True)
class IdentityLostPayload:
    previous_person_id: str


@dataclass(frozen=True)
class IdentityChangedPayload:
    old_person_id: str
    new_person_id: str
    new_confidence: float


@dataclass(frozen=True)
class FaceLearnedPayload:
    person_id: str


@dataclass(frozen=True)
class FaceEnrolledPayload:
    person_id: str
    bbox: tuple


@dataclass(frozen=True)
class FocusChangedPayload:
    old_track_id: Optional[int]
    new_track_id: int
    old_focus_score: float
    new_focus_score: float
    new_person_id: Optional[str]


@dataclass(frozen=True)
class FaceEngagedPayload:
    person_id: Optional[str]
    yaw: float
    pitch: float
    focus_score: float


@dataclass(frozen=True)
class FaceDisengagedPayload:
    person_id: Optional[str]
    reason: str  # "looked_away" | "lost_focus" | "disappeared"


@dataclass(frozen=True)
class EmotionChangedPayload:
    old_emotion: str
    new_emotion: str
    new_confidence: float
    person_id: Optional[str]


FaceEventPayload = Union[
    FaceAppearedPayload, FaceDisappearedPayload, FaceOccludedPayload,
    FaceRecoveredPayload, IdentityConfirmedPayload, IdentityLostPayload,
    IdentityChangedPayload, FaceLearnedPayload, FaceEnrolledPayload,
    FocusChangedPayload, FaceEngagedPayload, FaceDisengagedPayload,
    EmotionChangedPayload,
]


@dataclass(frozen=True)
class FaceEvent:
    type: FaceEventType
    timestamp: float
    track_id: Optional[int]
    payload: FaceEventPayload


FaceEventCallback = Callable[[FaceEvent], None]


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------

@dataclass
class TrackedFace:
    """A face tracked across frames with a stable ID. Contains no identity info."""
    track_id: int
    emotion: str = "neutral"
    emotion_confidence: float = 0.0
    encoding: np.ndarray = field(default_factory=lambda: np.zeros(128))
    # Most recent *raw* detection encoding (before the EMA blend in `encoding`).
    # Used for enrollment so we store a clean single-frame embedding, never the
    # smoothed-and-possibly-contaminated tracking encoding.
    last_encoding: np.ndarray = field(default_factory=lambda: np.zeros(128))
    bbox: tuple = (0, 0, 0, 0)  # (top, right, bottom, left)
    first_seen: float = 0.0
    last_seen: float = 0.0
    frames_visible: int = 0
    frames_since_seen: int = 0
    focus_score: float = 0.0
    # Head pose in degrees from the detector (0 when the backend has no pose,
    # e.g. dlib). yaw ~ left/right turn, pitch ~ up/down; ~0 means facing the
    # camera. Drives the engagement ("makes contact") gate.
    yaw: float = 0.0
    pitch: float = 0.0

    @property
    def center(self) -> tuple:
        top, right, bottom, left = self.bbox
        return ((left + right) // 2, (top + bottom) // 2)

    @property
    def area(self) -> int:
        top, right, bottom, left = self.bbox
        return max(0, (right - left) * (bottom - top))

    @property
    def is_visible(self) -> bool:
        return self.frames_since_seen == 0


@dataclass
class Identity:
    """Recognition result for a tracked face, keyed by stable person_id."""
    person_id: str
    confidence: float
    _matching_since: float = field(default=0.0, repr=False)
    _failing_since: float = field(default=0.0, repr=False)
    _candidate_person_id: str = field(default="", repr=False)
    _candidate_confidence: float = field(default=0.0, repr=False)
    _confirmed: bool = field(default=False, repr=False)


# ---------------------------------------------------------------------------
# FaceDatabase and EmotionDetector (unchanged)
# ---------------------------------------------------------------------------

class FaceDatabase:
    """Persistent face encoding database keyed by stable person_id.

    ``metric`` selects how stored embeddings are compared:
    ``"euclidean"`` for the legacy dlib 128-d encodings (schema v2) and
    ``"cosine"`` for the L2-normalized InsightFace/ArcFace 512-d embeddings
    (schema v3). The two are not interchangeable, so each metric uses its own
    db file (``faces.pkl`` for dlib/v2, ``faces_arcface.pkl`` for InsightFace/v3):
    the backends coexist for A/B comparison and one never clobbers the other's
    saved encodings. ``load()`` still refuses a file whose ``schema_version``
    does not match, warning and starting empty rather than mixing dimensions.
    """

    # Per-metric db file and the schema version it must carry.
    _DB_FILE = {"euclidean": "faces.pkl", "cosine": "faces_arcface.pkl"}
    _SCHEMA = {"euclidean": 2, "cosine": 3}

    def __init__(self, db_dir: str = _KNOWN_FACES_DIR, tolerance: float = 0.6,
                 recognition_k: int = 3, metric: str = "cosine"):
        self.db_dir = db_dir
        self.tolerance = tolerance
        self.metric = metric
        # Number of nearest stored samples per person to average when matching.
        # k=1 reproduces the old single-nearest behaviour; k>1 makes a single
        # lucky-close (or poisoned) sample less able to win a false match.
        self.recognition_k = max(1, recognition_k)
        self._db = self._empty_db()
        self._lock = threading.Lock()

    @property
    def schema_version(self) -> int:
        return self._SCHEMA[self.metric]

    @property
    def db_file(self) -> str:
        return os.path.join(self.db_dir, self._DB_FILE[self.metric])

    def _empty_db(self) -> dict:
        return {
            "encodings": [],
            "person_ids": [],
            "last_seen": {},
            "schema_version": self.schema_version,
        }

    def load(self):
        db_file = self.db_file
        if os.path.exists(db_file):
            with open(db_file, "rb") as f:
                loaded = pickle.load(f)
            version = loaded.get("schema_version", 2)
            if version != self.schema_version:
                # The file holds embeddings of the wrong dimension/metric for the
                # active backend. Refuse to mix: warn and start empty so
                # auto-enroll can repopulate. For the InsightFace backend, run
                # migrate_db.py to re-embed the saved face crops without losing
                # anyone.
                logger.warning(
                    "Ignoring face db at %s: schema v%s != expected v%s. "
                    "Starting empty -- run migrate_db.py to re-embed known_faces/.",
                    db_file, version, self.schema_version,
                )
                self._db = self._empty_db()
            else:
                self._db = loaded
                self._db.setdefault("last_seen", {})
                self._db.setdefault("person_ids", [])
                self._db.setdefault("schema_version", self.schema_version)
        logger.info(
            f"Database loaded: {len(self.known_person_ids)} people, "
            f"{self.encoding_count} encodings"
        )

    def save(self):
        os.makedirs(self.db_dir, exist_ok=True)
        with self._lock:
            with open(self.db_file, "wb") as f:
                pickle.dump(self._db, f)

    def recognize(self, encoding: np.ndarray) -> tuple:
        with self._lock:
            if not self._db["encodings"]:
                return (None, 0.0)
            distances = _embedding_distances(
                self._db["encodings"], encoding, self.metric)
            # Aggregate per person: score each candidate by the mean of their
            # k nearest stored samples, then pick the closest person. This is
            # steadier than a single nearest encoding -- one good sample can no
            # longer pull a confusion across the line on its own.
            by_pid: dict[str, list[float]] = {}
            for dist, pid in zip(distances, self._db["person_ids"]):
                by_pid.setdefault(pid, []).append(float(dist))
            best_pid, best_dist = None, float("inf")
            for pid, dists in by_pid.items():
                dists.sort()
                score = float(np.mean(dists[:self.recognition_k]))
                if score < best_dist:
                    best_dist, best_pid = score, pid
            if best_pid is not None and best_dist < self.tolerance:
                confidence = max(0.0, 1.0 - best_dist) * 100
                return (best_pid, confidence)
            return (None, 0.0)

    def add_face(self, person_id: str, encoding: np.ndarray,
                 frame: np.ndarray, bbox: tuple):
        with self._lock:
            self._db["encodings"].append(encoding)
            self._db["person_ids"].append(person_id)
            sample_count = self._db["person_ids"].count(person_id)
        self.save()
        self._save_face_image(person_id, frame, bbox)
        logger.info(f"Added face for {person_id!r} ({sample_count} samples)")

    def add_encoding(self, person_id: str, encoding: np.ndarray):
        """Append one encoding without persisting or saving a crop.

        For bulk/migration use (e.g. migrate_db.py rebuilding the db from saved
        face crops); call ``save()`` once after adding everything.
        """
        with self._lock:
            self._db["encodings"].append(np.asarray(encoding, dtype=np.float64))
            self._db["person_ids"].append(person_id)

    def update_last_seen(self, person_id: str):
        with self._lock:
            self._db["last_seen"][person_id] = datetime.now().timestamp()

    def get_last_seen(self, person_id: str) -> Optional[float]:
        return self._db["last_seen"].get(person_id)

    def clear(self):
        with self._lock:
            self._db = self._empty_db()
        self.save()
        logger.info("Database cleared")

    def remove_person(self, person_id: str) -> int:
        """Drop all encodings for ``person_id``. Returns number removed."""
        with self._lock:
            keep_enc = []
            keep_ids = []
            removed = 0
            for enc, pid in zip(self._db["encodings"], self._db["person_ids"]):
                if pid == person_id:
                    removed += 1
                else:
                    keep_enc.append(enc)
                    keep_ids.append(pid)
            self._db["encodings"] = keep_enc
            self._db["person_ids"] = keep_ids
            self._db["last_seen"].pop(person_id, None)
        if removed:
            self.save()
            logger.info(f"Removed {removed} encodings for {person_id!r}")
        return removed

    def rename_person(self, old_id: str, new_id: str) -> int:
        """Reassign all of ``old_id``'s encodings and saved crops to ``new_id``.

        Merges into ``new_id`` if it already exists. Returns the number of
        encodings moved; a no-op (0) when the ids match or ``old_id`` is absent.
        Used when a face that was auto-enrolled as ``pNNN`` is later given a name,
        so the two never coexist as competing identities.
        """
        if old_id == new_id:
            return 0
        with self._lock:
            moved = 0
            for i, pid in enumerate(self._db["person_ids"]):
                if pid == old_id:
                    self._db["person_ids"][i] = new_id
                    moved += 1
            if old_id in self._db["last_seen"]:
                ts = self._db["last_seen"].pop(old_id)
                self._db["last_seen"][new_id] = max(
                    self._db["last_seen"].get(new_id, 0.0), ts)
        if moved:
            self.save()
            self._move_face_images(old_id, new_id)
            logger.info(f"Renamed {old_id!r} -> {new_id!r} ({moved} encodings)")
        return moved

    def _move_face_images(self, old_id: str, new_id: str):
        old_dir = os.path.join(self.db_dir, old_id)
        if not os.path.isdir(old_dir):
            return
        new_dir = os.path.join(self.db_dir, new_id)
        os.makedirs(new_dir, exist_ok=True)
        for fname in os.listdir(old_dir):
            src = os.path.join(old_dir, fname)
            dst = os.path.join(new_dir, fname)
            if os.path.exists(dst):  # avoid clobbering on merge
                base, ext = os.path.splitext(fname)
                dst = os.path.join(new_dir, f"{base}_{old_id}{ext}")
            os.rename(src, dst)
        try:
            os.rmdir(old_dir)
        except OSError:
            pass

    @property
    def known_person_ids(self) -> set:
        return set(self._db["person_ids"])

    @property
    def encoding_count(self) -> int:
        return len(self._db["encodings"])

    @property
    def last_seen_map(self) -> dict:
        return dict(self._db["last_seen"])

    def _save_face_image(self, person_id, frame, bbox):
        top, right, bottom, left = bbox
        face_img = frame[top:bottom, left:right]
        person_dir = os.path.join(self.db_dir, person_id)
        os.makedirs(person_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(os.path.join(person_dir, f"{timestamp}.jpg"), face_img)


class EmotionDetector:
    """ONNX-based facial emotion detection."""

    def __init__(self, model_dir: str = EMOTION_MODEL_DIR):
        self.model_dir = model_dir
        self.session = None
        self._ensure_model()

    def _ensure_model(self):
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, "emotion-ferplus-8.onnx")
        if not os.path.exists(model_path):
            logger.info("Downloading emotion detection model...")
            import urllib.request
            urllib.request.urlretrieve(EMOTION_MODEL_URL, model_path)
            logger.info("Emotion model downloaded.")
        self.session = ort.InferenceSession(model_path)

    def detect(self, face_bgr: np.ndarray) -> tuple:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64))
        input_data = resized.astype(np.float32).reshape(1, 1, 64, 64)
        input_name = self.session.get_inputs()[0].name
        result = self.session.run(None, {input_name: input_data})
        scores = result[0][0]
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()
        idx = np.argmax(probs)
        return EMOTION_LABELS[idx], float(probs[idx])


# ---------------------------------------------------------------------------
# Detection / embedding backends
# ---------------------------------------------------------------------------

class DlibBackend:
    """Legacy backend: dlib HOG detector + 128-d encoder via face_recognition.

    Faces are detected on a downscaled frame (``frame_scale``) for speed, then
    the bboxes are scaled back to full-frame coordinates. Embeddings compare
    under the Euclidean metric.
    """

    name = "dlib"
    metric = "euclidean"

    def __init__(self, frame_scale: float = 0.5):
        self.frame_scale = frame_scale

    def detect(self, frame):
        s = self.frame_scale
        small = cv2.resize(frame, (0, 0), fx=s, fy=s)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb_small)
        encodings = face_recognition.face_encodings(rgb_small, locations)
        locations = [
            (int(t / s), int(r / s), int(b / s), int(l / s))
            for t, r, b, l in locations
        ]
        # dlib carries no head pose; extras are None so downstream yaw/pitch = 0.
        return locations, encodings, [None] * len(locations)


class InsightFaceBackend:
    """InsightFace backend: SCRFD detector + ArcFace 512-d embedder on onnxruntime.

    One ``FaceAnalysis.get(frame)`` pass per frame yields bbox, det_score, 5-point
    kps, head pose, and an L2-normalized 512-d embedding for every face. The model
    takes BGR directly (no RGB conversion) and returns bboxes as ``[x1,y1,x2,y2]``,
    which we map to the tracker's ``(top,right,bottom,left)`` convention.
    Embeddings are unit-norm, so they compare under the cosine metric.

    The ``buffalo_l`` pack (~300 MB) is downloaded by InsightFace on first use to
    ``~/.insightface/models``; expect a one-time cold start.
    """

    name = "insightface"
    metric = "cosine"

    def __init__(self, det_size=640, det_thresh: float = 0.5, providers=None,
                 model_name: str = "buffalo_l"):
        from insightface.app import FaceAnalysis

        if isinstance(det_size, (int, float)):
            det_size = (int(det_size), int(det_size))
        else:
            det_size = (int(det_size[0]), int(det_size[1]))
        providers = _select_providers(providers)
        # ctx_id selects the CUDA device; -1 forces CPU. CoreML/CPU ignore it.
        ctx_id = 0 if "CUDAExecutionProvider" in providers else -1
        logger.info("InsightFace backend: model=%s providers=%s det_size=%s",
                    model_name, providers, det_size)
        self.app = FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size, det_thresh=det_thresh)

    def detect(self, frame):
        faces = self.app.get(frame)
        locations, encodings, extras = [], [], []
        for f in faces:
            x1, y1, x2, y2 = (int(v) for v in f.bbox)
            locations.append((y1, x2, y2, x1))  # (top, right, bottom, left)
            encodings.append(np.asarray(f.normed_embedding, dtype=np.float64))
            # pose is [pitch, yaw, roll] in degrees; kps/det_score kept for future use.
            extras.append({
                "kps": getattr(f, "kps", None),
                "pose": getattr(f, "pose", None),
                "det_score": float(getattr(f, "det_score", 0.0)),
            })
        return locations, encodings, extras


# ---------------------------------------------------------------------------
# FaceTracker
# ---------------------------------------------------------------------------

class FaceTracker:
    """
    Detects, recognizes, and tracks faces across video frames.

    Each detected face gets a stable track_id that persists across frames.
    Identity (name) is separate from tracking — query via get_identity().

    Subscribe to typed FaceEvent callbacks via subscribe().
    """

    def __init__(self,
                 db: FaceDatabase,
                 emotion_detector: EmotionDetector,
                 backend: str = "insightface",
                 frame_scale: float = 0.5,
                 det_size=640,
                 det_thresh: float = 0.5,
                 providers=None,
                 track_encoding_threshold: Optional[float] = None,
                 track_iou_threshold: float = 0.3,
                 max_missing_seconds: float = 2.0,
                 recognition_confirm_seconds: float = 0.15,
                 recognition_revoke_seconds: float = 0.25,
                 focus_switch_threshold: float = 0.1,
                 focus_switch_seconds: float = 0.5,
                 focus_min_area_frac: float = 0.0,
                 focus_dwell_seconds: float = 0.0,
                 engage_max_yaw: float = 25.0,
                 engage_max_pitch: float = 20.0,
                 engage_dwell_seconds: float = 0.3,
                 emotion_debounce_seconds: float = 0.3,
                 auto_enroll: bool = True,
                 enroll_min_frames: int = 10,
                 enroll_min_face_px: int = 0,
                 enroll_min_sharpness: float = 0.0,
                 enroll_frontal_tolerance: float = 1.0,
                 enroll_max_yaw: float = 0.0):
        self.db = db
        self.emotion_detector = emotion_detector
        self.frame_scale = frame_scale

        # Detection/embedding backend. dlib stays available as a fallback so the
        # two can be A/B compared by flipping [tracker] backend in config.
        if backend == "dlib":
            self._backend = DlibBackend(frame_scale=frame_scale)
        elif backend == "insightface":
            self._backend = InsightFaceBackend(
                det_size=det_size, det_thresh=det_thresh, providers=providers)
        else:
            raise ValueError(f"Unknown backend {backend!r} (expected 'insightface' or 'dlib')")
        self._metric = self._backend.metric
        # The db's metric must already match the backend: it picks the db file and
        # schema version at load() time, which ran before this tracker was built.
        # A mismatch means the wrong db file was loaded -- realign and warn rather
        # than silently comparing embeddings across metrics.
        if self.db.metric != self._metric:
            logger.warning(
                "FaceDatabase metric %r != backend %r metric %r; realigning. "
                "Build FaceDatabase with metric=backend_metric(backend) so the "
                "right db file is loaded.",
                self.db.metric, self._backend.name, self._metric,
            )
            self.db.metric = self._metric

        # Frame-to-frame association threshold. Same identity across adjacent
        # frames sits very close in either metric; the sensible cutoff differs by
        # metric, so resolve a default only when the caller did not set one.
        if track_encoding_threshold is None:
            track_encoding_threshold = 0.6 if self._metric == "cosine" else 0.5
        self._track_enc_thresh = track_encoding_threshold
        self._track_iou_thresh = track_iou_threshold
        self._max_missing_s = max_missing_seconds
        self._confirm_s = recognition_confirm_seconds
        self._revoke_s = recognition_revoke_seconds
        self._emotion_debounce_s = emotion_debounce_seconds
        self._auto_enroll = auto_enroll
        self._enroll_min_frames = enroll_min_frames
        # Enrollment quality gate (0 / 1.0 = disabled, so defaults are a no-op).
        self._enroll_min_face_px = enroll_min_face_px
        self._enroll_min_sharpness = enroll_min_sharpness
        self._enroll_frontal_tolerance = enroll_frontal_tolerance
        # Real-yaw frontal gate for enrollment (degrees, 0 = off). Uses the
        # backend's head pose; preferred over the dlib nose-centering proxy when
        # the backend supplies pose (InsightFace).
        self._enroll_max_yaw = enroll_max_yaw

        self._tracks: list[TrackedFace] = []
        self._identities: dict[int, Identity] = {}
        # Track ids that have already been enrolled or named, so they are never
        # auto-enrolled a second time. A track keeps its physical identity; if
        # recognition briefly drops (revoking the Identity), we must re-recognize
        # it -- not mint a competing new pNNN, which causes id flip-flop.
        self._enrolled_track_ids: set[int] = set()
        self._last_frames: dict[int, np.ndarray] = {}
        self._next_id = 1
        self._lock = threading.Lock()
        self._skip_frame = False

        # Focus hysteresis
        self._focus_id: Optional[int] = None
        self._focus_switch_threshold = focus_switch_threshold
        self._focus_switch_s = focus_switch_seconds
        # Engagement gate: a face must cover at least this fraction of the frame
        # (proximity) and be held for this long before it can take focus. Both
        # default to off, so background passers-by are filtered only when set.
        self._focus_min_area_frac = focus_min_area_frac
        self._focus_dwell_s = focus_dwell_seconds
        self._focus_challenger_id: Optional[int] = None
        self._focus_challenger_since: float = 0.0

        # Engagement ("makes contact") gate, layered on top of focus: the focused
        # face must point at the camera within these pose limits, held for the
        # dwell, to count as engaged. Distinct from focus (which is who, not
        # whether they're looking). Requires a backend with head pose (InsightFace).
        self._engage_max_yaw = engage_max_yaw
        self._engage_max_pitch = engage_max_pitch
        self._engage_dwell_s = engage_dwell_seconds
        self._engaged_id: Optional[int] = None
        self._engage_candidate_id: Optional[int] = None
        self._engage_candidate_since: float = 0.0

        # Emotion debounce: track_id -> (emotion, since_timestamp)
        self._emotion_stable: dict[int, tuple[str, float]] = {}

        # Event system
        self._dispatcher = EventDispatcher(owner="face_tracker")

    # --- Public API ---

    def subscribe(self, callback: FaceEventCallback,
                  event_types: Optional[set] = None) -> Callable[[], None]:
        """Register a callback to receive face events.

        Args:
            callback: Called with a FaceEvent.
            event_types: If provided, only these event types are delivered.
                         If None, all events are delivered.
        Returns:
            An unsubscribe function.
        """
        return self._dispatcher.subscribe(callback, event_types)

    def unsubscribe(self, callback: FaceEventCallback) -> bool:
        return self._dispatcher.unsubscribe(callback)

    def process_frame(self, frame: np.ndarray) -> list[TrackedFace]:
        """Process a video frame. Returns tracked faces sorted by focus score."""
        frame_h, frame_w = frame.shape[:2]
        pending: list[FaceEvent] = []

        self._skip_frame = not self._skip_frame
        if self._skip_frame:
            with self._lock:
                focus_events = self._update_focus_scores(frame_w, frame_h)
                pending.extend(focus_events)
                pending.extend(self._update_engagement())
                result = sorted(self._tracks, key=lambda f: f.focus_score, reverse=True)
            self._dispatch_all(pending)
            return result

        locations, encodings, extras = self._detect_faces(frame)
        detections = list(zip(locations, encodings))
        now = time.time()

        with self._lock:
            matches, unmatched_dets, unmatched_tracks = self._match(detections)

            # --- Update matched tracks ---
            identity_changes = []
            for det_idx, track_idx in matches:
                bbox, enc = detections[det_idx]
                track = self._tracks[track_idx]
                was_occluded = not track.is_visible
                prev_ident = self._identities.get(track.track_id)
                old_pid = prev_ident.person_id if prev_ident else None
                old_emotion = track.emotion

                self._update_track(track, enc, bbox, frame, extras[det_idx])

                # Recovery event
                if was_occluded:
                    pid = self.get_person_id(track.track_id)
                    pending.append(self._make_event(
                        FaceEventType.FACE_RECOVERED, track.track_id,
                        FaceRecoveredPayload(
                            bbox=bbox, person_id=pid,
                            seconds_missing=now - track.last_seen + (now - track.last_seen),
                        )
                    ))

                # Identity change
                new_ident = self._identities.get(track.track_id)
                new_pid = new_ident.person_id if new_ident else None
                if old_pid != new_pid:
                    identity_changes.append((track.track_id, old_pid, new_pid,
                                             new_ident.confidence if new_ident else 0.0))

                # Emotion change (with debounce)
                if track.emotion != old_emotion:
                    stable = self._emotion_stable.get(track.track_id)
                    if stable is None or stable[0] != track.emotion:
                        self._emotion_stable[track.track_id] = (track.emotion, now)
                    elif now - stable[1] >= self._emotion_debounce_s:
                        pid = self.get_person_id(track.track_id)
                        pending.append(self._make_event(
                            FaceEventType.EMOTION_CHANGED, track.track_id,
                            EmotionChangedPayload(
                                old_emotion=old_emotion, new_emotion=track.emotion,
                                new_confidence=track.emotion_confidence, person_id=pid,
                            )
                        ))

            # --- Occluded tracks ---
            for track_idx in unmatched_tracks:
                track = self._tracks[track_idx]
                was_visible = track.is_visible
                track.frames_since_seen += 1
                if was_visible:
                    pid = self.get_person_id(track.track_id)
                    pending.append(self._make_event(
                        FaceEventType.FACE_OCCLUDED, track.track_id,
                        FaceOccludedPayload(last_bbox=track.bbox, person_id=pid)
                    ))

            # --- Evict lost tracks ---
            lost = []
            for track_idx in unmatched_tracks:
                track = self._tracks[track_idx]
                if now - track.last_seen > self._max_missing_s:
                    lost.append(track)

            for track in lost:
                ident = self._identities.pop(track.track_id, None)
                self._emotion_stable.pop(track.track_id, None)
                self._last_frames.pop(track.track_id, None)
                self._enrolled_track_ids.discard(track.track_id)
                pending.append(self._make_event(
                    FaceEventType.FACE_DISAPPEARED, track.track_id,
                    FaceDisappearedPayload(
                        last_bbox=track.bbox,
                        person_id=ident.person_id if ident else None,
                        duration_visible=track.last_seen - track.first_seen,
                        total_frames=track.frames_visible,
                    )
                ))
            if lost:
                lost_ids = {t.track_id for t in lost}
                self._tracks = [t for t in self._tracks if t.track_id not in lost_ids]

            # --- New tracks ---
            for det_idx in unmatched_dets:
                bbox, enc = detections[det_idx]
                track = self._create_track(enc, bbox, frame, extras[det_idx])
                self._tracks.append(track)
                ident = self._identities.get(track.track_id)
                pending.append(self._make_event(
                    FaceEventType.FACE_APPEARED, track.track_id,
                    FaceAppearedPayload(
                        bbox=bbox, emotion=track.emotion,
                        emotion_confidence=track.emotion_confidence,
                        initial_person_id=ident.person_id if ident else None,
                        initial_confidence=ident.confidence if ident else 0.0,
                    )
                ))

            # --- Auto-enroll stable unknown faces ---
            if self._auto_enroll:
                pending.extend(self._auto_enroll_unknown())

            # --- Identity change events ---
            for track_id, old_pid, new_pid, new_conf in identity_changes:
                if old_pid is None and new_pid:
                    pending.append(self._make_event(
                        FaceEventType.IDENTITY_CONFIRMED, track_id,
                        IdentityConfirmedPayload(
                            person_id=new_pid, confidence=new_conf,
                            last_seen_timestamp=self.db.get_last_seen(new_pid),
                        )
                    ))
                elif old_pid and new_pid is None:
                    pending.append(self._make_event(
                        FaceEventType.IDENTITY_LOST, track_id,
                        IdentityLostPayload(previous_person_id=old_pid)
                    ))
                elif old_pid and new_pid and old_pid != new_pid:
                    pending.append(self._make_event(
                        FaceEventType.IDENTITY_CHANGED, track_id,
                        IdentityChangedPayload(
                            old_person_id=old_pid, new_person_id=new_pid,
                            new_confidence=new_conf,
                        )
                    ))

            # --- Focus & engagement ---
            focus_events = self._update_focus_scores(frame_w, frame_h)
            pending.extend(focus_events)
            pending.extend(self._update_engagement())

            result = sorted(self._tracks, key=lambda f: f.focus_score, reverse=True)

        # Dispatch all events outside lock
        self._dispatch_all(pending)
        return result

    @property
    def focus_track_id(self) -> Optional[int]:
        return self._focus_id

    @property
    def engaged_track_id(self) -> Optional[int]:
        """Track id of the face currently making contact (looking at the camera),
        or None. A subset of the focused face -- see FACE_ENGAGED."""
        return self._engaged_id

    def is_engaged(self, track_id: int) -> bool:
        return self._engaged_id == track_id

    def get_identity(self, track_id: int) -> Optional[Identity]:
        with self._lock:
            return self._identities.get(track_id)

    def get_person_id(self, track_id: int) -> Optional[str]:
        ident = self._identities.get(track_id)
        return ident.person_id if ident else None

    def get_confidence(self, track_id: int) -> float:
        ident = self._identities.get(track_id)
        return ident.confidence if ident else 0.0

    def is_recognized(self, track_id: int) -> bool:
        return track_id in self._identities

    def get_visible_faces(self) -> list[TrackedFace]:
        with self._lock:
            return [t for t in self._tracks if t.is_visible]

    def get_face_by_id(self, track_id: int) -> Optional[TrackedFace]:
        with self._lock:
            for t in self._tracks:
                if t.track_id == track_id:
                    return t
        return None

    def get_primary_face(self) -> Optional[TrackedFace]:
        if self._focus_id is not None:
            face = self.get_face_by_id(self._focus_id)
            if face and face.is_visible:
                return face
        visible = self.get_visible_faces()
        return max(visible, key=lambda f: f.focus_score) if visible else None

    def get_recognized_person_ids(self) -> list[str]:
        result = []
        for f in self.get_visible_faces():
            pid = self.get_person_id(f.track_id)
            if pid:
                result.append(pid)
        return result

    def learn_face(self, track_id: int, person_id: str, frame: np.ndarray) -> bool:
        face = self.get_face_by_id(track_id)
        if face is None:
            return False
        # If this track is already bound to another id (typically an auto-enrolled
        # pNNN), rename that id to the new label rather than adding a second,
        # competing identity for the same face -- otherwise recognition keeps
        # flip-flopping between the two ids frame to frame.
        with self._lock:
            existing = self._identities.get(track_id)
            old_pid = existing.person_id if existing else None
        if old_pid and old_pid != person_id:
            self.db.rename_person(old_pid, person_id)
            with self._lock:
                # Repoint any in-memory identities that referenced the old id.
                for ident in self._identities.values():
                    if ident.person_id == old_pid:
                        ident.person_id = person_id
                    if ident._candidate_person_id == old_pid:
                        ident._candidate_person_id = person_id
        self.db.add_face(person_id, face.last_encoding, frame, face.bbox)
        with self._lock:
            self._enrolled_track_ids.add(track_id)
            self._identities[track_id] = Identity(
                person_id=person_id, confidence=100.0,
                _matching_since=0.0, _confirmed=True,
            )
        self._dispatcher.dispatch(self._make_event(
            FaceEventType.FACE_LEARNED, track_id,
            FaceLearnedPayload(person_id=person_id)
        ))
        return True

    def _allocate_person_id(self) -> str:
        """Pick the next free pNNN id by scanning the face database."""
        max_n = 0
        for pid in self.db.known_person_ids:
            m = re.match(r"^p(\d+)$", pid)
            if m:
                max_n = max(max_n, int(m.group(1)))
        return f"p{max_n + 1:03d}"

    def _auto_enroll_unknown(self) -> list[FaceEvent]:
        """Save any visible, unrecognized, stable faces to the face DB.

        Caller must hold self._lock. Returns events to be dispatched after
        the lock is released.
        """
        events = []
        for track in self._tracks:
            if track.track_id in self._identities:
                continue
            # Already enrolled/named once: never mint a competing id, even if its
            # identity was just revoked by a momentary recognition miss.
            if track.track_id in self._enrolled_track_ids:
                continue
            if not track.is_visible:
                continue
            if track.frames_visible < self._enroll_min_frames:
                continue
            frame = self._last_frames.get(track.track_id)
            if frame is None:
                continue
            if not self._passes_enroll_quality(frame, track.bbox, track.yaw):
                continue
            person_id = self._allocate_person_id()
            # Enroll the clean raw encoding, not the EMA-smoothed tracking one.
            self.db.add_face(person_id, track.last_encoding, frame, track.bbox)
            self._enrolled_track_ids.add(track.track_id)
            self._identities[track.track_id] = Identity(
                person_id=person_id, confidence=100.0,
                _matching_since=0.0, _confirmed=True,
            )
            events.append(self._make_event(
                FaceEventType.FACE_ENROLLED, track.track_id,
                FaceEnrolledPayload(person_id=person_id, bbox=track.bbox),
            ))
        return events

    def _passes_enroll_quality(self, frame, bbox, yaw=0.0) -> bool:
        """Quality gate for auto-enrollment: size, sharpness, frontalness.

        Each check is skipped when its threshold is at the disabled sentinel
        (0 / 1.0), so the default configuration enrolls exactly as before.
        Only runs on enrollment candidates (rare), so the landmark pass below
        is not a per-frame cost.
        """
        top, right, bottom, left = bbox
        w, h = right - left, bottom - top
        if min(w, h) < self._enroll_min_face_px:
            return False

        if self._enroll_min_sharpness > 0.0:
            roi = frame[max(0, top):max(0, bottom), max(0, left):max(0, right)]
            if roi.size == 0:
                return False
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            if cv2.Laplacian(gray, cv2.CV_64F).var() < self._enroll_min_sharpness:
                return False

        # Frontal gate: prefer the backend's real head pose (InsightFace) when
        # enabled; otherwise fall back to the dlib nose-centering proxy.
        if self._enroll_max_yaw > 0.0:
            if abs(yaw) > self._enroll_max_yaw:
                return False
        elif self._enroll_frontal_tolerance < 1.0 and not self._is_frontal(frame, bbox):
            return False

        return True

    def _is_frontal(self, frame, bbox) -> bool:
        """Cheap yaw proxy: is the nose roughly centred between the eyes?

        Uses dlib landmarks (via face_recognition) on the single enrollment
        crop. Returns True on any landmark failure so we never *block* on a
        detector hiccup -- the gate is meant to reject obvious profiles, not
        to be a hard authority. A true head-pose estimator arrives in the
        InsightFace phase.
        """
        top, right, bottom, left = bbox
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            landmarks = face_recognition.face_landmarks(rgb, [(top, right, bottom, left)])
        except Exception:
            return True
        if not landmarks:
            return False
        lm = landmarks[0]
        left_eye, right_eye, nose = lm.get("left_eye"), lm.get("right_eye"), lm.get("nose_tip")
        if not (left_eye and right_eye and nose):
            return False
        lx = np.mean([p[0] for p in left_eye])
        rx = np.mean([p[0] for p in right_eye])
        nx = np.mean([p[0] for p in nose])
        span = rx - lx
        if abs(span) < 1e-3:
            return False
        ratio = (nx - lx) / span  # ~0.5 looking straight ahead
        return abs(ratio - 0.5) <= self._enroll_frontal_tolerance

    @property
    def active_tracks(self) -> list[TrackedFace]:
        with self._lock:
            return list(self._tracks)

    # --- Internal ---

    def _make_event(self, etype, track_id, payload):
        return FaceEvent(type=etype, timestamp=time.time(),
                         track_id=track_id, payload=payload)

    def _dispatch_all(self, events):
        for e in events:
            logger.info(f"[{e.type.name}] track={e.track_id} {e.payload}")
            self._dispatcher.dispatch(e)

    def _focus_change_event(self, old_id, new_id, old_score, new_score, new_pid):
        """Build a FOCUS_CHANGED event. new_id=None means focus was cleared
        (payload.new_track_id reports 0, matching the historical contract)."""
        return self._make_event(
            FaceEventType.FOCUS_CHANGED, new_id,
            FocusChangedPayload(
                old_track_id=old_id,
                new_track_id=new_id if new_id is not None else 0,
                old_focus_score=old_score, new_focus_score=new_score,
                new_person_id=new_pid,
            )
        )

    def _update_focus_scores(self, frame_w, frame_h) -> list[FaceEvent]:
        """Compute focus scores and return any focus-change events.

        Focus only lands on a face that passes the proximity gate (covers at
        least ``focus_min_area_frac`` of the frame) and is held for
        ``focus_dwell_seconds`` -- so background passers-by never trigger it.
        With both at their disabled defaults this reduces to the original
        centrality+size selection with switch hysteresis.
        """
        events = []
        now = time.time()

        if not self._tracks:
            if self._focus_id is not None:
                events.append(self._focus_change_event(self._focus_id, None, 0.0, 0.0, None))
            self._focus_id = None
            self._focus_challenger_id = None
            self._focus_challenger_since = 0.0
            return events

        cx, cy = frame_w / 2, frame_h / 2
        frame_area = max(1, frame_w * frame_h)
        max_area = max(t.area for t in self._tracks) or 1

        for track in self._tracks:
            fx, fy = track.center
            dx = (fx - cx) / cx if cx > 0 else 0
            dy = (fy - cy) / cy if cy > 0 else 0
            centrality = 1.0 - np.sqrt(0.75 * dx ** 2 + 0.25 * dy ** 2)
            size = min(1.0, track.area / max_area) if max_area > 0 else 0.0
            track.focus_score = 0.75 * centrality + 0.25 * size

        visible = [t for t in self._tracks if t.is_visible]
        if not visible:
            return events  # keep the current focus as a ghost while occluded

        cur = next((t for t in self._tracks if t.track_id == self._focus_id), None)
        if cur is not None and not cur.is_visible:
            return events  # focused face briefly occluded -> hold focus

        # Proximity gate: only faces close enough to the camera are eligible.
        eligible = [t for t in visible
                    if t.area / frame_area >= self._focus_min_area_frac]
        if not eligible:
            # Only distant/background faces remain -> nobody is in focus.
            if self._focus_id is not None:
                old_score = cur.focus_score if cur else 0.0
                events.append(self._focus_change_event(self._focus_id, None, old_score, 0.0, None))
            self._focus_id = None
            self._focus_challenger_id = None
            self._focus_challenger_since = 0.0
            return events

        best = max(eligible, key=lambda f: f.focus_score)
        cur_eligible = cur is not None and cur in eligible

        # --- Acquisition: no valid current focus (none, gone, or now too far) ---
        if not cur_eligible:
            old_id = self._focus_id
            if self._focus_dwell_s <= 0:
                self._focus_id = best.track_id
                self._focus_challenger_id = None
                self._focus_challenger_since = 0.0
                if old_id != best.track_id:
                    events.append(self._focus_change_event(
                        old_id, best.track_id, 0.0, best.focus_score,
                        self.get_person_id(best.track_id)))
                return events
            # Require the candidate to dwell before we engage.
            if best.track_id == self._focus_challenger_id:
                if now - self._focus_challenger_since >= self._focus_dwell_s:
                    self._focus_id = best.track_id
                    self._focus_challenger_id = None
                    self._focus_challenger_since = 0.0
                    if old_id != best.track_id:
                        events.append(self._focus_change_event(
                            old_id, best.track_id, 0.0, best.focus_score,
                            self.get_person_id(best.track_id)))
            else:
                self._focus_challenger_id = best.track_id
                self._focus_challenger_since = now
            return events

        # --- Switch hysteresis: a valid current focus exists ---
        current_score = cur.focus_score if cur else 0.0
        if best.track_id != self._focus_id and \
           best.focus_score > current_score + self._focus_switch_threshold:
            if best.track_id == self._focus_challenger_id:
                if now - self._focus_challenger_since >= self._focus_switch_s:
                    old_id = self._focus_id
                    self._focus_id = best.track_id
                    self._focus_challenger_id = None
                    self._focus_challenger_since = 0.0
                    events.append(self._focus_change_event(
                        old_id, best.track_id, current_score, best.focus_score,
                        self.get_person_id(best.track_id)))
            else:
                self._focus_challenger_id = best.track_id
                self._focus_challenger_since = now
        else:
            self._focus_challenger_id = None
            self._focus_challenger_since = 0.0

        return events

    # Extra yaw/pitch slack before an engaged face is declared disengaged, so a
    # small wobble at the threshold does not flicker FACE_ENGAGED/DISENGAGED.
    _ENGAGE_HYSTERESIS_DEG = 8.0

    def _disengage_event(self, track_id, reason):
        return self._make_event(
            FaceEventType.FACE_DISENGAGED, track_id,
            FaceDisengagedPayload(person_id=self.get_person_id(track_id), reason=reason),
        )

    def _update_engagement(self) -> list[FaceEvent]:
        """Track whether the focused face is looking at the camera.

        Engagement is layered on focus: the focused face must point at the camera
        within ``engage_max_yaw`` / ``engage_max_pitch`` and hold it for
        ``engage_dwell_seconds`` to emit FACE_ENGAGED; turning away (past a
        hysteresis margin), losing focus, or disappearing emits FACE_DISENGAGED.
        Caller must hold ``self._lock``.
        """
        events: list[FaceEvent] = []
        now = time.time()

        focus = next((t for t in self._tracks if t.track_id == self._focus_id), None)
        if focus is None or not focus.is_visible:
            if self._engaged_id is not None:
                reason = "disappeared" if focus is not None else "lost_focus"
                events.append(self._disengage_event(self._engaged_id, reason))
                self._engaged_id = None
            self._engage_candidate_id = None
            self._engage_candidate_since = 0.0
            return events

        if self._engaged_id is not None:
            if self._engaged_id != focus.track_id:
                # Focus moved to a different face -> the old one disengages.
                events.append(self._disengage_event(self._engaged_id, "lost_focus"))
                self._engaged_id = None
            elif (abs(focus.yaw) > self._engage_max_yaw + self._ENGAGE_HYSTERESIS_DEG or
                  abs(focus.pitch) > self._engage_max_pitch + self._ENGAGE_HYSTERESIS_DEG):
                events.append(self._disengage_event(self._engaged_id, "looked_away"))
                self._engaged_id = None
            else:
                return events  # still engaged

        # Not (or no longer) engaged: require the focused face to face the camera
        # for the dwell window before declaring engagement.
        aligned = (abs(focus.yaw) <= self._engage_max_yaw and
                   abs(focus.pitch) <= self._engage_max_pitch)
        if aligned:
            if self._engage_candidate_id != focus.track_id:
                self._engage_candidate_id = focus.track_id
                self._engage_candidate_since = now
            # Same pass handles dwell=0 (engage immediately) and dwell>0 (wait).
            if now - self._engage_candidate_since >= self._engage_dwell_s:
                self._engaged_id = focus.track_id
                self._engage_candidate_id = None
                self._engage_candidate_since = 0.0
                events.append(self._make_event(
                    FaceEventType.FACE_ENGAGED, focus.track_id,
                    FaceEngagedPayload(
                        person_id=self.get_person_id(focus.track_id),
                        yaw=focus.yaw, pitch=focus.pitch,
                        focus_score=focus.focus_score),
                ))
        else:
            self._engage_candidate_id = None
            self._engage_candidate_since = 0.0
        return events

    def _detect_faces(self, frame):
        return self._backend.detect(frame)

    def _match(self, detections):
        if not detections or not self._tracks:
            return ([], list(range(len(detections))), list(range(len(self._tracks))))

        n_det = len(detections)
        n_trk = len(self._tracks)
        costs = np.zeros((n_det, n_trk))
        for i, (bbox_d, enc_d) in enumerate(detections):
            for j, track in enumerate(self._tracks):
                costs[i, j] = _pair_distance(enc_d, track.encoding, self._metric)

        matches = []
        used_dets = set()
        used_tracks = set()
        candidates = sorted(
            ((costs[i, j], i, j) for i in range(n_det) for j in range(n_trk))
        )
        for dist, i, j in candidates:
            if i in used_dets or j in used_tracks:
                continue
            if dist < self._track_enc_thresh:
                matches.append((i, j))
                used_dets.add(i)
                used_tracks.add(j)

        remaining_dets = [i for i in range(n_det) if i not in used_dets]
        remaining_tracks = [j for j in range(n_trk) if j not in used_tracks]
        for i in list(remaining_dets):
            best_iou, best_j = 0.0, -1
            for j in remaining_tracks:
                iou = self._compute_iou(detections[i][0], self._tracks[j].bbox)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou > self._track_iou_thresh and best_j >= 0:
                matches.append((i, best_j))
                remaining_dets.remove(i)
                remaining_tracks.remove(best_j)

        return matches, remaining_dets, remaining_tracks

    def _compute_iou(self, bbox1, bbox2):
        t1, r1, b1, l1 = bbox1
        t2, r2, b2, l2 = bbox2
        it, il = max(t1, t2), max(l1, l2)
        ib, ir = min(b1, b2), min(r1, r2)
        if ib <= it or ir <= il:
            return 0.0
        inter = (ib - it) * (ir - il)
        union = (b1 - t1) * (r1 - l1) + (b2 - t2) * (r2 - l2) - inter
        return inter / union if union > 0 else 0.0

    def _update_track(self, track, encoding, bbox, frame, extra=None):
        now = time.time()
        track.last_encoding = encoding
        track.encoding = 0.3 * encoding + 0.7 * track.encoding
        if self._metric == "cosine":
            # The EMA blend of two unit vectors is no longer unit-norm; cosine
            # distance assumes unit vectors, so re-normalize.
            norm = np.linalg.norm(track.encoding)
            if norm > 0:
                track.encoding = track.encoding / norm
        track.yaw, track.pitch = _pose_yaw_pitch(extra)
        track.bbox = bbox
        track.last_seen = now
        track.frames_visible += 1
        track.frames_since_seen = 0
        self._last_frames[track.track_id] = frame

        top, right, bottom, left = bbox
        face_roi = frame[top:bottom, left:right]
        if face_roi.size > 0 and self.emotion_detector:
            try:
                label, conf = self.emotion_detector.detect(face_roi)
                track.emotion = label
                track.emotion_confidence = conf
            except Exception:
                pass

        self._recognize_and_stabilize(track)

    def _create_track(self, encoding, bbox, frame, extra=None):
        now = time.time()
        yaw, pitch = _pose_yaw_pitch(extra)
        track = TrackedFace(
            track_id=self._next_id, encoding=encoding.copy(),
            last_encoding=encoding.copy(), bbox=bbox,
            first_seen=now, last_seen=now, frames_visible=1,
            yaw=yaw, pitch=pitch,
        )
        self._next_id += 1
        self._last_frames[track.track_id] = frame

        top, right, bottom, left = bbox
        face_roi = frame[top:bottom, left:right]
        if face_roi.size > 0 and self.emotion_detector:
            try:
                label, conf = self.emotion_detector.detect(face_roi)
                track.emotion = label
                track.emotion_confidence = conf
            except Exception:
                pass

        person_id, confidence = self.db.recognize(encoding)
        if person_id is not None:
            self._identities[track.track_id] = Identity(
                person_id=person_id, confidence=confidence,
                _matching_since=now, _candidate_person_id=person_id,
                _candidate_confidence=confidence,
            )

        return track

    def _recognize_and_stabilize(self, track):
        raw_pid, raw_conf = self.db.recognize(track.encoding)
        tid = track.track_id
        now = time.time()
        ident = self._identities.get(tid)

        if raw_pid is not None:
            if ident is None:
                ident = Identity(
                    person_id=raw_pid, confidence=raw_conf,
                    _matching_since=now, _candidate_person_id=raw_pid,
                    _candidate_confidence=raw_conf,
                )
                self._identities[tid] = ident
            else:
                if raw_pid != ident._candidate_person_id:
                    ident._candidate_person_id = raw_pid
                    ident._candidate_confidence = raw_conf
                    ident._matching_since = now
                ident._failing_since = 0.0
                if now - ident._matching_since >= self._confirm_s:
                    ident.person_id = raw_pid
                    ident.confidence = raw_conf
                    ident._confirmed = True
        else:
            if ident is not None:
                if ident._failing_since == 0.0:
                    ident._failing_since = now
                ident._matching_since = now
                if now - ident._failing_since >= self._revoke_s:
                    del self._identities[tid]


# ---------------------------------------------------------------------------
# Standalone mode
# ---------------------------------------------------------------------------

def _get_name_from_gui(frame, existing_match=None):
    name = ""
    while True:
        display = frame.copy()
        overlay = display.copy()
        h, w = display.shape[:2]
        box_h = 120
        y_start = h // 2 - box_h // 2
        cv2.rectangle(overlay, (0, y_start), (w, y_start + box_h), (0, 0, 0), cv2.FILLED)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
        if existing_match and not name:
            match_name, match_conf = existing_match
            cv2.putText(display, f"Known as: {match_name} ({match_conf:.0f}%) - ENTER=add sample, type to override",
                        (20, y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(display, "Type name and press ENTER (ESC to cancel):",
                        (20, y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display, name + "_", (20, y_start + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imshow("Face Tracker", display)

        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            return None
        elif key in (13, 10):
            if name.strip():
                return name.strip()
            elif existing_match:
                return existing_match[0]
            return None
        elif key in (8, 127):
            name = name[:-1]
        elif 32 <= key <= 126:
            name += chr(key)


_EVENT_SHORT_NAMES = {
    "FACE_APPEARED": "APPEARED",
    "FACE_DISAPPEARED": "GONE",
    "FACE_OCCLUDED": "OCCLUDED",
    "FACE_RECOVERED": "RECOVERED",
    "IDENTITY_CONFIRMED": "ID OK",
    "IDENTITY_LOST": "ID LOST",
    "IDENTITY_CHANGED": "ID CHANGE",
    "FACE_LEARNED": "LEARNED",
    "FACE_ENROLLED": "ENROLLED",
    "FOCUS_CHANGED": "FOCUS",
    "FACE_ENGAGED": "ENGAGED",
    "FACE_DISENGAGED": "DISENGAGED",
    "EMOTION_CHANGED": "EMOTION",
}

_EVENT_COLORS = {
    "FACE_APPEARED": (255, 200, 100),
    "FACE_DISAPPEARED": (120, 120, 255),
    "FACE_OCCLUDED": (180, 180, 120),
    "FACE_RECOVERED": (120, 255, 180),
    "IDENTITY_CONFIRMED": (120, 255, 120),
    "IDENTITY_LOST": (120, 120, 255),
    "IDENTITY_CHANGED": (220, 220, 120),
    "FACE_LEARNED": (255, 255, 120),
    "FACE_ENROLLED": (255, 200, 255),
    "FOCUS_CHANGED": (120, 220, 255),
    "FACE_ENGAGED": (100, 255, 100),
    "FACE_DISENGAGED": (120, 160, 220),
    "EMOTION_CHANGED": (220, 140, 255),
}


def _draw_log_window(log_lines, width=800, height=600):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)

    # Title bar
    cv2.rectangle(canvas, (0, 0), (width, 36), (50, 50, 50), cv2.FILLED)
    cv2.putText(canvas, "FACE TRACKER EVENTS", (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1, cv2.LINE_AA)
    count_text = f"{len(log_lines)} events"
    cv2.putText(canvas, count_text, (width - 150, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)

    line_h = 26
    max_lines = (height - 46) // line_h
    visible = log_lines[-max_lines:]

    tag_x = 70       # after timestamp
    msg_x = 175      # after tag
    dim_color = (90, 90, 90)

    for i, (ts, etype, msg) in enumerate(visible):
        y = 44 + i * line_h
        color = _EVENT_COLORS.get(etype, (180, 180, 180))
        short = _EVENT_SHORT_NAMES.get(etype, etype[:10])

        # Alternating row background
        if i % 2 == 0:
            cv2.rectangle(canvas, (0, y - 4), (width, y + line_h - 6), (38, 38, 38), cv2.FILLED)

        # Timestamp (dimmed)
        cv2.putText(canvas, ts, (8, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, dim_color, 1, cv2.LINE_AA)

        # Event tag (colored, fixed width)
        cv2.putText(canvas, short, (tag_x, y + 14),
                    cv2.FONT_HERSHEY_DUPLEX, 0.42, color, 1, cv2.LINE_AA)

        # Message (white, truncated to fit)
        max_chars = (width - msg_x - 10) // 8
        display_msg = msg if len(msg) <= max_chars else msg[:max_chars - 2] + ".."
        cv2.putText(canvas, display_msg, (msg_x, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)

        # Subtle separator
        cv2.line(canvas, (8, y + line_h - 5), (width - 8, y + line_h - 5), (45, 45, 45), 1)

    cv2.imshow("Face Tracker Log", canvas)


def main():
    parser = argparse.ArgumentParser(description="Standalone face tracker")
    parser.add_argument("--db-dir", default="known_faces", help="Face database directory")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--backend", choices=["insightface", "dlib"], default=None,
                        help="Detection/embedding backend (overrides face_config.toml)")
    parser.add_argument("--det-size", type=int, default=None,
                        help="InsightFace detector input size (square)")
    parser.add_argument("--det-thresh", type=float, default=None,
                        help="InsightFace minimum detector confidence")
    parser.add_argument("--scale", type=float, default=None,
                        help="dlib backend detection scale factor (overrides face_config.toml)")
    parser.add_argument("--engage-max-yaw", type=float, default=None,
                        help="Max |yaw| degrees for the focused face to count as engaged")
    parser.add_argument("--engage-max-pitch", type=float, default=None,
                        help="Max |pitch| degrees for the focused face to count as engaged")
    parser.add_argument("--engage-dwell-seconds", type=float, default=None,
                        help="Seconds the focused face must face the camera before FACE_ENGAGED")
    parser.add_argument("--fps", type=int, default=0, help="Max FPS (0 = unlimited)")
    parser.add_argument("--no-emotion", action="store_true", help="Disable emotion detection")
    parser.add_argument("--no-log-window", action="store_true", help="Disable log window")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    log_lines = []

    def on_event_display(event: FaceEvent):
        ts = datetime.now().strftime("%H:%M:%S")
        p = event.payload
        # Build a short summary
        if event.type == FaceEventType.FACE_APPEARED:
            msg = f"track={event.track_id} id={p.initial_person_id or '?'} emo={p.emotion}"
        elif event.type == FaceEventType.FACE_DISAPPEARED:
            msg = f"track={event.track_id} id={p.person_id or '?'} dur={p.duration_visible:.1f}s"
        elif event.type == FaceEventType.FOCUS_CHANGED:
            msg = f"{p.old_track_id} -> {p.new_track_id} ({p.new_person_id or '?'})"
        elif event.type == FaceEventType.IDENTITY_CONFIRMED:
            msg = f"track={event.track_id} -> {p.person_id} ({p.confidence:.0f}%)"
        elif event.type == FaceEventType.IDENTITY_CHANGED:
            msg = f"track={event.track_id} {p.old_person_id} -> {p.new_person_id} ({p.new_confidence:.0f}%)"
        elif event.type == FaceEventType.IDENTITY_LOST:
            msg = f"track={event.track_id} lost {p.previous_person_id}"
        elif event.type == FaceEventType.FACE_LEARNED:
            msg = f"track={event.track_id} learned as {p.person_id}"
        elif event.type == FaceEventType.FACE_ENROLLED:
            msg = f"track={event.track_id} enrolled as {p.person_id}"
        elif event.type == FaceEventType.FACE_OCCLUDED:
            msg = f"track={event.track_id} id={p.person_id or '?'}"
        elif event.type == FaceEventType.FACE_RECOVERED:
            msg = f"track={event.track_id} id={p.person_id or '?'} missing={p.seconds_missing:.1f}s"
        elif event.type == FaceEventType.FACE_ENGAGED:
            msg = f"track={event.track_id} id={p.person_id or '?'} yaw={p.yaw:.0f} pitch={p.pitch:.0f}"
        elif event.type == FaceEventType.FACE_DISENGAGED:
            msg = f"track={event.track_id} id={p.person_id or '?'} ({p.reason})"
        elif event.type == FaceEventType.EMOTION_CHANGED:
            msg = f"track={event.track_id} {p.old_emotion}->{p.new_emotion}"
        else:
            msg = str(p)[:60]
        log_lines.append((ts, event.type.name, msg))

    from face_config import build_db_kwargs, build_tracker_kwargs, backend_metric
    db_kwargs = build_db_kwargs()
    if args.backend is not None:
        # Keep the db's metric/file in step with a CLI backend override so
        # load() opens the matching db (it runs before the tracker is built).
        db_kwargs["metric"] = backend_metric(args.backend)
    face_db = FaceDatabase(db_dir=args.db_dir, **db_kwargs)
    face_db.load()

    emotion_detector = None
    if not args.no_emotion:
        emotion_detector = EmotionDetector()

    tracker_kwargs = build_tracker_kwargs()
    for cli_val, kw in (
        (args.backend, "backend"),
        (args.det_size, "det_size"),
        (args.det_thresh, "det_thresh"),
        (args.scale, "frame_scale"),
        (args.engage_max_yaw, "engage_max_yaw"),
        (args.engage_max_pitch, "engage_max_pitch"),
        (args.engage_dwell_seconds, "engage_dwell_seconds"),
    ):
        if cli_val is not None:
            tracker_kwargs[kw] = cli_val
    tracker = FaceTracker(db=face_db, emotion_detector=emotion_detector,
                          **tracker_kwargs)
    tracker.subscribe(on_event_display)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logger.error(f"Could not open camera {args.camera}")
        return

    cv2.namedWindow("Face Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face Tracker", 800, 600)
    if not args.no_log_window:
        cv2.namedWindow("Face Tracker Log", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Face Tracker Log", 800, 600)
        cv2.moveWindow("Face Tracker Log", 820, 0)

    selected_track_id = None
    frame_min_interval = 1.0 / args.fps if args.fps > 0 else 0
    last_frame_time = 0.0

    # --- Detector runs on its own thread; main thread handles capture + display. ---
    state_lock = threading.Lock()
    stats_lock = threading.Lock()
    pending_frame = [None]
    frame_signal = threading.Event()
    stop_event = threading.Event()
    latest = {
        "faces": [],           # list[TrackedFace] snapshot (copies, safe to read)
        "identities": {},      # track_id -> (person_id, confidence)
        "focus_id": None,
        "engaged_id": None,
        "result_id": 0,
        "process_ms": 0.0,
        "result_time": 0.0,
    }
    stats = {
        "captured": 0,
        "submitted": 0,
        "processed": 0,
        "dropped": 0,
        "display_frames": 0,
        "display_reused": 0,
        "detector_busy_s": 0.0,
    }

    def detector_worker():
        local_result_id = 0
        while not stop_event.is_set():
            frame_signal.wait(timeout=0.1)
            frame_signal.clear()
            with state_lock:
                work = pending_frame[0]
                pending_frame[0] = None
            if work is None:
                continue
            t0 = time.time()
            faces = tracker.process_frame(work)
            elapsed_ms = (time.time() - t0) * 1000.0
            faces_snap = [copy.copy(f) for f in faces]
            idents = {}
            for f in faces_snap:
                pid = tracker.get_person_id(f.track_id)
                if pid is not None:
                    idents[f.track_id] = (pid, tracker.get_confidence(f.track_id))
            focus_id_local = tracker.focus_track_id
            engaged_id_local = tracker.engaged_track_id
            local_result_id += 1
            with state_lock:
                latest["faces"] = faces_snap
                latest["identities"] = idents
                latest["focus_id"] = focus_id_local
                latest["engaged_id"] = engaged_id_local
                latest["result_id"] = local_result_id
                latest["process_ms"] = elapsed_ms
                latest["result_time"] = time.time()
            with stats_lock:
                stats["processed"] += 1
                stats["detector_busy_s"] += elapsed_ms / 1000.0

    worker = threading.Thread(target=detector_worker, daemon=True, name="face-detector")
    worker.start()

    hud_text = ""
    hud_last_time = time.time()
    hud_last_snapshot = dict(stats)
    prev_displayed_result_id = -1

    logger.info(f"Controls: L=learn  TAB=select  D=delete-db  Q=quit | FPS cap: {args.fps or 'unlimited'}")

    while True:
        if frame_min_interval > 0:
            now = time.time()
            elapsed = now - last_frame_time
            if elapsed < frame_min_interval:
                wait_ms = max(1, int((frame_min_interval - elapsed) * 1000))
                key = cv2.waitKey(wait_ms) & 0xFF
                if key == ord("q") or key == 27:
                    break
                continue
            last_frame_time = now

        ret, frame = cap.read()
        if not ret:
            break
        with stats_lock:
            stats["captured"] += 1

        # Handoff: overwrite pending slot. If something was already waiting, that's a drop.
        with state_lock:
            dropped_here = pending_frame[0] is not None
            pending_frame[0] = frame
        frame_signal.set()
        with stats_lock:
            stats["submitted"] += 1
            if dropped_here:
                stats["dropped"] += 1

        # Snapshot the latest detector result for this display frame.
        with state_lock:
            faces_snap = latest["faces"]
            idents_snap = latest["identities"]
            focus_id = latest["focus_id"]
            engaged_id = latest["engaged_id"]
            result_id = latest["result_id"]
            process_ms = latest["process_ms"]
            result_time = latest["result_time"]

        with stats_lock:
            stats["display_frames"] += 1
            if result_id == prev_displayed_result_id:
                stats["display_reused"] += 1
        prev_displayed_result_id = result_id

        visible = [f for f in faces_snap if f.is_visible]

        if selected_track_id is not None:
            if not any(f.track_id == selected_track_id for f in visible):
                selected_track_id = visible[0].track_id if visible else None
        elif visible:
            selected_track_id = visible[0].track_id

        focus_face = next((f for f in faces_snap if f.track_id == focus_id), None) if focus_id else None
        focus_is_ghost = focus_face is not None and not focus_face.is_visible

        if focus_is_ghost and focus_face:
            top, right, bottom, left = focus_face.bbox
            elapsed = time.time() - focus_face.last_seen
            alpha = max(0.0, 1.0 - elapsed / 2.0)
            ghost_color = (0, int(255 * alpha), int(100 * alpha))
            for i in range(0, right - left, 12):
                cv2.line(frame, (left + i, top), (left + min(i + 6, right - left), top), ghost_color, 2)
                cv2.line(frame, (left + i, bottom), (left + min(i + 6, right - left), bottom), ghost_color, 2)
            for i in range(0, bottom - top, 12):
                cv2.line(frame, (left, top + i), (left, top + min(i + 6, bottom - top)), ghost_color, 2)
                cv2.line(frame, (right, top + i), (right, top + min(i + 6, bottom - top)), ghost_color, 2)
            ghost_pid_conf = idents_snap.get(focus_face.track_id)
            pid = ghost_pid_conf[0] if ghost_pid_conf else None
            ghost_label = f"FOCUS (lost {elapsed:.1f}s)"
            if pid:
                ghost_label = f"{pid} - {ghost_label}"
            cv2.putText(frame, ghost_label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, ghost_color, 1)

        for rank, face in enumerate(visible):
            is_focus = (face.track_id == focus_id)
            is_engaged = (face.track_id == engaged_id)
            is_selected = (face.track_id == selected_track_id)
            pid_conf = idents_snap.get(face.track_id)
            pid = pid_conf[0] if pid_conf else None
            conf = pid_conf[1] if pid_conf else 0.0
            top, right, bottom, left = face.bbox

            if is_focus:
                color = (0, 255, 100)
                thickness = 4
                glow = frame.copy()
                pad = 6
                cv2.rectangle(glow, (left - pad, top - pad), (right + pad, bottom + pad),
                              (0, 255, 100), cv2.FILLED)
                cv2.addWeighted(glow, 0.15, frame, 0.85, 0, frame)
            elif pid:
                color = (0, 200, 0)
                thickness = 2
            else:
                color = (0, 0, 200)
                thickness = 1

            if not is_focus and len(visible) > 1:
                dim = frame.copy()
                cv2.rectangle(dim, (left, top), (right, bottom), (0, 0, 0), cv2.FILLED)
                cv2.addWeighted(dim, 0.15, frame, 0.85, 0, frame)

            cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)

            if is_selected:
                cv2.rectangle(frame, (left - 2, top - 2), (right + 2, bottom + 2),
                              (0, 255, 255), 1)

            if pid:
                label = f"[{rank+1}] #{face.track_id} {pid} {conf:.0f}%"
            else:
                label = f"[{rank+1}] #{face.track_id} Unknown"
            cv2.rectangle(frame, (left, bottom), (right, bottom + 30), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 6, bottom + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            if is_focus:
                badge_w = 70
                cv2.rectangle(frame, (left, top - 28), (left + badge_w, top - 4),
                              (0, 255, 100), cv2.FILLED)
                cv2.putText(frame, "FOCUS", (left + 6, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                if is_engaged:
                    cv2.rectangle(frame, (left + badge_w + 4, top - 28),
                                  (left + badge_w + 4 + 110, top - 4),
                                  (60, 255, 60), cv2.FILLED)
                    cv2.putText(frame, "ENGAGED", (left + badge_w + 10, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            emo_y = top - 32 if is_focus else top - 10
            if face.emotion and face.emotion != "neutral":
                cv2.putText(frame, f"{face.emotion} ({face.emotion_confidence:.0%})",
                            (left, emo_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            info = (f"focus:{face.focus_score:.2f} vis:{face.frames_visible} "
                    f"yaw:{face.yaw:.0f} pitch:{face.pitch:.0f}")
            cv2.putText(frame, info, (left + 6, bottom + 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        # Recompute HUD at ~1 Hz from counter deltas.
        now_t = time.time()
        dt_hud = now_t - hud_last_time
        if dt_hud >= 1.0:
            with stats_lock:
                snap = dict(stats)
            d_cap = snap["captured"] - hud_last_snapshot["captured"]
            d_proc = snap["processed"] - hud_last_snapshot["processed"]
            d_sub = snap["submitted"] - hud_last_snapshot["submitted"]
            d_drop = snap["dropped"] - hud_last_snapshot["dropped"]
            d_disp = snap["display_frames"] - hud_last_snapshot["display_frames"]
            drop_pct = (100.0 * d_drop / d_sub) if d_sub else 0.0
            stale_ms = max(0.0, (now_t - result_time) * 1000.0) if result_time else 0.0
            hud_text = (
                f"cap {d_cap/dt_hud:4.1f}fps  det {d_proc/dt_hud:4.1f}fps  "
                f"disp {d_disp/dt_hud:4.1f}fps  drop {drop_pct:4.0f}%  "
                f"proc {process_ms:5.0f}ms  stale {stale_ms:4.0f}ms"
            )
            hud_last_time = now_t
            hud_last_snapshot = snap

        status = f"Tracks: {len(visible)} | Known: {len(face_db.known_person_ids)} | DB: {face_db.encoding_count} encodings"
        cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if hud_text:
            cv2.putText(frame, hud_text, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (180, 255, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, "L=learn  TAB=select  D=delete-db  Q=quit",
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        cv2.imshow("Face Tracker", frame)
        if not args.no_log_window:
            _draw_log_window(log_lines)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == 9:
            if visible:
                ids = [f.track_id for f in visible]
                if selected_track_id in ids:
                    idx = (ids.index(selected_track_id) + 1) % len(ids)
                    selected_track_id = ids[idx]
                else:
                    selected_track_id = ids[0]
        elif key == ord("l"):
            if selected_track_id:
                face = tracker.get_face_by_id(selected_track_id)
                if face:
                    existing = None
                    if tracker.is_recognized(face.track_id):
                        existing = (tracker.get_person_id(face.track_id),
                                    tracker.get_confidence(face.track_id))
                    input_label = _get_name_from_gui(frame, existing)
                    if input_label:
                        tracker.learn_face(selected_track_id, input_label, frame)
        elif key == ord("d"):
            display = frame.copy()
            cv2.putText(display, "DELETE ALL FACES? Y/N", (50, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.imshow("Face Tracker", display)
            confirm = cv2.waitKey(0) & 0xFF
            if confirm == ord("y"):
                face_db.clear()

    stop_event.set()
    frame_signal.set()
    worker.join(timeout=2.0)
    cap.release()
    cv2.destroyAllWindows()

    with stats_lock:
        final = dict(stats)
    sub = final["submitted"] or 1
    disp = final["display_frames"] or 1
    avg_proc_ms = (final["detector_busy_s"] * 1000.0 / final["processed"]) if final["processed"] else 0.0
    logger.info(
        "Pipeline stats: captured=%d submitted=%d processed=%d dropped=%d (%.1f%%) "
        "display=%d reused=%d (%.1f%%) avg_proc=%.1fms",
        final["captured"], final["submitted"], final["processed"], final["dropped"],
        100.0 * final["dropped"] / sub,
        final["display_frames"], final["display_reused"],
        100.0 * final["display_reused"] / disp,
        avg_proc_ms,
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
