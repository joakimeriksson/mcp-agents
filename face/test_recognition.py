"""Headless logic tests for the cosine (InsightFace/ArcFace) recognition path.

Covers the parts of the Phase (a) backend swap that don't need a camera or the
model pack: the metric-aware distance helpers, the cosine mean-of-k aggregation
in FaceDatabase.recognize(), the per-metric db file / schema version, and the
load() guard that refuses a stale-schema db.

Run:  uv run python test_recognition.py
"""

import os
import pickle
import tempfile

import numpy as np

from face_tracker import (
    FaceDatabase,
    _embedding_distances,
    _pair_distance,
)


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def test_distance_helpers():
    a = _unit([1.0, 0.0, 0.0])
    b = _unit([1.0, 0.0, 0.0])
    c = _unit([0.0, 1.0, 0.0])
    # cosine distance: identical -> 0, orthogonal -> 1
    assert abs(_pair_distance(a, b, "cosine")) < 1e-9
    assert abs(_pair_distance(a, c, "cosine") - 1.0) < 1e-9
    # euclidean distance matches numpy
    assert abs(_pair_distance(a, c, "euclidean") - np.sqrt(2)) < 1e-9
    known = np.stack([b, c])
    d = _embedding_distances(known, a, "cosine")
    assert d.shape == (2,)
    assert abs(d[0]) < 1e-9 and abs(d[1] - 1.0) < 1e-9
    print("ok: distance helpers")


def test_cosine_mean_of_k():
    with tempfile.TemporaryDirectory() as d:
        db = FaceDatabase(db_dir=d, tolerance=0.4, recognition_k=2, metric="cosine")
        # Person A: a tight cluster near +x. Person B: near +y.
        a_dir = _unit([1.0, 0.05, 0.0])
        b_dir = _unit([0.0, 1.0, 0.05])
        for jitter in (0.0, 0.02, 0.04):
            db.add_encoding("pA", _unit(a_dir + jitter * np.array([0, 0, 1.0])))
            db.add_encoding("pB", _unit(b_dir + jitter * np.array([1.0, 0, 0])))
        # A query right on top of A's cluster resolves to pA with high confidence.
        pid, conf = db.recognize(_unit([1.0, 0.05, 0.0]))
        assert pid == "pA", pid
        assert conf > 80.0, conf
        # A query near B resolves to pB.
        pid, _ = db.recognize(_unit([0.0, 1.0, 0.05]))
        assert pid == "pB", pid
        # An orthogonal query is beyond tolerance -> no match.
        pid, conf = db.recognize(_unit([0.0, 0.0, 1.0]))
        assert pid is None and conf == 0.0, (pid, conf)
        print("ok: cosine mean-of-k recognition")


def test_per_metric_file_and_version():
    with tempfile.TemporaryDirectory() as d:
        cos = FaceDatabase(db_dir=d, metric="cosine")
        euc = FaceDatabase(db_dir=d, metric="euclidean")
        assert cos.db_file.endswith("faces_arcface.pkl"), cos.db_file
        assert euc.db_file.endswith("faces.pkl"), euc.db_file
        assert cos.schema_version == 3 and euc.schema_version == 2
        print("ok: per-metric db file and schema version")


def test_load_rejects_stale_schema():
    with tempfile.TemporaryDirectory() as d:
        # A v2-looking db written where the cosine backend expects v3.
        stale = {
            "encodings": [np.zeros(128)],
            "person_ids": ["pX"],
            "last_seen": {},
            "schema_version": 2,
        }
        cos = FaceDatabase(db_dir=d, metric="cosine")
        with open(cos.db_file, "wb") as f:
            pickle.dump(stale, f)
        cos.load()  # should warn and start empty, not import the 128-d vector
        assert cos.encoding_count == 0, cos.encoding_count
        assert "pX" not in cos.known_person_ids

        # A correct v3 db round-trips.
        cos.add_encoding("pA", _unit([1.0, 0.0, 0.0]))
        cos.save()
        again = FaceDatabase(db_dir=d, metric="cosine")
        again.load()
        assert again.encoding_count == 1
        assert "pA" in again.known_person_ids
        print("ok: load rejects stale schema, accepts matching schema")


def test_rename_person():
    with tempfile.TemporaryDirectory() as d:
        db = FaceDatabase(db_dir=d, tolerance=0.4, recognition_k=3, metric="cosine")
        a = _unit([1.0, 0.05, 0.0])
        # p001 auto-enrolled, plus a saved crop folder on disk.
        for _ in range(2):
            db.add_encoding("p001", a)
        os.makedirs(os.path.join(d, "p001"))
        open(os.path.join(d, "p001", "shot.jpg"), "wb").close()
        # Rename p001 -> Per: encodings, db, and crop folder all move.
        moved = db.rename_person("p001", "Per")
        assert moved == 2, moved
        assert "p001" not in db.known_person_ids
        assert "Per" in db.known_person_ids
        assert not os.path.isdir(os.path.join(d, "p001"))
        assert os.path.isfile(os.path.join(d, "Per", "shot.jpg"))
        assert db.recognize(a)[0] == "Per"
        # Merge case: a second id renamed into the existing one.
        db.add_encoding("p002", _unit([1.0, 0.06, 0.0]))
        moved = db.rename_person("p002", "Per")
        assert moved == 1
        assert db.known_person_ids == {"Per"}
        assert db._db["person_ids"].count("Per") == 3
        # No-op rename.
        assert db.rename_person("Per", "Per") == 0
        print("ok: rename_person moves encodings, crops, and merges")


def test_ema_renormalization_math():
    # Mirrors _update_track's cosine branch: an EMA blend of two unit vectors is
    # not unit-norm and must be renormalized for cosine distance to stay valid.
    new = _unit([1.0, 0.0, 0.0])
    old = _unit([0.6, 0.8, 0.0])
    blended = 0.3 * new + 0.7 * old
    assert abs(np.linalg.norm(blended) - 1.0) > 1e-3  # not unit before renorm
    renorm = blended / np.linalg.norm(blended)
    assert abs(np.linalg.norm(renorm) - 1.0) < 1e-9
    print("ok: EMA renormalization keeps unit norm")


if __name__ == "__main__":
    test_distance_helpers()
    test_cosine_mean_of_k()
    test_per_metric_file_and_version()
    test_load_rejects_stale_schema()
    test_rename_person()
    test_ema_renormalization_math()
    print("\nAll recognition logic tests passed.")
