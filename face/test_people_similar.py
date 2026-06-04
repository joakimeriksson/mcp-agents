"""Logic tests for the face-similarity helpers behind the `similar` CLI.

Covers the metric-aware face comparison used by ``people_memory.py similar``
after the InsightFace switch: cosine vs Euclidean distance, the per-metric
verdict thresholds, and the db loader preferring the ArcFace db.

Run:  uv run python test_people_similar.py
"""

import os
import pickle
import tempfile

import numpy as np

from people_memory import (
    _load_face_encodings_by_person,
    _pair_face_distances,
    _similarity_verdict,
)


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def test_pair_distances_metrics():
    a, b, c = _unit([1, 0, 0]), _unit([1, 0, 0]), _unit([0, 1, 0])
    # cosine: identical -> 0, orthogonal -> 1
    assert abs(_pair_face_distances([a], [b], "cosine")[0]) < 1e-9
    assert abs(_pair_face_distances([a], [c], "cosine")[0] - 1.0) < 1e-9
    # euclidean matches numpy
    assert abs(_pair_face_distances([a], [c], "euclidean")[0] - np.sqrt(2)) < 1e-9
    # min is taken across the cross-product of both sets
    mn, _ = _pair_face_distances([a, c], [b], "cosine")
    assert abs(mn) < 1e-9
    # cosine works even if a stored vector is not unit-norm (defensive normalize)
    assert abs(_pair_face_distances([3.0 * a], [b], "cosine")[0]) < 1e-9
    # empty sets -> inf
    assert _pair_face_distances([], [b], "cosine") == (float("inf"), float("inf"))
    print("ok: pair distances cosine/euclidean")


def test_verdict_thresholds():
    # cosine cutoffs: same < 0.30, maybe < 0.45
    assert _similarity_verdict(0.20, 0.0, 0.0, "cosine") == "LIKELY SAME"
    assert _similarity_verdict(0.40, 0.0, 0.0, "cosine") == "maybe"
    assert _similarity_verdict(0.40, 0.9, 0.0, "cosine") == "LIKELY SAME"
    assert _similarity_verdict(0.60, 0.0, 0.0, "cosine") is None
    # euclidean cutoffs: same < 0.40, maybe < 0.55 (legacy dlib scale)
    assert _similarity_verdict(0.35, 0.0, 0.0, "euclidean") == "LIKELY SAME"
    assert _similarity_verdict(0.50, 0.0, 0.0, "euclidean") == "maybe"
    assert _similarity_verdict(0.60, 0.0, 0.0, "euclidean") is None
    # a 0.50 cosine pair is below the dlib "maybe" line but not the cosine one
    assert _similarity_verdict(0.50, 0.0, 0.0, "cosine") is None
    print("ok: verdict thresholds per metric")


def test_loader_prefers_arcface():
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "faces.pkl"), "wb") as f:
            pickle.dump({"person_ids": ["p1"], "encodings": [np.zeros(128)]}, f)
        encs, metric = _load_face_encodings_by_person(d)
        assert metric == "euclidean" and "p1" in encs

        # With both present the ArcFace db wins.
        with open(os.path.join(d, "faces_arcface.pkl"), "wb") as f:
            pickle.dump({"person_ids": ["pA", "pA"],
                         "encodings": [np.zeros(512), np.zeros(512)]}, f)
        encs, metric = _load_face_encodings_by_person(d)
        assert metric == "cosine" and len(encs["pA"]) == 2 and "p1" not in encs

    with tempfile.TemporaryDirectory() as empty:
        assert _load_face_encodings_by_person(empty) == ({}, "cosine")
    print("ok: loader prefers arcface, reports metric")


if __name__ == "__main__":
    test_pair_distances_metrics()
    test_verdict_thresholds()
    test_loader_prefers_arcface()
    print("\nAll people_memory similar-CLI tests passed.")
