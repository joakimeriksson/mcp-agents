"""Tests for the local voice database (no server needed)."""
import json
import os

import numpy as np
import pytest

from voice_id import VoiceDB, PROVISIONAL_PREFIX


def unit(seed: int, dim: int = 192) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def near(base: np.ndarray, seed: int, noise: float = 0.15) -> np.ndarray:
    v = base + noise * unit(seed)
    return (v / np.linalg.norm(v)).astype(np.float32)


@pytest.fixture
def db(tmp_path):
    return VoiceDB(str(tmp_path / "voices.json"), persist_named=True)


def test_short_clip_is_never_a_decision(db):
    a = unit(1)
    assert db.enroll("p001", a, seconds=0.8) is False        # too short to enroll
    assert db.enroll("p001", a, seconds=3.0) is True
    m = db.verify("p001", unit(2), seconds=0.8)              # a stranger, but too short
    assert m.decision == "no-decision"


def test_verify_same_unsure_different(db):
    a = unit(1)
    db.enroll("p001", a, seconds=3.0)
    assert db.verify("p001", near(a, 5), 3.0).decision == "same"
    assert db.verify("p001", unit(9), 3.0).decision == "different"   # ~0 similarity
    # construct a vector at a chosen similarity to land in the unsure band
    mid = 0.5 * a + np.sqrt(1 - 0.25) * (unit(7) - (unit(7) @ a) * a) / np.linalg.norm(unit(7) - (unit(7) @ a) * a)
    m = db.verify("p001", mid.astype(np.float32), 3.0)
    assert db.reject <= m.similarity < db.confirm and m.decision == "unsure"


def test_same_learns_but_unsure_does_not(db):
    a = unit(1)
    db.enroll("p001", a, 3.0)
    db.verify("p001", near(a, 3), 3.0)
    assert len(db._voices["p001"].samples) == 2
    mid = 0.5 * a + 0.866 * unit(8)
    db.verify("p001", (mid / np.linalg.norm(mid)).astype(np.float32), 3.0)
    assert len(db._voices["p001"].samples) == 2


def test_unknown_person_fails_open(db):
    assert db.verify("nobody", unit(1), 3.0).decision == "no-decision"
    assert db.verify("nobody", None, 3.0).decision == "no-decision"


def test_identify_is_stricter_and_picks_best(db):
    a, b = unit(1), unit(2)
    db.enroll("p001", a, 3.0)
    db.enroll("p002", b, 3.0)
    m = db.identify(near(b, 4, noise=0.1), 3.0)
    assert m is not None and m.person_id == "p002"
    assert db.identify(unit(3), 3.0) is None                  # nobody close enough
    assert db.identify(near(b, 4, noise=0.1), 1.0) is None    # too short


def test_rename_carries_voice_over(db):
    a = unit(1)
    db.enroll("track:7", a, 3.0)
    db.rename("track:7", "p003")
    assert not db.has("track:7") and db.has("p003")
    assert db.verify("p003", near(a, 2), 3.0).decision == "same"


def test_ephemeral_by_default(tmp_path):
    db = VoiceDB(str(tmp_path / "v.json"), persist_named=False)
    db.enroll("p001", unit(1), 3.0)
    db.mark_named("p001")
    assert db.save() == 0 and not os.path.exists(db.path)


def test_persists_named_people_only(db):
    db.enroll("p001", unit(1), 3.0)            # named
    db.enroll("p002", unit(2), 3.0)            # anonymous visitor
    db.enroll("track:3", unit(3), 3.0)         # provisional
    db.mark_named("p001")
    db.mark_named("track:3")                   # must be ignored
    assert db.save() == 1
    people = json.load(open(db.path))["people"]
    assert set(people) == {"p001"}
    fresh = VoiceDB(db.path, persist_named=True)
    assert fresh.people() == ["p001"]


def test_purge_removes_memory_and_file(db):
    db.enroll("p001", unit(1), 3.0)
    db.mark_named("p001")
    db.save()
    assert os.path.exists(db.path)
    db.purge()
    assert db.people() == [] and not os.path.exists(db.path)


def _server_has_speaker():
    from voice_id import SpeakerClient
    return SpeakerClient().available()


@pytest.mark.skipif(not _server_has_speaker(), reason="voice server with --speaker not running")
def test_server_separates_two_synthetic_voices():
    """End to end: two Kokoro voices -> server embeddings -> the store tells them apart."""
    import io, requests, soundfile as sf
    from voice_id import SpeakerClient
    def clip(text, voice):
        wav = requests.post("http://127.0.0.1:8880/v1/audio/speech",
                            json={"input": text, "voice": voice, "language": "sv"}).content
        a, sr = sf.read(io.BytesIO(wav), dtype="float32")
        return np.interp(np.arange(0, len(a), sr / 16000), np.arange(len(a)), a).astype("float32")
    sc = SpeakerClient()
    db = VoiceDB(persist_named=False)
    e1, s1 = sc.embed(clip("Hej, jag skulle gärna vilja ha en Dumle om det går bra.", "Greta"))
    e2, s2 = sc.embed(clip("Kan du flytta den röda biten till mig, tack så mycket.", "Greta"))
    e3, s3 = sc.embed(clip("Vad har ni för godis på bordet idag, finns det choklad?", "Oskar"))
    assert e1 is not None and s1 >= 2.0
    assert db.enroll("p001", e1, s1)
    assert db.verify("p001", e2, s2).decision == "same"        # Greta again
    assert db.verify("p001", e3, s3).decision == "different"   # Oskar
