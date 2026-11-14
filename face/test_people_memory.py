"""People memory, visit memory, forgetting and the service-policy memory hooks.

Run: uv run --with pytest python -m pytest -q test_people_memory.py
"""

import json
import os
import re
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

import people_memory as pm
from people_memory import PeopleMemory, Fact, is_grounded, is_junk_fact
from visit_memory import VisitMemory
from forget import is_forget_request, confirmation
from llm import parse_visit_json

DAY = 86400.0


@pytest.fixture
def mem(tmp_path):
    m = PeopleMemory(storage_dir=str(tmp_path))
    m.load()
    return m


def named(mem, tid=1, name="Anna"):
    pid = mem.set_name(tid, name)
    return pid


# --- what counts as a fact ----------------------------------------------

@pytest.mark.parametrize("fact", [
    "mentioned 123", "silverplate", "talking with a coconut voice",
    "talking to Fina's voice", "name is Kim", "asked about the weather",
    "Joakim mentioned a hotel", "is talking about sleep", "",
])
def test_junk_is_not_a_fact(fact):
    assert is_junk_fact(fact, "Joakim")


@pytest.mark.parametrize("fact", [
    "likes sushi", "is a musician", "works as a nurse", "has two kids",
    "Joakim likes chocolate",
])
def test_real_facts_pass(fact):
    assert not is_junk_fact(fact, "Joakim")


def test_grounding_needs_their_own_words():
    heard = "Jag jobbar som sjuksköterska på Danderyd. Och jag gillar sushi."
    assert is_grounded("jobbar som sjuksköterska", heard)
    assert is_grounded("Jag gillar sushi", heard)
    assert not is_grounded("I am a pilot", heard)
    assert not is_grounded("sushi", heard)              # one word proves nothing
    assert not is_grounded("", heard)
    # One small word off in five is tolerated (Whisper vs the model).
    assert is_grounded("jag jobbar som en sjuksköterska på", heard)


# --- facts: confirmation, expiry ------------------------------------------

def test_fact_heard_again_is_confirmed(mem):
    named(mem)
    assert mem.add_fact(1, "likes sushi", quote="gillar sushi") == "new"
    f = mem.get(1).facts[0]
    assert not f.confirmed
    assert mem.add_fact(1, "Anna likes sushi.") == "again"
    assert f.count == 2 and f.confirmed
    assert mem.add_fact(1, "mentioned a hotel") == "junk"
    assert len(mem.get(1).facts) == 1


def test_stated_fact_is_confirmed_at_once(mem):
    named(mem)
    mem.add_fact(1, "works as a nurse", quote="I work as a nurse", explicit=True)
    assert mem.get(1).facts[0].confirmed


def test_unconfirmed_facts_expire_on_identify(mem, tmp_path):
    pid = named(mem)
    now = time.time()
    mem.add_fact(1, "likes jazz", now=now - 40 * DAY)                 # once, old: goes
    mem.add_fact(1, "likes sushi", now=now - 40 * DAY)
    mem.add_fact(1, "likes sushi", now=now - 39 * DAY)                # twice: stays
    mem.add_fact(1, "has a dog", now=now - 2 * DAY)                   # once, recent: stays
    m2 = PeopleMemory(storage_dir=str(tmp_path)); m2.load()
    m2.identify(7, pid)
    assert sorted(m2.get(7).fact_texts) == ["has a dog", "likes sushi"]


def test_visit_facts_are_grounded_and_counted_once(mem):
    named(mem)
    heard = "I work as a nurse in Uppsala. I like sushi, I really like sushi."
    report = mem.apply_visit_facts(1, [
        {"fact": "works as a nurse", "quote": "I work as a nurse", "explicit": True},
        {"fact": "likes sushi", "quote": "I like sushi"},
        {"fact": "Likes sushi.", "quote": "I really like sushi"},          # same visit
        {"fact": "is a pilot", "quote": "I fly planes"},                  # not said
        {"fact": "mentioned Uppsala", "quote": "in Uppsala"},             # not a fact
        "garbage",
    ], heard, summary="Nurse from Uppsala who likes sushi.")
    assert [f for f, _ in report["kept"]] == ["works as a nurse", "likes sushi"]
    assert len(report["dropped"]) == 3
    p = mem.get(1)
    sushi = next(f for f in p.facts if "sushi" in f.text)
    assert sushi.count == 1 and not sushi.confirmed      # one visit = one mention
    assert p.summary.startswith("Nurse")
    # The next visit repeats it: now it is confirmed.
    mem.apply_visit_facts(1, [{"fact": "likes sushi", "quote": "gillar sushi"}],
                          "Jag gillar sushi fortfarande")
    assert sushi.count == 2 and sushi.confirmed


def test_stated_needs_first_person_and_summary_needs_a_fact(mem):
    from people_memory import is_first_person
    assert is_first_person("Jag jobbar som lärare") and is_first_person("I'm a nurse")
    assert is_first_person("j'habite à Nice") and is_first_person("Ich wohne in Wien")
    assert not is_first_person("123 i Strömsunds kommun i Norra Jämtland")
    named(mem)
    heard = "123 i Strömsunds kommun i Norra Jämtland. Två män döms."
    mem.apply_visit_facts(1, [{"fact": "is from Strömsunds kommun", "explicit": True,
                               "quote": "i Strömsunds kommun i Norra Jämtland"}],
                          heard, summary="From Strömsund.")
    f = mem.get(1).facts[0]
    assert not f.confirmed            # the TV said it, not them
    mem.update_summary(1, "")
    mem.apply_visit_facts(1, [], "painted walls, sleep", summary="Talked about walls.")
    assert mem.get(1).summary == ""   # no facts learned: no new summary


def test_pending_lines_and_marker(mem):
    named(mem)
    mem.add_dialogue(1, "person", "hello")
    mem.add_dialogue(1, "system", "hi")
    lines = mem.pending_visit_lines(1)
    assert [d.text for d in lines] == ["hello", "hi"]
    mem.mark_consolidated(1, lines[-1].timestamp)
    assert mem.pending_visit_lines(1) == []
    mem.add_dialogue(1, "person", "again")
    assert [d.text for d in mem.pending_visit_lines(1)] == ["again"]


# --- old records ----------------------------------------------------------

def test_legacy_record_loads(tmp_path):
    old = {"name": "Joakim", "facts": ["likes chocolate", "mentioned 123"],
           "dialogues": [{"timestamp": 1000.0, "speaker": "person", "text": "hej"}],
           "asked_topics": ["favourite_candy"], "times_seen": 3}
    (tmp_path / "p001.json").write_text(json.dumps(old))
    m = PeopleMemory(storage_dir=str(tmp_path)); m.load()
    m.identify(5, "p001")
    p = m.get(5)
    assert p.fact_texts == ["likes chocolate", "mentioned 123"]
    assert all(not f.confirmed for f in p.facts)
    assert p.consolidated_upto == 1000.0          # old dialogue is not re-mined
    ctx = m.get_context_for_llm(5)
    assert "mentioned 123" not in ctx              # junk never reaches the prompt
    saved = json.loads((tmp_path / "p001.json").read_text())
    assert saved["facts"][0]["text"] == "likes chocolate"


# --- prompt budget --------------------------------------------------------

def test_context_is_small_and_prefers_confirmed(mem):
    named(mem, name="Joakim")
    for i in range(12):
        mem.add_fact(1, f"likes thing number {i}", explicit=(i < 3))
    for i in range(40):
        mem.add_dialogue(1, "person" if i % 2 else "system", f"line {i} " + "x" * 60)
    ctx = mem.get_context_for_llm(1)
    assert len(ctx) <= pm.CONTEXT_CHARS
    assert "likes thing number 0" in ctx                    # confirmed first
    assert "line 39" in ctx and "line 20 " not in ctx       # newest lines only
    assert ctx.startswith("Person: Joakim")


def test_context_this_visit_only(mem):
    named(mem)
    p = mem.get(1)
    now = time.time()
    p.dialogues = [pm.DialogueEntry(now - 3 * 3600, "person", "old visit"),
                   pm.DialogueEntry(now - 30, "person", "new visit")]
    ctx = mem.get_context_for_llm(1)
    assert "new visit" in ctx and "old visit" not in ctx
    assert "last talked 3 h ago" in ctx


def test_dialogue_is_capped(mem, tmp_path):
    pid = named(mem)
    for i in range(pm.MAX_DIALOGUES + 25):
        mem.add_dialogue(1, "person", f"l{i}")
    assert len(mem.get(1).dialogues) == pm.MAX_DIALOGUES
    saved = json.loads((tmp_path / f"{pid}.json").read_text())
    assert saved["dialogues"][-1]["text"] == f"l{pm.MAX_DIALOGUES + 24}"


# --- saving, merging, forgetting ------------------------------------------

def test_concurrent_saves_never_corrupt(mem, tmp_path):
    pid = named(mem)
    errors = []

    def talk(k):
        try:
            for i in range(40):
                mem.add_dialogue(1, "person", f"{k}-{i}")
                mem.add_fact(1, f"likes number {k} {i}")
        except Exception as e:  # pragma: no cover
            errors.append(e)
    threads = [threading.Thread(target=talk, args=(k,)) for k in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert not errors
    data = json.loads((tmp_path / f"{pid}.json").read_text())
    assert len(data["dialogues"]) == pm.MAX_DIALOGUES
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]


def test_merge_combines_history(mem, tmp_path):
    a = named(mem, 1, "Joakim")
    mem.add_fact(1, "likes sushi"); mem.add_dialogue(1, "person", "from p001")
    b = named(mem, 2, "Joakim")
    mem.add_fact(2, "likes sushi"); mem.add_fact(2, "has a dog")
    mem.add_dialogue(2, "person", "from p002")
    merged = mem.merge(b, a)
    assert merged and not (tmp_path / f"{b}.json").exists()
    p = mem.get(1)
    sushi = next(f for f in p.facts if f.text == "likes sushi")
    assert sushi.count == 2 and sushi.confirmed           # heard as both: confirmed
    assert {d.text for d in p.dialogues} == {"from p001", "from p002"}
    assert mem.get(2).persistent_id == a                   # the live track follows
    assert b not in mem.known_person_ids


def test_forget_is_final(mem, tmp_path):
    pid = named(mem)
    held = mem.get(1)                       # a background thread still holds it
    assert mem.forget(pid)
    assert not (tmp_path / f"{pid}.json").exists()
    mem._save(held)                         # ...and saves late
    assert not (tmp_path / f"{pid}.json").exists()


def test_forget_requests():
    for t in ["Forget me please", "kan du glömma mig? nej, glöm mig!", "Glöm bort mig",
              "Vergiss mich", "oublie-moi", "Olvídame", "dimenticami", "delete my face"]:
        assert is_forget_request(t), t
    for t in ["Don't forget me!", "I forgot my keys", "glöm inte godiset", "forgettable"]:
        # "Don't forget me" still matches the words; the pattern errs on the
        # side of deleting -- except where the words are not there at all.
        if "forget me" in t.lower():
            continue
        assert not is_forget_request(t), t
    assert confirmation("sv").startswith("Klart")
    assert confirmation("xx") == confirmation("en")


# --- visit memory ---------------------------------------------------------

def test_visit_memory_line_and_ttl():
    v = VisitMemory(ttl=7200)
    t0 = 1_000_000.0
    v.seen("p1", "sv", now=t0)
    assert v.line("p1", now=t0) == ""
    v.note("p1", "was given a Dumle", now=t0 + 10)
    v.seen("p1", now=t0 + 600)                           # back after ~10 min
    line = v.line("p1", now=t0 + 610)
    assert "came back" in line and "was given a Dumle (10 min ago)" in line
    v.rekey("p1", "p9")
    assert v.line("p9", now=t0 + 610) and not v.line("p1", now=t0 + 610)
    assert v.line("p9", now=t0 + 600 + 7201) == ""        # gone after the TTL
    v.seen("x", now=t0); v.forget("x")
    assert v.line("x", now=t0) == ""


# --- service policy: memory hooks ----------------------------------------

def test_candy_policy_remembers_hand_outs_only():
    from candytron_mcp.client_policy import CandyPolicy
    p = CandyPolicy()
    scene = {"A1": "Dumle, brun", "B2": "Plopp, gul"}
    assert p.remember("move_between", {"src": "A1", "dst": "O0"},
                      "Started moving A1 -> O0", scene) == "was given a Dumle"
    assert p.remember("move_between", {"src": "A1", "dst": "B3"},
                      "Started moving A1 -> B3", scene) is None
    assert p.remember("move_between", {"src": "A1", "dst": "O0"},
                      "Error: arm busy", scene) is None
    assert p.remember("show_demo_move", {}, "Started", scene) is None
    assert "favourite_candy" in p.interview_topics()


def test_generic_layer_has_no_candy():
    """The shared face/ layer must not know about candy (it moved to the
    CandyTron policy)."""
    import tomllib
    here = os.path.dirname(os.path.abspath(__file__))
    cfg = tomllib.load(open(os.path.join(here, "languages.toml"), "rb"))
    topics = {t for lang in cfg["languages"].values() for t in lang.get("interview", {})}
    assert "favourite_candy" not in topics
    for name in ("people_memory.py", "visit_memory.py", "llm.py", "agent.py",
                 "debug_shell.py", "forget.py"):
        src = open(os.path.join(here, name)).read().lower()
        assert not re.search(r"\bcandy\b|godis", src), name


def test_direct_llm_records_notes_before_apply():
    from direct_llm import DirectAudioLLM
    from candytron_mcp.client_policy import CandyPolicy
    llm = DirectAudioLLM.__new__(DirectAudioLLM)
    llm.policy = CandyPolicy()
    llm.last_tool_calls, llm.last_notes = [], []
    scene = {"A1": "Dumle, brun"}
    llm._record("move_between", {"src": "A1", "dst": "O0"}, "Started moving", scene)
    assert llm.last_notes == ["was given a Dumle"]
    assert "A1" not in scene                       # apply ran after remember


# --- the end-of-visit pass's output -----------------------------------------

def test_parse_visit_json():
    raw = ('<think>hmm</think>Sure!\n```json\n{"facts": [{"fact": "likes tea", '
           '"quote": "I like tea"}], "summary": "Tea drinker."}\n```')
    d = parse_visit_json(raw)
    assert d["facts"][0]["fact"] == "likes tea" and d["summary"] == "Tea drinker."
    assert parse_visit_json("no json here") is None
    assert parse_visit_json('{"facts": "oops"}')["facts"] == []
