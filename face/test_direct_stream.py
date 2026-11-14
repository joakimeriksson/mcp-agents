"""Streaming direct-audio turns: sentences go out as written, action claims
only once the arm has really moved. gemma4 is replaced by scripted replies."""
import numpy as np

import os
import sys

import direct_llm
from direct_llm import DirectAudioLLM
from service_policy import GENERIC

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from candytron_mcp.client_policy import CandyPolicy  # noqa: E402

MOVE = {"function": {"name": "move_between", "arguments": {"src": "A1", "dst": "O0"}}}


class FakeOllama:
    """Plays back one scripted (content, tool_calls) per chat() call."""

    def __init__(self, rounds):
        self.rounds = list(rounds)

    def chat(self, *, stream, **_):
        content, calls = self.rounds.pop(0)
        if not stream:
            return {"message": {"role": "assistant", "content": content,
                                "tool_calls": calls}}
        pieces = [{"message": {"content": content[i:i + 3]}}
                  for i in range(0, len(content), 3)]
        if calls:
            pieces.append({"message": {"content": "", "tool_calls": calls}})
        return iter(pieces)


def make(rounds, tools=True, policy=None):
    """CandyTron by default; generic-service tests pass policy=GENERIC."""
    llm = DirectAudioLLM.__new__(DirectAudioLLM)
    llm.policy = policy or CandyPolicy()
    direct_llm.set_known_tools(["move_between", "show_demo_move", "default_action", "express"])
    llm._client = FakeOllama(rounds)
    llm._model, llm._options, llm._agent_name = "gemma4", {}, "CandyTron"
    llm._service_prompt = llm._augmentation_provider = None
    llm._sample_rate = 16000
    llm._tools = [{"type": "function", "function": {"name": "move_between"}}] if tools else []
    llm.moves = []
    # Answer like candytron_mcp: only "Started moving ..." means the arm moved.
    llm.arm_result = "Started moving the candy from {src} to {dst}; the arm is moving now."
    llm._call_tool = lambda name, args: llm.moves.append(args) or llm.arm_result.format(**args)
    return llm


def run(llm):
    heard = []
    reply, lang = llm.respond(np.zeros(1600, np.float32), language_hint="sv",
                              on_sentence=lambda s, l: heard.append((s, l, len(llm.moves))))
    return reply, lang, heard


def test_plain_reply_streams_sentence_by_sentence():
    reply, lang, heard = run(make([("[sv] Ja, Plopp har vi! Vill du ha en?", None)], tools=False))
    assert [s for s, _, _ in heard] == ["Ja, Plopp har vi!", "Vill du ha en?"]
    assert {l for _, l, _ in heard} == {"sv"}
    assert (reply, lang) == ("Ja, Plopp har vi! Vill du ha en?", "sv")


def test_claim_waits_for_the_tool():
    reply, _, heard = run(make([
        ("[sv] Okej! Här får du en Plopp.", [MOVE]),
        ("[sv] Varsågod, en Plopp till dig!", None),
    ]))
    assert heard == [("Okej!", "sv", 0), ("Varsågod, en Plopp till dig!", "sv", 1)]
    assert reply == "Okej! Varsågod, en Plopp till dig!"


def test_claim_without_tool_is_said_only_after_the_retry_moved_the_arm():
    reply, _, heard = run(make([
        ("[sv] Här får du en Plopp.", None),   # lie: no tool call
        ("[sv]", [MOVE]),                       # retry: the move, no words
        ("[sv]", None),
    ]))
    assert heard == [("Här får du en Plopp.", "sv", 1)]
    assert reply == "Här får du en Plopp."


MARIANNE = {"function": {"name": "move_between", "arguments": {"src": "A3", "dst": "O0"}}}
SCENE = "De senast observerade positionerna:\nA1 : Dumle, kola.\nA3 : Marianne, mint."


def run_scene(llm, scene=None):
    heard = []
    reply, lang = llm.respond(np.zeros(32000, np.float32), language_hint="sv",
                              augmentation=scene or SCENE,
                              on_sentence=lambda s, l: heard.append((s, l, len(llm.moves))))
    return reply, lang, heard


def test_har_ar_din_without_tool_is_retried():
    # The demo failure: "Här är din Marianne" and nothing happened.
    llm = make([("[sv] Här är din Marianne.", None),
                ("[sv]", [MARIANNE]),
                ("[sv]", None)])
    reply, _, heard = run_scene(llm)
    assert llm.moves == [{"src": "A3", "dst": "O0"}]
    assert heard == [("Här är din Marianne.", "sv", 1)]


def test_refused_move_does_not_count_as_done():
    # Arm busy -> the server refuses; the claim must not pass as true.
    llm = make([("[sv] Här är din Marianne.", [MARIANNE]),
                ("[sv] Här är din Marianne.", None),
                ("[sv]", [MARIANNE]),
                ("[sv]", None)])
    results = iter(["Refused: the arm is still finishing the previous move.",
                    "Queued: moving the candy from A3 to O0 after 1 earlier move."])
    llm._call_tool = lambda name, args: llm.moves.append(args) or next(results)
    reply, _, heard = run_scene(llm)
    assert len(llm.moves) == 2          # the retry made the move that counts
    assert heard and heard[0][2] == 2   # the claim was held until then


def test_two_candies_claimed_one_moved_retries_for_the_other():
    DUMLE = {"function": {"name": "move_between", "arguments": {"src": "A1", "dst": "O0"}}}
    llm = make([("[sv] Här är din Dumle och din Marianne.", [DUMLE]),
                ("[sv] Här är din Dumle och din Marianne.", None),
                ("[sv]", [MARIANNE]),
                ("[sv]", None)])
    reply, _, heard = run_scene(llm)
    assert llm.moves == [{"src": "A1", "dst": "O0"}, {"src": "A3", "dst": "O0"}]


def test_short_utterance_pins_the_language():
    llm = make([("[sv] Vill du ha godis?", None)], tools=False)
    captured = {}
    orig = llm._client.chat
    def spy(**kw):
        captured["system"] = kw["messages"][0]["content"]
        return orig(**kw)
    llm._client.chat = spy
    llm.respond(np.zeros(8000, np.float32), language_hint="sv", on_sentence=lambda s, l: None)
    assert "very short" in captured["system"] and "Swedish" in captured["system"]


def test_greet_streams_without_tools():
    llm = make([("[sv] Hej! Vilket godis vill du ha?", None)])
    seen = {}
    orig = llm._client.chat
    def spy(**kw):
        seen["tools"] = kw.get("tools")
        return orig(**kw)
    llm._client.chat = spy
    heard = []
    reply, lang = llm.greet("A new person is here. Greet briefly.", language="sv",
                            on_sentence=lambda s, l: heard.append(s))
    assert seen["tools"] is None
    assert heard == ["Hej!", "Vilket godis vill du ha?"] and lang == "sv"


def test_tool_call_written_as_text_is_executed_and_not_spoken():
    # Seen at the demo: the call arrived as text and was read out loud.
    raw = ('[toolcall] movebetween{dst:<|"|>O0<|"|>,src:<|"|>A1<|"|>} '
           '[toolcall] movebetween{dst:<|"|>O0<|"|>,src:<|"|>A3<|"|>}')
    llm = make([(raw, None), ("[sv] Här är din Dumle.", None)])
    reply, _, heard = run_scene(llm)
    assert llm.moves == [{"dst": "O0", "src": "A1"}, {"dst": "O0", "src": "A3"}]
    assert all("toolcall" not in s and "<|" not in s for s, _, _ in heard)


def test_demo_turn_text_calls_then_the_real_answer():
    # The exact reply from the robot at 21:11: four calls as text, then prose.
    raw = ('[toolcall] movebetween{dst:<|"|>O0<|"|>,src:<|"|>A1<|"|>} '
           '[toolcall] movebetween{dst:<|"|>O0<|"|>,src:<|"|>A3<|"|>} '
           '[toolcall] movebetween{dst:<|"|>O0<|"|>,src:<|"|>D1<|"|>} '
           '[toolcall] movebetween{dst:<|"|>O0<|"|>,src:<|"|>D3<|"|>} [en] Here is a Dumle for you.')
    scene = "Positions:\nA1 : Dumle, red.\nA3 : Dumle, red.\nD1 : Dumle, red.\nD3 : Dumle, red."
    llm = make([(raw, None), ("[en] Here is a Dumle for you.", None)])
    heard = []
    reply, lang = llm.respond(np.zeros(32000, np.float32), language_hint="sv", augmentation=scene,
                              on_sentence=lambda s, l: heard.append((s, l)))
    assert llm.moves == [{"dst": "O0", "src": "A1"}]          # ONE Dumle, not four
    assert heard == [("Here is a Dumle for you.", "en")]       # no tool text spoken


NO_MARIANNE = "Positions:\nA1 : Dumle, red.\nC2 : Plopp, red rectangular."


def test_absent_candy_claim_is_not_retried_and_not_said():
    # Demo: "Here is your Marianne" with no Marianne on the table.
    llm = make([("[en] Here is your Marianne.", None),
                ("[en] Sorry, there is no Marianne on the table, but I have Dumle and Plopp.", None)])
    reply, lang, heard = run_scene(llm, NO_MARIANNE)
    assert llm.moves == []
    assert [s for s, _, _ in heard] == ["Sorry, there is no Marianne on the table, but I have Dumle and Plopp."]


def test_honest_reply_falls_back_when_the_model_claims_again():
    llm = make([("[sv] Här är din Marianne.", None), ("[sv] Här är din Marianne!", None)])
    reply, lang, heard = run_scene(llm, NO_MARIANNE)
    assert llm.moves == [] and "Här är" not in reply and reply.startswith("Förlåt")


def test_retry_may_not_hand_over_a_different_candy():
    # Demo: the retry for a promised candy moved a Plopp instead.
    PLOPP = {"function": {"name": "move_between", "arguments": {"src": "C2", "dst": "O0"}}}
    llm = make([("[en] Here is your Dumle.", None),     # claim, no tool
                ("[en]", [PLOPP]),                      # retry grabs the wrong candy
                ("[en]", None),                         # retry round 2: nothing
                ("[en] Sorry, that did not work. Which candy would you like?", None)])
    reply, lang, heard = run_scene(llm, NO_MARIANNE)
    assert llm.moves == []                              # the Plopp move was refused
    assert all("Here is your Dumle" not in s for s, _, _ in heard)


def test_bare_text_tool_call_is_executed_and_not_spoken():
    # Demo: "movebetween{dst: O0 ,src: C1 } [sv] Här är din Snickers." was read out.
    scene = "Positions:\nC1 : Snickers, brown.\nA1 : Dumle, red."
    llm = make([('movebetween{dst:<|"|>O0<|"|>,src:<|"|>C1<|"|>} [sv] Här är din Snickers.', None),
                ("[sv] Här är din Snickers.", None)])
    reply, lang, heard = run_scene(llm, scene)
    assert llm.moves == [{"dst": "O0", "src": "C1"}]
    assert all("{" not in s and "movebetween" not in s for s, _, _ in heard)


def test_written_demo_move_is_dropped_not_executed_or_spoken():
    llm = make([("showdemomove{}", None), ("[sv] Vill du se vad jag kan?", None)])
    reply, lang, heard = run_scene(llm)
    assert llm.moves == []
    assert all("showdemomove" not in s for s, _, _ in heard)


LIGHT_TOOLS = [{"type": "function", "function": {"name": "set_light",
                "description": "Turn a lamp on or off, or dim it."}}]


def test_generic_service_claim_without_tool_is_retried():
    # Lighting, not candy: "I turned on the kitchen light" with no tool call.
    ON = {"function": {"name": "set_light", "arguments": {"room": "kitchen", "on": True}}}
    llm = make([("[en] I've turned on the kitchen light.", None),
                ("YES", None),                                   # the judge
                ("[en]", [ON]),                                  # retry calls the tool
                ("[en] The kitchen light is on.", None)], policy=GENERIC)
    llm._tools = LIGHT_TOOLS
    direct_llm.set_known_tools(["set_light", "express"])
    llm._call_tool = lambda name, args: llm.moves.append(args) or "OK: kitchen light on"
    reply, lang, heard = run(llm)
    assert llm.moves == [{"room": "kitchen", "on": True}]


def test_generic_service_honest_when_retry_does_nothing():
    llm = make([("[sv] Nu har jag släckt lampan.", None),
                ("YES", None),                                   # judge
                ("[sv] Nu har jag släckt lampan.", None),        # retry: still no tool
                ("[sv] Förlåt, jag har inte släckt lampan.", None)], policy=GENERIC)
    llm._tools = LIGHT_TOOLS
    direct_llm.set_known_tools(["set_light", "express"])
    reply, lang, heard = run(llm)
    assert llm.moves == []
    # An unknown phrasing streams before the judge rules; the honest
    # correction must follow it, so the user is never left believing it.
    assert [s for s, _, _ in heard][-1] == "Förlåt, jag har inte släckt lampan."


def test_generic_written_tool_call_is_recovered():
    llm = make([('set_light{room:<|"|>hall<|"|>,on:<|"|>false<|"|>} [en] Done.', None),
                ("[en] The hall light is off.", None)], policy=GENERIC)
    llm._tools = LIGHT_TOOLS
    direct_llm.set_known_tools(["set_light", "express"])
    llm._call_tool = lambda name, args: llm.moves.append(args) or "OK"
    reply, lang, heard = run(llm)
    assert llm.moves == [{"room": "hall", "on": "false"}]
    assert all("set_light" not in s and "{" not in s for s, _, _ in heard)


# --- Whisper ear: the judge sees the request; transcripts snap to candy names ---

PLOPP_SCENE = "Positions:\nC1 : Plopp, chocolate.\nD2 : Dumle, toffee."


class FakeEar:
    """Stands in for lang_id.LanguageID with a fixed transcript."""
    available = True

    def __init__(self, text, lang="en"):
        from lang_id import Heard
        self.heard = Heard(text, lang, "ok")

    def transcribe(self, audio, hint_words=""):
        self.hint_words = hint_words
        return self.heard

    def detect(self, audio):
        return self.heard.language, 0.99


def whisper(llm, text, lang="en"):
    llm.ear, llm.lang_id = "whisper", FakeEar(text, lang)
    return llm


def run_heard(llm, lang="en"):
    heard = []
    reply, rlang = llm.respond(np.zeros(32000, np.float32), language_hint=lang,
                               augmentation=PLOPP_SCENE,
                               on_sentence=lambda s, l: heard.append((s, l, len(llm.moves))))
    return reply, rlang, heard


def test_agreement_without_tool_is_retried_with_the_request():
    # Seen in simulation: "Can I have a plop?" -> "Of course, you can have a
    # Plopp!" and nothing moved; the reply-only judge said NO.
    PLOPP = {"function": {"name": "move_between", "arguments": {"src": "C1", "dst": "O0"}}}
    llm = whisper(make([("[en] Of course, you can have a Plopp!", None),
                        ("YES", None),                          # judge (with request)
                        ("[en]", [PLOPP]),                      # retry moves it
                        ("[en] There you go, one Plopp!", None)]), "Can I have a plop please?")
    prompts_seen = []
    chat = llm._client.chat
    llm._client.chat = lambda **kw: prompts_seen.append(kw["messages"]) or chat(**kw)
    reply, lang, heard = run_heard(llm)
    assert llm.moves == [{"src": "C1", "dst": "O0"}]
    judge = prompts_seen[1][0]["content"]
    assert "The person said" in judge and "Plopp" in judge     # request, name snapped
    assert "plop please" not in judge


def test_question_answered_is_not_a_claim():
    llm = whisper(make([("[en] Yes, we have Plopp! Would you like one?", None),
                        ("NO", None)]), "Do you have Plopp?")
    reply, lang, heard = run_heard(llm)
    assert llm.moves == [] and reply == "Yes, we have Plopp! Would you like one?"


def test_judge_yes_naming_nothing_on_the_table_keeps_the_reply():
    # "What can you do?" -> "I can hand you candy!" judged YES must not turn
    # into "Sorry, that didn't work".
    llm = whisper(make([("[en] I hand out candy with my robot arm!", None),
                        ("YES", None)]), "What can you do?")
    reply, lang, heard = run_heard(llm)
    assert llm.moves == [] and reply == "I hand out candy with my robot arm!"


def test_transcript_snaps_to_candy_names():
    p = CandyPolicy()
    scene = {"C1": "Plopp, chocolate", "B2": "Geisha, nougat",
             "A2": "Riesen toffee (Riesen-kola), toffee"}
    assert p.normalize_heard("Can I have a plop please?", scene) == "Can I have a Plopp please?"
    assert p.normalize_heard("Kann ich ein Geischa haben?", scene) == "Kann ich ein Geisha haben?"
    for keep in ("Kan jag få en dumma?", "Can I have a toffee", "Kan jag få en kola"):
        assert p.normalize_heard(keep, scene) == keep
    assert GENERIC.normalize_heard("plop", scene) == "plop"
