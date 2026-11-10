"""Streaming direct-audio turns: sentences go out as written, action claims
only once the arm has really moved. gemma4 is replaced by scripted replies."""
import numpy as np

from direct_llm import DirectAudioLLM

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


def make(rounds, tools=True):
    llm = DirectAudioLLM.__new__(DirectAudioLLM)
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


def run_scene(llm):
    heard = []
    reply, lang = llm.respond(np.zeros(32000, np.float32), language_hint="sv",
                              augmentation=SCENE,
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
