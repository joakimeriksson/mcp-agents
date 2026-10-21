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
    llm._call_tool = lambda name, args: llm.moves.append(args) or "ok"
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
