"""Tests for cutting replies into sentences, whole and token by token."""
import pytest

from sentences import SentenceStream, split_sentences


@pytest.mark.parametrize("text, want", [
    ("Ja, Plopp har vi! Vill du ha en?", ["Ja, Plopp har vi!", "Vill du ha en?"]),
    ("Hallå där! 😊 För en sexåring tar jag Dumle.",
     ["Hallå där!", "För en sexåring tar jag Dumle."]),
    # abbreviations and numbers stay inside the sentence
    ("Vi har t.ex. Plopp och kl. 3 kommer mer.", ["Vi har t.ex. Plopp och kl. 3 kommer mer."]),
    ('Han sa "Hej." Sen gick han.', ['Han sa "Hej."', "Sen gick han."]),
    ("Hmm… Okej, här kommer en Geisha.", ["Hmm…", "Okej, här kommer en Geisha."]),
    ("**Varsågod** 🍬✨", ["Varsågod"]),
    ("🍭", []),
])
def test_split(text, want):
    assert split_sentences(text) == want


def test_stream_matches_split_and_emits_early():
    text = "Ja, Plopp har vi! Geisha är slut just nu. Vill du ha något annat?"
    s, got, first_at = SentenceStream(), [], None
    for i in range(0, len(text), 3):           # ~token-sized pieces
        out = s.feed(text[i:i + 3])
        if out and first_at is None:
            first_at = i
        got += out
    got += s.flush()
    assert got == split_sentences(text)
    assert first_at < len("Ja, Plopp har vi! G")   # not held until the end
