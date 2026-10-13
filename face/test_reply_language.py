"""Tests for picking the TTS voice language from the reply text.

Cases are real CandyTron replies from the robot test on 2026-10-06, each with
the (lagging) conversation language it was spoken under.
"""
import pytest

from reply_language import reply_language

LANGS = {"sv", "en", "fr", "es", "it", "de"}


@pytest.mark.parametrize("text, hint, want", [
    # Wrong voice that night: the reply's own language must win.
    ("Jag kan prata svenska nu. Vad vill du att jag ska göra?", "en", "sv"),
    ("Bonjour! Je peux parler plusieurs langues pour vous.", "sv", "fr"),
    ("Bien sûr, le système que je représente intègre la reconnaissance vocale.", "it", "fr"),
    ("Certo, posso descrivere tutto il sistema in italiano.", "fr", "it"),
    ("Can you move the Geisha to D1?", "it", "en"),
    ("¿Quieres que te dé el dulce de Geisha?", "sv", "es"),
    # Already right: must stay right.
    ("Jag är Candytron 4000 och jag kan prata flera språk.", "sv", "sv"),
    ("Varsågod, här kommer en Dumle till dig.", "sv", "sv"),
    ("Parlo italiano, sì. Posso anche parlare francese e inglese.", "it", "it"),
])
def test_voice_follows_the_reply(text, hint, want):
    assert reply_language(text, hint, LANGS)[0] == want


@pytest.mark.parametrize("text, hint", [
    ("Ja!", "sv"), ("OK", "fr"), ("Sure!", "en"), ("Absolut.", "sv"),
])
def test_fragments_keep_the_conversation_language(text, hint):
    assert reply_language(text, hint, LANGS)[0] == hint


def test_no_text_or_too_few_languages_keeps_hint():
    assert reply_language("", "sv", LANGS) == ("sv", 0.0)
    assert reply_language("Hej på dig, hur mår du idag?", "en", {"en"}) == ("en", 0.0)
