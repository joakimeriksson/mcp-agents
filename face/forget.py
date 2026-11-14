"""'Forget me': a person asks the robot to delete what it knows about them.

Matched on the transcript with a plain pattern, not by the model: deleting
someone must not depend on gemma4 deciding to call a tool, and must never
happen because it misread an ordinary sentence. The caller deletes the
memory record, the face encodings and the face crops (PeopleMemory.forget,
FaceTracker.forget_person) and the short-term visit memory, then says one
of the confirmations below.
"""

import re

_FORGET_RE = re.compile(
    r"\b("
    r"forget (about )?me|delete (me|my (data|face|picture|photo))|"
    r"glöm (bort )?mig|radera (mig|min (bild|data))|"
    r"vergiss mich|lösche? (mich|meine daten|mein (bild|foto))|"
    r"oublie[ -]moi|oubliez[ -]moi|efface (moi|mes données)|"
    r"olv[ií]da(me|te de m[ií])|borra (mis datos|mi (cara|foto))|"
    r"dimentica(mi|ti di me)|cancella (i miei dati|la mia foto)"
    r")\b",
    re.IGNORECASE)

_CONFIRM = {
    "en": "Done, I have forgotten you.",
    "sv": "Klart, nu har jag glömt bort dig.",
    "de": "Erledigt, ich habe dich vergessen.",
    "fr": "C'est fait, je t'ai oublié.",
    "es": "Hecho, ya te he olvidado.",
    "it": "Fatto, ti ho dimenticato.",
}


def is_forget_request(text: str) -> bool:
    """True if ``text`` asks to be forgotten ("forget me", "glöm mig", ...)."""
    return bool(text) and bool(_FORGET_RE.search(text))


def confirmation(lang: str) -> str:
    return _CONFIRM.get((lang or "en")[:2], _CONFIRM["en"])
