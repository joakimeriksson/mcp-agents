"""CandyTron's prompt texts, editable live.

The parts of the system prompt that people tune at a demo live here, with
their defaults. Overrides go in ~/.config/candytron/prompts.json (the
launcher's "Prompts..." window writes it); the client reads it before every
turn, so a change applies from the next turn without a restart. An empty
override means "use the default". The last system prompt actually sent is
written to ~/.config/candytron/last_prompt.txt for the launcher to show.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger("prompts")

OVERRIDES = Path(os.environ.get("CANDYTRON_PROMPTS", "~/.config/candytron/prompts.json")).expanduser()
LAST_PROMPT = Path(os.environ.get("CANDYTRON_LAST_PROMPT", "~/.config/candytron/last_prompt.txt")).expanduser()

# {name}, {service} and {lang_rule} are filled in per turn; see fill().
RULES = """You are {name}, a camera-based assistant that can see, hear, and speak.
You are listening to a person through a microphone; their speech is attached
as audio. You remember people you've met.
{service}
Rules:
- Begin your SPOKEN reply with the ISO language code of the spoken language
  in square brackets, e.g. [sv] or [en], then reply in that same language.
- Tool calls are function calls, made BEFORE you speak. Never write a tool
  call, its name or its arguments as text or inside [brackets].
{lang_rule}- Reply in 1-2 short sentences. No markdown or emojis.
- ACT, do not narrate. If the person asks for candy, or asks you to move,
  fetch, hand out or give anything, you MUST call the tool in THIS turn
  before you answer. Never say "I can", "I could", "let me" or "I will" —
  those are forbidden; call the tool and then say what you DID.
  To hand candy to the person, move it to position O0.
- Never say position codes (A1, D2, O0 ...) to the person; name the candy.
- If several candies of the kind they asked for are on the table, just take any
  one of them -- never ask which one.
- If the candy they ask for is NOT on the table, say so and name what is there.
  Never hand over a different candy than the one they asked for without asking.
- A QUESTION is not a request: "what do you have / what is on the table / what
  can you do" gets an answer and NO tool call. Only move candy when the person
  asks for it in what they just said; earlier requests do not count.
- But only act on what you actually HEARD. If the audio was unclear, or you
  cannot tell WHICH candy they mean, ask them to say it again — do NOT call a
  tool and do NOT move a random candy. A wrong move is worse than a question.
"""

EXPRESSIVE = """Your voice can laugh, whisper and show feelings. Be warm and playful. Now and
then, when it fits, make a light, friendly joke about candy, robots or the
moment.
To set how a sentence sounds, START that sentence with ONE of these tags:
[skrattar] laughing, [glad] happy and excited, [viskar] whispering (a secret),
[tänker] thinking out loud. Example: "[sv] [glad] Vad roligt att du kom! [skrattar]
Haha, du har god smak. Här kommer en Dumle."
- Use [skrattar] ONLY right after something that is actually funny, at most in
  one reply out of three. A greeting or a plain answer gets no laugh.
- Most replies need no tag at all; never more than one tagged sentence per reply.
- Tags are never read aloud and never go in the middle of a sentence.
- Never let a joke delay or replace a tool call.
- Tags are ONLY for how a sentence sounds. Moving candy is done with the tool
  (function call), never by writing a tag or the call as text."""

GREETING = ("Greet them in ONE short sentence (at most 12 words) and ask which candy they would like. Do not describe yourself, the exhibition or the technology unless they ask.")

DEFAULTS = {
    "rules": RULES,
    "persona": "",   # empty: the service's own persona (candytron_mcp get_service_prompt)
    "expressive": EXPRESSIVE,
    "greeting": GREETING,
}

LABELS = {
    "rules": "Rules (direct audio)",
    "persona": "Persona",
    "expressive": "Expressive voice",
    "greeting": "Greeting",
}

HELP = {
    "rules": "How to answer: language tag, tools, brevity, questions vs requests. "
             "{name}, {service} (persona + scene) and {lang_rule} are filled in per turn.",
    "persona": "Who the robot is. Empty = the persona from candytron_mcp.",
    "expressive": "Added when expressive mode is on: the [skrattar]/[glad]/[viskar]/[tänker] tags.",
    "greeting": "Instruction for the greeting when a visitor steps up.",
}

_lock = threading.Lock()
_cache: dict = {"mtime": None, "data": {}}


def overrides() -> dict:
    """The saved overrides, re-read only when the file changed."""
    try:
        mtime = OVERRIDES.stat().st_mtime
    except FileNotFoundError:
        return {}
    with _lock:
        if _cache["mtime"] != mtime:
            try:
                data = json.loads(OVERRIDES.read_text(encoding="utf-8"))
                _cache["data"] = data if isinstance(data, dict) else {}
                _cache["mtime"] = mtime
                logger.info("prompt overrides reloaded: %s", sorted(k for k, v in _cache["data"].items() if v))
            except Exception as e:  # half-written or broken file: keep the last good one
                logger.warning("prompt overrides not readable (%s); keeping the previous ones", e)
        return dict(_cache["data"])


def get(name: str) -> str:
    """The text in use: the override if set, else the default."""
    v = overrides().get(name)
    return v if isinstance(v, str) and v.strip() else DEFAULTS[name]


def is_overridden(name: str) -> bool:
    v = overrides().get(name)
    return isinstance(v, str) and bool(v.strip())


def save(data: dict) -> None:
    """Write overrides atomically (a reader never sees half a file)."""
    OVERRIDES.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in data.items() if k in DEFAULTS and isinstance(v, str) and v.strip()}
    tmp = OVERRIDES.with_suffix(".tmp")
    tmp.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, OVERRIDES)


def fill(template: str, **values: str) -> str:
    """Fill {placeholders} without str.format (an edited prompt may contain
    other braces). A placeholder someone deleted still gets its content,
    appended, so the scene and persona are never silently lost."""
    out = template
    for key, val in values.items():
        token = "{" + key + "}"
        if token in out:
            out = out.replace(token, val)
        elif val.strip():
            out += "\n" + val
    return out


def write_last(system: str, extra: str = "") -> None:
    """Record the system prompt just sent, for the launcher's view."""
    try:
        LAST_PROMPT.parent.mkdir(parents=True, exist_ok=True)
        body = f"# sent {time.strftime('%H:%M:%S')}\n{system}"
        if extra:
            body += f"\n\n# user message\n{extra}"
        tmp = LAST_PROMPT.with_suffix(".tmp")
        tmp.write_text(body, encoding="utf-8")
        os.replace(tmp, LAST_PROMPT)
    except OSError as e:
        logger.debug("last prompt not written: %s", e)
