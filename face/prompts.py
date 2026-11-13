"""The voice agent's prompt texts, editable live.

Only GENERIC rules live here (language tag, brevity, tools are function
calls). How to use a particular service's tools comes from that service
(``get_service_rules`` / ``get_service_prompt`` on its MCP server).


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
- If the person asks for something your tools can do, call the tool in THIS
  turn before you answer, then say what you DID. A question is not a request.
{service_rules}"""

EXPRESSIVE = """Your voice can laugh, whisper and show feelings. Be warm and playful. Now and
then, when it fits, make a light, friendly joke about robots or the moment.
To set how a sentence sounds, START that sentence with ONE of these tags:
[skrattar] laughing, [glad] happy and excited, [viskar] whispering (a secret),
[tänker] thinking out loud. Example: "[sv] [glad] Vad roligt att du kom! [skrattar]
Haha, du har god smak. Här kommer en Dumle."
- Use [skrattar] ONLY right after something that is actually funny, at most in
  one reply out of three. A greeting or a plain answer gets no laugh.
- Most replies need no tag at all; never more than one tagged sentence per reply.
- Tags are never read aloud and never go in the middle of a sentence.
- Never let a joke delay or replace a tool call.
- Tags are ONLY for how a sentence sounds. Actions are done with the tools
  (function calls), never by writing a tag or the call as text."""

EXPRESSIVE_TOOL = """Your voice can laugh, whisper and show feelings. Be warm and playful. Now and
then, when it fits, make a light, friendly joke about robots or the moment.
To set how your reply sounds, call the express tool (mood: laugh, happy,
whisper or thinking) together with your reply. Most replies need no mood at
all. Use laugh ONLY after something actually funny. express never replaces
another tool: if the person asks for an action, you still call its tool.
Your spoken text is always plain sentences: no tags, no brackets except the
leading language code."""

# Generic; a service's policy may supply its own (CandyTron: "ask which candy").
GREETING = ("Greet them in ONE short sentence (at most 12 words) and ask how you can help. Do not describe yourself, the exhibition or the technology unless they ask.")

DEFAULTS = {
    "rules": RULES,
    "persona": "",   # empty: the service's own persona (its get_service_prompt)
    "expressive": EXPRESSIVE,
    "expressive_tool": EXPRESSIVE_TOOL,
    "greeting": GREETING,
}

LABELS = {
    "rules": "Rules (direct audio)",
    "persona": "Persona",
    "expressive": "Expressive voice",
    "expressive_tool": "Expressive voice (express tool)",
    "greeting": "Greeting",
}

HELP = {
    "rules": "How to answer: language tag, tools, brevity, questions vs requests. "
             "{name}, {service} (persona + state), {lang_rule} and {service_rules} "
             "(the service's own rules) are filled in per turn.",
    "persona": "Who the robot is. Empty = the persona from the service.",
    "expressive": "Added when expressive mode is on: the [skrattar]/[glad]/[viskar]/[tänker] tags.",
    "expressive_tool": "Used instead of the above with [voice] express_tool = true: mood via the express() tool.",
    "greeting": "Instruction for the greeting when a visitor steps up. Empty = the service's default.",
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


def get(name: str, default: str | None = None) -> str:
    """The text in use: the override if set, else *default* (e.g. the
    service's own greeting), else the built-in default."""
    v = overrides().get(name)
    if isinstance(v, str) and v.strip():
        return v
    return default if default else DEFAULTS[name]


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
