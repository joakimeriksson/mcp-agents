"""CandyTron's client-side policy: what the voice client must know about the
candy arm. Loaded by the client (face/service_policy.load_policy) because
candytron_mcp advertises it via ``url://client_policy``.

Standard library only: this module runs inside the CLIENT process, not the
MCP server, and must not pull in the robot or camera stacks.

All of this used to live in face/direct_llm.py and the speech client; it was
moved here so the voice layer stays usable for other services (lights,
screens, ...). The behaviour is unchanged.
"""

from __future__ import annotations

import re
from typing import Optional

try:   # the base class lives in face/ (on sys.path in the client)
    from service_policy import ServicePolicy
except ImportError:  # pragma: no cover - standalone use
    ServicePolicy = object

HAND_OUT = "O0"   # the position in front of the visitor

# Positions the vision system reports, e.g. "A2 : Riesen-kola, brun ..." and
# "L1 : ..." for loose candy outside the boxes.
_SCENE_POS = re.compile(r"^\s*([A-Z]\d+)\s*:\s*(.+?)\s*$", re.MULTILINE)
# What the server answers when the arm really started (or queued) a move.
_MOVE_OK = re.compile(r"^(Started moving|Queued|Successfully demonstrated)", re.IGNORECASE)


def parse_scene_positions(augmentation: Optional[str]) -> dict:
    """{position: description} from the vision system's scene text."""
    if not augmentation:
        return {}
    return {m.group(1): m.group(2) for m in _SCENE_POS.finditer(augmentation)}


def candy_aliases(scene: dict) -> dict:
    """{candy name: {ways it is said}} for the candies on the table. The name
    is the head of the vision system's description ("Riesen-kola", "Dumle");
    its first word is an alias too unless another candy shares it
    ("Nötkräm päron" / "Nötkräm original")."""
    heads = sorted({desc.split(",")[0].strip().lower() for desc in scene.values()} - {""})
    # "licorice (lakrits)": the Swedish name in brackets is a name too.
    extra = {h: set(re.findall(r"\(([^)]+)\)", h)) for h in heads}
    first: dict = {}
    for h in heads:
        w = re.split(r"[\s\-]+", h)[0]
        first.setdefault(w, []).append(h)
    out = {}
    for h in heads:
        w = re.split(r"[\s\-]+", h)[0]
        bare = re.sub(r"\s*\([^)]*\)", "", h).strip()
        out[h] = ({h, bare} | {x.strip() for x in extra[h]}
                  | ({w} if len(w) >= 4 and len(first[w]) == 1 else set()))
    return out


def _pos(args: dict, key: str) -> str:
    return str((args or {}).get(key, "")).upper().strip()


def _kind(scene: dict, pos: str) -> str:
    return (scene.get(pos) or "").split(",")[0].strip().lower()


class CandyPolicy(ServicePolicy):
    action_tools = frozenset({"move_between", "show_demo_move"})
    not_recovered = frozenset({"show_demo_move", "default_action"})

    # --- per-turn state ---------------------------------------------------
    def parse_state(self, augmentation):
        return parse_scene_positions(augmentation)

    def state_line(self, augmentation):
        items = re.findall(r"^\s*([A-Z]\d+)\s*:\s*([^,\n]+)", augmentation or "", re.M)
        return "Scene:  " + (" · ".join(f"{p} {c.strip()}" for p, c in items) or "no candy seen")

    def hint_words(self, state):
        # Both names: "Licorice (Lakrits)" -> Licorice, Lakrits.
        kinds = sorted({n.strip() for d in state.values()
                        for n in re.split(r"[()]", d.split(",")[0]) if n.strip()})
        return ", ".join(kinds)

    # --- tool calls -------------------------------------------------------
    def reject(self, name, args, state):
        """The server happily reports "Successfully moved from B1 to O0" for an
        empty square -- the arm grips air while the robot says it worked.
        Only the client knows what the vision system saw this turn."""
        if name != "move_between" or not state or not isinstance(args, dict):
            return None
        src, dst = _pos(args, "src"), _pos(args, "dst")
        known = ", ".join(f"{p} ({d.split(',')[0]})" for p, d in sorted(state.items()))
        if not src or not dst:
            return f"Missing src or dst. Candy is at: {known}."
        if src == dst:
            return f"src and dst are both {src}; pick a different destination."
        if src not in state:
            return (f"Position {src} is EMPTY -- there is no candy there, so "
                    f"nothing was moved. Candy is at: {known}. "
                    f"Call move_between again with one of those as src.")
        return None

    def apply(self, name, args, result, state):
        """Track the board within a turn: B3 -> O0 after A1 -> B3 is fine,
        A1 -> O0 twice is not."""
        if name != "move_between" or not _MOVE_OK.match(result or ""):
            return
        src, dst = _pos(args, "src"), _pos(args, "dst")
        desc = state.pop(src, None)
        if desc is not None and dst and dst != HAND_OUT:
            state[dst] = desc

    def dedupe_text_calls(self, calls, state):
        """At most one move per kind of candy: written out, gemma4 tends to
        list every Dumle on the table for "a Dumle"."""
        kept, kinds = [], set()
        for c in calls:
            fn = c["function"]
            if fn["name"] == "move_between":
                src = _pos(fn["arguments"], "src")
                kind = (state.get(src) or src).split(",")[0].strip().lower()
                if kind in kinds:
                    continue
                kinds.add(kind)
            kept.append(c)
        return kept

    def describe_call(self, name, args):
        if name == "move_between":
            return f"move between {args.get('src', '?')} and {args.get('dst', '?')}"
        if name == "show_demo_move":
            return "demo move (random candy)"
        return None

    # --- claims -----------------------------------------------------------
    def claimed_items(self, sentences, state):
        aliases = candy_aliases(state)
        seen = set()
        for sentence in sentences:
            low = sentence.lower()
            seen.update(h for h, names in aliases.items()
                        if any(re.search(r"\b" + re.escape(n) + r"\b", low) for n in names))
        return seen

    def retry_refusal(self, name, args, promised, state_at_start):
        if name != "move_between" or not promised:
            return None
        src = _pos(args, "src")
        kind = _kind(state_at_start, src)
        if kind not in promised:
            return (f"Refused: you promised {', '.join(sorted(promised))}, "
                    f"but {src} holds {kind or 'nothing'}. Never hand over a "
                    f"different candy than the one promised.")
        return None

    def nudge_no_action(self):
        return ("You just told the person you were moving or handing over "
                "candy, but no move happened (no tool call, or the arm "
                "refused it), so nothing happened. Call move_between NOW "
                "for exactly that action, then confirm in one sentence "
                "starting with the ISO language code in brackets.")

    def nudge_partial(self, claimed, done):
        return (f"You told the person you are handing over {claimed} "
                f"candies but only {done} move(s) happened. Call "
                f"move_between NOW for each candy that was not moved yet "
                f"(one call per candy), then confirm in one sentence "
                f"starting with the ISO language code in brackets. If you "
                f"only meant to hand over one candy, call no tool and just "
                f"answer.")

    _NOTHING_MOVED = {
        "sv": "[sv] Förlåt, det gick inte. Vilket godis vill du ha?",
        "en": "[en] Sorry, that didn't work. Which candy would you like?",
        "de": "[de] Entschuldigung, das hat nicht geklappt. Welche Süßigkeit möchtest du?",
        "fr": "[fr] Désolé, ça n'a pas marché. Quel bonbon veux-tu ?",
        "es": "[es] Lo siento, no ha funcionado. ¿Qué caramelo quieres?",
        "it": "[it] Scusa, non ha funzionato. Quale caramella vuoi?",
    }

    def fallback_reply(self, lang):
        return self._NOTHING_MOVED.get(lang, self._NOTHING_MOVED["en"])

    # --- prompts ----------------------------------------------------------
    def language_tail(self):
        return (" The language only changes your words: if they ask for candy, you "
                "still call move_between FIRST, exactly as you would in English.\n")

    def greeting(self):
        return ("Greet them in ONE short sentence (at most 12 words) and ask which "
                "candy they would like. Do not describe yourself, the exhibition or "
                "the technology unless they ask.")
