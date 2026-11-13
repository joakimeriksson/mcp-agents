"""Service policy: the domain knowledge a voice client needs about ONE service.

The voice/face layer (this package) is generic: it hears, thinks with gemma4,
calls MCP tools and speaks. Everything that only makes sense for a particular
service -- which tool "hands something over", which positions exist, what a
claim like "here is your Dumle" means, the fallback sentence when nothing
happened -- belongs to that service and lives next to it (for CandyTron:
``candytron_mcp/client_policy.py``).

A service advertises its policy with the MCP resource ``url://client_policy``
returning ``"module.path:ClassName"``; the client imports it with the repo
root on ``sys.path``. A service without one gets this no-op base class, and
the generic checks (the model judging "did you claim an action?") still apply.

Every hook takes and returns plain data so a policy never needs to import the
client. ``state`` is whatever ``parse_state`` returns (for CandyTron a
{position: description} dict) -- the generic layer only copies it.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from typing import Optional

logger = logging.getLogger("service_policy")


class ServicePolicy:
    """Defaults for a service that needs no special handling."""

    #: Tools that DO something in the world (counted as "an action happened").
    #: Empty = every tool except the no-op ones.
    action_tools: frozenset = frozenset()
    #: Tools that never count as an action and are never re-run from text.
    noop_tools: frozenset = frozenset({"default_action", "express"})
    #: Tools never executed when the model only WROTE them as text (they act
    #: at random or do nothing).
    not_recovered: frozenset = frozenset()

    # --- per-turn state ---------------------------------------------------
    def parse_state(self, augmentation: Optional[str]) -> dict:
        """Structured state from the service's per-turn augmentation text."""
        return {}

    def state_line(self, augmentation: Optional[str]) -> str:
        """One-line summary of the state for the console transcript."""
        return ""

    def hint_words(self, state: dict) -> str:
        """Vocabulary for speech recognition (names the user may say)."""
        return ""

    # --- tool calls -------------------------------------------------------
    def reject(self, name: str, args: dict, state: dict) -> Optional[str]:
        """Reason to refuse a call before it reaches the service, or None."""
        return None

    def apply(self, name: str, args: dict, result: str, state: dict) -> None:
        """Update *state* in place after a call the service accepted."""

    def dedupe_text_calls(self, calls: list, state: dict) -> list:
        """Filter tool calls recovered from text (the model tends to list
        every matching item when it writes calls out)."""
        return calls

    def describe_call(self, name: str, args: dict) -> Optional[str]:
        """Human-readable action for the transcript, or None for the default."""
        return None

    # --- claims -----------------------------------------------------------
    def claimed_items(self, sentences: list, state: dict) -> set:
        """Items in *state* named in *sentences* (the claim sentences of a
        reply, picked by the generic layer). Empty set = unknown."""
        return set()

    def retry_refusal(self, name: str, args: dict, promised: set, state_at_start: dict) -> Optional[str]:
        """During the claim retry: refuse a call that would act on something
        other than what was promised."""
        return None

    def nudge_no_action(self) -> Optional[str]:
        """Retry instruction when an action was claimed but none happened.
        None = the generic one."""
        return None

    def nudge_partial(self, claimed: int, done: int) -> Optional[str]:
        """Retry instruction when more was claimed than done. None = no retry."""
        return None

    def fallback_reply(self, lang: str) -> Optional[str]:
        """Canned honest sentence ("[en] Sorry, that didn't work ...")."""
        return None

    # --- prompts ----------------------------------------------------------
    def language_tail(self) -> str:
        """Appended to the language order: what must NOT change with language."""
        return (" The language only changes your words: if they ask you to do "
                "something, you still call the tool FIRST, exactly as you would "
                "in English.\n")

    def greeting(self) -> Optional[str]:
        """Default greeting instruction. None = the generic one."""
        return None


GENERIC = ServicePolicy()

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_policy(spec: Optional[str]) -> ServicePolicy:
    """Instantiate ``"module.path:ClassName"``; the generic policy when *spec*
    is empty or cannot be loaded (logged, never fatal)."""
    if not spec:
        return GENERIC
    try:
        mod_name, _, cls_name = spec.partition(":")
        if _REPO_ROOT not in sys.path:
            sys.path.append(_REPO_ROOT)
        cls = getattr(importlib.import_module(mod_name), cls_name or "Policy")
        policy = cls()
        logger.info("service policy: %s", spec)
        return policy
    except Exception as e:  # noqa: BLE001
        logger.warning("service policy %r not loadable (%s); using the generic one", spec, e)
        return GENERIC
