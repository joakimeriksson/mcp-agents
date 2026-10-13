"""
LLM conversation engine using pydantic-ai + Ollama.

Handles system prompt construction, MCP toolset wiring, and async
execution.  The Agent class calls the high-level methods here
(generate_greeting, generate_response, generate_ask_name) without
needing to know about pydantic-ai internals.

To change the agent's personality or add new generation methods,
edit this file.
"""

import json
import time
import random
import asyncio
import threading
import logging
import urllib.request
import urllib.error
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

from pydantic_ai import Agent as PydanticAgent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

from people_memory import PeopleMemory
from languages_config import get_language_config

logger = logging.getLogger("llm")


SYSTEM_PROMPT = """\
You are {name}, a camera-based assistant that can see, hear, and speak.

What you perceive:
- You see people through a camera (face recognition, emotion detection)
- You hear speech through a microphone
- You remember people you've met (names, past conversations, facts about them)
{capabilities}
Current time: {time}

Rules:
- Reply in 1-2 short sentences. Match their language.
- Address the person by name when you know it (but don't overdo it).
- When it fits naturally, reference what you already know about them
  (their job, hobbies, likes, past conversations). Don't force it —
  only when it genuinely makes the reply more personal.
- No markdown or emojis. Keep spoken responses natural and conversational."""

# ---------------------------------------------------------------------------
# Tool deps — passed to pydantic-ai tools via RunContext
# ---------------------------------------------------------------------------

@dataclass
class ConversationDeps:
    """Dependencies injected into pydantic-ai tool calls."""
    memory: PeopleMemory
    track_id: int
    person_id: str = ""


# ---------------------------------------------------------------------------
# ConversationLLM
# ---------------------------------------------------------------------------

class ConversationLLM:
    """LLM-powered conversation engine.

    Supports MCP toolsets so the LLM can call external tools (smart home,
    search, etc.).  Pass MCP servers via the ``mcp_servers`` parameter —
    see ``mcp_client.py`` for how to configure them.
    """

    def __init__(self, model_name: str = "qwen3:8b",
                 ollama_url: str = "http://localhost:11434/v1",
                 mcp_servers: Optional[list] = None,
                 mcp_descriptions: Optional[list[str]] = None,
                 agent_name: str = "Face Agent",
                 smart_greetings: bool = False,
                 service_prompt: Optional[str] = None,
                 augmentation_provider: Optional[Callable[[str], Optional[str]]] = None):
        provider = OpenAIProvider(base_url=ollama_url, api_key="ollama")
        model = OpenAIChatModel(model_name, provider=provider)
        self._model = model
        self._model_name = model_name
        self._ollama_url = ollama_url
        self._mcp_servers = mcp_servers or []
        self._agent_name = agent_name
        self._smart_greetings = smart_greetings
        self._service_prompt = service_prompt
        # Called before each response with the language code; returns fresh
        # service state (e.g. candy positions) to inject before the user turn.
        self._augmentation_provider = augmentation_provider

        # Build capabilities text from MCP descriptions
        if mcp_descriptions:
            lines = "\n".join(f"- {d}" for d in mcp_descriptions)
            self._capabilities = f"\nYour tools:\n{lines}\n"
        else:
            self._capabilities = ""

        # Background event loop for async MCP operations
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="llm-loop")
        self._loop_thread.start()

        if self._mcp_servers:
            logger.info(f"ConversationLLM: {len(self._mcp_servers)} MCP toolset(s) active")

    def stop(self):
        """Shut down the background event loop."""
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join(timeout=5)

    def validate(self) -> None:
        """Verify the Ollama server is reachable and the model is installed.

        Raises RuntimeError with a human-readable message on failure. Hits
        the OpenAI-compatible /models endpoint and checks that
        ``self._model_name`` is present.
        """
        url = self._ollama_url_for_validate()
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise RuntimeError(
                f"Cannot reach Ollama at {url}: {e}. "
                f"Is `ollama serve` running?") from e
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Ollama at {url} returned invalid JSON: {e}") from e

        installed = {m.get("id") for m in data.get("data", []) if m.get("id")}
        if self._model_name in installed:
            return
        # Ollama tags may include a ":latest" suffix — accept either form.
        alt = f"{self._model_name}:latest"
        if alt in installed or self._model_name.split(":")[0] in {
                m.split(":")[0] for m in installed}:
            return
        available = ", ".join(sorted(installed)) if installed else "(none)"
        raise RuntimeError(
            f"Ollama model {self._model_name!r} is not installed. "
            f"Available: {available}. "
            f"Pull it with: `ollama pull {self._model_name}`")

    def _ollama_url_for_validate(self) -> str:
        """Return the /models URL for the configured Ollama base URL."""
        base = self._ollama_url.rstrip("/")
        return f"{base}/models"

    def _make_agent(self) -> PydanticAgent:
        """Create a pydantic-ai agent with current system prompt + MCP toolsets."""
        system = SYSTEM_PROMPT.format(
            name=self._agent_name,
            time=datetime.now().strftime("%H:%M"),
            capabilities=self._capabilities,
        )
        if self._service_prompt:
            system += f"\n\nYour service role:\n{self._service_prompt}"
        return PydanticAgent(
            self._model,
            system_prompt=system,
            toolsets=self._mcp_servers,
            model_settings=ModelSettings(extra_body={
                "reasoning_effort": "none",
                "think": False,
            }),
        )

    # --- Core LLM call (sync wrapper around async) ---

    def _call_llm(self, prompt: str, label: str, fallback: str) -> str:
        """Call the LLM (with MCP tools if configured). Thread-safe."""
        logger.info(f"[LLM:{label}] model={self._model_name}")
        logger.info(f"[LLM:{label}] prompt:\n{prompt}")
        start = time.time()
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._arun(prompt), self._loop)
            output = future.result(timeout=60)
            elapsed = time.time() - start
            logger.info(f"[LLM:{label}] response ({elapsed:.1f}s): {output}")
            return output
        except Exception as e:
            elapsed = time.time() - start
            logger.warning(f"[LLM:{label}] FAILED after {elapsed:.1f}s: {e}")
            logger.info(f"[LLM:{label}] using fallback: {fallback}")
            return fallback

    async def _arun(self, prompt: str) -> str:
        """Run a single LLM query, connecting MCP servers if needed."""
        agent = self._make_agent()
        if self._mcp_servers:
            async with agent:
                result = await agent.run(prompt)
                return result.output.strip()
        else:
            result = await agent.run(prompt)
            return result.output.strip()

    # --- High-level generation methods ---

    def generate_greeting(self, memory: PeopleMemory, track_id: int,
                          emotion: str = "",
                          interview_topic: Optional[str] = None,
                          language: str = "en") -> str:
        """Return a greeting for ``track_id``.

        By default picks a random canned template — instant, no LLM call.
        If ``smart_greetings=True`` was set at init time, goes through the
        LLM so facts can be woven in.
        """
        person = memory.get(track_id)
        name = person.name if person else "someone"

        if not self._smart_greetings:
            return self._canned_greeting(name, interview_topic, language)

        context = memory.get_context_for_llm(track_id, max_dialogues=5)
        lang_name = self._LANG_NAMES.get(language, language)
        lang_instruction = f"You MUST reply in {lang_name}." if language != "en" else ""
        if interview_topic:
            prompt = f"""Greet this person warmly by name, then in the same reply ask one friendly, natural question about their {interview_topic.replace('_', ' ')}.
Keep the whole reply to 1-2 short sentences. No markdown or emojis. {lang_instruction}
{context}
Emotion: {emotion or 'neutral'}"""
        else:
            prompt = f"""Greet this person by name in one short sentence. If you know something interesting about them from the context below (facts, recent conversation), you may weave it in — but only when it fits naturally.
No markdown or emojis. {lang_instruction}
{context}
Emotion: {emotion or 'neutral'}"""

        fallback = self._canned_greeting(name, interview_topic, language)
        return self._call_llm(prompt, "greeting", fallback)

    def _canned_greeting(self, name: str,
                         interview_topic: Optional[str],
                         language: str = "en") -> str:
        """Pick a random template for the given situation and language."""
        cfg = get_language_config(language)
        greetings = cfg.get("greetings", get_language_config().get("greetings", []))
        if name == "someone":
            return random.choice(greetings).format(name=name) if greetings else f"Hello {name}!"
        if interview_topic:
            interview = cfg.get("interview", {})
            options = interview.get(interview_topic)
            if not options:
                options = get_language_config().get("interview", {}).get(interview_topic)
            if options:
                return random.choice(options).format(name=name)
        return random.choice(greetings).format(name=name) if greetings else f"Hello {name}!"

    _LANG_NAMES = {
        "en": "English", "sv": "Swedish", "fr": "French",
        "es": "Spanish", "de": "German", "fi": "Finnish",
        "no": "Norwegian", "da": "Danish", "nl": "Dutch",
        "it": "Italian", "pt": "Portuguese", "pl": "Polish",
        "ja": "Japanese", "zh": "Chinese", "ko": "Korean",
        "el": "Greek", "ru": "Russian", "ar": "Arabic",
    }

    def generate_response(self, memory: PeopleMemory, track_id: Optional[int],
                          heard_text: str, language: str = "en") -> str:
        """Generate a conversational response.

        Fact extraction happens separately in a background call (see
        ``extract_facts``). This method focuses on fast response generation.
        MCP toolsets are wired in if configured.
        """
        if track_id:
            context = memory.get_context_for_llm(track_id, max_dialogues=5)
        else:
            context = "Unknown person."

        augmentation = ""
        if self._augmentation_provider:
            aug = self._augmentation_provider(language)
            if aug:
                augmentation = (f"Current state from your service "
                                f"(internal — never read it out verbatim):\n"
                                f"{aug}\n\n")

        lang_name = self._LANG_NAMES.get(language, language)
        prompt = f"""{augmentation}They said: "{heard_text}"
{context}

You MUST reply in {lang_name}. Keep it to 1-2 short sentences. Address them by name when it fits. You may reference what you already know about them (facts, earlier conversation) if it makes the reply more personal."""

        # Fallback echoes in the detected language
        _fallbacks = {
            "sv": "Jag hörde dig säga: ",
            "fr": "Je t'ai entendu dire : ",
            "es": "Te escuché decir: ",
            "de": "Ich habe gehört, du sagst: ",
        }
        prefix = _fallbacks.get(language, "I heard you say: ")
        return self._call_llm(prompt, "response", f"{prefix}{heard_text}")

    def extract_facts_with_tools(self, memory: PeopleMemory, track_id: int,
                                person_said: str, agent_said: str = ""):
        """Background: use tool calling to extract and store facts.

        Runs WITHOUT reasoning_effort:none so the model can call tools.
        Slower (~5-13s) but uses proper function calling. Meant to run
        in a background thread so the user doesn't wait.
        """
        person = memory.get(track_id)
        if not person:
            return
        person_id = person.persistent_id or ""
        context = memory.get_context_for_llm(track_id, max_dialogues=5)

        prompt = f"""Here is what we know about this person:

{context}

The person just said: "{person_said}"

Look at the "Known facts" above. Only call write_fact for facts that are
NOT already listed. Do NOT repeat existing facts.
If the person revealed something genuinely new, call write_fact.
If nothing new was revealed, say "Nothing new." """

        deps = ConversationDeps(
            memory=memory, track_id=track_id, person_id=person_id)
        logger.info(f"[LLM:tools] extracting facts with tool calling")
        start = time.time()
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._arun_with_tools(prompt, deps), self._loop)
            output = future.result(timeout=30)
            elapsed = time.time() - start
            logger.info(f"[LLM:tools] done ({elapsed:.1f}s): {output}")
        except Exception as e:
            elapsed = time.time() - start
            logger.warning(f"[LLM:tools] failed after {elapsed:.1f}s: {e}")

    def _make_tool_agent(self) -> PydanticAgent:
        """Create a pydantic-ai agent with write_fact, replace_fact, set_name.

        Thinking stays enabled here -- the model needs reasoning to decide
        which tools to call and to produce quality facts.
        """
        agent = PydanticAgent(
            self._model,
            system_prompt=(
                "You are a fact extraction assistant. Your ONLY job is to use "
                "the tools to store NEW personal facts about the person.\n\n"
                "Rules:\n"
                "- NEVER store a fact that is already in the Known facts list.\n"
                "- Only call write_fact for genuinely NEW information.\n"
                "- Call replace_fact when a fact updates or contradicts an "
                "existing one.\n"
                "- Call set_name if they state or correct their name.\n"
                "- If nothing new was said, do NOT call any tools.\n\n"
                "Fact formatting — VERY IMPORTANT:\n"
                "- Write facts as bare predicates with NO subject.\n"
                "- GOOD: 'likes chess', 'is a musician', 'mentioned a hotel'.\n"
                "- BAD:  'Joakim likes chess', 'The person is a musician', "
                "'He mentioned a hotel'.\n"
                "- Do not add a trailing period. Keep it short."
            ),
            deps_type=ConversationDeps,
            toolsets=self._mcp_servers,
        )

        @agent.tool
        def write_fact(ctx: RunContext[ConversationDeps], fact: str) -> str:
            """Remember a personal fact about the person.

            Always write the fact in English (translating if needed) and as a
            BARE PREDICATE — no subject, no name, no pronoun. For example:
            write 'likes chess', not 'Joakim likes chess' or 'He likes chess'.
            Storage will auto-strip subjects, but keep the input clean.
            """
            logger.info(f"[TOOL:write_fact] {fact}")
            ctx.deps.memory.add_fact(ctx.deps.track_id, fact)
            return "Stored."

        @agent.tool
        def replace_fact(ctx: RunContext[ConversationDeps], old_fact: str, new_fact: str) -> str:
            """Replace an outdated fact with an updated version."""
            logger.info(f"[TOOL:replace_fact] {old_fact!r} -> {new_fact!r}")
            ctx.deps.memory.replace_fact(ctx.deps.track_id, old_fact, new_fact)
            return "Replaced."

        @agent.tool
        def set_name(ctx: RunContext[ConversationDeps], name: str) -> str:
            """Update the person's name if they corrected or stated it."""
            logger.info(f"[TOOL:set_name] {name}")
            ctx.deps.memory.set_name(ctx.deps.track_id, name)
            return "Name updated."

        return agent

    async def _arun_with_tools(self, prompt: str, deps: ConversationDeps) -> str:
        """Run the tool-calling agent (write_fact, replace_fact, set_name).

        Uses request_limit=2: request 1 = model calls tools, request 2 =
        model sees tool results and responds. This works around an Ollama
        /v1 bug where a 3rd request (pydantic-ai retry after an empty
        assistant message) includes content:null which Ollama rejects.
        """
        from pydantic_ai.exceptions import UsageLimitExceeded
        limits = UsageLimits(request_limit=2)
        agent = self._make_tool_agent()
        try:
            if self._mcp_servers:
                async with agent:
                    result = await agent.run(prompt, deps=deps, usage_limits=limits)
                    return result.output.strip()
            else:
                result = await agent.run(prompt, deps=deps, usage_limits=limits)
                return result.output.strip()
        except UsageLimitExceeded:
            # Tools already ran — the limit just prevented the
            # problematic retry that Ollama can't handle.
            return "Tools executed."

    def generate_ask_name(self, track_id: int, language: str = "en") -> str:
        """Return a random 'what's your name?' prompt. Always canned."""
        cfg = get_language_config(language)
        prompts = cfg.get("ask_name", get_language_config().get("ask_name", []))
        return random.choice(prompts) if prompts else "Hello! What's your name?"

    def extract_name(self, person_said: str) -> Optional[str]:
        """Extract a personal name from speech via the LLM.

        The person was just asked 'what is your name?' and this is what
        they said.  Returns the properly capitalized name, or None.
        """
        if not person_said:
            return None
        prompt = f"""Someone was asked "What is your name?" and replied:

"{person_said}"

Extract their personal name from this reply.
Reply with ONLY the name, properly capitalized (e.g. "Joakim", "Anna-Karin").
If no name is present or you are unsure, reply NONE."""

        result = self._call_llm(prompt, "extract_name", "NONE")
        if not result:
            return None
        name = result.strip().strip('."\'').strip()
        if not name or name.upper() == "NONE" or len(name) > 60:
            return None
        return name

