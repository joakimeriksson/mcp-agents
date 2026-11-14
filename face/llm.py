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
from datetime import datetime
from typing import Callable, Optional

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

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

def parse_visit_json(raw: str) -> Optional[dict]:
    """The end-of-visit pass's JSON, tolerating a <think> block, code
    fences or prose around it. None if there is no JSON object."""
    import re
    text = re.sub(r"<think>.*?</think>", " ", raw or "", flags=re.S)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    if not isinstance(data.get("facts", []), list):
        data["facts"] = []
    return data


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
                 augmentation_provider: Optional[Callable[[str], Optional[str]]] = None,
                 interview_topics: Optional[dict] = None):
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
        # service state (e.g. what its camera sees) to inject before the user turn.
        self._augmentation_provider = augmentation_provider
        # Small-talk topics the service adds ({topic: {lang: [templates]}},
        # ServicePolicy.interview_topics); the generic ones are in languages.toml.
        self._service_topics = interview_topics or {}

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
                          language: str = "en",
                          extra_context: str = "") -> str:
        """Return a greeting for ``track_id``.

        By default picks a random canned template — instant, no LLM call.
        If ``smart_greetings=True`` was set at init time, goes through the
        LLM so facts can be woven in.
        """
        person = memory.get(track_id)
        name = person.name if person else "someone"

        if not self._smart_greetings:
            return self._canned_greeting(name, interview_topic, language)

        context = memory.get_context_for_llm(track_id)
        if extra_context:   # the visit memory: "They came back ... Earlier: ..."
            context += "\n" + extra_context
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

    def interview_topics(self) -> list[str]:
        """Topics to ask about, in order: languages.toml's, then the service's."""
        generic = list(get_language_config().get("interview", {}).keys())
        return generic + [t for t in self._service_topics if t not in generic]

    def _canned_greeting(self, name: str,
                         interview_topic: Optional[str],
                         language: str = "en") -> str:
        """Pick a random template for the given situation and language."""
        cfg = get_language_config(language)
        greetings = cfg.get("greetings", get_language_config().get("greetings", []))
        if name == "someone":
            return random.choice(greetings).format(name=name) if greetings else f"Hello {name}!"
        if interview_topic:
            service = self._service_topics.get(interview_topic, {})
            options = (service.get(language) or service.get("en")
                       or cfg.get("interview", {}).get(interview_topic))
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

        Facts are learned separately, once per visit (``consolidate_visit``).
        This method focuses on fast response generation.
        MCP toolsets are wired in if configured.
        """
        if track_id:
            context = memory.get_context_for_llm(track_id)
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

    # --- End-of-visit memory pass -------------------------------------

    _VISIT_SYSTEM = (
        "You maintain a robot's long-term memory of one person. You get what "
        "was said during one visit (speech recognition: it can be wrong) and "
        "what is already known. Output ONLY JSON:\n"
        '{"facts": [{"fact": "...", "quote": "...", "explicit": true}], '
        '"summary": "..."}\n\n'
        "facts: only lasting things the person said about THEMSELVES (work, "
        "home town, family, hobbies, likes and dislikes, plans). Each fact is "
        "a short English bare predicate with no subject: 'works as a nurse', "
        "'likes black coffee'. quote: the exact words from a 'Them:' line it "
        "came from, copied in the original language. explicit: true only if "
        "they said it plainly about themselves ('I work as a nurse').\n"
        "NOT facts: what they asked, said or talked about ('mentioned X', "
        "'asked about Y'), what they want right now, their name, anything "
        "about the robot, and anything garbled or out of place -- that is "
        "probably misheard. A TV, radio or other people may be in the "
        "transcript: only what the person says about THEMSELVES, in the "
        "first person, counts. Repeat a known fact only if they said it "
        "again. No facts is fine and common: {\"facts\": [], \"summary\": \"\"}.\n"
        "summary: who this person is, for next time (1-2 sentences, English, "
        "at most 200 characters), merging the previous summary with what you "
        "learned. Describe the PERSON, never the conversation or its topics. "
        "Empty string if you learned nothing about them.")

    def consolidate_visit(self, memory: PeopleMemory, track_id: int,
                          timeout: float = 90.0) -> Optional[dict]:
        """Turn the dialogue since the last pass into facts and a summary
        (one LLM call, no tools, off the critical path). The result goes
        through ``memory.apply_visit_facts``, which keeps only facts whose
        quote really is in what the person said. Returns its report, or
        None when there was nothing new or the call failed (the lines then
        stay pending for the next pass)."""
        person = memory.get(track_id)
        lines = memory.pending_visit_lines(track_id)
        heard = [d.text for d in lines if d.speaker == "person" and d.text.strip()]
        if not person or not heard:
            if lines:
                memory.mark_consolidated(track_id, lines[-1].timestamp)
            return None
        known = "; ".join(f.text for f in person.facts) or "(nothing yet)"
        transcript = "\n".join(
            f"{'Them' if d.speaker == 'person' else 'Robot'}: {d.text}" for d in lines)
        prompt = (f"Known facts: {known}\n"
                  f"Previous summary: {person.summary or '(none)'}\n\n"
                  f"This visit:\n{transcript}")
        start = time.time()
        try:
            future = asyncio.run_coroutine_threadsafe(self._arun_plain(
                self._VISIT_SYSTEM, prompt), self._loop)
            raw = future.result(timeout=timeout)
        except Exception as e:
            logger.warning(f"[LLM:visit] failed after {time.time() - start:.1f}s: {e}")
            return None
        parsed = parse_visit_json(raw)
        if parsed is None:
            logger.warning(f"[LLM:visit] unparseable output: {raw[:200]!r}")
            return None
        report = memory.apply_visit_facts(
            track_id, parsed.get("facts") or [], "\n".join(heard),
            summary=parsed.get("summary") if isinstance(parsed.get("summary"), str) else None)
        memory.mark_consolidated(track_id, lines[-1].timestamp)
        logger.info(f"[LLM:visit] {time.time() - start:.1f}s: kept {report['kept']}, "
                    f"dropped {len(report['dropped'])}")
        return report

    async def _arun_plain(self, system: str, prompt: str) -> str:
        """One LLM call with NO tools (memory work must never move an arm)."""
        agent = PydanticAgent(self._model, system_prompt=system)
        result = await agent.run(prompt)
        return result.output.strip()

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

