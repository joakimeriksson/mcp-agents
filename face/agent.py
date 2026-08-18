"""
Agent module: the intelligence loop that ties perception to action.

Subscribes to face_tracker and voice_input events, queries people_memory,
decides what to do, and speaks via voice_output.

Emits its own AgentEvent for UI/logging.

Can be run standalone with face_tracker + voice_input + voice_output:
    python agent.py [--no-voice] [--no-auto-ask]
"""

import os
import sys
import time
import threading
import logging
import argparse
from enum import Enum, auto
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Callable, Union

from events import EventDispatcher
from face_tracker import (
    FaceTracker, FaceDatabase, EmotionDetector, FaceEvent, FaceEventType,
)
import numpy as np
from voice_input import VoiceInput, AudioMonitor, ContinuousListener, EchoDetector
from voice_output import VoiceOutput
from people_memory import PeopleMemory
from llm import ConversationLLM
from languages_config import get_goodbye, get_default_language

logger = logging.getLogger("agent")

_SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Agent event types
# ---------------------------------------------------------------------------

class SpeakMode(Enum):
    SIMPLE = "simple"   # pause mic while speaking, no AEC
    AEC = "aec"         # full-duplex with echo cancellation + barge-in


class AgentEventType(Enum):
    GREETING = auto()           # agent decided to greet someone
    GOODBYE = auto()            # agent said goodbye when someone left
    ASKING_NAME = auto()        # agent is asking an unknown face for their name
    RESPONDING = auto()         # agent is responding to speech
    LEARNED_NAME = auto()       # agent learned someone's name from speech
    NAME_EXTRACT_FAILED = auto()  # heard speech but couldn't extract a name
    THINKING = auto()           # agent is deciding what to do (reasoning)


@dataclass(frozen=True)
class GreetingPayload:
    track_id: int
    name: str
    text: str
    emotion: str


@dataclass(frozen=True)
class GoodbyePayload:
    track_id: int
    name: str
    text: str


@dataclass(frozen=True)
class AskingNamePayload:
    track_id: int
    text: str


@dataclass(frozen=True)
class RespondingPayload:
    track_id: Optional[int]
    name: Optional[str]
    heard: str
    response: str
    language: str


@dataclass(frozen=True)
class LearnedNamePayload:
    track_id: int
    name: str
    raw_speech: str


@dataclass(frozen=True)
class NameExtractFailedPayload:
    track_id: int
    raw_speech: str


@dataclass(frozen=True)
class ThinkingPayload:
    reason: str


AgentEventPayload = Union[
    GreetingPayload, GoodbyePayload, AskingNamePayload, RespondingPayload,
    LearnedNamePayload, NameExtractFailedPayload, ThinkingPayload,
]


@dataclass(frozen=True)
class AgentEvent:
    type: AgentEventType
    timestamp: float
    payload: AgentEventPayload


AgentEventCallback = Callable[[AgentEvent], None]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    """The intelligence loop: perceives, remembers, decides, acts.

    Subscribes to face and voice events, manages people memory,
    and speaks via voice output.
    """

    def __init__(self, *,
                 tracker: FaceTracker,
                 voice_input: VoiceInput,
                 voice_output: VoiceOutput,
                 memory: PeopleMemory,
                 llm: ConversationLLM,
                 audio_monitor: Optional[AudioMonitor] = None,
                 greeting_cooldown_s: float = 60.0,
                 ask_name_cooldown_s: float = 30.0,
                 min_frames_before_ask: int = 3,
                 auto_ask: bool = True,
                 auto_greet: bool = True,
                 speak_mode: SpeakMode = SpeakMode.SIMPLE):
        self.tracker = tracker
        self.voice_in = voice_input
        self.voice_out = voice_output
        self.memory = memory
        self.llm = llm

        self._greeting_cooldown = greeting_cooldown_s
        self._ask_cooldown = ask_name_cooldown_s
        self._min_frames_ask = min_frames_before_ask
        self.auto_ask = auto_ask
        self.auto_greet = auto_greet
        self._audio_monitor = audio_monitor
        self.speak_mode = speak_mode

        self._busy = False
        self._busy_lock = threading.Lock()
        self._busy_since: float = 0.0
        self._busy_reason: str = ""
        self._busy_timeout: float = 90.0  # auto-clear after 90s

        # High-level state for UI display
        self.state: str = "IDLE"  # IDLE, LISTENING, TRANSCRIBING, THINKING, TALKING

        # AEC auto-disable: skip after repeated failures
        self._aec_failures: int = 0
        self._aec_disabled: bool = False

        # Cooldown tracking
        self._greeted: dict[str, float] = {}      # person_id -> timestamp
        self._asked: dict[int, float] = {}         # track_id -> timestamp

        # Echo detector reference — exposed for UI drawing
        self._echo_detector: Optional[EchoDetector] = None

        # Continuous listener
        self._listener: Optional[ContinuousListener] = None

        # Event system
        self._dispatcher = EventDispatcher(owner="agent")

        # Subscribe to face events
        self.tracker.subscribe(self._on_face_event)

        # Watchdog: force-clears stale _busy, dumps thread stacks, and
        # resumes the listener. Runs every 5s on a daemon thread.
        self._watchdog_stop = threading.Event()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=True, name="agent-watchdog")
        self._watchdog_thread.start()

    def _watchdog_loop(self):
        import faulthandler, sys
        while not self._watchdog_stop.wait(5.0):
            with self._busy_lock:
                if not self._busy or self._busy_since <= 0:
                    continue
                elapsed = time.time() - self._busy_since
                if elapsed <= self._busy_timeout:
                    continue
                reason = self._busy_reason
                self._busy = False
                self._busy_reason = ""
                self._busy_since = 0
            logger.error(
                f"[WATCHDOG] busy stuck {elapsed:.0f}s (reason={reason!r}) — "
                f"force-clearing; dumping thread stacks")
            try:
                faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
            except Exception as e:
                logger.warning(f"[WATCHDOG] traceback dump failed: {e}")
            try:
                if self._echo_detector is not None:
                    self._echo_detector.stop()
                    self._echo_detector = None
            except Exception as e:
                logger.warning(f"[WATCHDOG] echo stop failed: {e}")
            try:
                self.voice_out.speaking = False
                self.resume_listening()
            except Exception as e:
                logger.warning(f"[WATCHDOG] resume failed: {e}")

    def subscribe(self, callback: AgentEventCallback,
                  event_types: Optional[set] = None) -> Callable[[], None]:
        return self._dispatcher.subscribe(callback, event_types)

    # --- Control ---

    def start(self):
        """Start continuous listening (paused until first interaction completes)."""
        self._listener = ContinuousListener(
            self.voice_in, on_heard=self._on_heard_speech)
        self._listener.start()
        self._listener.paused = True  # don't listen until first greet is done
        self.state = "IDLE"
        logger.info("Agent started (listening paused until first interaction)")

    def stop(self):
        """Stop continuous listening and clean up."""
        self._watchdog_stop.set()
        if self._listener:
            self._listener.stop()
            # Cancel any blocking listen() so the thread exits promptly
            self.voice_in._cancel_listen = True
            self._listener = None
        self.llm.stop()
        self.memory.save_all()
        logger.info("Agent stopped")

    @property
    def busy(self) -> bool:
        with self._busy_lock:
            if self._busy and self._busy_since > 0:
                elapsed = time.time() - self._busy_since
                if elapsed > self._busy_timeout:
                    logger.warning(
                        f"[AGENT] busy timeout ({elapsed:.0f}s > {self._busy_timeout}s), "
                        f"reason was: {self._busy_reason}. Force-clearing.")
                    self._busy = False
                    self._busy_reason = ""
                    self._busy_since = 0
                    self.resume_listening()
            return self._busy

    def _set_busy(self, reason: str):
        """Must be called with _busy_lock held."""
        self._busy = True
        self._busy_since = time.time()
        self._busy_reason = reason
        logger.info(f"[AGENT] busy: {reason}")

    def _clear_busy(self):
        with self._busy_lock:
            self._busy = False
            self._busy_reason = ""
            self._busy_since = 0

    def pause_listening(self):
        if self._listener:
            self._listener.paused = True
            # Cancel any in-progress listen() so the pause takes effect
            # immediately instead of waiting for the 8s VAD timeout.
            self.voice_in._cancel_listen = True

    def resume_listening(self):
        if self._listener:
            self._listener.paused = False
            self.state = "LISTENING"

    def set_speak_mode(self, mode):
        if isinstance(mode, str):
            try:
                mode = SpeakMode(mode.lower())
            except ValueError:
                raise ValueError(
                    f"unknown speak mode {mode!r}; use "
                    f"{[m.value for m in SpeakMode]}")
        if mode == SpeakMode.AEC:
            self._aec_failures = 0
            self._aec_disabled = False
        self.speak_mode = mode

    # --- Manual actions ---

    def speak(self, text: str, language: str = None):
        """Speak text with WebRTC AEC barge-in support.

        *language* selects the TTS voice (e.g. ``"fr"``, ``"de"``).
        Uses a full-duplex audio stream: TTS plays through speakers while
        the mic is processed by WebRTC's echo canceller. The echo-cancelled
        residual is monitored — if a human voice is detected, TTS stops.

        Falls back to simple output-only playback if the AEC stream fails.
        """
        self.pause_listening()
        use_aec = (self.speak_mode == SpeakMode.AEC) and not self._aec_disabled
        aec_ok = False
        if use_aec:
            aec_ok = self._speak_aec(text, language)
            if not aec_ok:
                self._aec_failures += 1
                if self._aec_failures >= 2:
                    self._aec_disabled = True
                    logger.warning("[AGENT] AEC failed twice, disabling for this session")
                logger.warning("[AGENT] AEC playback failed, falling back to simple TTS")
        if not aec_ok:
            self.voice_out.speak(text, language)
        # Let room reverb decay before the listener reopens the mic
        time.sleep(0.5)
        self.resume_listening()

    def _speak_aec(self, text: str, language: str = None) -> bool:
        """Try to speak via AEC full-duplex stream.

        Returns True if audio played successfully, False if it failed
        (caller should fall back to simple playback).
        """
        echo = EchoDetector()
        self._echo_detector = echo

        try:
            voice = self.voice_out._get_voice(language)
            if voice is None:
                logger.warning("[AGENT] TTS not ready, skipping speak")
                return False

            # Synthesize TTS audio and resample to mic rate for the
            # full-duplex stream
            tts_sr = voice.config.sample_rate
            stream_sr = echo._stream_rate

            logger.info(f"[AGENT] synthesizing TTS ({language or 'default'}): {text!r}")
            for audio_chunk in voice.synthesize(text):
                raw = audio_chunk.audio_float_array
                # Resample from TTS rate to stream rate if needed
                if tts_sr != stream_sr:
                    import scipy.signal
                    resampled = scipy.signal.resample(
                        raw, int(len(raw) * stream_sr / tts_sr))
                    echo.feed(resampled.astype(np.float32))
                else:
                    echo.feed(raw)
            echo.finish_feeding()

            # Start full-duplex playback + AEC monitoring
            echo.start(tts_sample_rate=stream_sr)
            self.voice_out.speaking = True
            start = time.time()
            max_duration = 30.0  # safety timeout
            stall_limit = 2.0   # no output for this long = stuck
            last_output_time = start
            logger.info(f"[AEC] playing TTS via full-duplex stream")

            # Wait for playback to finish or barge-in
            aec_failed = False
            while echo.active:
                now = time.time()
                elapsed = now - start

                if echo.user_speaking:
                    echo.stop()
                    logger.info(
                        f"[AGENT] barge-in detected — stopping TTS "
                        f"(clean_rms={echo.clean_rms:.4f})")
                    break

                # Track when output was last non-zero
                if echo.output_rms > 0.001:
                    last_output_time = now

                # Stall detection: stream claims active but nothing playing
                if now - last_output_time > stall_limit and elapsed > stall_limit:
                    echo.stop()
                    logger.warning(
                        f"[AEC] stall detected — no output for "
                        f"{now - last_output_time:.1f}s, forcing stop")
                    aec_failed = True
                    break

                if elapsed > max_duration:
                    echo.stop()
                    logger.warning(
                        f"[AEC] timeout after {elapsed:.1f}s, forcing stop")
                    aec_failed = True
                    break

                time.sleep(0.05)

            duration = time.time() - start
            self.voice_out.speaking = False
            logger.info(f"[AEC] TTS done ({duration:.1f}s)")

            if duration < 1.0:
                logger.warning(f"[AEC] playback too short ({duration:.1f}s), treating as failure")
                return False
            if aec_failed and duration < 3.0:
                # Stalled early — likely no useful audio was played
                return False
            if aec_failed:
                # Stalled late — most audio likely played, don't replay
                logger.info(f"[AEC] stalled after {duration:.1f}s, skipping fallback (enough audio played)")
            return True

        except Exception as e:
            logger.warning(f"[AGENT] AEC speak failed: {e}", exc_info=True)
            self.voice_out.speaking = False
            return False
        finally:
            echo.stop()
            self._echo_detector = None

    def ask_name(self, track_id: int):
        """Manually trigger asking an unknown face for their name."""
        threading.Thread(target=self._do_ask_name,
                         args=(track_id,), daemon=True).start()

    def greet(self, track_id: int):
        """Manually trigger greeting a known face."""
        threading.Thread(target=self._do_greet,
                         args=(track_id,), daemon=True).start()

    # --- Face event handler ---

    def _on_face_event(self, event: FaceEvent):
        if event.type in (FaceEventType.FACE_APPEARED, FaceEventType.IDENTITY_CONFIRMED):
            logger.debug(f"[AGENT] face event: {event.type.name} track={event.track_id} busy={self._busy}")

        # Goodbye is lightweight (no LLM) — handle even when busy
        if event.type == FaceEventType.FACE_DISAPPEARED:
            self._handle_face_disappeared(event)
            return

        # Always update memory identification even when busy — only
        # skip the greeting speech. Otherwise "Unknown person" persists
        # and facts/names can't be recorded.
        if event.type == FaceEventType.IDENTITY_CONFIRMED:
            self._handle_identity_confirmed(event)
        elif event.type == FaceEventType.FACE_APPEARED:
            self._handle_face_appeared(event)
        elif event.type == FaceEventType.FACE_ENROLLED:
            self._handle_face_enrolled(event)

    def _handle_face_enrolled(self, event: FaceEvent):
        """Tracker auto-saved a new face — register it in memory."""
        tid = event.track_id
        pid = event.payload.person_id
        logger.info(f"[AGENT] FACE_ENROLLED: track={tid} person_id={pid}")
        self.memory.register_enrolled(tid, pid)

    def _handle_identity_confirmed(self, event: FaceEvent):
        """A face was just identified — update memory and greet."""
        p = event.payload
        tid = event.track_id
        face = self.tracker.get_face_by_id(tid)
        self.memory.identify(tid, p.person_id)
        self.memory.update_seen(tid, face.emotion if face else "")
        self._try_greet(tid, p.person_id)

    def _handle_face_appeared(self, event: FaceEvent):
        """A new face appeared — track in memory, greet if known."""
        p = event.payload
        tid = event.track_id
        logger.info(f"[AGENT] FACE_APPEARED: track={tid} person_id={p.initial_person_id} emotion={p.emotion}")
        self.memory.get_or_create(tid)
        if p.initial_person_id:
            self.memory.identify(tid, p.initial_person_id)
            self.memory.update_seen(tid, p.emotion)
            self._try_greet(tid, p.initial_person_id)

    def _try_greet(self, track_id: int, person_id: str):
        """Greet a person if auto_greet is on, agent is free, and cooldown expired."""
        if not self.auto_greet or self._busy or not person_id:
            return
        now = time.time()
        if person_id in self._greeted and (now - self._greeted[person_id]) < self._greeting_cooldown:
            return
        logger.info(f"[AGENT] triggering greet for {person_id} (track {track_id})")
        self._greeted[person_id] = now
        threading.Thread(target=self._do_greet, args=(track_id,), daemon=True).start()

    def _handle_face_disappeared(self, event: FaceEvent):
        """A face left the frame — say goodbye if it was a known person."""
        p = event.payload
        pid = p.person_id
        if not pid:
            return

        tid = event.track_id
        person = self.memory.get(tid) or self.memory.get_by_id(pid)
        name = person.name if person else None
        if not name:
            logger.info(f"[AGENT] FACE_DISAPPEARED: track={tid} pid={pid} (no name, skipping goodbye)")
            return
        logger.info(f"[AGENT] FACE_DISAPPEARED: track={tid} name={name} duration={p.duration_visible:.1f}s")

        lang = person.last_language if person else None
        text = get_goodbye(name, lang or "en")
        self._emit(AgentEventType.GOODBYE, GoodbyePayload(
            track_id=tid, name=name, text=text,
        ))
        # Acquire busy lock to prevent overlapping speech
        with self._busy_lock:
            if self.voice_out.speaking or self._busy:
                logger.info(f"[AGENT] skipping goodbye speech (busy)")
                return
            self._set_busy("goodbye")

        def _do_goodbye():
            try:
                self.speak(text, language=lang or None)
            finally:
                self._clear_busy()

        # Run on a background thread — this callback runs on the main
        # (camera) thread, and speak() is blocking.
        threading.Thread(target=_do_goodbye, daemon=True).start()

    # --- Periodic check for unknown faces (called from outside or a timer) ---

    def check_unknown_faces(self, frame=None):
        """Check if any visible unnamed people should be asked for their name.

        Auto-enrollment means the tracker saves an encoding for every new
        face on its own, so here we iterate by what ``memory`` knows —
        the unnamed placeholder persons registered on FACE_ENROLLED.
        ``frame`` is accepted for backwards compatibility but ignored.
        """
        if not self.auto_ask or self._busy:
            return
        if not self.voice_in.ready or not self.voice_out.ready:
            return

        for face in self.tracker.get_visible_faces():
            person = self.memory.get(face.track_id)
            if not person or person.is_identified:
                continue
            if face.frames_visible < self._min_frames_ask:
                continue
            tid = face.track_id
            now = time.time()
            if tid in self._asked and (now - self._asked[tid]) < self._ask_cooldown:
                continue

            self._asked[tid] = now
            threading.Thread(target=self._do_ask_name,
                             args=(tid,), daemon=True).start()
            break  # one at a time

    # --- Speech handler ---

    def _on_heard_speech(self, text: str):
        """Called by ContinuousListener when speech is transcribed."""
        if self._busy or not text:
            return

        with self._busy_lock:
            if self._busy:
                return
            self._set_busy("heard_speech")

        try:
            self.pause_listening()

            # Who are we talking to?
            primary = self.tracker.get_primary_face()
            tid = primary.track_id if primary else None
            person = self.memory.get(tid) if tid else None
            name = person.name if person else None
            lang = self.voice_in.detected_language or "en"

            # Log dialogue
            if tid:
                self.memory.add_dialogue(tid, "person", text,
                                         language=lang,
                                         emotion=primary.emotion if primary else "")

            # If the person is unidentified, try to learn their name
            if tid and person and not person.is_identified:
                threading.Thread(target=self._try_learn_name,
                                 args=(tid, text),
                                 daemon=True).start()

            # Generate response via LLM
            self.state = "THINKING"
            response = self.llm.generate_response(self.memory, tid, text, language=lang)

            self._emit(AgentEventType.RESPONDING, RespondingPayload(
                track_id=tid, name=name, heard=text,
                response=response, language=lang,
            ))

            # Log and speak
            if tid:
                self.memory.add_dialogue(tid, "system", response, language=lang)
            self.state = "TALKING"
            self.speak(response, language=lang)

            # Background fact extraction — runs for both identified and
            # unidentified people so set_name can catch names from
            # normal conversation (not just the ask-name flow).
            if tid and person:
                threading.Thread(target=self._extract_facts,
                                 args=(tid, text),
                                 daemon=True).start()

        finally:
            self._clear_busy()
            self.state = "LISTENING"
            self.resume_listening()

    # --- Internal action implementations ---

    def _do_greet(self, track_id: int):
        with self._busy_lock:
            if self._busy:
                logger.info(f"[AGENT] _do_greet skipped for track {track_id} — busy")
                return
            self._set_busy("greeting")

        try:
            logger.info(f"[AGENT] _do_greet starting for track {track_id}")
            self.pause_listening()

            face = self.tracker.get_face_by_id(track_id)
            person = self.memory.get(track_id)
            name = person.name if person else None
            emotion = face.emotion if face else ""

            if not name:
                logger.info(f"[AGENT] _do_greet: no name for track {track_id}, aborting")
                return

            # Pick an unasked interview topic if this person still has any
            person = self.memory.get(track_id)
            interview_topic: Optional[str] = None
            if person and person.is_identified:
                missing = person.missing_topics()
                if missing:
                    interview_topic = missing[0]
                    logger.info(f"[AGENT] interview: will ask {name} about {interview_topic}")

            lang = person.last_language if person else "en"
            logger.info(f"[AGENT] _do_greet: generating greeting for {name} (emotion={emotion}, lang={lang})")
            greeting = self.llm.generate_greeting(
                self.memory, track_id, emotion, interview_topic=interview_topic,
                language=lang or "en")
            logger.info(f"[AGENT] _do_greet: greeting ready")

            if interview_topic:
                self.memory.mark_topic_asked(track_id, interview_topic)

            self._emit(AgentEventType.GREETING, GreetingPayload(
                track_id=track_id, name=name, text=greeting, emotion=emotion,
            ))

            self.memory.add_dialogue(track_id, "system", greeting)
            self.memory.update_seen(track_id, emotion)
            lang = person.last_language if person else None
            self.speak(greeting, language=lang or None)
            logger.info(f"[AGENT] _do_greet: done, resuming listener")

        finally:
            self._clear_busy()
            self.resume_listening()

    def _do_ask_name(self, track_id: int):
        """Ask an unnamed face for their name.

        Speaks the question via TTS, then returns — the continuous
        listener picks up the response and ``_on_heard_speech`` handles
        name extraction for unidentified people. The face encoding is
        already saved by the tracker (auto-enroll), so no frame needs
        to be stashed.
        """
        with self._busy_lock:
            if self._busy:
                return
            self._set_busy("ask_name")

        try:
            ask_text = self.llm.generate_ask_name(track_id)
            self._emit(AgentEventType.ASKING_NAME, AskingNamePayload(
                track_id=track_id, text=ask_text,
            ))

            self.memory.add_dialogue(track_id, "system", ask_text)
            self.speak(ask_text)

        finally:
            self._clear_busy()

    def _try_learn_name(self, track_id: int, person_said: str):
        """Background: try to extract a name from what an unidentified person said."""
        extracted = self.llm.extract_name(person_said)
        if not extracted:
            self._emit(AgentEventType.NAME_EXTRACT_FAILED,
                       NameExtractFailedPayload(
                           track_id=track_id, raw_speech=person_said,
                       ))
            return

        self._emit(AgentEventType.LEARNED_NAME, LearnedNamePayload(
            track_id=track_id, name=extracted, raw_speech=person_said,
        ))

        person_id = self.memory.set_name(track_id, extracted)
        logger.info(f"[AGENT] learned name {extracted!r} for track {track_id} ({person_id})")

    def _extract_facts(self, track_id: int, person_said: str):
        """Background: extract facts via tool calling (write_fact, replace_fact, set_name).

        Uses proper function calling — the LLM calls tools to store facts.
        Runs without reasoning_effort:none so tools work. Slower but runs
        in a background thread so the user doesn't wait.
        """
        person = self.memory.get(track_id)
        if not person:
            return
        last_agent_said = ""
        for d in reversed(person.dialogues):
            if d.speaker == "system":
                last_agent_said = d.text
                break
        self.llm.extract_facts_with_tools(
            self.memory, track_id, person_said, agent_said=last_agent_said)

    def _emit(self, etype, payload):
        event = AgentEvent(type=etype, timestamp=time.time(), payload=payload)
        logger.info(f"[{etype.name}] {payload}")
        self._dispatcher.dispatch(event)


# ---------------------------------------------------------------------------
# Standalone mode
# ---------------------------------------------------------------------------

def _draw_echo_state(frame, agent):
    """Draw AEC state on the camera frame: raw mic, clean residual, threshold."""
    import cv2
    h, w = frame.shape[:2]
    x0 = 55
    y0 = 70
    bar_w = 12
    bar_h = h - 140
    gap = 4

    echo = agent._echo_detector
    if echo is None:
        cv2.putText(frame, "AEC", (x0, y0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
        return

    scale_max = 1.0  # RMS is 0-1 for float32 audio

    # --- Bar 1: Raw mic (green) ---
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + bar_w, y0 + bar_h), (30, 30, 30), cv2.FILLED)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    raw_h = int(min(1.0, echo.current_rms / scale_max) * bar_h)
    if raw_h > 0:
        for y in range(raw_h):
            cv2.line(frame, (x0 + 1, y0 + bar_h - y),
                     (x0 + bar_w - 1, y0 + bar_h - y), (0, 180, 0), 1)
    cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (80, 80, 80), 1)
    cv2.putText(frame, "RAW", (x0 - 2, y0 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 180, 0), 1)

    # --- Bar 2: Clean / echo-cancelled residual (blue/red) ---
    x1 = x0 + bar_w + gap
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (x1, y0), (x1 + bar_w, y0 + bar_h), (30, 30, 30), cv2.FILLED)
    cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)

    clean_scale = 0.3  # scale clean signal for visibility
    clean_h = int(min(1.0, echo.clean_rms / clean_scale) * bar_h)
    clean_color = (0, 0, 255) if echo.user_speaking else (255, 180, 0)
    if clean_h > 0:
        for y in range(clean_h):
            cv2.line(frame, (x1 + 1, y0 + bar_h - y),
                     (x1 + bar_w - 1, y0 + bar_h - y), clean_color, 1)

    # Threshold line on clean bar
    thresh_h = int(min(1.0, echo._speech_threshold / clean_scale) * bar_h)
    thresh_y = y0 + bar_h - thresh_h
    cv2.line(frame, (x1 - 3, thresh_y), (x1 + bar_w + 3, thresh_y), (0, 255, 255), 2)

    cv2.rectangle(frame, (x1, y0), (x1 + bar_w, y0 + bar_h), (80, 80, 80), 1)

    label = "VOICE!" if echo.user_speaking else "CLEAN"
    label_color = (0, 0, 255) if echo.user_speaking else (255, 180, 0)
    cv2.putText(frame, label, (x1 - 2, y0 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, label_color, 1)

    # --- Numeric readout below bars ---
    y_text = y0 + bar_h + 15
    cv2.putText(frame, f"r:{echo.current_rms:.3f}", (x0 - 2, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 180, 0), 1)
    cv2.putText(frame, f"c:{echo.clean_rms:.3f}", (x0 - 2, y_text + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, clean_color, 1)
    cv2.putText(frame, f"t:{echo._speech_threshold:.3f}", (x0 - 2, y_text + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 255), 1)


def main():
    parser = argparse.ArgumentParser(description="Standalone agent")
    parser.add_argument("--db-dir", default=os.path.join(_SOURCE_DIR, "known_faces"))
    parser.add_argument("--people-dir", default=os.path.join(_SOURCE_DIR, "people"))
    parser.add_argument("--camera", type=int, default=-1,
                        help="Camera index; -1 (default) auto-picks the first that delivers video")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--no-auto-ask", action="store_true")
    parser.add_argument("--no-auto-greet", action="store_true")
    parser.add_argument("--llm-model", default="qwen3:8b", help="Ollama model name")
    parser.add_argument("--ollama-url", default="http://localhost:11434/v1")
    parser.add_argument("--en-voice", default="en_US-lessac-medium",
                        help="English TTS voice")
    parser.add_argument("--mcp-config", default=None,
                        help="Path to MCP servers JSON config (default: mcp_servers.json)")
    parser.add_argument("--mcp-server", action="append", default=[],
                        help="MCP server SSE URL (can be repeated)")
    parser.add_argument("--service-server", default=None,
                        help="SSE URL of a service MCP server (e.g. candytron_mcp): "
                             "its tools are added like --mcp-server, and its "
                             "persona prompt, name, and init/exit lifecycle are "
                             "adopted (see ServiceHost in mcp_client.py)")
    parser.add_argument("--agent-name", default="Face Agent",
                        help="Name the agent uses to describe itself")
    parser.add_argument("--smart-greeting", action="store_true",
                        help="Use the LLM for greetings (slower, but can reference facts). "
                             "Default is canned templates — instant.")
    parser.add_argument("--shell", action="store_true",
                        help="Start an interactive debug shell alongside the agent")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    # Initialize all components
    from face_config import build_db_kwargs, build_tracker_kwargs
    face_db = FaceDatabase(db_dir=args.db_dir, **build_db_kwargs())
    face_db.load()
    emotion_detector = EmotionDetector()
    tracker = FaceTracker(db=face_db, emotion_detector=emotion_detector,
                          **build_tracker_kwargs())

    voice_in = VoiceInput()
    voice_out = VoiceOutput(model_name=args.en_voice)
    print("Loading speech models...")
    voice_in.load_sync()
    if voice_in.load_error:
        print(f"ERROR: {voice_in.load_error}", file=sys.stderr)
        sys.exit(1)
    voice_out.load_sync()
    if voice_out.load_error:
        print(f"ERROR: {voice_out.load_error}", file=sys.stderr)
        sys.exit(1)

    memory = PeopleMemory(storage_dir=args.people_dir)
    memory.load()

    from mcp_client import load_servers, ServiceHost
    server_urls = list(args.mcp_server)
    if args.service_server:
        server_urls.append(args.service_server)
    mcp_servers, mcp_descriptions = load_servers(
        config_path=args.mcp_config, server_urls=server_urls)

    # A service server (candytron_mcp etc.) also provides a persona, a
    # display name, and init/exit lifecycle — adopt what it offers.
    service = ServiceHost(args.service_server) if args.service_server else None
    agent_name = args.agent_name
    service_prompt = None
    if service:
        if not service.init():
            print(f"ERROR: service_init failed at {args.service_server}",
                  file=sys.stderr)
            sys.exit(1)
        name = service.fetch_name()
        if name and args.agent_name == "Face Agent":  # not overridden by CLI
            agent_name = name
        service_prompt = service.fetch_prompt(get_default_language())
        logger.info(f"Service: name={name!r}, "
                    f"persona={'yes' if service_prompt else 'no'}")

    llm = ConversationLLM(
        model_name=args.llm_model,
        ollama_url=args.ollama_url,
        mcp_servers=mcp_servers,
        mcp_descriptions=mcp_descriptions,
        agent_name=agent_name,
        smart_greetings=args.smart_greeting,
        service_prompt=service_prompt,
        augmentation_provider=service.fetch_augmentation if service else None,
    )
    try:
        llm.validate()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    logger.info(f"LLM: {args.llm_model} via {args.ollama_url}")

    monitor = AudioMonitor()
    monitor.start()

    agent = Agent(
        tracker=tracker,
        voice_input=voice_in,
        voice_output=voice_out,
        memory=memory,
        llm=llm,
        audio_monitor=monitor,
        auto_ask=not args.no_auto_ask,
        auto_greet=not args.no_auto_greet,
    )

    # Log agent events
    def on_agent_event(event: AgentEvent):
        ts = datetime.now().strftime("%H:%M:%S")
        p = event.payload
        if event.type == AgentEventType.GREETING:
            print(f"  [{ts}] GREET {p.name}: \"{p.text}\"")
        elif event.type == AgentEventType.GOODBYE:
            print(f"  [{ts}] GOODBYE {p.name}: \"{p.text}\"")
        elif event.type == AgentEventType.ASKING_NAME:
            print(f"  [{ts}] ASK track {p.track_id}: \"{p.text}\"")
        elif event.type == AgentEventType.RESPONDING:
            print(f"  [{ts}] HEARD from {p.name or '?'}: \"{p.heard}\"")
            print(f"  [{ts}] REPLY: \"{p.response}\"")
        elif event.type == AgentEventType.LEARNED_NAME:
            print(f"  [{ts}] LEARNED: track {p.track_id} = {p.name}")
        elif event.type == AgentEventType.NAME_EXTRACT_FAILED:
            print(f"  [{ts}] NAME FAIL: \"{p.raw_speech}\"")

    agent.subscribe(on_agent_event)

    # Per-turn voice latency: speech-end -> assistant-audio-start (STT/LLM/TTS).
    from latency import LatencyTracker
    LatencyTracker(voice_in, voice_out)

    print("Models loaded. Starting agent.\n")

    agent.start()

    if args.shell:
        from debug_shell import run_shell
        threading.Thread(
            target=run_shell, args=(memory, agent),
            daemon=True, name="debug-shell",
        ).start()

    import cv2
    from camera_utils import open_camera
    cap, cam_idx = open_camera(args.camera)
    if cap is None:
        logger.error("Could not open a working camera (all black/unavailable)")
        return

    cv2.namedWindow("Agent", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Agent", 800, 600)

    frame_interval = 1.0 / args.fps if args.fps > 0 else 0
    last_frame = 0.0

    print(f"Running at {args.fps} FPS. Q to quit.\n")

    try:
        while True:
            if frame_interval > 0:
                now = time.time()
                if now - last_frame < frame_interval:
                    wait_ms = max(1, int((frame_interval - (now - last_frame)) * 1000))
                    key = cv2.waitKey(wait_ms) & 0xFF
                    if key == ord("q") or key == 27:
                        break
                    continue
                last_frame = now

            ret, frame = cap.read()
            if not ret:
                break

            faces = tracker.process_frame(frame)
            visible = [f for f in faces if f.is_visible]

            # Let agent check for unknown faces
            agent.check_unknown_faces(frame)

            # Draw faces + audio meters
            from main import draw_faces, draw_audio_meter
            draw_faces(frame, tracker, memory)
            draw_audio_meter(frame, monitor, voice_in)

            # Echo detector overlay (only visible while agent is speaking)
            _draw_echo_state(frame, agent)

            # --- Agent state indicator (large, top-right) ---
            state = agent.state
            # Refine state from voice_input phase
            vi_phase = voice_in.listen_phase
            if state == "LISTENING" and vi_phase == "recording":
                state = "LISTENING..."
            elif state == "LISTENING" and vi_phase == "transcribing":
                state = "TRANSCRIBING"

            state_colors = {
                "IDLE": (120, 120, 120),
                "LISTENING": (0, 200, 0),
                "LISTENING...": (0, 255, 100),
                "TRANSCRIBING": (0, 200, 255),
                "THINKING": (0, 140, 255),
                "TALKING": (255, 200, 0),
            }
            state_color = state_colors.get(state, (200, 200, 200))

            # Draw state badge (top-right)
            h_frame, w_frame = frame.shape[:2]
            text_size = cv2.getTextSize(state, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            badge_x = w_frame - text_size[0] - 20
            badge_y = 10
            overlay_badge = frame.copy()
            cv2.rectangle(overlay_badge, (badge_x - 10, badge_y),
                          (w_frame - 5, badge_y + text_size[1] + 16),
                          state_color, cv2.FILLED)
            cv2.addWeighted(overlay_badge, 0.7, frame, 0.3, 0, frame)
            cv2.putText(frame, state, (badge_x, badge_y + text_size[1] + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

            # --- Info bar (bottom) ---
            status = f"Faces: {len(visible)} | Known: {len(face_db.known_person_ids)} | People: {memory.active_count}"
            cv2.putText(frame, status, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            cv2.imshow("Agent", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    except KeyboardInterrupt:
        pass

    # Force-exit on second Ctrl+C if cleanup hangs
    import signal
    signal.signal(signal.SIGINT, lambda *_: os._exit(1))

    print("\nShutting down...")
    try:
        if service:
            service.exit()
        agent.stop()
        monitor.stop()
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        logger.warning(f"Cleanup error: {e}")
    print("Done.")
    os._exit(0)


if __name__ == "__main__":
    main()
