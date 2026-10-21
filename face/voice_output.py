"""
Voice output module: text-to-speech using piper.

Provides:
- VoiceOutput: synthesize text and play through speakers
- Typed event system with subscribe/unsubscribe
- Blocking and non-blocking speak modes

Can be run standalone:
    python voice_output.py "Hello, world!"
    python voice_output.py --interactive
"""

import collections
import itertools
import os
import re
import time
import threading
import logging
import argparse
from enum import Enum, auto
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Callable, Union

import numpy as np
import sounddevice as sd
from piper import PiperVoice

from events import EventDispatcher
from reply_language import reply_language
from sentences import split_sentences

logger = logging.getLogger("voice_output")

_SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))

PIPER_MODEL_DIR = os.path.join(_SOURCE_DIR, "piper_models")
PIPER_MODEL_NAME = "en_US-lessac-medium"

from languages_config import (
    get_language_models, get_language_pronunciations, get_default_language,
    get_language_tts_servers,
)

LANGUAGE_MODELS = get_language_models()
DEFAULT_LANGUAGE = get_default_language()
LANG_PRONUNCIATIONS = get_language_pronunciations()
LANGUAGE_TTS_SERVERS = get_language_tts_servers()


# ---------------------------------------------------------------------------
# Event types and payloads
# ---------------------------------------------------------------------------

class TtsEventType(Enum):
    MODEL_LOADING = auto()
    MODEL_READY = auto()
    MODEL_LOAD_FAILED = auto()
    TTS_STARTED = auto()
    TTS_AUDIO_START = auto()  # first real audio sample reached the output device
    TTS_FINISHED = auto()
    TTS_BUSY = auto()       # speak() called while already speaking


@dataclass(frozen=True)
class ModelLoadingPayload:
    model_name: str


@dataclass(frozen=True)
class ModelReadyPayload:
    model_name: str


@dataclass(frozen=True)
class ModelLoadFailedPayload:
    model_name: str
    error: str


@dataclass(frozen=True)
class TtsStartedPayload:
    text: str
    tts_text: Optional[str] = None  # phonetic text if pronunciations were applied


@dataclass(frozen=True)
class TtsAudioStartPayload:
    text: str
    synth_latency_s: float   # speak() start -> first audible sample
    audio_start_ts: float    # absolute time.time() of the first audible sample


@dataclass(frozen=True)
class TtsFinishedPayload:
    text: str
    duration_s: float


@dataclass(frozen=True)
class TtsBusyPayload:
    rejected_text: str


TtsEventPayload = Union[
    ModelLoadingPayload, ModelReadyPayload, ModelLoadFailedPayload,
    TtsStartedPayload, TtsAudioStartPayload, TtsFinishedPayload, TtsBusyPayload,
]


@dataclass(frozen=True)
class TtsEvent:
    type: TtsEventType
    timestamp: float
    payload: TtsEventPayload


TtsEventCallback = Callable[[TtsEvent], None]


# ---------------------------------------------------------------------------
# VoiceOutput
# ---------------------------------------------------------------------------

class VoiceOutput:
    """Text-to-speech engine using piper.

    Subscribe to TtsEvent callbacks via subscribe().
    """

    def __init__(self, *,
                 model_dir: str = PIPER_MODEL_DIR,
                 model_name: str = PIPER_MODEL_NAME,
                 language_models: Optional[dict] = None):
        self._model_dir = model_dir
        self._default_model = model_name
        self._language_models = dict(language_models or LANGUAGE_MODELS)
        self._tts_servers = dict(LANGUAGE_TTS_SERVERS)
        self._lang_pron = LANG_PRONUNCIATIONS
        # Loaded PiperVoice instances keyed by model name
        self._voices: dict[str, PiperVoice] = {}
        self._model_lock = threading.Lock()  # protects _voices lazy loading
        self._ready = False
        self._load_error: Optional[str] = None
        self._speaking = False
        self._stop_event = threading.Event()
        self._interrupted = False
        self._lock = threading.Lock()
        self._filler_playing = threading.Event()
        self._dispatcher = EventDispatcher(owner="voice_output")
        # Rolling window of the last ~1s of played samples, for scope views.
        # Resized to the active stream's sample rate at playback time so the
        # buffer always spans exactly one second regardless of engine.
        self.out_waveform = np.zeros(24000, dtype=np.float32)
        self.out_last_write = 0.0   # time.time() of the last playback write

    # --- Event API ---

    def subscribe(self, callback: TtsEventCallback,
                  event_types: Optional[set] = None) -> Callable[[], None]:
        return self._dispatcher.subscribe(callback, event_types)

    def unsubscribe(self, callback: TtsEventCallback) -> bool:
        return self._dispatcher.unsubscribe(callback)

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def load_error(self) -> Optional[str]:
        """Error message from the last default-model load attempt, or None."""
        return self._load_error

    @property
    def speaking(self) -> bool:
        return self._speaking

    @speaking.setter
    def speaking(self, value: bool):
        self._speaking = value

    # --- Lifecycle ---

    def load(self):
        """Load TTS model in background thread."""
        threading.Thread(target=self._load, daemon=True).start()

    def load_sync(self):
        """Load TTS model synchronously (blocking)."""
        self._load()

    def _load(self):
        """Load the default model so the engine is ready quickly."""
        self._load_model(self._default_model)

    def _load_model(self, model_name: str):
        """Load a single piper model by name from disk.

        Never downloads — run `python download_models.py` beforehand.
        Thread-safe: uses _model_lock to prevent duplicate loads.
        """
        with self._model_lock:
            if model_name in self._voices:
                return
            self._emit(TtsEventType.MODEL_LOADING,
                       ModelLoadingPayload(model_name))
            model_path = os.path.join(self._model_dir, f"{model_name}.onnx")
            if not os.path.exists(model_path):
                msg = (f"Piper model {model_name!r} not found at "
                       f"{model_path} — run download_models.py")
                if model_name == self._default_model:
                    self._load_error = msg
                self._emit(TtsEventType.MODEL_LOAD_FAILED,
                           ModelLoadFailedPayload(model_name, msg))
                logger.error(msg)
                return
            try:
                logger.info(f"Loading piper model: {model_name}...")
                self._voices[model_name] = PiperVoice.load(model_path)
                self._ready = True
                self._emit(TtsEventType.MODEL_READY,
                           ModelReadyPayload(model_name))
                logger.info(f"Piper TTS loaded: {model_name}")
            except Exception as e:
                msg = f"Failed to load piper model {model_name}: {e}"
                if model_name == self._default_model:
                    self._load_error = msg
                self._emit(TtsEventType.MODEL_LOAD_FAILED,
                           ModelLoadFailedPayload(model_name, str(e)))
                logger.error(msg)

    def _get_voice(self, language: Optional[str] = None) -> Optional[PiperVoice]:
        """Return the PiperVoice for a language, or None if not loaded.

        Never downloads. Falls back to the default model if the requested
        language's model is not available on disk.
        """
        model_name = self._language_models.get(
            language, self._default_model) if language else self._default_model
        if model_name not in self._voices:
            self._load_model(model_name)
        if model_name not in self._voices:
            logger.warning(
                f"no model loaded for language={language!r} "
                f"({model_name}); falling back to default")
            model_name = self._default_model
            if model_name not in self._voices:
                self._load_model(model_name)
        return self._voices.get(model_name)

    # --- Server voice selection ---

    def server_voice(self, language: str) -> Optional[str]:
        """Voice name used on the TTS server for *language* (None = server default)."""
        return self._tts_servers.get(language, {}).get("voice")

    def server_voices(self, language: str) -> list:
        """Selectable server voices for *language* (from languages.toml tts_voices)."""
        return list(self._tts_servers.get(language, {}).get("voices", []))

    def set_server_voice(self, language: str, voice: str):
        """Switch the server voice for *language* live (next utterance uses it)."""
        if language not in self._tts_servers:
            raise ValueError(f"no TTS server configured for language {language!r}")
        self._tts_servers[language]["voice"] = voice
        logger.info(f"TTS voice for {language} -> {voice}")

    # --- Pronunciation ---

    def _apply_pronunciations(self, text: str, language: Optional[str]) -> str:
        """Replace words with phonetic spellings for better TTS output."""
        if language and language in self._lang_pron:
            for word, replacement in self._lang_pron[language].items():
                text = re.sub(rf"\b{re.escape(word)}\b", replacement,
                              text, flags=re.IGNORECASE)
        return text

    # --- Core ---

    @property
    def interrupted(self) -> bool:
        """True if the last speak() was cut short by stop_speaking()."""
        return self._interrupted

    def stop_speaking(self):
        """Interrupt the current TTS playback. Safe to call from any thread."""
        self._stop_event.set()

    def speak(self, text, language: Optional[str] = None):
        """Blocking TTS: synthesize and play, one sentence at a time. Thread-safe.

        *text* is a string or an iterable of sentences that may still be
        arriving (e.g. a queue fed by a streaming LLM). Either way each
        sentence is its own TTS request, made while the previous one plays,
        so speech starts as soon as the first sentence exists.

        *language* selects the voice model (e.g. ``"fr"``, ``"de"``); the
        first sentence decides it for the whole reply.
        Falls back to the default (English) if the language is unknown.
        Can be interrupted mid-playback via ``stop_speaking()``.
        After returning, check ``interrupted`` to know if it was cut short.
        """
        # While a thinking filler plays, the first sentence is synthesized
        # meanwhile and played the moment the filler ends; otherwise the
        # lock is taken now, so a second concurrent speak() is turned away.
        held = False
        if not self._filler_playing.is_set():
            if not self._lock.acquire(timeout=0.5):
                self._busy(text)
                return
            held = True

        try:
            sentences = iter(split_sentences(text) if isinstance(text, str) else text)
            first = next(sentences, None)
            if not first:
                return
            spoken = [first]

            def rest():
                for sentence in sentences:
                    spoken.append(sentence)
                    yield sentence

            language = self._speak_language(first, language)
            server = self._tts_servers.get(language)
            voice = None
            start = time.time()
            self._emit(TtsEventType.TTS_STARTED, TtsStartedPayload(first, None))
            first_audio = None
            if server is not None:
                # Pronunciation replacements are piper-specific phonetic hacks —
                # the HTTP server (Kokoro, real g2p) gets the raw text instead.
                try:
                    first_audio = self._http_synth(first, language, server)
                except Exception as e:
                    logger.warning(
                        f"HTTP TTS ({server['url']}) failed: {e} — "
                        f"falling back to piper")
            if first_audio is None:
                voice = self._get_voice(language)
                if voice is None:
                    logger.warning(f"TTS not ready, would say: {first} ...")
                    return
            if not held:   # wait for the filler (they are ~1-2 s)
                if not self._lock.acquire(timeout=3.0):
                    self._busy(text)
                    return
                held = True
            self._speaking = True
            self._interrupted = False
            self._stop_event.clear()
            if first_audio is not None:
                audio, sr = first_audio
                self._play_stream(first, itertools.chain(
                    [audio], self._http_rest(rest(), language, server, sr)), sr)
            else:
                self._play(itertools.chain([first], rest()), voice, language)
            duration = time.time() - start
            self._emit(TtsEventType.TTS_FINISHED,
                       TtsFinishedPayload(" ".join(spoken), duration))
        except Exception as e:
            logger.error(f"TTS playback failed: {e}")
        finally:
            if held:
                self._speaking = False
                self._lock.release()

    def _busy(self, text):
        preview = text if isinstance(text, str) else "(sentence stream)"
        self._emit(TtsEventType.TTS_BUSY, TtsBusyPayload(preview))
        logger.warning(f"TTS busy, rejected: {preview}")

    def speak_async(self, text, language: Optional[str] = None) -> threading.Thread:
        """Non-blocking TTS: speak in a background thread (returned, so the
        caller can wait for the whole reply, filler hand-over included)."""
        thread = threading.Thread(target=self.speak, args=(text, language), daemon=True)
        thread.start()
        return thread

    def filler(self, language: Optional[str] = None) -> bool:
        """Start a short "Hmm…" / "Mm, vänta lite…" in the background, to fill
        the silence while the reply is being thought up. Returns at once;
        False if nothing plays: the voice has no fillers on the server
        (kokoro-voice-server serves them for its --vox-voice voices), the
        language is not served over HTTP, or something is already playing.
        A speak() that comes along waits for the filler to finish.
        """
        import io
        import wave
        import requests

        language = language if language in self._tts_servers else DEFAULT_LANGUAGE
        server = self._tts_servers.get(language)
        if server is None or not server["url"].endswith("/v1/audio/speech") \
                or not self._lock.acquire(blocking=False):
            return False
        try:
            url = server["url"][:-len("speech")] + "filler"
            resp = requests.get(url, params={"voice": server.get("voice") or "",
                                             "language": language}, timeout=0.3)
            if resp.status_code != 200 or not resp.content:
                self._lock.release()
                return False
            with wave.open(io.BytesIO(resp.content)) as wf:
                sr = wf.getframerate()
                pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        except Exception as e:
            logger.debug(f"no filler: {e}")
            self._lock.release()
            return False
        audio = pcm.astype("float32") / 32768.0
        self._filler_playing.set()

        def play():
            try:
                self._speaking = True
                self._stop_event.clear()
                self._play_stream("(filler)", iter([audio]), sr, events=False)
            except Exception as e:
                logger.warning(f"filler playback failed: {e}")
            finally:
                self._speaking = False
                self._filler_playing.clear()
                self._lock.release()

        threading.Thread(target=play, daemon=True).start()
        return True

    def _http_rest(self, sentences, language, server, sr):
        """Audio for the sentences after the first, one request each. A
        sentence that fails is skipped: the stream is already playing at
        *sr*, so piper (another rate) cannot step in mid-reply."""
        for sentence in sentences:
            if self._stop_event.is_set():
                return
            try:
                audio, rate = self._http_synth(sentence, language, server)
            except Exception as e:
                logger.warning(f"HTTP TTS failed on {sentence!r}: {e} — skipped")
                continue
            if rate != sr:
                logger.warning(f"HTTP TTS answered at {rate} Hz, not {sr} — skipped")
                continue
            yield audio

    def _speak_language(self, text: str, language: Optional[str],
                        log: bool = True) -> str:
        """The language speak() voices *text* in."""
        # The voice follows the reply text, not the conversation language:
        # that one is sticky on purpose and lags a language switch (see
        # reply_language.py). It still decides fragments like "Ja!".
        detected, conf = reply_language(
            text, language, set(self._tts_servers) | set(self._language_models))
        if detected and detected != language:
            if log:
                logger.info(f"TTS: reply reads as {detected} ({conf:.2f}), not "
                            f"{language} — using the {detected} voice")
            language = detected
        # Normalize: no/unknown language means the default language —
        # otherwise a caller passing None/False/"hi" silently bypasses
        # the configured TTS server and lands on the default Piper voice.
        if (not language or (language not in self._tts_servers
                             and language not in self._language_models)):
            language = DEFAULT_LANGUAGE
        return language

    def voice_for(self, text: str, language: Optional[str] = None) -> tuple:
        """(language, voice name) that speak() will use for *text*, for display."""
        language = self._speak_language(text, language, log=False)
        server = self._tts_servers.get(language)
        if server is not None:
            return language, server.get("voice") or "server default"
        return language, "Piper"

    def _http_synth(self, text: str, language: Optional[str], server: dict):
        """Synthesize via an OpenAI-compatible /v1/audio/speech server.

        Returns (float32 mono numpy array, sample_rate). Raises on any
        network/decode error so the caller can fall back to piper.
        """
        import io
        import wave
        import numpy as np
        import requests

        body = {"input": text, "response_format": "wav"}
        if server.get("voice"):
            body["voice"] = server["voice"]
        if server.get("speed"):
            body["speed"] = server["speed"]
        # Advisory, not authoritative: the client's per-turn language
        # detection is flaky (a Swedish turn mislabeled 'it' would make the
        # server speak Swedish text through the Italian pipeline). The server
        # detects the language from the reply text itself — lingua restricted
        # to its allow-list, with this hint only breaking low-confidence ties
        # (the machinery debugged in reachy_mini_conversation_app).
        if language:
            body["language_hint"] = language
        resp = requests.post(server["url"], json=body, timeout=60)
        resp.raise_for_status()
        with wave.open(io.BytesIO(resp.content)) as wf:
            if wf.getsampwidth() != 2 or wf.getnchannels() != 1:
                raise ValueError(
                    f"expected mono PCM16 wav, got {wf.getnchannels()}ch "
                    f"{wf.getsampwidth() * 8}-bit")
            sr = wf.getframerate()
            pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        return pcm.astype("float32") / 32768.0, sr

    def _play(self, sentences, voice: PiperVoice, language: Optional[str]):
        sentences = iter(sentences)
        first = next(sentences, "")

        def chunks():
            for sentence in itertools.chain([first], sentences):
                text = self._apply_pronunciations(sentence, language)
                for chunk in voice.synthesize(text):
                    yield chunk.audio_float_array

        self._play_stream(first, chunks(), voice.config.sample_rate)

    def _play_stream(self, text: str, chunks, sr: int, events: bool = True):
        if len(self.out_waveform) != sr:
            # keep the scope window at exactly 1s for this stream's rate
            self.out_waveform = np.zeros(sr, dtype=np.float32)
        buffer = collections.deque()
        finished = threading.Event()
        stop = self._stop_event
        play_start = time.time()
        first_audio = {}   # 't' set (in the audio thread) at the first audible sample

        def callback(outdata, frames, time_info, status):
            if stop.is_set():
                outdata[:, 0] = 0
                raise sd.CallbackStop
            if buffer:
                if "t" not in first_audio:        # first real sample leaving the device
                    first_audio["t"] = time.time()
                chunk = buffer.popleft()
                if len(chunk) < frames:
                    outdata[:len(chunk), 0] = chunk
                    outdata[len(chunk):, 0] = 0
                else:
                    outdata[:, 0] = chunk[:frames]
                    if len(chunk) > frames:
                        buffer.appendleft(chunk[frames:])
            else:
                outdata[:, 0] = 0
                if finished.is_set():
                    raise sd.CallbackStop
            n = min(frames, len(self.out_waveform))
            self.out_waveform = np.roll(self.out_waveform, -n)
            self.out_waveform[-n:] = outdata[-n:, 0]
            self.out_last_write = time.time()

        emitted = False

        def _maybe_emit_audio_start():
            nonlocal emitted
            if events and not emitted and "t" in first_audio:
                ts = first_audio["t"]
                self._emit(TtsEventType.TTS_AUDIO_START,
                           TtsAudioStartPayload(text, ts - play_start, ts))
                emitted = True

        stream = sd.OutputStream(
            samplerate=sr, channels=1, dtype="float32",
            blocksize=4096, callback=callback,
        )
        stream.start()
        for audio_chunk in chunks:
            if stop.is_set():
                break
            buffer.append(audio_chunk)
            _maybe_emit_audio_start()
        finished.set()
        while stream.active:
            _maybe_emit_audio_start()
            if stop.is_set():
                stream.abort()
                break
            sd.sleep(50)
        stream.close()
        if stop.is_set():
            self._interrupted = True
            logger.info("[TTS] Playback interrupted by stop_speaking()")

    def _emit(self, etype, payload):
        event = TtsEvent(type=etype, timestamp=time.time(), payload=payload)
        logger.info(f"[{etype.name}] {payload}")
        self._dispatcher.dispatch(event)


# ---------------------------------------------------------------------------
# Standalone mode
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Standalone voice output (TTS)")
    parser.add_argument("text", nargs="?", help="Text to speak (omit for interactive)")
    parser.add_argument("--language", "-l", default=None,
                        choices=sorted(LANGUAGE_MODELS.keys()),
                        help=f"Language code (one of: {', '.join(sorted(LANGUAGE_MODELS.keys()))})")
    parser.add_argument("--model-dir", default=PIPER_MODEL_DIR)
    parser.add_argument("--model-name", default=None,
                        help="Override Piper model name (default: from --language)")
    parser.add_argument("--no-pronunciations", "-P", action="store_true",
                        help="Disable pronunciation replacements")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode: type text, press Enter to speak")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    # Validate language
    if args.language and args.language not in LANGUAGE_MODELS:
        print(f"Unknown language: {args.language}  "
              f"(available: {', '.join(sorted(LANGUAGE_MODELS))})")
        return

    # Resolve model name: explicit --model-name > language lookup > default
    model_name = args.model_name
    if model_name is None:
        model_name = LANGUAGE_MODELS.get(args.language, PIPER_MODEL_NAME)

    tts = VoiceOutput(model_dir=args.model_dir, model_name=model_name)
    if args.no_pronunciations:
        tts._lang_pron = {}

    def on_event(event: TtsEvent):
        ts = datetime.now().strftime("%H:%M:%S")
        if event.type == TtsEventType.TTS_STARTED:
            print(f"  [{ts}] Speaking: \"{event.payload.text}\"")
            if event.payload.tts_text:
                print(f"  [{ts}] Pronounce: \"{event.payload.tts_text}\"")
        elif event.type == TtsEventType.TTS_FINISHED:
            print(f"  [{ts}] Done ({event.payload.duration_s:.1f}s)")
        elif event.type == TtsEventType.MODEL_READY:
            print(f"  [{ts}] Model loaded: {event.payload.model_name}")

    tts.subscribe(on_event)

    print("Loading TTS model...")
    tts.load_sync()
    if not tts.ready:
        print("Failed to load TTS model.")
        return

    language = args.language

    if args.text and not args.interactive:
        tts.speak(args.text, language=language)
    else:
        print("Type text and press Enter to speak. Ctrl+C to quit.\n")
        try:
            while True:
                text = input("> ")
                if text.strip():
                    tts.speak(text.strip(), language=language)
        except (KeyboardInterrupt, EOFError):
            pass

    print("\nDone.")


if __name__ == "__main__":
    main()
