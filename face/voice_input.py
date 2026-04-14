"""
Voice input module: microphone monitoring, VAD, recording, and transcription.

Provides:
- AudioMonitor: real-time mic level (RMS, peak, dB)
- VoiceInput: VAD-based speech detection + whisper transcription
- ContinuousListener: always-on background listening
- Typed event system with subscribe/unsubscribe

Does NOT handle TTS (text-to-speech) or conversation logic.

Can be run standalone:
    python voice_input.py [--whisper-model base] [--vad-threshold 0.7]
"""

import numpy as np
import os
import time
import threading
import logging
import argparse
import collections
from enum import Enum, auto
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Callable, Union

import sounddevice as sd
from faster_whisper import WhisperModel

from events import EventDispatcher

logger = logging.getLogger("voice_input")

# Default constants
SAMPLE_RATE = 16000
RECORD_SECONDS = 4
VAD_THRESHOLD = 0.7
VAD_SILENCE_MS = 1200
VAD_MAX_SPEECH_S = 15
VAD_PRE_SPEECH_MS = 500
AUDIO_METER_DECAY = 0.92
NOISE_REDUCE = True


# ---------------------------------------------------------------------------
# Event types and payloads
# ---------------------------------------------------------------------------

class VoiceEventType(Enum):
    # Model lifecycle
    MODEL_LOADING = auto()
    MODEL_READY = auto()
    MODEL_LOAD_FAILED = auto()

    # VAD
    VAD_SPEECH_START = auto()
    VAD_SPEECH_END = auto()
    VAD_TIMEOUT = auto()

    # Transcription pipeline
    RECORDING_STARTED = auto()
    TRANSCRIPTION_STARTED = auto()
    TRANSCRIPTION_SEGMENT = auto()
    TRANSCRIPTION_COMPLETE = auto()
    TRANSCRIPTION_EMPTY = auto()
    LISTEN_FAILED = auto()

    # Continuous listener
    CONTINUOUS_STARTED = auto()
    CONTINUOUS_STOPPED = auto()
    CONTINUOUS_PAUSED = auto()
    CONTINUOUS_RESUMED = auto()


@dataclass(frozen=True)
class ModelLoadingPayload:
    model_name: str  # "whisper" or "vad"


@dataclass(frozen=True)
class ModelReadyPayload:
    model_name: str
    detail: str  # e.g. "base (int8)"


@dataclass(frozen=True)
class ModelLoadFailedPayload:
    model_name: str
    error: str


@dataclass(frozen=True)
class VadSpeechStartPayload:
    speech_prob: float


@dataclass(frozen=True)
class VadSpeechEndPayload:
    duration_ms: int
    speech_prob: float


@dataclass(frozen=True)
class VadTimeoutPayload:
    reason: str  # "max_speech_length" or "no_speech_detected"
    waited_ms: int


@dataclass(frozen=True)
class RecordingStartedPayload:
    mode: str  # "vad" or "fixed"


@dataclass(frozen=True)
class TranscriptionStartedPayload:
    audio_duration_ms: int
    noise_reduced: bool


@dataclass(frozen=True)
class TranscriptionSegmentPayload:
    segment_text: str
    cumulative_text: str
    start_time: float
    end_time: float


@dataclass(frozen=True)
class TranscriptionCompletePayload:
    text: str
    language: str
    language_probability: float
    audio_duration_ms: int


@dataclass(frozen=True)
class TranscriptionEmptyPayload:
    reason: str


@dataclass(frozen=True)
class ListenFailedPayload:
    error: str
    fallback_used: bool


@dataclass(frozen=True)
class ContinuousStatePayload:
    pass


VoiceEventPayload = Union[
    ModelLoadingPayload, ModelReadyPayload, ModelLoadFailedPayload,
    VadSpeechStartPayload, VadSpeechEndPayload, VadTimeoutPayload,
    RecordingStartedPayload, TranscriptionStartedPayload,
    TranscriptionSegmentPayload, TranscriptionCompletePayload,
    TranscriptionEmptyPayload, ListenFailedPayload,
    ContinuousStatePayload,
]


@dataclass(frozen=True)
class VoiceEvent:
    type: VoiceEventType
    timestamp: float
    payload: VoiceEventPayload


VoiceEventCallback = Callable[[VoiceEvent], None]


# ---------------------------------------------------------------------------
# AudioMonitor
# ---------------------------------------------------------------------------

class AudioMonitor:
    """Monitors microphone input levels (RMS, peak, dB)."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, decay: float = AUDIO_METER_DECAY):
        self.rms: float = 0.0
        self.peak: float = 0.0
        self.max_seen: float = 0.001
        self._sample_rate = sample_rate
        self._decay = decay
        self._stream = None

    def start(self):
        blocksize = int(self._sample_rate * 0.05)

        def callback(indata, frames, time_info, status):
            rms = float(np.sqrt(np.mean(indata ** 2)))
            self.rms = rms
            if rms > self.max_seen:
                self.max_seen = rms
            if rms > self.peak:
                self.peak = rms
            else:
                self.peak = self.peak * self._decay

        self._stream = sd.InputStream(
            samplerate=self._sample_rate, channels=1, dtype='float32',
            blocksize=blocksize, callback=callback
        )
        self._stream.start()

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    @property
    def db(self) -> float:
        return 20 * np.log10(self.rms + 1e-10)


# ---------------------------------------------------------------------------
# EchoDetector — lightweight mic monitor for barge-in during TTS
# ---------------------------------------------------------------------------

class EchoDetector:
    """WebRTC AEC-based barge-in detector for TTS playback.

    Uses a full-duplex ``sd.Stream`` to play TTS and record mic
    simultaneously. The TTS render reference is fed to
    ``livekit.rtc.AudioProcessingModule`` (WebRTC AEC3 + noise
    suppression) which subtracts the echo. The residual signal is
    the user's voice (if any). When the residual RMS exceeds a
    threshold, ``user_speaking`` goes True.

    Architecture::

        TTS audio ──► speakers (via sd.Stream output)
            │
            └──► APM.process_reverse_stream()  (render reference)
                                │
        mic ──► APM.process_stream() ──► clean audio ──► RMS check
    """

    def __init__(self,
                 stream_rate: int = SAMPLE_RATE,
                 speech_threshold: float = 0.08,
                 min_duration_ms: int = 200):
        self._stream_rate = stream_rate
        self._speech_threshold = speech_threshold
        self._min_chunks = max(1, min_duration_ms // 10)  # 10ms frames
        self._stream: Optional[sd.Stream] = None
        self._output_buffer: collections.deque[np.ndarray] = collections.deque()
        self._output_finished = False
        self._stop_event = threading.Event()
        self._above_count: int = 0

        # Observable state
        self.user_speaking: bool = False
        self.current_rms: float = 0.0       # raw mic RMS
        self.clean_rms: float = 0.0         # RMS after echo cancellation
        self.output_rms: float = 0.0        # TTS output RMS
        self.playing: bool = False

        # Waveform buffers for oscilloscope display (last ~100ms)
        self._wave_len = 1600  # 100ms at 16kHz
        self.wave_output: np.ndarray = np.zeros(self._wave_len, dtype=np.float32)
        self.wave_raw: np.ndarray = np.zeros(self._wave_len, dtype=np.float32)
        self.wave_clean: np.ndarray = np.zeros(self._wave_len, dtype=np.float32)

        # For display/logging compatibility with agent's _draw_echo_state
        self._baseline: float = 0.0
        self._factor: float = speech_threshold
        self._calibrated: bool = True

    def feed(self, audio_chunk: np.ndarray):
        """Feed a TTS audio chunk for playback."""
        self._output_buffer.append(audio_chunk)

    def finish_feeding(self):
        """Signal that all TTS audio has been fed."""
        self._output_finished = True

    def start(self, tts_sample_rate: int = 22050):
        """Start the full-duplex stream with WebRTC AEC."""
        from livekit.rtc import AudioProcessingModule, AudioFrame

        self.user_speaking = False
        self._above_count = 0
        self._stop_event.clear()
        self._output_finished = False
        self.playing = True

        apm = AudioProcessingModule(
            echo_cancellation=True,
            noise_suppression=False,  # off — was suppressing user voice too
            high_pass_filter=True,
            auto_gain_control=False,
        )
        # Tell the AEC the approximate speaker→mic delay so it can
        # properly separate echo from user voice during double-talk.
        # ~50ms is typical for laptop speakers→mic on macOS.
        apm.set_stream_delay_ms(50)

        sr = self._stream_rate
        # WebRTC APM requires exactly 10ms frames
        frame_samples = sr // 100  # 160 samples at 16kHz

        def callback(indata, outdata, frames, time_info, status):
            if self._stop_event.is_set():
                outdata[:, 0] = 0
                raise sd.CallbackStop

            # --- Output: play from buffer ---
            if self._output_buffer:
                chunk = self._output_buffer.popleft()
                if len(chunk) <= frames:
                    outdata[:len(chunk), 0] = chunk
                    outdata[len(chunk):, 0] = 0
                else:
                    outdata[:, 0] = chunk[:frames]
                    self._output_buffer.appendleft(chunk[frames:])
            else:
                outdata[:, 0] = 0
                if self._output_finished:
                    raise sd.CallbackStop

            # --- Process in 10ms frames through WebRTC APM ---
            inp = indata[:, 0]
            out = outdata[:, 0]
            self.current_rms = float(np.sqrt(np.mean(inp ** 2)))
            self.output_rms = float(np.sqrt(np.mean(out ** 2)))
            self._baseline = self.output_rms

            # Update waveform ring buffers
            n = len(inp)
            wl = self._wave_len
            self.wave_output = np.roll(self.wave_output, -n)
            self.wave_output[-n:] = out[:n]
            self.wave_raw = np.roll(self.wave_raw, -n)
            self.wave_raw[-n:] = inp[:n]

            all_clean = []
            max_clean_rms = 0.0
            for i in range(0, min(len(inp), len(out)), frame_samples):
                end = i + frame_samples
                if end > len(inp) or end > len(out):
                    break

                # Convert float32 [-1,1] to int16 for APM
                out_i16 = (out[i:end] * 32767).astype(np.int16)
                inp_i16 = (inp[i:end] * 32767).astype(np.int16)

                # Feed render reference (what the speakers are playing)
                ref_frame = AudioFrame(
                    out_i16.tobytes(), sample_rate=sr,
                    num_channels=1, samples_per_channel=frame_samples)
                apm.process_reverse_stream(ref_frame)

                # Process mic input (echo cancellation happens in-place)
                mic_frame = AudioFrame(
                    bytearray(inp_i16.tobytes()), sample_rate=sr,
                    num_channels=1, samples_per_channel=frame_samples)
                apm.process_stream(mic_frame)

                # Check residual for user speech
                clean_i16 = np.frombuffer(mic_frame.data, dtype=np.int16)
                clean_f32 = clean_i16.astype(np.float32) / 32767.0
                all_clean.append(clean_f32)
                frame_rms = float(np.sqrt(np.mean(clean_f32 ** 2)))
                max_clean_rms = max(max_clean_rms, frame_rms)

                # Per-frame barge-in check (10ms resolution)
                if frame_rms > self._speech_threshold:
                    self._above_count += 1
                    if self._above_count >= self._min_chunks and not self.user_speaking:
                        self.user_speaking = True
                        logger.info(
                            f"[AEC] barge-in: clean_rms={frame_rms:.4f} "
                            f"threshold={self._speech_threshold}")
                else:
                    self._above_count = max(0, self._above_count - 1)

            self.clean_rms = max_clean_rms
            if all_clean:
                clean_all = np.concatenate(all_clean)
                nc = len(clean_all)
                self.wave_clean = np.roll(self.wave_clean, -nc)
                self.wave_clean[-nc:] = clean_all[:nc]

        self._stream = sd.Stream(
            samplerate=sr, channels=1, dtype='float32',
            blocksize=frame_samples * 10,  # 100ms blocks (10 APM frames)
            callback=callback,
        )
        self._stream.start()
        logger.info(f"[AEC] started full-duplex stream at {sr}Hz, "
                    f"threshold={self._speech_threshold}")

    def stop(self, beep: bool = False):
        """Stop playback and close the stream.

        If ``beep`` is True, plays a short confirmation tone (~100ms)
        through the default output device after stopping, so the user
        gets audible feedback that their barge-in was detected.
        """
        self._stop_event.set()
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self.playing = False
        if beep:
            self._play_beep()

    @staticmethod
    def _play_beep(freq: float = 880, duration_ms: int = 80,
                   volume: float = 0.3, sample_rate: int = SAMPLE_RATE):
        """Play a short confirmation beep."""
        try:
            t = np.linspace(0, duration_ms / 1000, int(sample_rate * duration_ms / 1000),
                            dtype=np.float32)
            # Sine with quick fade-in/out to avoid click
            tone = np.sin(2 * np.pi * freq * t) * volume
            fade = min(len(tone) // 4, 50)
            tone[:fade] *= np.linspace(0, 1, fade)
            tone[-fade:] *= np.linspace(1, 0, fade)
            sd.play(tone, samplerate=sample_rate, blocking=True)
        except Exception:
            pass

    @property
    def active(self) -> bool:
        return self._stream is not None and self._stream.active


# ---------------------------------------------------------------------------
# VoiceInput
# ---------------------------------------------------------------------------

class VoiceInput:
    """Speech-to-text engine: VAD + whisper transcription.

    Subscribe to VoiceEvent callbacks via subscribe().
    """

    def __init__(self, *,
                 whisper_model_size: str = "medium",
                 whisper_compute_type: str = "float16",
                 sample_rate: int = SAMPLE_RATE,
                 vad_threshold: float = VAD_THRESHOLD,
                 vad_silence_ms: int = VAD_SILENCE_MS,
                 vad_max_speech_s: float = VAD_MAX_SPEECH_S,
                 vad_pre_speech_ms: int = VAD_PRE_SPEECH_MS,
                 noise_reduce: bool = NOISE_REDUCE,
                 record_seconds: int = RECORD_SECONDS):
        self._whisper_size = whisper_model_size
        self._whisper_compute = whisper_compute_type
        self._sample_rate = sample_rate
        self._vad_threshold = vad_threshold
        self._vad_silence_ms = vad_silence_ms
        self._vad_max_speech_s = vad_max_speech_s
        self._vad_pre_speech_ms = vad_pre_speech_ms
        self._noise_reduce = noise_reduce
        self._record_seconds = record_seconds

        self._whisper_model = None
        self._vad_model = None
        self._loading = False
        self._ready = False
        self._lock = threading.Lock()

        # Observable state (for GUI polling)
        self.vad_prob: float = 0.0
        self.listen_phase: str = ""  # "", "waiting", "recording", "transcribing"
        self._cancel_listen: bool = False
        self.detected_language: str = ""
        self.detected_language_prob: float = 0.0

        self._dispatcher = EventDispatcher(owner="voice_input")

    # --- Event API ---

    def subscribe(self, callback: VoiceEventCallback,
                  event_types: Optional[set] = None) -> Callable[[], None]:
        return self._dispatcher.subscribe(callback, event_types)

    def unsubscribe(self, callback: VoiceEventCallback) -> bool:
        return self._dispatcher.unsubscribe(callback)

    @property
    def vad_threshold(self) -> float:
        return self._vad_threshold

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def loading(self) -> bool:
        return self._loading

    # --- Lifecycle ---

    def load(self):
        """Start async model loading."""
        self._loading = True
        threading.Thread(target=self._load_models, daemon=True).start()

    def load_sync(self):
        """Load models synchronously (blocking)."""
        self._loading = True
        self._load_models()

    def _load_models(self):
        self._emit(VoiceEventType.MODEL_LOADING, ModelLoadingPayload("whisper"))
        try:
            logger.info(f"Loading whisper model ({self._whisper_size})...")
            self._whisper_model = WhisperModel(self._whisper_size,
                                               compute_type=self._whisper_compute,
                                               device="cuda")
            self._emit(VoiceEventType.MODEL_READY,
                       ModelReadyPayload("whisper", f"{self._whisper_size} ({self._whisper_compute})"))
            logger.info("Whisper model loaded.")
        except Exception as e:
            self._emit(VoiceEventType.MODEL_LOAD_FAILED,
                       ModelLoadFailedPayload("whisper", str(e)))
            logger.error(f"Failed to load whisper: {e}")

        self._ready = self._whisper_model is not None
        self._loading = False

    def _ensure_vad(self):
        if self._vad_model is not None:
            return
        import torch
        self._emit(VoiceEventType.MODEL_LOADING, ModelLoadingPayload("vad"))
        logger.info("Loading Silero VAD model...")
        self._vad_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad', model='silero_vad',
            trust_repo=True
        )
        self._emit(VoiceEventType.MODEL_READY, ModelReadyPayload("vad", "silero"))
        logger.info("Silero VAD loaded.")

    # --- Core operations ---

    def listen(self, seconds: int = None, on_segment: Callable = None) -> Optional[str]:
        """Blocking STT. Tries VAD, falls back to fixed recording.
        Returns transcribed text or None."""
        if not self._ready:
            logger.warning("STT not ready")
            return None

        if not self._lock.acquire(timeout=1):
            logger.warning("Voice input busy, skipping listen")
            return None
        try:
            try:
                return self._listen_vad(on_segment=on_segment)
            except Exception as e:
                logger.warning(f"VAD listen failed: {e}, falling back to fixed recording")
                self._emit(VoiceEventType.LISTEN_FAILED,
                           ListenFailedPayload(str(e), fallback_used=True))
                secs = seconds or self._record_seconds
                return self._listen_fixed(secs, on_segment=on_segment)
        finally:
            self._lock.release()

    def _listen_vad(self, on_segment=None):
        import torch

        self._ensure_vad()
        self.listen_phase = "waiting"
        self.vad_prob = 0.0
        self._cancel_listen = False

        vad_model = self._vad_model
        vad_model.reset_states()

        chunk_ms = 32
        chunk_samples = 512
        silence_chunks_needed = int(self._vad_silence_ms / chunk_ms)
        max_chunks = int(self._vad_max_speech_s * 1000 / chunk_ms)
        pre_speech_chunks = int(self._vad_pre_speech_ms / chunk_ms)

        pre_buffer = collections.deque(maxlen=pre_speech_chunks)
        speech_chunks = []
        silence_count = 0
        speech_started = False
        total_chunks = 0

        self._emit(VoiceEventType.RECORDING_STARTED, RecordingStartedPayload("vad"))

        stream = sd.InputStream(
            samplerate=self._sample_rate, channels=1, dtype='float32',
            blocksize=chunk_samples
        )
        stream.start()

        try:
            while True:
                if self._cancel_listen:
                    logger.info("[VAD] listen cancelled")
                    return None

                data, _ = stream.read(chunk_samples)
                chunk = data[:, 0]
                total_chunks += 1

                tensor = torch.from_numpy(chunk)
                speech_prob = vad_model(tensor, self._sample_rate).item()
                self.vad_prob = speech_prob

                if not speech_started:
                    pre_buffer.append(chunk.copy())
                    if speech_prob >= self._vad_threshold:
                        speech_started = True
                        self.listen_phase = "recording"
                        silence_count = 0
                        speech_chunks.extend(pre_buffer)
                        speech_chunks.append(chunk.copy())
                        self._emit(VoiceEventType.VAD_SPEECH_START,
                                   VadSpeechStartPayload(speech_prob))
                        if on_segment:
                            on_segment("(listening...)")
                    elif total_chunks * chunk_ms > 8000:
                        self._emit(VoiceEventType.VAD_TIMEOUT,
                                   VadTimeoutPayload("no_speech_detected",
                                                     total_chunks * chunk_ms))
                        return None
                else:
                    speech_chunks.append(chunk.copy())
                    if speech_prob < self._vad_threshold:
                        silence_count += 1
                        if silence_count >= silence_chunks_needed:
                            duration_ms = len(speech_chunks) * chunk_ms
                            self._emit(VoiceEventType.VAD_SPEECH_END,
                                       VadSpeechEndPayload(duration_ms, speech_prob))
                            break
                    else:
                        silence_count = 0

                    if len(speech_chunks) >= max_chunks:
                        duration_ms = len(speech_chunks) * chunk_ms
                        self._emit(VoiceEventType.VAD_TIMEOUT,
                                   VadTimeoutPayload("max_speech_length", duration_ms))
                        break
        finally:
            stream.stop()
            stream.close()

        if not speech_chunks:
            self.listen_phase = ""
            self.vad_prob = 0.0
            self._emit(VoiceEventType.TRANSCRIPTION_EMPTY,
                       TranscriptionEmptyPayload("no_speech"))
            return None

        self.listen_phase = "transcribing"
        self.vad_prob = 0.0
        audio = np.concatenate(speech_chunks)
        audio_duration_ms = int(len(audio) / self._sample_rate * 1000)

        noise_reduced = False
        if self._noise_reduce:
            try:
                import noisereduce as nr
                audio = nr.reduce_noise(y=audio, sr=self._sample_rate,
                                        stationary=True, prop_decrease=0.75)
                noise_reduced = True
            except Exception as e:
                logger.warning(f"Noise reduction failed: {e}")

        self._emit(VoiceEventType.TRANSCRIPTION_STARTED,
                   TranscriptionStartedPayload(audio_duration_ms, noise_reduced))

        segments, info = self._whisper_model.transcribe(audio, beam_size=5)
        self.detected_language = info.language
        self.detected_language_prob = info.language_probability

        full_text = ""
        for segment in segments:
            full_text += segment.text
            self._emit(VoiceEventType.TRANSCRIPTION_SEGMENT,
                       TranscriptionSegmentPayload(
                           segment.text.strip(), full_text.strip(),
                           segment.start, segment.end))
            if on_segment:
                on_segment(full_text.strip())

        text = full_text.strip()
        self.listen_phase = ""

        if text:
            self._emit(VoiceEventType.TRANSCRIPTION_COMPLETE,
                       TranscriptionCompletePayload(
                           text, info.language, info.language_probability,
                           audio_duration_ms))
        else:
            self._emit(VoiceEventType.TRANSCRIPTION_EMPTY,
                       TranscriptionEmptyPayload("empty_segments"))

        return text or None

    def _listen_fixed(self, seconds, on_segment=None):
        self.listen_phase = "recording"
        self._emit(VoiceEventType.RECORDING_STARTED, RecordingStartedPayload("fixed"))

        audio = sd.rec(int(seconds * self._sample_rate),
                       samplerate=self._sample_rate, channels=1, dtype="float32")
        sd.wait()
        audio = audio.flatten()
        self.listen_phase = "transcribing"
        audio_duration_ms = int(len(audio) / self._sample_rate * 1000)

        noise_reduced = False
        if self._noise_reduce:
            try:
                import noisereduce as nr
                audio = nr.reduce_noise(y=audio, sr=self._sample_rate,
                                        stationary=True, prop_decrease=0.75)
                noise_reduced = True
            except Exception as e:
                logger.warning(f"Noise reduction failed: {e}")

        self._emit(VoiceEventType.TRANSCRIPTION_STARTED,
                   TranscriptionStartedPayload(audio_duration_ms, noise_reduced))

        segments, info = self._whisper_model.transcribe(audio, beam_size=5)
        self.detected_language = info.language
        self.detected_language_prob = info.language_probability

        full_text = ""
        for segment in segments:
            full_text += segment.text
            self._emit(VoiceEventType.TRANSCRIPTION_SEGMENT,
                       TranscriptionSegmentPayload(
                           segment.text.strip(), full_text.strip(),
                           segment.start, segment.end))
            if on_segment:
                on_segment(full_text.strip())

        text = full_text.strip()
        self.listen_phase = ""

        if text:
            self._emit(VoiceEventType.TRANSCRIPTION_COMPLETE,
                       TranscriptionCompletePayload(
                           text, info.language, info.language_probability,
                           audio_duration_ms))
        else:
            self._emit(VoiceEventType.TRANSCRIPTION_EMPTY,
                       TranscriptionEmptyPayload("empty_segments"))

        return text or None

    def _emit(self, etype, payload):
        event = VoiceEvent(type=etype, timestamp=time.time(), payload=payload)
        logger.info(f"[{etype.name}] {payload}")
        self._dispatcher.dispatch(event)


# ---------------------------------------------------------------------------
# ContinuousListener
# ---------------------------------------------------------------------------

class ContinuousListener:
    """Background thread that continuously listens and emits events on speech."""

    def __init__(self, voice_input: VoiceInput, *,
                 on_heard: Optional[Callable[[str], None]] = None):
        self.voice = voice_input
        self.on_heard = on_heard
        self._running = False
        self._paused = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.voice._emit(VoiceEventType.CONTINUOUS_STARTED, ContinuousStatePayload())

    def stop(self):
        self._running = False
        self.voice._cancel_listen = True
        self.voice._emit(VoiceEventType.CONTINUOUS_STOPPED, ContinuousStatePayload())
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    @property
    def paused(self):
        return self._paused

    @paused.setter
    def paused(self, val):
        old = self._paused
        self._paused = val
        if val and not old:
            self.voice._emit(VoiceEventType.CONTINUOUS_PAUSED, ContinuousStatePayload())
        elif not val and old:
            self.voice._emit(VoiceEventType.CONTINUOUS_RESUMED, ContinuousStatePayload())

    def _run(self):
        while self._running and not self.voice.ready:
            time.sleep(0.5)
        logger.info("Continuous listener started")
        while self._running:
            if self._paused:
                time.sleep(0.1)
                continue
            try:
                text = self.voice.listen()
                if text and self.on_heard and not self._paused:
                    self.on_heard(text)
                time.sleep(0.3)
            except Exception as e:
                logger.error(f"Continuous listener error: {e}")
                time.sleep(1)


# ---------------------------------------------------------------------------
# Standalone mode
# ---------------------------------------------------------------------------

_EVENT_SHORT_NAMES = {
    "MODEL_LOADING": "LOADING",
    "MODEL_READY": "READY",
    "MODEL_LOAD_FAILED": "FAIL",
    "VAD_SPEECH_START": "SPEECH",
    "VAD_SPEECH_END": "END",
    "VAD_TIMEOUT": "TIMEOUT",
    "RECORDING_STARTED": "REC",
    "TRANSCRIPTION_STARTED": "TRANSCR",
    "TRANSCRIPTION_SEGMENT": "SEGMENT",
    "TRANSCRIPTION_COMPLETE": "HEARD",
    "TRANSCRIPTION_EMPTY": "EMPTY",
    "LISTEN_FAILED": "FAIL",
    "CONTINUOUS_STARTED": "CONT ON",
    "CONTINUOUS_STOPPED": "CONT OFF",
    "CONTINUOUS_PAUSED": "PAUSED",
    "CONTINUOUS_RESUMED": "RESUMED",
}

_EVENT_COLORS = {
    "MODEL_LOADING": (180, 180, 180),
    "MODEL_READY": (120, 255, 120),
    "MODEL_LOAD_FAILED": (100, 100, 255),
    "VAD_SPEECH_START": (120, 255, 255),
    "VAD_SPEECH_END": (200, 200, 120),
    "VAD_TIMEOUT": (180, 180, 120),
    "RECORDING_STARTED": (255, 200, 100),
    "TRANSCRIPTION_STARTED": (200, 200, 200),
    "TRANSCRIPTION_SEGMENT": (200, 220, 255),
    "TRANSCRIPTION_COMPLETE": (120, 255, 120),
    "TRANSCRIPTION_EMPTY": (180, 120, 120),
    "LISTEN_FAILED": (100, 100, 255),
    "CONTINUOUS_STARTED": (120, 200, 255),
    "CONTINUOUS_STOPPED": (180, 180, 180),
    "CONTINUOUS_PAUSED": (220, 200, 120),
    "CONTINUOUS_RESUMED": (120, 255, 200),
}


def main():
    parser = argparse.ArgumentParser(description="Standalone voice input (STT)")
    parser.add_argument("--whisper-model", default="base", help="Whisper model size")
    parser.add_argument("--vad-threshold", type=float, default=0.7)
    parser.add_argument("--no-continuous", action="store_true",
                        help="Press Enter to listen instead of always-on")
    parser.add_argument("--no-noise-reduce", action="store_true")
    parser.add_argument("--echo", action="store_true",
                        help="Print transcriptions to stdout (for piping)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    voice = VoiceInput(
        whisper_model_size=args.whisper_model,
        vad_threshold=args.vad_threshold,
        noise_reduce=not args.no_noise_reduce,
    )

    # Event display
    def on_event(event: VoiceEvent):
        ts = datetime.now().strftime("%H:%M:%S")
        short = _EVENT_SHORT_NAMES.get(event.type.name, event.type.name[:10])
        p = event.payload
        if event.type == VoiceEventType.TRANSCRIPTION_COMPLETE:
            detail = f'"{p.text}" [{p.language} {p.language_probability:.0%}]'
        elif event.type == VoiceEventType.VAD_SPEECH_START:
            detail = f"prob={p.speech_prob:.2f}"
        elif event.type == VoiceEventType.VAD_SPEECH_END:
            detail = f"{p.duration_ms}ms"
        elif event.type == VoiceEventType.TRANSCRIPTION_SEGMENT:
            detail = f'"{p.segment_text}"'
        elif event.type == VoiceEventType.MODEL_READY:
            detail = f"{p.model_name}: {p.detail}"
        else:
            detail = str(p)[:60]

        if not args.echo:
            print(f"  [{ts}] {short:>10}  {detail}")

    voice.subscribe(on_event)

    # Echo mode: only print final transcriptions, clean for piping
    if args.echo:
        def on_transcription(event: VoiceEvent):
            print(event.payload.text, flush=True)
        voice.subscribe(on_transcription,
                        event_types={VoiceEventType.TRANSCRIPTION_COMPLETE})

    # Load models
    print("Loading models...")
    voice.load_sync()
    if not voice.ready:
        print("Failed to load models. Exiting.")
        return

    # Start audio monitor
    monitor = AudioMonitor(sample_rate=voice.sample_rate)
    monitor.start()

    if not args.no_continuous:
        print("Listening continuously. Speak anytime. Ctrl+C to quit.\n")

        def on_heard(text):
            if not args.echo:
                print(f"\n>>> {text}\n")

        listener = ContinuousListener(voice, on_heard=on_heard)
        listener.start()
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            listener.stop()
    else:
        print("Press Enter to listen, Ctrl+C to quit.\n")
        try:
            while True:
                input("  [Press Enter to listen] ")
                text = voice.listen()
                if text:
                    print(f"\n>>> {text}\n")
                else:
                    print("  (no speech detected)\n")
        except (KeyboardInterrupt, EOFError):
            pass

    monitor.stop()
    print("\nDone.")


if __name__ == "__main__":
    main()
