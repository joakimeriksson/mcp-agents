"""End-to-end voice latency tracker.

Measures, per spoken turn, the wall-clock time from when the user STOPS
talking (``VAD_SPEECH_END``) to when the assistant's voice actually STARTS
(``TTS_AUDIO_START``) — the number you feel as "how long until it answers".

It correlates events already emitted by ``VoiceInput`` and ``VoiceOutput``
(plus the new ``TTS_AUDIO_START``) and splits the total into stages:

    speech-end ──STT──► transcribed ──LLM──► speak ──TTS──► audio-start

- **STT**  = transcription (Gemma native audio, incl. noise reduction)
- **LLM**  = reply generation + the agent's decision to speak
- **TTS**  = synth time to the first audible sample

Greeting / name-asking speech that has no preceding ``VAD_SPEECH_END`` is
ignored (no user utterance to measure against).

    tracker = LatencyTracker(voice_in, voice_out)   # logs each turn
"""

import logging
from typing import Callable, Optional

from voice_input import VoiceInput, VoiceEventType
from voice_output import VoiceOutput, TtsEventType

logger = logging.getLogger("latency")


class LatencyTracker:
    def __init__(self, voice_in: VoiceInput, voice_out: VoiceOutput,
                 on_measure: Optional[Callable[[dict], None]] = None):
        self._t_capture_end: Optional[float] = None
        self._t_transcribed: Optional[float] = None
        self._t_speak: Optional[float] = None
        self._on_measure = on_measure
        self.last: Optional[dict] = None
        voice_in.subscribe(self._on_voice_in)
        voice_out.subscribe(self._on_voice_out)

    def _on_voice_in(self, e):
        if e.type == VoiceEventType.VAD_SPEECH_END:
            # New utterance — start a fresh measurement window.
            self._t_capture_end = e.timestamp
            self._t_transcribed = None
            self._t_speak = None
        elif e.type == VoiceEventType.TRANSCRIPTION_COMPLETE:
            self._t_transcribed = e.timestamp

    def _on_voice_out(self, e):
        if e.type == TtsEventType.TTS_STARTED:
            self._t_speak = e.timestamp
        elif e.type == TtsEventType.TTS_AUDIO_START:
            self._report(getattr(e.payload, "audio_start_ts", e.timestamp))

    def _report(self, t_audio: float):
        t0 = self._t_capture_end
        if t0 is None:
            return  # speech with no preceding user utterance (e.g. a greeting)

        total = t_audio - t0
        stt = (self._t_transcribed - t0) if self._t_transcribed else None
        llm = (self._t_speak - self._t_transcribed) \
            if (self._t_speak and self._t_transcribed) else None
        tts = (t_audio - self._t_speak) if self._t_speak else None

        self.last = {"total": total, "stt": stt, "llm": llm, "tts": tts}

        def f(x):
            return f"{x:.2f}s" if x is not None else "?"

        logger.info(
            f"⏱️  voice latency: speech-end → audio-start = {total:.2f}s "
            f"(STT {f(stt)} | LLM {f(llm)} | TTS {f(tts)})")

        if self._on_measure:
            try:
                self._on_measure(self.last)
            except Exception as ex:
                logger.warning(f"on_measure callback failed: {ex}")

        # Consume the window so a follow-up TTS (same turn) isn't double-counted.
        self._t_capture_end = None
