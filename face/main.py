"""
Face Agent UI — thin visual shell around the Agent.

All intelligence (greetings, responses, name asking, goodbye) lives in
agent.py.  This file only handles:
  - OpenCV camera display and face box drawing
  - Audio meter and event log overlays
  - Keyboard controls (L=learn, T=talk, C=continuous, A=auto-ask, etc.)
"""

import cv2
import numpy as np
import os
import sys
import time
import threading
import logging
import argparse
from datetime import datetime

from face_tracker import FaceTracker, FaceDatabase, EmotionDetector
from voice_input import VoiceInput, AudioMonitor
from voice_output import VoiceOutput
from people_memory import PeopleMemory
from agent import Agent, AgentEvent, AgentEventType
from llm import ConversationLLM

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_DIR = os.path.join(_SOURCE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"face_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logger = logging.getLogger("face_app")
logger.setLevel(logging.DEBUG)
_fh = logging.FileHandler(LOG_FILE)
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(_fh)
_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(_ch)

# ---------------------------------------------------------------------------
# Event log (on-screen ring buffer)
# ---------------------------------------------------------------------------

class EventLog:
    """In-memory ring buffer of recent log events for on-screen display."""

    def __init__(self, max_entries=200):
        self._entries = []
        self._max = max_entries
        self._lock = threading.Lock()

    def add(self, category, message, detail=None):
        ts = datetime.now().strftime("%H:%M:%S")
        entry = {"ts": ts, "cat": category, "msg": message}
        if detail:
            entry["detail"] = detail
        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self._max:
                self._entries.pop(0)
        detail_str = f" | {detail}" if detail else ""
        logger.info(f"[{category.upper()}] {message}{detail_str}")

    def recent(self, n=12):
        with self._lock:
            return list(self._entries[-n:])


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_audio_meter(frame, audio_monitor, voice_engine=None):
    h, w = frame.shape[:2]
    mx, my = 15, 70
    mw, mh = 14, h - 140

    max_val = max(audio_monitor.max_seen, 0.001)
    level = min(1.0, audio_monitor.rms / max_val)
    peak = min(1.0, audio_monitor.peak / max_val)

    overlay = frame.copy()
    cv2.rectangle(overlay, (mx, my), (mx + mw, my + mh), (30, 30, 30), cv2.FILLED)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    bar_h = int(level * mh)
    if bar_h > 0:
        for y in range(bar_h):
            frac = y / mh
            if frac < 0.6:
                color = (0, 200, 0)
            elif frac < 0.85:
                color = (0, 220, 220)
            else:
                color = (0, 0, 255)
            y_pos = my + mh - y
            cv2.line(frame, (mx + 1, y_pos), (mx + mw - 1, y_pos), color, 1)

    peak_y = my + mh - int(peak * mh)
    cv2.line(frame, (mx - 2, peak_y), (mx + mw + 2, peak_y), (255, 255, 255), 2)
    cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (100, 100, 100), 1)

    db = 20 * np.log10(audio_monitor.rms + 1e-10)
    cv2.putText(frame, f"{db:.0f}dB", (mx - 2, my - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

    if voice_engine:
        vx = mx + mw + 8
        vw = 14

        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (vx, my), (vx + vw, my + mh), (30, 30, 30), cv2.FILLED)
        cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)

        vad_level = voice_engine.vad_prob
        vad_h = int(vad_level * mh)
        if vad_h > 0:
            vad_color = (0, 200, 255) if vad_level >= voice_engine.vad_threshold else (180, 100, 0)
            for y in range(vad_h):
                y_pos = my + mh - y
                cv2.line(frame, (vx + 1, y_pos), (vx + vw - 1, y_pos), vad_color, 1)

        thresh_y = my + mh - int(voice_engine.vad_threshold * mh)
        cv2.line(frame, (vx - 3, thresh_y), (vx + vw + 3, thresh_y), (0, 255, 255), 2)
        cv2.putText(frame, f"{voice_engine.vad_threshold:.1f}", (vx + vw + 5, thresh_y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
        cv2.rectangle(frame, (vx, my), (vx + vw, my + mh), (100, 100, 100), 1)
        cv2.putText(frame, "VAD", (vx - 2, my + mh + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

        phase = voice_engine.listen_phase
        if phase:
            phase_colors = {
                "waiting": (0, 200, 255),
                "recording": (0, 0, 255),
                "transcribing": (255, 200, 0),
            }
            phase_color = phase_colors.get(phase, (200, 200, 200))
            if phase == "recording":
                pulse = abs(np.sin(time.time() * 5))
                phase_color = (0, 0, int(150 + 105 * pulse))
            px = vx + vw + 8
            cv2.putText(frame, phase.upper(), (px, my + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, phase_color, 2)

        lang = voice_engine.detected_language
        if lang:
            lang_prob = voice_engine.detected_language_prob
            px = vx + vw + 8
            cv2.putText(frame, f"Lang: {lang} ({lang_prob:.0%})", (px, my + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


LOG_CATEGORY_COLORS = {
    "face":     (255, 200, 100),
    "voice":    (100, 255, 100),
    "action":   (100, 200, 255),
    "system":   (200, 200, 200),
    "emotion":  (200, 100, 255),
    "reasoning":(100, 255, 255),
    "agent":    (100, 200, 255),
}


def draw_event_log_window(event_log_inst, max_lines=30, width=700, height=600):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (25, 25, 25)

    entries = event_log_inst.recent(max_lines)

    cv2.rectangle(canvas, (0, 0), (width, 30), (40, 40, 40), cv2.FILLED)
    cv2.putText(canvas, "EVENT LOG", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    entry_count = len(event_log_inst.recent(200))
    cv2.putText(canvas, f"({entry_count} events)", (width - 130, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    if not entries:
        cv2.putText(canvas, "No events yet...", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
        cv2.imshow("Event Log", canvas)
        return

    line_h = 19
    y_start = 40
    for i, entry in enumerate(entries):
        y = y_start + i * line_h
        if y + line_h > height:
            break
        cat = entry["cat"]
        color = LOG_CATEGORY_COLORS.get(cat, (180, 180, 180))
        ts = entry["ts"]
        msg = entry["msg"]
        detail = entry.get("detail", "")

        cv2.putText(canvas, ts, (8, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (100, 100, 100), 1)
        cv2.putText(canvas, f"[{cat.upper()}]", (72, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, color, 1)
        max_msg_chars = 40
        display_msg = msg if len(msg) <= max_msg_chars else msg[:max_msg_chars - 2] + ".."
        cv2.putText(canvas, display_msg, (150, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1)
        if detail:
            detail_x = 150 + max_msg_chars * 8
            max_detail = 35
            display_detail = detail if len(detail) <= max_detail else detail[:max_detail - 2] + ".."
            cv2.putText(canvas, display_detail, (detail_x, y + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (130, 130, 130), 1)
        cv2.line(canvas, (8, y + line_h - 1), (width - 8, y + line_h - 1), (40, 40, 40), 1)

    cv2.imshow("Event Log", canvas)


def show_overlay(frame, lines, window="Face Recognition"):
    display = frame.copy()
    overlay = display.copy()
    h, w = display.shape[:2]
    box_h = 40 * len(lines) + 40
    y_start = h // 2 - box_h // 2
    cv2.rectangle(overlay, (0, y_start), (w, y_start + box_h), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
    for i, (text, color, scale) in enumerate(lines):
        cv2.putText(display, text, (20, y_start + 35 + i * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
    cv2.imshow(window, display)


def get_name_from_gui(frame, existing_match=None):
    name = ""
    while True:
        lines = []
        if existing_match and not name:
            match_name, match_conf = existing_match
            lines.append((f"Already known as: {match_name} ({match_conf:.0f}%)", (0, 255, 0), 0.7))
            lines.append(("ENTER=add sample  Type new name to override  ESC=cancel", (200, 200, 200), 0.5))
            lines.append((name + "_", (0, 255, 255), 1.0))
        else:
            lines.append(("Type name and press ENTER (ESC to cancel):", (255, 255, 255), 0.7))
            lines.append((name + "_", (0, 255, 255), 1.0))
        show_overlay(frame, lines)

        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            return None
        elif key == 13 or key == 10:
            if name.strip():
                return name.strip()
            elif existing_match:
                return existing_match[0]
            return None
        elif key == 8 or key == 127:
            name = name[:-1]
        elif 32 <= key <= 126:
            name += chr(key)


def draw_faces(frame, tracker, memory):
    """Draw face boxes, names, emotions, and focus indicator.

    Display names are resolved via ``memory`` (the tracker only knows
    stable person IDs). Also draws a fading ghost box for the focused
    face if it briefly disappears (grace period before track is evicted).
    """
    faces = tracker.get_visible_faces()
    focus_id = tracker.focus_track_id

    # Ghost box for the focused face if it's in the grace period
    if focus_id is not None:
        focus_face = tracker.get_face_by_id(focus_id)
        if focus_face and not focus_face.is_visible:
            elapsed = time.time() - focus_face.last_seen
            alpha = max(0.0, 1.0 - elapsed / 2.0)
            ghost_color = (0, int(255 * alpha), int(100 * alpha))
            top, right, bottom, left = focus_face.bbox
            # Dashed border
            for i in range(0, right - left, 12):
                cv2.line(frame, (left + i, top),
                         (left + min(i + 6, right - left), top), ghost_color, 2)
                cv2.line(frame, (left + i, bottom),
                         (left + min(i + 6, right - left), bottom), ghost_color, 2)
            for i in range(0, bottom - top, 12):
                cv2.line(frame, (left, top + i),
                         (left, top + min(i + 6, bottom - top)), ghost_color, 2)
                cv2.line(frame, (right, top + i),
                         (right, top + min(i + 6, bottom - top)), ghost_color, 2)
            person = memory.get(focus_face.track_id)
            name = person.name if person and person.is_identified else None
            ghost_label = f"FOCUS (lost {elapsed:.1f}s)"
            if name:
                ghost_label = f"{name} - {ghost_label}"
            cv2.putText(frame, ghost_label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, ghost_color, 1)

    for rank, face in enumerate(faces):
        is_focus = (face.track_id == focus_id)
        person = memory.get(face.track_id)
        name = person.name if person and person.is_identified else None
        conf = tracker.get_confidence(face.track_id)
        top, right, bottom, left = face.bbox
        color = (0, 255, 100) if is_focus else (0, 200, 0) if name else (0, 0, 200)
        thickness = 4 if is_focus else 2

        cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)
        label = f"[{rank+1}] {name or '?'} #{face.track_id}"
        if conf:
            label += f" {conf:.0f}%"
        cv2.rectangle(frame, (left, bottom), (right, bottom + 28), color, cv2.FILLED)
        cv2.putText(frame, label, (left + 4, bottom + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
        if face.emotion and face.emotion != "neutral":
            cv2.putText(frame, face.emotion, (left, top - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        if is_focus:
            cv2.putText(frame, "FOCUS", (left, top - 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 100), 2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Face Agent UI")
    parser.add_argument("--db-dir", default=os.path.join(_SOURCE_DIR, "known_faces"))
    parser.add_argument("--people-dir", default=os.path.join(_SOURCE_DIR, "people"))
    parser.add_argument("--camera", type=int, default=-1,
                        help="Camera index; -1 (default) auto-picks the first that delivers video")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--no-auto-ask", action="store_true")
    parser.add_argument("--no-auto-greet", action="store_true")
    parser.add_argument("--llm-model", default="qwen3:8b")
    parser.add_argument("--ollama-url", default="http://localhost:11434/v1")
    parser.add_argument("--en-voice", default="en_US-lessac-medium")
    parser.add_argument("--mcp-config", default=None)
    parser.add_argument("--mcp-server", action="append", default=[])
    parser.add_argument("--agent-name", default="Face Agent")
    parser.add_argument("--smart-greeting", action="store_true",
                        help="Use the LLM for greetings (slower, but can reference facts). "
                             "Default is canned templates — instant.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    # --- Initialize components ---
    face_db = FaceDatabase(db_dir=args.db_dir)
    face_db.load()
    emotion_detector = EmotionDetector()
    tracker = FaceTracker(db=face_db, emotion_detector=emotion_detector)

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

    audio_monitor = AudioMonitor()
    audio_monitor.start()

    from mcp_client import load_servers
    mcp_servers, mcp_descriptions = load_servers(
        config_path=args.mcp_config, server_urls=args.mcp_server)

    llm = ConversationLLM(
        model_name=args.llm_model,
        ollama_url=args.ollama_url,
        mcp_servers=mcp_servers,
        mcp_descriptions=mcp_descriptions,
        agent_name=args.agent_name,
        smart_greetings=args.smart_greeting,
    )
    try:
        llm.validate()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    agent = Agent(
        tracker=tracker,
        voice_input=voice_in,
        voice_output=voice_out,
        memory=memory,
        llm=llm,
        auto_ask=not args.no_auto_ask,
        auto_greet=not args.no_auto_greet,
    )

    # --- Event log (bridge agent events to on-screen log) ---
    event_log = EventLog()
    event_log.add("system", f"App started, {len(face_db.known_person_ids)} known people")
    event_log.add("system", f"Log file: {LOG_FILE}")

    def _resolve_name(person_id):
        if not person_id:
            return None
        person = memory.get_by_id(person_id)
        return person.name if person else person_id

    # Bridge face tracker events to the event log
    from face_tracker import FaceEvent, FaceEventType
    def on_face_event(event: FaceEvent):
        p = event.payload
        if event.type == FaceEventType.FACE_APPEARED:
            name = _resolve_name(getattr(p, 'initial_person_id', None)) or '?'
            event_log.add("face", f"Appeared: {name} (track {event.track_id})")
        elif event.type == FaceEventType.FACE_DISAPPEARED:
            name = _resolve_name(getattr(p, 'person_id', None)) or '?'
            event_log.add("face", f"Left: {name} (track {event.track_id})")
        elif event.type == FaceEventType.IDENTITY_CONFIRMED:
            name = _resolve_name(p.person_id) or p.person_id
            event_log.add("face", f"Identified: {name} (track {event.track_id})")
        elif event.type == FaceEventType.EMOTION_CHANGED:
            name = _resolve_name(getattr(p, 'person_id', None)) or f"track {event.track_id}"
            event_log.add("emotion", f"{name}: {p.old_emotion} -> {p.new_emotion}")
    tracker.subscribe(on_face_event)

    # Bridge agent events to the event log
    def on_agent_event(event: AgentEvent):
        p = event.payload
        if event.type == AgentEventType.GREETING:
            event_log.add("agent", f"Greeting {p.name}", f'"{p.text}"')
        elif event.type == AgentEventType.GOODBYE:
            event_log.add("agent", f"Goodbye {p.name}", f'"{p.text}"')
        elif event.type == AgentEventType.ASKING_NAME:
            event_log.add("agent", f"Asking name (track {p.track_id})", f'"{p.text}"')
        elif event.type == AgentEventType.RESPONDING:
            event_log.add("voice", f"Heard from {p.name or '?'}", f'"{p.heard}"')
            event_log.add("agent", f"Responding", f'"{p.response}"')
        elif event.type == AgentEventType.LEARNED_NAME:
            event_log.add("agent", f"Learned: {p.name}")
        elif event.type == AgentEventType.NAME_EXTRACT_FAILED:
            event_log.add("agent", f"Name extract failed", f'"{p.raw_speech}"')
    agent.subscribe(on_agent_event)

    # --- Start agent ---
    print("Models loaded. Starting agent.\n")

    agent.start()

    # --- Camera + UI loop ---
    from camera_utils import open_camera
    cap, cam_idx = open_camera(args.camera)
    if cap is None:
        logger.error("Could not open a working camera (all black/unavailable)")
        return

    cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face Recognition", 800, 600)
    cv2.namedWindow("Event Log", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Event Log", 700, 600)
    cv2.moveWindow("Event Log", 820, 0)

    frame_interval = 1.0 / args.fps if args.fps > 0 else 0
    last_frame = 0.0

    print(f"Running at {args.fps} FPS.")
    print("  L=learn  T=talk  C=continuous  A=auto-ask  TAB=select  D=delete  Q=quit\n")

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

            # --- Face tracking ---
            tracker.process_frame(frame)
            agent.check_unknown_faces(frame)

            # --- Draw ---
            draw_faces(frame, tracker, memory)
            draw_audio_meter(frame, audio_monitor, voice_in)
            draw_event_log_window(event_log)

            # Status bar
            visible = tracker.get_visible_faces()
            status = f"Faces: {len(visible)} | Known: {len(face_db.known_person_ids)} | People: {memory.active_count}"
            if agent.busy:
                status += " | BUSY"
            if agent.auto_ask:
                status += " | AUTO-ASK"
            cv2.putText(frame, status, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            keys_hint = "L=learn T=talk C=continuous A=auto-ask D=delete Q=quit"
            cv2.putText(frame, keys_hint, (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

            cv2.imshow("Face Recognition", frame)

            # --- Keyboard controls ---
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

            elif key == ord("l"):
                # Learn: label a face via keyboard
                primary = tracker.get_primary_face()
                if primary:
                    existing = None
                    if tracker.is_recognized(primary.track_id):
                        active = memory.get(primary.track_id)
                        existing_name = active.name if active else tracker.get_person_id(primary.track_id)
                        existing = (existing_name, tracker.get_confidence(primary.track_id))
                    name = get_name_from_gui(frame, existing)
                    if name:
                        # Prefer the auto-enrolled pid if one already exists;
                        # set_name handles both "rename existing" and
                        # "allocate new" paths. Also add an extra encoding
                        # sample from this frame for better recognition.
                        person_id = memory.set_name(primary.track_id, name)
                        if person_id:
                            tracker.learn_face(primary.track_id, person_id, frame)
                        event_log.add("face", f"Manually learned: {name} ({person_id})")

            elif key == ord("t"):
                # Talk: manual one-shot listen + respond
                if not agent.busy:
                    primary = tracker.get_primary_face()
                    tid = primary.track_id if primary else None
                    def _do_talk(track_id=tid):
                        agent.pause_listening()
                        response = voice_in.listen()
                        if response:
                            lang = voice_in.detected_language or "en"
                            if track_id:
                                memory.add_dialogue(track_id, "person", response, language=lang)
                            reply = llm.generate_response(memory, track_id, response, language=lang)
                            if track_id:
                                memory.add_dialogue(track_id, "system", reply, language=lang)
                            agent.speak(reply, language=lang)
                        agent.resume_listening()
                    threading.Thread(target=_do_talk, daemon=True).start()

            elif key == ord("c"):
                # Toggle continuous listening
                if agent._listener:
                    agent._listener.paused = not agent._listener.paused
                    state = "OFF" if agent._listener.paused else "ON"
                    event_log.add("action", f"Continuous listen: {state}")

            elif key == ord("a"):
                agent.auto_ask = not agent.auto_ask
                event_log.add("action", f"Auto-ask: {'ON' if agent.auto_ask else 'OFF'}")

            elif key == ord("d"):
                show_overlay(frame, [
                    ("Delete all known faces? Y/N", (0, 0, 255), 0.8),
                ])
                confirm = cv2.waitKey(0) & 0xFF
                if confirm == ord("y"):
                    event_log.add("action", "Database cleared",
                                  f"was {len(face_db.known_person_ids)} people")
                    face_db.clear()

    except KeyboardInterrupt:
        pass

    # --- Shutdown ---
    event_log.add("system", "App shutting down")
    agent.stop()
    audio_monitor.stop()
    cap.release()
    cv2.destroyAllWindows()
    logger.info(f"Session log saved to {LOG_FILE}")
    print(f"\nLog saved to {LOG_FILE}")
    print("Goodbye!")


if __name__ == "__main__":
    main()
