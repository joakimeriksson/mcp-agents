#
# Generic MCP client, communicating with an MCP server running sse,
# and using a speechbased LLM interface with whisper and piper.
#

import argparse
import asyncio
import atexit
from fastmcp import Client, exceptions
from fastmcp.client.transports import SSETransport
import json
import logging
import os
import subprocess
import threading
import time
import sys
import re

import openai
from openai import OpenAI

# voice_input / voice_output / face_tracker live in ../face/
_FACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'face')
if _FACE_DIR not in sys.path:
    sys.path.insert(0, _FACE_DIR)

from readnb import *
from eyewindow import *
from voice_input import (
    VoiceInput, ContinuousListener, VoiceEventType, AudioMonitor,
    list_input_devices, RAW_AUDIO_TEXT,
)
from voice_output import VoiceOutput
from face_tracker import (
    FaceTracker, FaceDatabase, FaceEventType,
)
from face_config import build_db_kwargs, build_tracker_kwargs, backend_metric
import cv2
import sounddevice as sd
from config import load_config
from interaction_logger import InteractionLogger

logger = logging.getLogger("mcpclient_speech")

interaction_log: InteractionLogger | None = None


def _ilog(event_type: str, **fields) -> None:
    if interaction_log is not None:
        interaction_log.log(event_type, **fields)


def _git_commit_short() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode().strip() or None
    except Exception:
        return None

default_lang = "sv"
messages_trunclen = 8
messages = []
state = {'evtime': 0, 'statetime': 0, 'newstate': None, 'currstate': None}
omit_names_and_prefs = False
talk_to_unknown = True   # greet faces the DB doesn't know yet

muted = False

has_sysprompt = False
has_sysprompt_lang = False
has_augprompt = False
has_augprompt_lang = False
has_name = False
has_init = False
has_exit = False

class Person:
    def __init__(self, name):
        self.name = name
        self.lang = default_lang
        self.lasttime = None
        self.lastmessages = []
        self.profileinfo = None

persondict = {}

curr_person = None

curr_prompt = ""

win: EyeWindow | None = None
cam_win: CameraWindow | None = None
voice_in: VoiceInput | None = None
voice_out: VoiceOutput | None = None
listener: ContinuousListener | None = None
audio_monitor: AudioMonitor | None = None   # debug panel meters (may be None)
direct_llm = None   # DirectAudioLLM when --direct-audio (speech straight into gemma4)
# Voice identity (face/voice_id.py): speaker embeddings from the local voice
# server, matched on this machine. voice_db is None when disabled/unavailable.
voice_db = None
speaker_client = None
voice_cfg: dict = {}
_voice_pending = {"future": None}     # embedding of the utterance being processed
_voice_pool = None
tracker: FaceTracker | None = None
model: str | None = None


def list_cameras(max_index=10):
    """Cameras OpenCV can open, annotated with their device names."""
    try:
        from camera_utils import camera_name
    except Exception:
        camera_name = lambda i: ""
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            info = {
                'index': i,
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'backend': cap.getBackendName(),
                'name': camera_name(i),
            }
            available.append(info)
            cap.release()
    return available



def parse_args():
    parser = argparse.ArgumentParser(description="MCP Speech Client with Face Tracking")
    parser.add_argument('-l', '--list-cameras', action='store_true', help='List available cameras and exit')
    parser.add_argument('--camera', default=None,
                        help="Camera index, or part of its name (e.g. 'macbook', 'brio'). "
                             "Names are stable; indices shuffle when an iPhone wakes up as a "
                             "Continuity Camera or a USB webcam is plugged in. Default: auto-detect.")
    parser.add_argument('--server', default="http://127.0.0.1:8000/sse", help='MCP server SSE URL')
    parser.add_argument('--llm-model', default=None, help='LLM model name')
    parser.add_argument('--llm-url', default=None, help='LLM base URL')
    parser.add_argument('-m', '--list-mics', action='store_true', help='List available microphones and exit')
    parser.add_argument('--mic', type=int, default=None, help='Microphone device index (default: system default)')
    parser.add_argument('-v', '--verbose', action='count', default=0, help='Increase verbosity (-v for INFO, -vv for DEBUG)')
    parser.add_argument('--config', default=None, help='Path to config TOML file (default: config.toml next to this script)')

    # Face-tracker tuning. Each overrides the matching key in
    # face/face_config.toml [tracker]; left unset, the config value (then the
    # face_tracker.py default) applies.
    tune = parser.add_argument_group('face tracker tuning')
    tune.add_argument('--backend', choices=['insightface', 'dlib'], default=None,
                      help='Face detection/embedding backend (default from config: insightface)')
    tune.add_argument('--det-size', type=int, default=None,
                      help='InsightFace detector input size (square); larger = better on distant faces, slower')
    tune.add_argument('--det-thresh', type=float, default=None,
                      help='InsightFace minimum detector confidence')
    tune.add_argument('--frame-scale', type=float, default=None,
                      help='dlib backend detection downscale factor (higher = better on distant faces, more CPU)')
    tune.add_argument('--recognition-tolerance', type=float, default=None,
                      help='Max embedding distance to accept a match (lower = stricter)')
    tune.add_argument('--recognition-k', type=int, default=None,
                      help='Average the k nearest stored samples per person when matching')
    tune.add_argument('--max-missing-seconds', type=float, default=None,
                      help='Grace period before a missing face is dropped (survives look-aways)')
    tune.add_argument('--vad-near-field-ratio', type=float, default=None,
                      help='How much louder than the room a voice must be to count as '
                           'talking TO the robot (1.0 = off). Raise in a noisy hall.')
    tune.add_argument('--focus-min-area-frac', type=float, default=None,
                      help='Min fraction of frame a face must cover to take focus (0 = off)')
    tune.add_argument('--focus-dwell-seconds', type=float, default=None,
                      help='Seconds a candidate must be held before it takes focus (0 = off)')
    tune.add_argument('--engage-max-yaw', type=float, default=None,
                      help='Max |yaw| degrees for the focused face to count as engaged (FACE_ENGAGED)')
    tune.add_argument('--engage-max-pitch', type=float, default=None,
                      help='Max |pitch| degrees for the focused face to count as engaged')
    tune.add_argument('--engage-dwell-seconds', type=float, default=None,
                      help='Seconds the focused face must face the camera before FACE_ENGAGED')

    parser.add_argument('--direct-audio', action='store_true', default=None,
                        help='Send the captured speech straight into the (audio-capable) LLM: '
                             'one call does hearing + reasoning + tool calls. The transcript '
                             'for the history/log is produced in the background. '
                             '(default from config [llm] direct_audio)')
    parser.add_argument('--detect-fps', type=int, default=6,
                        help='Max face-detection rate (Hz). Lower = far less memory '
                             '(InsightFace/onnxruntime leaks ~40MB per inference on this build).')
    parser.add_argument('--debug-audio', action='store_true', default=None,
                        help='Show the audio debug panel (VU meters + oscilloscope) '
                             'in the eye window (default from config [debug] audio_panel)')

    log_group = parser.add_mutually_exclusive_group()
    log_group.add_argument('--log-file', default=None, help='Log every interaction event to this JSONL file (overwrites on each run)')
    log_group.add_argument('--log-dir', default=None, help='Log every interaction event to a new timestamped JSONL file in this directory')
    return parser.parse_args()


def extract_dialog_messages(messages):
    return [ msg for msg in messages if (msg['role'] == 'user' if type(msg)==dict else msg.content) ]

def distill_user_info(messages):
    prompt = {'role':'user', 'content':'Summarize the above conversation in this form about the user, leaving fields blank if no information:\nName: \nLanguage: \nPreferences: \n'}
    response = openai.chat.completions.create(
        model=model,
        messages=messages + [prompt],
    )
    return response.choices[0].message.content

def extract_value(key, info):
    reg = "[-+ *#]*" + key + "[-+ *#]*"
    lst = info.split("\n")
    for s in lst:
        m = re.match(reg, s)
        if m:
            return s[m.end():]
    return None

def extract_language(info):
    languages = {"English": "en",
                 "Swedish": "sv",
                 "Svenska": "sv",
                 "German": "de",
                 "Deutch": "de",
                 "French": "fr",
                 "Française": "fr",
                 "Francaise": "fr",
                 "Spanish": "es",
                 "Espanol": "es",
                 "Español": "es"}
    s = extract_value("Language:", info)
    if s:
        for l in languages:
            if re.search(l, s):
                return languages[l]
    return "en"

def kp_toggle_mute(_event, _obj):
    global muted
    muted = not muted
    interrupted = False
    if muted:
        if listener:
            listener.paused = True
        if voice_in is not None:
            voice_in._cancel_listen = True
        if voice_out is not None:
            interrupted = bool(getattr(voice_out, "speaking", False))
            voice_out.stop_speaking()
        logger.info("Muted")
        if win:
            win.set_state('muted')
    else:
        logger.info("Unmuted")
        if win:
            win.set_state(state.get('currstate', 'wait'))
    _ilog("mute", muted=muted, interrupted_speech=interrupted)

def kp_force_process(_event, _obj):
    if state.get('currstate') == 'listen' and voice_in is not None:
        voice_in.flush_listen()
    else:
        logger.debug("force-process pressed but ignored (state=%s)", state.get('currstate'))

def _current_mic() -> tuple[int | None, str]:
    """(index, name) of the microphone in use; resolves the system default."""
    idx = voice_in.device if voice_in is not None else None
    if idx is None:
        try:
            idx = sd.default.device[0]
        except Exception:
            return None, "system default"
    try:
        return idx, sd.query_devices(idx)['name']
    except Exception:
        return idx, f"device {idx}"

def _show_mic_indicator() -> None:
    """Indicator line: active mic + active voice, with their hotkeys."""
    if win is None:
        return
    idx, name = _current_mic()
    text = f"Mic {idx}: {name} (i)"
    if voice_out is not None and voice_out.server_voices(default_lang):
        text += f"   |   Voice: {voice_out.server_voice(default_lang)} (v)"
    win.set_indicator(text)
    win.check_events()

def kp_cycle_voice(_event, _obj):
    """'v': next server voice for the default language (Swedish), live.
    Speaks a short sample in the new voice when the robot is idle."""
    if voice_out is None:
        return
    voices = voice_out.server_voices(default_lang)
    if not voices:
        logger.info("No selectable voices configured for %s", default_lang)
        return
    cur = voice_out.server_voice(default_lang)
    pos = voices.index(cur) if cur in voices else -1
    new_voice = voices[(pos + 1) % len(voices)]
    voice_out.set_server_voice(default_lang, new_voice)
    _ilog("voice_change", language=default_lang, voice=new_voice)
    _show_mic_indicator()
    if state.get('currstate') in ('wait', 'listen') and not voice_out.speaking:
        # Audition: pause the mic while the sample plays, then restore.
        if listener:
            listener.paused = True
        def _sample():
            voice_out.speak(f"Hej, jag heter {new_voice}.", default_lang)
            time.sleep(0.3)
            if listener:
                listener.paused = (state.get('currstate') != 'listen')
        threading.Thread(target=_sample, daemon=True).start()

def kp_cycle_mic(_event, _obj):
    """'i': switch to the next input device, live — VAD/STT and the
    debug-panel meters follow. The active mic is shown in the indicator."""
    if voice_in is None:
        return
    devices = list_input_devices()
    if not devices:
        return
    cur, _ = _current_mic()
    idxs = [d[0] for d in devices]
    pos = idxs.index(cur) if cur in idxs else -1
    new_idx, name = devices[(pos + 1) % len(devices)][:2]
    voice_in.set_device(new_idx)
    if audio_monitor is not None:
        audio_monitor.set_device(new_idx)
    logger.info("Microphone -> %d: %s", new_idx, name)
    _ilog("mic_change", device=new_idx, name=name)
    _show_mic_indicator()

def _refresh_save_indicator(count: int) -> None:
    if win is None:
        return
    if count > 0:
        win.set_indicator(f"Save next {count}")
        win.check_events()
    else:
        _show_mic_indicator()

def kp_save_recording(_event, _obj):
    if voice_in is None:
        return
    pending = voice_in.save_next_recordings(1)
    _refresh_save_indicator(pending)
    logger.info("Save-next-recordings count is now %d", pending)

def on_exit(state):
    if state.get('newstate') == 'exit':
        return
    logger.info("Exit event")
    if voice_db is not None:
        try:
            voice_db.save()      # writes only if persist_named, and only named people
        except Exception as e:
            logger.warning("voice db save failed: %s", e)
    _ilog("state_change", **{"from": state.get('currstate'), "to": "exit"})
    state['evtime'] = time.time()
    state['newstate'] = 'exit'

def on_face_change(id):
    global messages, state, curr_person
    logger.info("Face change event")
    if muted or state['newstate'] == 'exit':
        return
    reset_language_switch()   # a new person starts the switch hysteresis over
    prev_face_id = next((pid for pid, p in persondict.items() if p is curr_person), None)
    if curr_person:
        logger.info("Storing current person")
        curr_person.lasttime = time.time()
        if not omit_names_and_prefs and curr_person.lastmessages is not messages: # Does this work? I try to see if anything new has been said, otherwise there is no point extracting again. If there are many switches between people.
            curr_person.lastmessages = messages
            info = distill_user_info(extract_dialog_messages(messages))
            name = extract_value("Name:", info)
            if name:
                curr_person.name = name
                if voice_db is not None and prev_face_id:
                    voice_db.mark_named(prev_face_id)   # eligible for opt-in persistence
            lang = extract_language(info)
            if lang:
                curr_person.lang = lang
            pref = extract_value("Preferences:", info)
            if pref:
                curr_person.profileinfo = pref
            _ilog("person_distillation",
                  face_id=prev_face_id,
                  raw_summary=info,
                  parsed_name=name,
                  parsed_lang=lang,
                  parsed_preferences=pref)
        else:
            print("(Nothing new to extract)")
    if id is None:
        _ilog("face_change", face_id=None, status="none", name=None, lang=None, last_seen_seconds_ago=None)
        messages = []
        curr_person = None
        state['evtime'] = time.time()
        state['newstate'] = 'wait'
    else:
        if id in persondict:
            curr_person = persondict[id]
            messages = list(curr_person.lastmessages or [])
            logger.info("Retrieving person %s from memory", id)
            last_seen = (time.time() - curr_person.lasttime) if curr_person.lasttime else None
            _ilog("face_change",
                  face_id=id,
                  status="known",
                  name=curr_person.name,
                  lang=curr_person.lang,
                  last_seen_seconds_ago=last_seen)
            _ilog("person_resumed",
                  face_id=id,
                  name=curr_person.name,
                  lang=curr_person.lang,
                  profileinfo=curr_person.profileinfo,
                  restored_message_count=len(messages))
        else:
            curr_person = Person(None)
            persondict[id] = curr_person
            messages = []
            logger.info("Creating person %s", id)
            _ilog("face_change", face_id=id, status="new", name=None, lang=None, last_seen_seconds_ago=None)
        state['evtime'] = time.time()
        if curr_person.lasttime and state['evtime'] - curr_person.lasttime < 60:
            state['newstate'] = 'listen'
        else:
            state['newstate'] = 'greet'

def on_speech(txt):
    global curr_prompt, state
    if muted:
        return
    logger.info("Speech event")
    if state['newstate'] == 'exit':
        return
    if state['currstate'] == 'listen' and (state['newstate'] is None or state['newstate'] == 'listen'):
        curr_prompt = txt
        if voice_db is not None and voice_in is not None and voice_in.last_audio is not None:
            # Fetch the speaker embedding in parallel with everything else the
            # turn does (LLM, scene fetch); the gate collects it just before
            # the model is called. ~20 ms on the server, so it is never late.
            audio = voice_in.last_audio
            _voice_pending["future"] = _voice_pool.submit(
                speaker_client.embed, audio, voice_in.sample_rate)
        state['evtime'] = time.time()
        state['newstate'] = 'process'


def voice_gate(person_id):
    """Decide whether the utterance just captured came from *person_id*.

    Returns (accept, match). Fails OPEN on every uncertainty: no embedding,
    no enrolled voice, a clip too short to judge, or an in-between score all
    ACCEPT. Only a clear "different" rejects. The first usable utterance from
    a person enrolls their voice; later "same" verdicts refine it.
    """
    fut = _voice_pending.pop("future", None)
    if voice_db is None or fut is None or not person_id:
        return True, None
    try:
        emb, seconds = fut.result(timeout=1.5)
    except Exception as e:
        logger.warning("voice gate: embedding unavailable (%s) — accepting", e)
        return True, None
    if not voice_db.has(person_id):
        if voice_db.enroll(person_id, emb, seconds):
            _ilog("voice_enrolled", person_id=person_id, seconds=round(seconds, 2))
            logger.info("voice id: enrolled %s from a %.1fs utterance", person_id, seconds)
        return True, None
    m = voice_db.verify(person_id, emb, seconds)
    _ilog("voice_verify", person_id=person_id, similarity=round(m.similarity, 3),
          decision=m.decision, seconds=round(seconds, 2), reason=m.reason)
    if m.decision == "different":
        return False, m
    return True, m

def check_statechange(state):
    win.check_events()
    if state['newstate'] and state['newstate'] != state['currstate']:
        return state['newstate']
    elif state['newstate'] == 'exit':
        return 'exit'
    else:
        return False

def set_state(state, newstate):
    prev = state.get('currstate')
    if listener:
        listener.paused = muted or (newstate != 'listen')
    state['currstate'] = newstate
    state['statetime'] = time.time()
    state['newstate'] = None
    win.set_state('muted' if muted else newstate)
    win.check_events()
    if prev != newstate:
        _ilog("state_change", **{"from": prev, "to": newstate})

def set_win_state(newstate):
    win.set_state('muted' if muted else newstate)
    win.check_events()

def init_llm(conf):
    default_config = conf
    if "api_key" in default_config:
        openai.api_key = default_config["api_key"]
    if "base_url" in default_config:
        openai.base_url = default_config["base_url"]
    model = default_config["model"]
    llm = OpenAI(api_key=openai.api_key, base_url=openai.base_url)

    # Verify connection and model availability
    try:
        available = llm.models.list()
        model_ids = [m.id for m in available.data]
        if model not in model_ids:
            print(f"ERROR: Model '{model}' not available in Ollama.")
            print(f"Available models: {', '.join(model_ids)}")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Cannot connect to LLM server at {openai.base_url}: {e}")
        sys.exit(1)

    return (llm, model)

def map_tool_definition(f):
        tool_param = {
            'type': 'function',
            'function': {
                'name': f.name,
                'description': f.description,
                'parameters': f.inputSchema,
            },
        }
        return tool_param

async def system_message(client, lang):
    if has_sysprompt:
        if has_sysprompt_lang:
            pr = await client.get_prompt("get_service_prompt", {"lang": lang})
        else:
            pr = await client.get_prompt("get_service_prompt", {})
        txt = pr.messages[0].content.text
        _ilog("mcp_prompt", kind="system", lang=lang, content=txt)
    else:
        txt = "You are a helpful assistant that can control various devices."
    return {"role": "system", "content": txt}

async def augmentation_message(client, lang):
    if has_augprompt:
        if has_augprompt_lang:
            pr = await client.get_prompt("get_service_augmentation", {"lang": lang})
        else:
            pr = await client.get_prompt("get_service_augmentation", {})
        txt = pr.messages[0].content.text
        _ilog("mcp_prompt", kind="augmentation", lang=lang, content=txt)
        return {"role": "system", "content": txt}
    else:
        return False

def user_message(prompt):
    return {"role": "user", "content": prompt}

def _text_history(mlst, max_turns):
    """Recent user/assistant turns as plain dicts (for the direct-audio call).
    Skips tool traffic and works for both dict and OpenAI message objects."""
    out = []
    for m in mlst:
        role = m.get('role') if isinstance(m, dict) else getattr(m, 'role', None)
        content = m.get('content') if isinstance(m, dict) else getattr(m, 'content', None)
        if role in ('user', 'assistant') and content:
            out.append({'role': role, 'content': str(content)})
    return out[-2 * max_turns:]

def _direct_transcribe(audio, user_msg, lang):
    """Background: transcribe the utterance for the history/log (the reply
    was already produced straight from the audio)."""
    try:
        tr = direct_llm.transcriber
        tr.language_hint = lang
        segs, info = tr.transcribe(audio)
        text = "".join(seg.text for seg in segs).strip()
    except Exception as e:
        logger.warning("background transcription failed: %s", e)
        text = ""
    user_msg['content'] = text or "(unintelligible audio)"
    print(f"\n  Heard: ({info.language or lang if text else '?'}) {text}")
    _ilog("user_transcript", content=text, lang=(info.language if text else "") or lang)

KNOWN_LANGS = ('en', 'sv', 'de', 'fr', 'es', 'it')
LANG_SWITCH_MIN_WORDS = 4   # a shorter utterance can't flip the conversation language
LANG_SWITCH_TURNS = 2       # ...and it takes this many consecutive turns to flip
# Pending language switch: {lang, count}. Module state so a single stray turn
# cannot flip the conversation on its own (see choose_language).
_lang_pending = {"lang": None, "count": 0}


def reset_language_switch():
    """Forget a half-finished language switch (new person / new session)."""
    _lang_pending["lang"] = None
    _lang_pending["count"] = 0


def choose_language(current, detected, text):
    """Sticky conversation language.

    Two guards, because the STT's failure modes need both:
      * word count — a one-word reply ("Yeah.", "Ja tack") carries almost no
        language signal, so it can never flip the conversation, and
      * hysteresis — gemma's other failure mode is transcribing Swedish audio
        as a fluent, LONG English sentence, which the word-count rule alone
        happily trusts. Requiring LANG_SWITCH_TURNS consecutive turns in the
        new language means one bad transcript costs nothing, while a person
        who really switched language is followed one turn later.
    """
    if not detected or detected not in KNOWN_LANGS:
        return current or default_lang
    if not current or current not in KNOWN_LANGS:
        reset_language_switch()
        return detected
    if detected == current:
        reset_language_switch()
        return current

    words = len((text or "").split())
    if words < LANG_SWITCH_MIN_WORDS:
        logger.info("Ignoring language %s for a %d-word utterance, staying in %s",
                    detected, words, current)
        return current

    if _lang_pending["lang"] == detected:
        _lang_pending["count"] += 1
    else:
        _lang_pending["lang"] = detected
        _lang_pending["count"] = 1

    if _lang_pending["count"] >= LANG_SWITCH_TURNS:
        logger.info("Language switch %s -> %s (%d consecutive turns)",
                    current, detected, _lang_pending["count"])
        reset_language_switch()
        return detected
    logger.info("Heard %s (%d words) but staying in %s — needs %d turns in a row "
                "(a mis-transcription should not flip the language)",
                detected, words, current, LANG_SWITCH_TURNS)
    return current

def language_message(lang):
    languages = { "en": "English",
                  "sv": "Swedish",
                  "de": "Deutch",
                  "fr": "French",
                  "es": "Spanish"}
    if not lang in languages:
        lang = 'en'
    reply_language = languages[lang]
    msg = f"Reply in {reply_language}!"
    return {"role": "system", "content": msg}

def greet_prompt_noname():
    if curr_person.lasttime is None:
        return {'role':'user', 'content':'There is a new person in front of you. Produce a suitable greeting.'}
    else:
        duration = int((time.time() - curr_person.lasttime) / 60)
        return {'role':'user', 'content': f'A person has appeared in front of you. It was {duration} minutes since you last met. Produce a suitable greeting.'}
   
def greet_prompt():
    if not curr_person.name and not curr_person.lasttime:
        return {'role':'user', 'content':'There is a new person in front of you. Produce a greeting and ask for the name.'}
    if curr_person.lasttime is None: # This alternative was added by Claude but it should actually never happen
        return {'role':'user', 'content': f'The person {curr_person.name} has appeared in front of you. Produce a suitable greeting.'}
    duration = int((time.time() - curr_person.lasttime) / 60)
    pref = ("Known preferences: " + curr_person.profileinfo) if curr_person.profileinfo else ""
    if not curr_person.name:
        return {'role':'user', 'content': f'A person has appeared in front of you. {pref} It was {duration} minutes since you last met, but you still dont know the name. Produce a suitable greeting and ask for the name.'}
    else:
        return {'role':'user', 'content': f'The person {curr_person.name} has appeared in front of you. {pref} It was {duration} minutes since you last met. Produce a suitable greeting.'}

def compose_messages(sysp, mlst, augs):
    n = 0
    i1 = 0
    i2 = 0
    for i in reversed(range(len(mlst))):
        if type(mlst[i])==dict and mlst[i]["role"] == 'user':
            n += 1
            if n == 1:
                i2 = i
            if n == messages_trunclen:
                i1 = i
                break
    return [sysp] + mlst[i1:i2] + augs + mlst[i2:]

def clear_messages():
    global messages
    messages = []
    _ilog("clear_history")

### remove next 2?
def trim_last_message():
    global messages
    for i in reversed(range(len(messages))):
        if type(messages[i])==dict and messages[i]["role"] == 'user':
            messages = messages[0:i+1]
            return True
    return False

def kp_clear_messages(_event, _state):
    print("\n  (Cleared history)")
    clear_messages()

def messagedump(messages):
    print("\nMessages:")
    for msg in messages:
        print(msg)

def _serialize_tool_calls(tool_calls):
    if not tool_calls:
        return []
    out = []
    for tc in tool_calls:
        args_str = tc.function.arguments
        try:
            args_parsed = json.loads(args_str)
        except (TypeError, ValueError):
            args_parsed = None
        out.append({
            "id": tc.id,
            "name": tc.function.name,
            "arguments": args_str,
            "arguments_parsed": args_parsed,
        })
    return out

async def main(args):
    global messages
    global tools
    global win
    global cam_win
    global model
    global has_sysprompt
    global has_sysprompt_lang
    global has_augprompt
    global has_augprompt_lang
    global has_name
    global has_init
    global has_exit
    global voice_in, voice_out, listener, tracker, audio_monitor, direct_llm, voice_db, speaker_client, _voice_pool
    global curr_prompt

    # Connect via SSE to the MCP server
    async with Client(transport=SSETransport(args.server)) as client:
        ### Initialization phase

        # Check MCP server capabilities
        ress = await client.list_resources()
        print("\nAvailable resources:")
        for res in ress:
            print(res)
            if res.name == 'get_service_name':
                has_name = True
            elif res.name == 'service_init':
                has_init = True
            elif res.name == 'service_exit':
                has_exit = True
        
        prompts = await client.list_prompts()
        print("\nAvailable prompts:")
        for prompt in prompts:
            print(prompt)
            if prompt.name == 'get_service_prompt':
                has_sysprompt = True
                for arg in prompt.arguments:
                    if arg.name == 'lang':
                        has_sysprompt_lang = True
            if prompt.name == 'get_service_augmentation':
                has_augprompt = True
                for arg in prompt.arguments:
                    if arg.name == 'lang':
                        has_augprompt_lang = True

        tools = await client.list_tools()
        print("\nAvailable tools:")
        for tool in tools:
            print(tool)
        tools = [map_tool_definition(tool) for tool in tools]
        print("\n")

        make_nonblocking(sys.stdin)

        llm_config = {
            "model": args.llm_model,
            "base_url": args.llm_url,
            "api_key": "ollama",
        }
        llm, model = init_llm(llm_config)
        print(f'LLM Chatbot using model {model}')

        # new states: wait, listen, greet, process, talk
        sdict = {'wait':      ((0, 0.7, 0.2), "Ready", ""),
                 'listen':    ((0, 0.6, 0.8), "Listening", ""),
                 'greet':     ((0.9, 0.5, 0), "Contact", "Please wait"),
                 'process':   ((0.9, 0.5, 0), "Processing", "Please wait"),
                 'talk':      ((0.95, 0.75, 0), "Speaking", ""),
                 'muted':     ((0.4, 0.4, 0.4), "MUTED", "Press 'm' to unmute"),
                 }
        if has_name:
            tmp = await client.read_resource("url://get_service_name")
            name = tmp[0].text
        else:
            name = "MCP Speech Client"
        win = EyeWindow(name, sdict, 'ready', debug=args.debug_audio)
        win.set_exit_callback(on_exit, state)
        win.keydict["m"] = (kp_toggle_mute, None)
        win.keydict[" "] = (kp_force_process, None)
        win.keydict["s"] = (kp_save_recording, None)
        win.keydict["i"] = (kp_cycle_mic, None)
        win.keydict["v"] = (kp_cycle_voice, None)
        cam_win = CameraWindow(f"{name} - People camera", keydict=win.keydict)
        cam_win.set_exit_callback(on_exit, state)
        win.attach_camera_window(cam_win)
        win.check_events()
        print('Created the interaction window')

        ### Initialize voice_input library, as ContinuousListener with on_speech as callback here
        vin_kwargs = {}
        if args.vad_near_field_ratio is not None:
            vin_kwargs["vad_near_field_ratio"] = float(args.vad_near_field_ratio)
        voice_in = VoiceInput(device=args.mic,
                              stt_backend="raw" if args.direct_audio else "gemma4",
                              **vin_kwargs)
        voice_in.subscribe(
            lambda ev: _ilog("transcription",
                             text=ev.payload.text,
                             language=ev.payload.language,
                             language_probability=ev.payload.language_probability,
                             audio_duration_ms=ev.payload.audio_duration_ms),
            event_types={VoiceEventType.TRANSCRIPTION_COMPLETE},
        )
        def _log_vad(ev):
            # The fair failure mode leaves no trace otherwise: in a noisy hall
            # every utterance runs to the length cap and arrives as a blob of
            # crowd + visitor. Log the shape of each capture so it is visible.
            if ev.type == VoiceEventType.VAD_SPEECH_END:
                _ilog("vad_capture", duration_ms=ev.payload.duration_ms,
                      noise_floor=round(getattr(voice_in, "noise_floor", 0.0), 5))
            elif ev.type == VoiceEventType.VAD_TIMEOUT:
                _ilog("vad_timeout", reason=ev.payload.reason,
                      waited_ms=ev.payload.waited_ms,
                      noise_floor=round(getattr(voice_in, "noise_floor", 0.0), 5))
                if ev.payload.reason == "max_speech_length":
                    logger.warning(
                        "VAD hit the %.0fs length cap — the room is noisy enough that "
                        "silence never registers. Raise [audio] near_field_ratio.",
                        ev.payload.waited_ms / 1000)

        voice_in.subscribe(_log_vad, event_types={VoiceEventType.VAD_SPEECH_END,
                                                  VoiceEventType.VAD_TIMEOUT})
        voice_in.subscribe(
            lambda ev: on_speech(ev.payload.text),
            event_types={VoiceEventType.TRANSCRIPTION_COMPLETE},
        )
        voice_in.subscribe(
            lambda ev: _refresh_save_indicator(ev.payload.remaining),
            event_types={VoiceEventType.RECORDING_SAVED},
        )
        voice_in.subscribe(
            lambda ev: _ilog("recording_saved",
                             remaining=ev.payload.remaining,
                             wav_path=ev.payload.path),
            event_types={VoiceEventType.RECORDING_SAVED},
        )
        print('Loading whisper model...')
        voice_in.load_sync()
        if not voice_in.ready:
            print('Failed to load whisper model')
            return False
        listener = ContinuousListener(voice_in)
        listener.start()
        listener.paused = True
        voice_in.language_hint = default_lang
        print('Continuous listener started')

        if voice_cfg.get("enabled", True):
            from concurrent.futures import ThreadPoolExecutor
            from voice_id import SpeakerClient, VoiceDB
            speaker_client = SpeakerClient(voice_cfg.get("url", "http://127.0.0.1:8880/v1/audio/speaker"))
            if speaker_client.available():
                persist = bool(voice_cfg.get("persist_named", False))
                voice_db = VoiceDB(
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "known_voices", "voices.json"),
                    persist_named=persist,
                    confirm=float(voice_cfg.get("confirm", 0.65)),
                    reject=float(voice_cfg.get("reject", 0.40)),
                    min_seconds=float(voice_cfg.get("min_seconds", 2.0)))
                _voice_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voice-id")
                print(f"Voice id: on (local only; {'named people persisted' if persist else 'nothing written to disk'}; "
                      f"{len(voice_db.people())} known)")
                _ilog("voice_id", enabled=True, persist_named=persist, known=len(voice_db.people()))
            else:
                print("Voice id: server has no --speaker endpoint; running without (fail-open)")
                _ilog("voice_id", enabled=False, reason="server lacks speaker endpoint")

        if args.direct_audio:
            from direct_llm import DirectAudioLLM
            ollama_host = re.sub(r'/v1/?$', '', args.llm_url.rstrip('/'))
            # Built in a worker thread: it fetches the MCP tools with
            # asyncio.run(), which is not allowed inside this running loop.
            direct_llm = await asyncio.to_thread(
                DirectAudioLLM, model=args.llm_model, host=ollama_host,
                agent_name=name, tools_url=args.server)
            direct_llm.transcriber.check()   # fail fast if the model can't hear
            if not direct_llm._tools:
                print('WARNING: direct-audio mode loaded no tools from the server')
            print(f'Direct-audio mode: speech goes straight into {args.llm_model} '
                  f'({len(direct_llm._tools)} tools)')
        _show_mic_indicator()

        ### Initialize voice_output (piper TTS)
        voice_out = VoiceOutput()
        print('Loading piper model...')
        voice_out.load_sync()
        if not voice_out.ready:
            print('Failed to load piper model')
            return False

        # Debug panel in the eye window: VU meters (mic level, VAD
        # probability, silence countdown) + in/out oscilloscope
        if args.debug_audio:
            audio_monitor = AudioMonitor(device=args.mic)
            audio_monitor.start()
            win.set_audio_sources(audio_monitor, voice_in, voice_out)

        ### Initialize the face_tracker here, with on_face_change as callback
        # Tuning comes from face/face_config.toml [tracker]; CLI flags override.
        db_kwargs = build_db_kwargs()
        if args.recognition_tolerance is not None:
            db_kwargs["tolerance"] = args.recognition_tolerance
        if args.recognition_k is not None:
            db_kwargs["recognition_k"] = args.recognition_k
        if args.backend is not None:
            # Keep the db's metric/file in step with a CLI backend override so
            # load() opens the matching db (it runs before the tracker is built).
            db_kwargs["metric"] = backend_metric(args.backend)

        tracker_kwargs = build_tracker_kwargs()
        for cli_val, kw in (
            (args.backend, "backend"),
            (args.det_size, "det_size"),
            (args.det_thresh, "det_thresh"),
            (args.frame_scale, "frame_scale"),
            (args.max_missing_seconds, "max_missing_seconds"),
            (args.focus_min_area_frac, "focus_min_area_frac"),
            (args.focus_dwell_seconds, "focus_dwell_seconds"),
            (args.engage_max_yaw, "engage_max_yaw"),
            (args.engage_max_pitch, "engage_max_pitch"),
            (args.engage_dwell_seconds, "engage_dwell_seconds"),
        ):
            if cli_val is not None:
                tracker_kwargs[kw] = cli_val

        face_db = FaceDatabase(**db_kwargs)
        face_db.load()
        tracker = FaceTracker(db=face_db, emotion_detector=None, **tracker_kwargs)

        focus_state = {"track_id": None, "person_id": None}

        def _emit(person_id):
            if person_id != focus_state["person_id"]:
                focus_state["person_id"] = person_id
                on_face_change(person_id)

        def _on_face_event(ev):
            if ev.type == FaceEventType.FOCUS_CHANGED:
                tid = ev.payload.new_track_id
                focus_state["track_id"] = tid
                pid = ev.payload.new_person_id
                if pid is None and tid is not None and talk_to_unknown:
                    # Talk to a visitor the face DB doesn't know yet. Otherwise
                    # the robot sits in 'wait' (green eye) until auto-enrollment
                    # fires, which needs a big, sharp, frontal face -- at a stand
                    # most visitors never clear that bar and are never greeted.
                    pid = f"track:{tid}"
                _emit(pid)
            elif ev.type in (FaceEventType.IDENTITY_CONFIRMED,
                             FaceEventType.FACE_ENROLLED):
                if ev.track_id == focus_state["track_id"]:
                    provisional = f"track:{ev.track_id}"
                    if focus_state["person_id"] == provisional:
                        # Same human, now identified: adopt the real id in place
                        # so the ongoing conversation is NOT restarted.
                        if provisional in persondict:
                            persondict[ev.payload.person_id] = persondict.pop(provisional)
                        if voice_db is not None:
                            voice_db.rename(provisional, ev.payload.person_id)
                        focus_state["person_id"] = ev.payload.person_id
                        logger.info("Provisional %s identified as %s",
                                    provisional, ev.payload.person_id)
                    else:
                        _emit(ev.payload.person_id)

        tracker.subscribe(
            _on_face_event,
            event_types={FaceEventType.FOCUS_CHANGED,
                         FaceEventType.IDENTITY_CONFIRMED,
                         FaceEventType.FACE_ENROLLED},
        )

        from camera_utils import resolve_camera
        cap, cam_index, cam_name = resolve_camera(args.camera)
        if cap is None:
            print(f"ERROR: no working camera for --camera {args.camera!r}")
            sys.exit(1)
        print(f"People camera: {cam_index}"
              f"{f' ({cam_name})' if cam_name else ''}")
        _ilog("camera_selected", requested=args.camera, index=cam_index, name=cam_name)

        def _camera_loop():
            detect_interval = 1.0 / max(1, args.detect_fps)
            last_detect = [0.0]
            try:
                while state.get('currstate') != 'exit' and state.get('newstate') != 'exit':
                    ret, frame = cap.read()
                    if not ret:
                        time.sleep(0.05)
                        continue
                    # Throttle detection to DETECT_FPS. onnxruntime retains
                    # ~40 MB per InsightFace inference on this build (a known
                    # leak, provider-independent), so running it at full camera
                    # fps grows RSS by gigabytes/minute. A few detections per
                    # second is plenty to notice someone walking up.
                    now = time.time()
                    if now - last_detect[0] < detect_interval:
                        time.sleep(0.005)
                        continue
                    last_detect[0] = now
                    faces = tracker.process_frame(frame)
                    focus_id = tracker.focus_track_id
                    for face in (faces or []):
                        top, right, bottom, left = face.bbox
                        is_focus = (focus_id is not None and face.track_id == focus_id)
                        pid = tracker.get_person_id(face.track_id)
                        if is_focus:
                            color, thickness = (100, 255, 0), 3
                        elif pid:
                            color, thickness = (0, 200, 0), 2
                        else:
                            color, thickness = (200, 0, 0), 1
                        cv2.rectangle(frame, (left, top), (right, bottom), color[::-1], thickness)
                        label = f"#{face.track_id}"
                        if pid:
                            label += f" {pid}"
                        if is_focus:
                            label += " [FOCUS]"
                        cv2.rectangle(frame, (left, bottom), (right, bottom + 22), color[::-1], cv2.FILLED)
                        cv2.putText(frame, label, (left + 4, bottom + 16),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    cam_win.set_camera_frame(rgb)
            except Exception:
                logger.exception("camera_loop crashed")
            finally:
                cap.release()

        threading.Thread(target=_camera_loop, daemon=True).start()
        print('Face tracker started')

        if has_init:
            ok = await client.read_resource("url://service_init")
            if ok:
                if has_name:
                    print('Initialized service '+name)
                else:
                    print('Initialized service')
            else:
                print('Failed to initialize service')
                return False

        ### Main loop 

        lang = default_lang
        prompt = ""
        txtlang = default_lang
        langprompt = False
        sysprompt = False
        augprompt = False

        ### New loop:
        # Initial is wait
        # Triggered by face in focus -> greet (handled as process)
        # As long same face in focus, process -> talk -> listen
        # When listening, sound triggers -> process (above)
        # Face out of focus -> wait

        # Persistent UI pump: repaints the eye window (state changes, VU
        # meters, camera thumbnail) whenever the main coroutine is awaiting —
        # e.g. during LLM inference or MCP tool calls. Without it the
        # 'Processing' state never gets painted before the blocking work.
        async def _pump_ui():
            while state.get('currstate') != 'exit':
                win.check_events()
                await asyncio.sleep(0.03)
        pump_task = asyncio.ensure_future(_pump_ui())

        set_state(state, 'wait')
        newstate = False
        prompt_source = None
        while True:

            # In this loop, state is either wait or listen
            # only in listen, audio recording is active, and may cause event
            # Otherwise we expect focus events
            # focus-out -> store profile, stop listen, go to wait
            # focus-in -> if listen first do focus out, fetch profile, go to greet
            # sound-ready -> go to process (stop listen?)
            while not newstate:
                await asyncio.sleep(0.05)
                newstate = check_statechange(state)
                ### Remove?
                if nb_available(sys.stdin):
                    res = nb_readline(sys.stdin)
                    if res:
                        res = res.strip(" \n")
                        if res[0:5] == "/lang":
                            txtlang = res[5:].strip(" ")
                            _ilog("stdin_command", cmd="/lang", arg=txtlang)
                        elif res[0:5] == "/exit":
                            _ilog("stdin_command", cmd="/exit", arg=None)
                            newstate = 'exit'
                        elif len(res):
                            _ilog("stdin_command", cmd="text", arg=res)
                            prompt = res
                            lang = txtlang
                            prompt_source = "stdin"
                            newstate = 'process'
                            set_state(state, newstate)
    
            if newstate == 'exit':
                break

            if newstate in ('wait', 'listen'):
                set_state(state, newstate)
                newstate = False
                continue

            if newstate == 'process' or newstate == 'greet':
                set_state(state, newstate)

                if newstate == 'process':
                    # prompt came from speech (curr_prompt) or stdin (prompt)
                    if curr_prompt:
                        prompt = curr_prompt
                        if voice_in and voice_in.detected_language:
                            lang = choose_language(lang, voice_in.detected_language, prompt)
                        curr_prompt = ""
                        prompt_source = "speech"
                        accept, vm = voice_gate(focus_state.get("person_id"))
                        if not accept:
                            # Someone else's voice: not the person we are
                            # talking to. Drop it and keep listening.
                            print(f"\n  (ignored: not {focus_state.get('person_id')}'s voice, "
                                  f"similarity {vm.similarity:.2f})")
                            _ilog("speech_rejected_voice", person_id=focus_state.get("person_id"),
                                  similarity=round(vm.similarity, 3))
                            set_state(state, 'listen')
                            newstate = False
                            continue

                if newstate == 'greet':
                    if curr_person and curr_person.lang and curr_person.lang in ['en','sv','de','fr','es']:
                        lang = curr_person.lang
                    else:
                        lang = default_lang

                if voice_in is not None and lang:
                    voice_in.language_hint = lang   # prior for the next STT call
                if curr_person is not None and lang in KNOWN_LANGS:
                    curr_person.lang = lang
                langprompt = language_message(lang)
                sysprompt = await system_message(client, lang)
                augprompt = await augmentation_message(client, lang)
                augpromptlist = []
                if augprompt:
                    print("\n  Augmentation:")
                    print(augprompt['content'])
                    augpromptlist.append(augprompt)
                augpromptlist.append(langprompt)
                direct_turn = (direct_llm is not None and newstate == 'process'
                               and prompt == RAW_AUDIO_TEXT
                               and voice_in is not None and voice_in.last_audio is not None)
                if direct_turn:
                    turn_audio = voice_in.last_audio
                    print("\n  User: (", lang, ") [audio, %.1fs]" % (len(turn_audio) / voice_in.sample_rate))
                    _ilog("user_turn", kind="speech_audio", content="",
                          lang=lang, seconds=round(len(turn_audio) / voice_in.sample_rate, 2))
                    prompt_source = None
                    pending_user_msg = {"role": "user", "content": "(audio — transcribing)"}
                    messages.append(pending_user_msg)
                elif newstate == 'process':
                    print("\n  User: (", lang, ") ", prompt)
                    _ilog("user_turn", kind=prompt_source or "unknown", content=prompt, lang=lang)
                    prompt_source = None
                    messages.append(user_message(prompt))
                elif newstate == 'greet':
                    if omit_names_and_prefs:
                        greetprompt = greet_prompt_noname()
                    else:
                        greetprompt = greet_prompt()
                    print("\n  Greeting:", greetprompt['content'])
                    _ilog("user_turn",
                          kind="greet_noname" if omit_names_and_prefs else "greet",
                          content=greetprompt['content'],
                          lang=lang)
                    messages.append(greetprompt)

                if direct_turn:
                    # One multimodal call: hear + persona + live scene + tools.
                    history = _text_history(messages[:-1], messages_trunclen)
                    t0 = time.time()
                    _ilog("direct_request", lang=lang, history_turns=len(history),
                          tool_count=len(direct_llm._tools))
                    try:
                        reply_text, tag_lang = await asyncio.to_thread(
                            direct_llm.respond, turn_audio,
                            language_hint=lang,
                            service_prompt=sysprompt['content'],
                            augmentation=augprompt['content'] if augprompt else None,
                            history=history)
                    except Exception as e:
                        logger.exception("direct-audio turn failed")
                        _ilog("llm_error", error_class=type(e).__name__, error_message=str(e))
                        reply_text, tag_lang = "", lang
                    if tag_lang in KNOWN_LANGS and tag_lang != lang:
                        # The model heard a language switch; trust it only on a
                        # real sentence (the tag misfires on short turns).
                        lang = choose_language(lang, tag_lang, reply_text)
                    for _tn, _ta, _tr in getattr(direct_llm, "rejected_tool_calls", []):
                        print(f"\n  Rejected: {_tn} ( {json.dumps(_ta)} )")
                        print(f"  Reason:   {_tr[:90]}")
                        _ilog("tool_call_rejected", name=_tn, arguments=_ta, reason=_tr)
                    for _tn, _ta, _tr in getattr(direct_llm, "last_tool_calls", []):
                        print(f"\n  Function: {_tn} ( {json.dumps(_ta)} )")
                        print(f"  Result:   {_tr}")
                        _ilog("tool_call", name=_tn, arguments=_ta, result=_tr)
                    messages.append({"role": "assistant", "content": reply_text})
                    _ilog("direct_response", content=reply_text, lang=lang,
                          tool_calls=[{"name": n, "arguments": a}
                                      for n, a, _ in getattr(direct_llm, "last_tool_calls", [])],
                          seconds=round(time.time() - t0, 2))
                    print(f"  (direct-audio turn: {time.time() - t0:.2f}s)")
                    threading.Thread(target=_direct_transcribe,
                                     args=(turn_audio, pending_user_msg, lang),
                                     daemon=True).start()
                else:
                    iteration = 0
                    msg = compose_messages(sysprompt, messages, augpromptlist)
                    #messagedump(msg)
                    _ilog("llm_request",
                          iteration=iteration,
                          model=model,
                          messages_full=list(messages),
                          messages_sent=msg,
                          tool_count=len(tools or []))
                    try:
                        # In a thread so the UI pump keeps the window alive during
                        # inference (and 'Processing' actually shows).
                        response = await asyncio.to_thread(
                            openai.chat.completions.create,
                            model=model,
                            messages=msg,
                            tools=tools,
                        )
                    except Exception as e:
                        _ilog("llm_error",
                              iteration=iteration,
                              error_class=type(e).__name__,
                              error_message=str(e))
                        raise
                    _ilog("llm_response",
                          iteration=iteration,
                          content=response.choices[0].message.content,
                          tool_calls=_serialize_tool_calls(response.choices[0].message.tool_calls))

                    tool_calls = response.choices[0].message.tool_calls
                    while tool_calls:
                        messages.append(response.choices[0].message)
                        for tool_call in tool_calls:
                            try:
                                args_parsed = json.loads(tool_call.function.arguments)
                            except (TypeError, ValueError):
                                args_parsed = None
                            try:
                                result = await client.call_tool(tool_call.function.name,
                                                                json.loads(tool_call.function.arguments))
                                if type(result)==list:
                                    resulttxt = result[0].text
                                else:
                                    resulttxt = result.content[0].text
                                result_message = {
                                    "role": "tool",
                                    "content": json.dumps({
                                        "result": resulttxt
                                    }),
                                    "tool_call_id": tool_call.id
                                }
                                print("\n  Function: ", tool_call.function.name, "(", tool_call.function.arguments, ")")
                                print(  "  Result:   ", resulttxt)
                                _ilog("mcp_tool_call",
                                      name=tool_call.function.name,
                                      arguments=tool_call.function.arguments,
                                      arguments_parsed=args_parsed,
                                      success=True,
                                      result_text=resulttxt)
                                messages.append(result_message)
                            except exceptions.ToolError as te:
                                result_message = {
                                    "role": "tool",
                                    "content": json.dumps({
                                        "result": "unknown function called"
                                    }),
                                    "tool_call_id": tool_call.id
                                }
                                print("\n  Unknown function: ", tool_call.function.name, "(", tool_call.function.arguments, ")")
                                _ilog("mcp_tool_call",
                                      name=tool_call.function.name,
                                      arguments=tool_call.function.arguments,
                                      arguments_parsed=args_parsed,
                                      success=False,
                                      error_message=f"ToolError: {te}")
                                messages.append(result_message)

                        iteration += 1
                        msg = compose_messages(sysprompt, messages, augpromptlist)
                        #messagedump(msg)
                        _ilog("llm_request",
                              iteration=iteration,
                              model=model,
                              messages_full=list(messages),
                              messages_sent=msg,
                              tool_count=len(tools or []))
                        try:
                            response = await asyncio.to_thread(
                                openai.chat.completions.create,
                                model=model,
                                messages=msg,
                                tools=tools,
                            )
                        except Exception as e:
                            _ilog("llm_error",
                                  iteration=iteration,
                                  error_class=type(e).__name__,
                                  error_message=str(e))
                            raise
                        _ilog("llm_response",
                              iteration=iteration,
                              content=response.choices[0].message.content,
                              tool_calls=_serialize_tool_calls(response.choices[0].message.tool_calls))
                        tool_calls = response.choices[0].message.tool_calls

                    # No tool calls, just print the response.
                    messages.append(response.choices[0].message)
                    reply_text = response.choices[0].message.content or ""
                print(f'\n  Response: {reply_text}  (lang={lang})')
                set_win_state('talk')
                if not reply_text:
                    _ilog("tts_speak", text="", lang=lang, status="empty")
                elif muted:
                    _ilog("tts_speak", text=reply_text, lang=lang, status="muted_suppressed")
                else:
                    # Simple TTS: pause mic for the whole utterance, no AEC.
                    # Resume is handled by the state-machine transition below.
                    listener.paused = True
                    # Show under "Speaking" which language and voice it uses.
                    tts_lang, tts_voice = voice_out.voice_for(reply_text, lang)
                    win.txt2.text.set_text(f"{tts_lang} · {tts_voice}")
                    voice_out.speak_async(reply_text, lang)
                    # Give the async thread a moment to flip speaking=True so we
                    # don't fall through (and skip the reverb wait) on a slow start.
                    deadline = time.monotonic() + 0.2
                    while not voice_out.speaking and time.monotonic() < deadline:
                        await asyncio.sleep(0.01)
                    if voice_out.speaking:
                        while voice_out.speaking:
                            win.check_events()
                            await asyncio.sleep(0.05)
                        if not voice_out.interrupted:
                            await asyncio.sleep(0.5)  # let room reverb decay before mic reopens
                    _ilog("tts_speak",
                          text=reply_text,
                          lang=lang,
                          status="interrupted" if getattr(voice_out, "interrupted", False) else "spoken")
                if state['newstate'] is None or state['newstate']=='listen':
                    set_state(state, 'listen')
                else:
                    state['currstate'] = None

                newstate = False
    
        if has_exit:
            ok = await client.read_resource("url://service_exit")
        ### Something corresponding to this in new library?
        #exit_audio()
        print('Exiting')

def run():
    global omit_names_and_prefs, talk_to_unknown

    args = parse_args()

    # Configure logging
    log_level = logging.WARNING
    if args.verbose >= 2:
        log_level = logging.DEBUG
    elif args.verbose >= 1:
        log_level = logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # List microphones and exit
    if args.list_mics:
        mics = list_input_devices()
        if mics:
            print("Available microphones:")
            for idx, name, channels, rate in mics:
                print(f"  Index {idx}: {name} ({channels}ch, {int(rate)}Hz)")
        else:
            print("No microphones found")
        sys.exit(0)

    # List cameras and exit
    if args.list_cameras:
        cameras = list_cameras()
        if cameras:
            print("Available cameras:")
            for c in cameras:
                label = f" {c['name']}" if c.get('name') else ""
                print(f"  Index {c['index']}:{label}  {c['width']}x{c['height']} "
                      f"@ {c['fps']:.1f} fps ({c['backend']})")
        else:
            print("No cameras found")
        sys.exit(0)

    # Load config file; CLI args take priority over config, config over built-in defaults
    cfg = load_config(args.config)

    if args.llm_model is None:
        args.llm_model = cfg["llm"]["model"]
    if args.llm_url is None:
        args.llm_url = cfg["llm"]["base_url"]
    if args.camera is None:
        args.camera = cfg["devices"].get("camera")
    if args.mic is None:
        args.mic = cfg["devices"].get("microphone")

    omit_names_and_prefs = cfg["face"]["omit_names_and_prefs"]
    talk_to_unknown = cfg["face"].get("talk_to_unknown", True)
    voice_cfg.update(cfg.get("voice_id", {}))
    if args.debug_audio is None:
        args.debug_audio = cfg["debug"].get("audio_panel", False)
    if args.direct_audio is None:
        args.direct_audio = cfg["llm"].get("direct_audio", False)
    if args.vad_near_field_ratio is None:
        args.vad_near_field_ratio = cfg["audio"].get("near_field_ratio")

    # A name or index may come from --camera or [devices] camera; resolving it
    # to a device that actually delivers video happens at open time.

    global interaction_log
    interaction_log = InteractionLogger.from_args(args)
    if interaction_log is not None:
        print(f"Interaction log: {interaction_log.path}")
        atexit.register(interaction_log.close, "atexit")
        _ilog("session_start",
              schema=1,
              pid=os.getpid(),
              git_commit=_git_commit_short(),
              args={
                  "server": args.server,
                  "llm_model": args.llm_model,
                  "llm_url": args.llm_url,
                  "lang_default": default_lang,
                  "mic": args.mic,
                  "camera": args.camera,
                  "omit_names_and_prefs": omit_names_and_prefs,
                  "log_path": interaction_log.path,
                  "log_dir": args.log_dir,
              })

    try:
        asyncio.run(main(args))
    except (ConnectionError, OSError) as e:
        print(f"ERROR: Cannot connect to MCP server at {args.server}: {e}")
        if interaction_log is not None:
            interaction_log.close("connection_error",
                                  error_class=type(e).__name__,
                                  error_message=str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nShutting down")
        if interaction_log is not None:
            interaction_log.close("keyboard_interrupt")
    except Exception as e:
        import traceback
        logger.error("Unexpected error: %s", e)
        if interaction_log is not None:
            interaction_log.close("exception",
                                  error_class=type(e).__name__,
                                  error_message=str(e),
                                  traceback=traceback.format_exc())
        sys.exit(1)
    else:
        if interaction_log is not None:
            interaction_log.close("normal")


if __name__ == "__main__":
    run()
