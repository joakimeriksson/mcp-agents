#
# Generic MCP client, communicating with an MCP server running sse,
# and using a speechbased LLM interface with whisper and piper.
#

import argparse
import asyncio
from fastmcp import Client, exceptions
from fastmcp.client.transports import SSETransport
import json
import logging
import os
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
from voice_input import VoiceInput, ContinuousListener, VoiceEventType, list_input_devices
from voice_output import VoiceOutput
from face_tracker import (
    FaceTracker, FaceDatabase, EmotionDetector, FaceEventType,
)
import cv2

logger = logging.getLogger("mcpclient_speech")

ollama_config = {
    "model": "PetrosStav/gemma3-tools:12b",
    "base_url": "http://localhost:11434/v1/",
    "api_key": "ollama"
}

default_lang = "sv"
messages_trunclen = 8
messages = []
state = {'evtime': 0, 'statetime': 0, 'newstate': None, 'currstate': None}

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
voice_in: VoiceInput | None = None
voice_out: VoiceOutput | None = None
listener: ContinuousListener | None = None
tracker: FaceTracker | None = None
model: str | None = None


def list_cameras(max_index=10):
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
            }
            available.append(info)
            cap.release()
    return available


def find_first_camera(max_index=10):
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="MCP Speech Client with Face Tracking")
    parser.add_argument('-l', '--list-cameras', action='store_true', help='List available cameras and exit')
    parser.add_argument('--camera', type=int, default=None, help='Camera index (default: auto-detect)')
    parser.add_argument('--server', default="http://127.0.0.1:7999/sse", help='MCP server SSE URL')
    parser.add_argument('--llm-model', default="PetrosStav/gemma3-tools:12b", help='LLM model name')
    parser.add_argument('--llm-url', default="http://localhost:11434/v1/", help='LLM base URL')
    parser.add_argument('-m', '--list-mics', action='store_true', help='List available microphones and exit')
    parser.add_argument('--mic', type=int, default=None, help='Microphone device index (default: system default)')
    parser.add_argument('-v', '--verbose', action='count', default=0, help='Increase verbosity (-v for INFO, -vv for DEBUG)')
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
    if muted:
        if listener:
            listener.paused = True
        logger.info("Muted")
        if win:
            win.set_state('muted')
    else:
        logger.info("Unmuted")
        if win:
            win.set_state(state.get('currstate', 'wait'))

def on_exit(state):
    logger.info("Exit event")
    state['evtime'] = time.time()
    state['newstate'] = 'exit'

def on_face_change(id):
    global messages, state, curr_person
    logger.info("Face change event")
    if muted or state['newstate'] == 'exit':
        return
    if curr_person:
        logger.info("Storing current person")
        curr_person.lasttime = time.time()
        if curr_person.lastmessages is not messages: # Does this work? I try to see if anything new has been said, otherwise there is no point extracting again. If there are many switches between people.
            curr_person.lastmessages = messages
            info = distill_user_info(extract_dialog_messages(messages))
            name = extract_value("Name:", info)
            if name:
                curr_person.name = name
            lang = extract_language(info)
            if lang:
                curr_person.lang = lang
            pref = extract_value("Preferences:", info)
            if pref:
                curr_person.profileinfo = pref
        else:
            print("(Nothing new to extract)")
    if id is None:
        messages = []
        curr_person = None
        state['evtime'] = time.time()
        state['newstate'] = 'wait'
    else:
        if id in persondict:
            curr_person = persondict[id]
            messages = list(curr_person.lastmessages or [])
            logger.info("Retrieving person %s from memory", id)
        else:
            curr_person = Person(None)
            persondict[id] = curr_person
            messages = []
            logger.info("Creating person %s", id)
        state['evtime'] = time.time()
        if state['evtime'] - curr_person.lasttime < 60:
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
        state['evtime'] = time.time()
        state['newstate'] = 'process'

def check_statechange(state):
    win.check_events()
    if state['newstate'] and state['newstate'] != state['currstate']:
        return state['newstate']
    elif state['newstate'] == 'exit':
        return 'exit'
    else:
        return False

def set_state(state, newstate):
    if listener:
        listener.paused = muted or (newstate != 'listen')
    state['currstate'] = newstate
    state['statetime'] = time.time()
    state['newstate'] = None
    win.set_state(newstate)
    win.check_events()

def set_win_state(newstate):
    win.set_state(newstate)
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
        return {"role": "system", "content": txt}
    else:
        return False

def user_message(prompt):
    return {"role": "user", "content": prompt}

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

async def main(args):
    global messages
    global tools
    global win
    global model
    global has_sysprompt
    global has_sysprompt_lang
    global has_augprompt
    global has_augprompt_lang
    global has_name
    global has_init
    global has_exit
    global voice_in, voice_out, listener, tracker
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
                 'talk':      ((0.95, 0.75, 0), "~~~", ""),
                 'muted':     ((0.4, 0.4, 0.4), "MUTED", "Press 'm' to unmute"),
                 }
        if has_name:
            tmp = await client.read_resource("url://get_service_name")
            name = tmp[0].text
        else:
            name = "MCP Speech Client"
        win = EyeWindow(name, sdict, 'ready')
        win.set_exit_callback(on_exit, state)
        win.keydict["m"] = (kp_toggle_mute, None)
        win.check_events()
        print('Created the interaction window')

        ### Initialize voice_input library, as ContinuousListener with on_speech as callback here
        voice_in = VoiceInput(device=args.mic)
        voice_in.subscribe(
            lambda ev: on_speech(ev.payload.text),
            event_types={VoiceEventType.TRANSCRIPTION_COMPLETE},
        )
        print('Loading whisper model...')
        voice_in.load_sync()
        if not voice_in.ready:
            print('Failed to load whisper model')
            return False
        listener = ContinuousListener(voice_in)
        listener.start()
        listener.paused = True
        print('Continuous listener started')

        ### Initialize voice_output (piper TTS)
        voice_out = VoiceOutput()
        print('Loading piper model...')
        voice_out.load_sync()
        if not voice_out.ready:
            print('Failed to load piper model')
            return False

        ### Initialize the face_tracker here, with on_face_change as callback
        face_db = FaceDatabase()
        face_db.load()
        emotion_detector = EmotionDetector()
        tracker = FaceTracker(db=face_db, emotion_detector=emotion_detector)

        def _on_face_event(ev):
            new_id = ev.payload.new_track_id
            on_face_change(new_id if new_id else None)

        tracker.subscribe(_on_face_event,
                          event_types={FaceEventType.FOCUS_CHANGED})

        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"ERROR: Could not open camera {args.camera}")
            sys.exit(1)

        def _camera_loop():
            try:
                while state.get('currstate') != 'exit' and state.get('newstate') != 'exit':
                    ret, frame = cap.read()
                    if not ret:
                        time.sleep(0.05)
                        continue
                    tracker.process_frame(frame)
                    if win and win._pending_frame is None:
                        win.set_camera_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
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

        set_state(state, 'wait')
        newstate = False
        while True:

            # In this loop, state is either wait or listen
            # only in listen, audio recording is active, and may cause event
            # Otherwise we expect focus events
            # focus-out -> store profile, stop listen, go to wait
            # focus-in -> if listen first do focus out, fetch profile, go to greet
            # sound-ready -> go to process (stop listen?)
            while not newstate:
                time.sleep(0.05)
                newstate = check_statechange(state)
                ### Remove?
                if nb_available(sys.stdin):
                    res = nb_readline(sys.stdin)
                    if res:
                        res = res.strip(" \n")
                        if res[0:5] == "/lang":
                            txtlang = res[5:].strip(" ")
                        elif res[0:5] == "/exit":
                            newstate = 'exit'
                        elif len(res):
                            prompt = res
                            lang = txtlang
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
                            lang = voice_in.detected_language
                        curr_prompt = ""

                if newstate == 'greet':
                    if curr_person and curr_person.lang and curr_person.lang in ['en','sv','de','fr','es']:
                        lang = curr_person.lang
                    else:
                        lang = default_lang

                langprompt = language_message(lang)
                sysprompt = await system_message(client, lang)
                augprompt = await augmentation_message(client, lang)
                augpromptlist = []
                if augprompt:
                    print("\n  Augmentation:")
                    print(augprompt['content'])
                    augpromptlist.append(augprompt)
                augpromptlist.append(langprompt)
                if newstate == 'process':
                    print("\n  User: (", lang, ") ", prompt)
                    messages.append(user_message(prompt))
                elif newstate == 'greet':
                    greetprompt = greet_prompt()
                    print("\n  Greeting:", greetprompt['content'])
                    messages.append(greetprompt)

                msg = compose_messages(sysprompt, messages, augpromptlist)
                #messagedump(msg)
                response = openai.chat.completions.create(
                    model=model,
                    messages=msg,
                    tools=tools,
                )
    
                tool_calls = response.choices[0].message.tool_calls
                while tool_calls:
                    messages.append(response.choices[0].message)
                    for tool_call in tool_calls:
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
                            messages.append(result_message)
                        except exceptions.ToolError:
                            result_message = {
                                "role": "tool",
                                "content": json.dumps({
                                    "result": "unknown function called"
                                }),
                                "tool_call_id": tool_call.id
                            }
                            print("\n  Unknown function: ", tool_call.function.name, "(", tool_call.function.arguments, ")")
                            messages.append(result_message)
    
                    msg = compose_messages(sysprompt, messages, augpromptlist)
                    #messagedump(msg)
                    response = openai.chat.completions.create(
                        model=model,
                        messages=msg,
                        tools=tools,
                    )
                    tool_calls = response.choices[0].message.tool_calls

                # No tool calls, just print the response.
                messages.append(response.choices[0].message)
                reply_text = response.choices[0].message.content or ""
                print(f'\n  Response: {reply_text}  (lang={lang})')
                set_win_state('talk')
                if reply_text:
                    # Simple TTS: pause mic for the whole utterance, no AEC.
                    # Resume is handled by the state-machine transition below.
                    listener.paused = True
                    voice_out.speak(reply_text, lang)
                    time.sleep(0.5)  # let room reverb decay before mic reopens
                if state['newstate'] is None or state['newstate']=='listen':
                    set_state(state, 'listen')

                newstate = False
    
        if has_exit:
            ok = await client.read_resource("url://service_exit")
        ### Something corresponding to this in new library?
        #exit_audio()
        print('Exiting')

def run():
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
                print(f"  Index {c['index']}: {c['width']}x{c['height']} @ {c['fps']:.1f} fps ({c['backend']})")
        else:
            print("No cameras found")
        sys.exit(0)

    # Resolve camera index
    if args.camera is None:
        args.camera = find_first_camera()
        if args.camera is not None:
            print(f"Auto-selected camera at index {args.camera}")
        else:
            print("ERROR: No cameras found. Use --camera N.")
            sys.exit(1)

    try:
        asyncio.run(main(args))
    except (ConnectionError, OSError) as e:
        print(f"ERROR: Cannot connect to MCP server at {args.server}: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nShutting down")
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    run()
