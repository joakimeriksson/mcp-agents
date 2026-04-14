#
# Generic MCP client, communicating with an MCP server running sse,
# and using a speechbased LLM interface with whisper and piper.
#

import asyncio
from fastmcp import Client, exceptions
from fastmcp.client.transports import SSETransport
import base64
import json
import os
import threading
import time
import sys

import openai
from openai import OpenAI

# voice_input / voice_output / face_tracker live in ../face/
_FACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'face')
if _FACE_DIR not in sys.path:
    sys.path.insert(0, _FACE_DIR)

from readnb import *
from eyewindow import *
from record import *
from voice_input import VoiceInput, ContinuousListener, VoiceEventType
from voice_output import VoiceOutput
from face_tracker import (
    FaceTracker, FaceDatabase, EmotionDetector, FaceEventType,
)
import cv2

ollama_config = {
    "model": "llama3.1",
    #"model": "google/gemma-4-E4B-it",
    #"model": "PetrosStav/gemma3-tools:12b",
    "base_url": "http://localhost:11434/v1/",
    "api_key": "ollama"
}

messages_trunclen = 8
messages = []
state = {'evtime': 0, 'statetime': 0, 'newstate': None, 'currstate': None}

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
        self.lang = "en"
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

def on_exit(state):
    print("\n(Exit event)")
    state['evtime'] = time.time()
    state['newstate'] = 'exit'

def on_face_change(id):
    global messages, state, curr_person
    print("\n(Face change event)")
    if state['newstate'] == 'exit':
        return
    if curr_person:
        curr_person.lasttime = time.time()
        curr_person.lastmessages = messages
        print("(Storing current person)")

    if id is None:
        messages = []
        curr_person = None
        state['evtime'] = time.time()
        state['newstate'] = 'wait'
    else:
        if id in persondict:
            curr_person = persondict[id]
            messages = list(curr_person.lastmessages or [])
            print(f"(Retrieving person {id} from memory)")
        else:
            curr_person = Person(None)
            persondict[id] = curr_person
            messages = []
            print(f"(Creating person {id})")
        state['evtime'] = time.time()
        state['newstate'] = 'greet'

def on_speech(txt):
    global curr_prompt, state
    print("\n(Speech event)")
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
        listener.paused = (newstate != 'listen')
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
    llm = OpenAI(api_key=openai.api_key)
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
    if not curr_person.name:
        return {'role':'user', 'content':'There is a new person in front of you. Produce a greeting and ask for the name.'}
    if curr_person.lasttime is None:
        return {'role':'user', 'content': f'The person {curr_person.name} has appeared in front of you. Produce a suitable greeting.'}
    duration = int((time.time() - curr_person.lasttime) / 60)
    return {'role':'user', 'content': f'The person {curr_person.name} has appeared in front of you. It was {duration} minutes since you last met. Produce a suitable greeting.'}

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

async def main():
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

    # Connect via stdio to a local script
    async with Client(transport=SSETransport("http://127.0.0.1:7999/sse")) as client:
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

        llm, model = init_llm(ollama_config)
        print(f'LLM Chatbot using model {model}')

        # new states: wait, listen, greet, process, talk
        sdict = {'wait':      ((0, 0.7, 0.2), "Ready", ""),
                 'listen':    ((0, 0.6, 0.8), "Listening", ""),
                 'greet':     ((0.9, 0.5, 0), "Contact", "Please wait"),
                 'process':   ((0.9, 0.5, 0), "Processing", "Please wait"),
                 'talk':      ((0.95, 0.75, 0), "~~~", ""),
                 }
        if has_name:
            tmp = await client.read_resource("url://get_service_name")
            name = tmp[0].text
        else:
            name = "MCP Speech Client"
        win = EyeWindow(name, sdict, 'ready')
        win.set_exit_callback(on_exit, state)
        #win.keydict["c"] = (kp_clear_messages, None)
        win.check_events()
        print('Created the interaction window')

        ### Initialize voice_input library, as ContinuousListener with on_speech as callback here
        voice_in = VoiceInput()
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
        #emotion_detector = EmotionDetector()
        tracker = FaceTracker(db=face_db) #, emotion_detector=emotion_detector)

        def _on_face_event(ev):
            new_id = ev.payload.new_track_id
            on_face_change(new_id if new_id else None)

        tracker.subscribe(_on_face_event,
                          event_types={FaceEventType.FOCUS_CHANGED})

        def _camera_loop():
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print('Could not open camera 0')
                return
            try:
                while state.get('currstate') != 'exit' and state.get('newstate') != 'exit':
                    ret, frame = cap.read()
                    if not ret:
                        time.sleep(0.05)
                        continue
                    tracker.process_frame(frame)
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

        lang = False
        prompt = ""
        txtlang = 'en'
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
                messagedump(msg)
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
                    messagedump(msg)
                    response = openai.chat.completions.create(
                        model=model,
                        messages=msg,
                        tools=tools,
                    )
                    tool_calls = response.choices[0].message.tool_calls

                # No tool calls, just print the response.
                messages.append(response.choices[0].message)
                reply_text = response.choices[0].message.content or ""
                print(f'\n  Response: {reply_text}')
                set_win_state('talk')
                if reply_text:
                    voice_out.speak(reply_text, lang)
                    #speak(reply_text, lang)
                set_state(state, 'listen')

                newstate = False
    
        if has_exit:
            ok = await client.read_resource("url://service_exit")
        ### Something corresponding to this in new library?
        #exit_audio()
        print('Exiting')

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
