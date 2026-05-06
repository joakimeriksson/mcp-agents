#
# Generic MCP client, communicating with an MCP server running sse,
# and using a speechbased LLM interface with whisper and piper.
#

import asyncio
from fastmcp import Client, exceptions
from fastmcp.client.transports import SSETransport
import base64
import json
import time
import sys

import openai
from openai import OpenAI

from readnb import *
from eyewindow import *
from record import *
from config import load_config

messages_trunclen = 4
messages = []
state = {'evtime': 0, 'statetime': 0, 'newstate': None, 'currstate': None}
clicktimeout = 0.4

has_sysprompt = False
has_sysprompt_lang = False
has_augprompt = False
has_augprompt_lang = False
has_name = False
has_init = False
has_exit = False

win: EyeWindow | None = None

def on_exit(state):
    state['evtime'] = time.time()
    state['newstate'] = 'exit'

def on_press(_event, state):
    if state['currstate'] == 'ready':
        state['evtime'] = time.time()
        state['newstate'] = 'listening2'

def on_release(_event, state):
    if state['currstate'] == 'ready':
        state['evtime'] = time.time()
        state['newstate'] = 'listening1'
    elif state['currstate'] == 'listening1' or state['currstate'] == 'listening2':
        state['evtime'] = time.time()
        state['newstate'] = 'processing'

def check_statechange(state):
    win.check_events()
    t = time.time()
    if state['newstate'] == 'listening2' and t - state['evtime'] < clicktimeout:
        return False
    if state['newstate'] and state['newstate'] != state['currstate']:
        return state['newstate']
    elif state['newstate'] == 'exit':
        return 'exit'
    else:
        return False

def set_state(state, newstate):
    state['currstate'] = newstate
    state['statetime'] = time.time()
    state['newstate'] = None
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
    
def compose_messages(sysp, mlst, aug, langmsg):
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

    return [sysp] + mlst[i1:i2] + ([aug] if aug else []) + [langmsg] + mlst[i2:]

def clear_messages():
    global messages
    messages = []

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

def kp_repeat_last(_event, state):
    state['evtime'] = time.time()
    state['newstate'] = 'repeat'

async def main():
    global messages
    global tools
    global win
    global has_sysprompt
    global has_sysprompt_lang
    global has_augprompt
    global has_augprompt_lang
    global has_name
    global has_init
    global has_exit

    cfg = load_config()
    ollama_config = cfg["llm"]
    mic_index = cfg["devices"].get("microphone")

    # Connect via stdio to a local script
    async with Client(transport=SSETransport("http://127.0.0.1:8000/sse")) as client:
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

        if init_audio(device_idx=mic_index):
            print('Initialized speech recognition and synthesis')
        else:
            print('Error: failed to initialize audio')
            return False

        llm, model = init_llm(ollama_config)
        print(f'LLM Chatbot using model {model}')

        sdict = {'ready':      ((0, 0.7, 0.2), "Ready", "Press to start talking"),
                 'listening1': ((0, 0.6, 0.8), "Listening", "Press again when done talking"),
                 'listening2': ((0, 0.6, 0.8), "Listening", "Release when done talking"),
                 'processing': ((0.9, 0.5, 0), "Processing", "Please wait"),
                 'talking':    ((0.95, 0.75, 0), "~~~", ""),
                 }
        if has_name:
            tmp = await client.read_resource("url://service_name")
            name = tmp[0].text
        else:
            name = "MCP Speech Client"
        win = EyeWindow(name, sdict, 'ready')
        win.set_button_callbacks(on_press, on_release, state)
        win.set_exit_callback(on_exit, state)
        win.keydict["c"] = (kp_clear_messages, None)
        win.keydict["r"] = (kp_repeat_last, state)
        win.check_events()
        print('Created the interaction window')

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

        repeat = False
        lang = False
        prompt = ""
        txtlang = 'en'
        langprompt = False
        sysprompt = False
        augprompt = False
        while True:
            set_state(state, 'ready')
            newstate = False
            while not newstate:
                time.sleep(0.05)
                newstate = check_statechange(state)
                if newstate == 'repeat':
                    if sysprompt and lang and trim_last_message():
                        print("\n  (Repeating)")
                        repeat = True
                        newstate = 'processing'
                        set_state(state, newstate)
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
                            newstate = 'processing'
                            set_state(state, newstate)
    
            if newstate == 'exit':
                break
    
            if newstate == 'listening1' or newstate == 'listening2':
    
                set_state(state, newstate)
                tempfile = "temp.wav"
                if not record(tempfile, check_statechange, state):
                    break
                newstate = 'processing'
                set_state(state, newstate)
                prompt,lang = transcribe(tempfile)
                if not len(prompt):
                    newstate = 'ready'
    
            if newstate == 'processing':
                if not repeat:
                    langprompt = language_message(lang)
                    sysprompt = await system_message(client, lang)
                    augprompt = await augmentation_message(client, lang)
                    if augprompt:
                        print("\n  Augmentation:")
                        print(augprompt['content'])
                    print("\n  User: (", lang, ") ", prompt)
                    messages.append(user_message(prompt))
                else:
                    repeat = False

                response = openai.chat.completions.create(
                    model=model,
                    messages=compose_messages(sysprompt, messages, augprompt, langprompt),
                    tools=tools,
                )
    
                tool_calls = response.choices[0].message.tool_calls
                while tool_calls:
                    messages.append(response.choices[0].message)
                    for tool_call in tool_calls:
                        try:
                            result = await client.call_tool(tool_call.function.name,
                                                            json.loads(tool_call.function.arguments))
                            result_message = {
                                "role": "tool",
                                "content": json.dumps({
                                    "result": result[0].text
                                }),
                                "tool_call_id": tool_call.id
                            }
                            print("\n  Function: ", tool_call.function.name, "(", tool_call.function.arguments, ")")
                            print(  "  Result:   ", result[0].text)
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
    
                    response = openai.chat.completions.create(
                        model=model,
                        messages=compose_messages(sysprompt, messages, augprompt, lang),
                        tools=tools,
                    )
                    tool_calls = response.choices[0].message.tool_calls
    
                # No tool calls, just print the response.
                messages.append(response.choices[0].message)
                print(f'\n  Response: {response.choices[0].message.content}')
                check_statechange(state)
                set_state(state, 'talking')
                speak(response.choices[0].message.content, lang)
    
            if check_statechange(state) == 'exit':
                break

        if has_exit:
            ok = await client.read_resource("url://service_exit")
        exit_audio()
        print('Exiting')

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
