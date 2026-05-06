import pyaudio
import wave
import hardware_devices
from subprocess import run

DEFAULT_SAMPLE_RATE = 22050
audio = None
device_index = -1
device_sample_rate = DEFAULT_SAMPLE_RATE

whisperdir = "../../whisper.cpp/"
tempfile = "temp.wav"

def init_audio(microphone_name=None, sample_rate=0, device_idx=None) -> bool:
    global audio, device_index, device_sample_rate
    device_sample_rate = DEFAULT_SAMPLE_RATE if sample_rate == 0 else sample_rate
    audio = pyaudio.PyAudio()
    if device_idx is not None:
        try:
            info = audio.get_device_info_by_index(device_idx)
        except (IOError, OSError) as e:
            print(f"Error: invalid microphone index {device_idx}: {e}")
            return False
        device_index = device_idx
        print(f"Using microphone {info.get('name', str(device_idx))} at {device_idx}")
        return True
    if microphone_name is not None:
        device_index, device_name = hardware_devices.find_microphone_index(microphone_name, audio=audio, sample_rate=device_sample_rate)
        if device_index >= 0:
            print(f"Found microphone {device_name} at {device_index}")
            return True
        print(f"Error: could not find microphone matching {microphone_name} supporting sample rate {device_sample_rate}")
        return False
    try:
        info = audio.get_default_input_device_info()
    except IOError as e:
        print(f"Error: no default input device available: {e}")
        return False
    device_index = info["index"]
    print(f"Using default microphone {info.get('name', '')} at {device_index}")
    return True

def exit_audio():
    global audio
    if audio:
        audio.terminate()

def record(name, stopfunc, sfobj) -> bool:
    if device_index < 0:
        print("No microphone available")
        return False
    #stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    stream = audio.open(format=pyaudio.paInt16, channels=1,
                        rate=device_sample_rate, input=True,
                        frames_per_buffer=22050, input_device_index=device_index)
    with wave.open(name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        while not stopfunc(sfobj):
            data = stream.read(1024)
            wf.writeframes(data)
    stream.stop_stream()
    stream.close()
    return True

def transcribe(name):
    cmd = [whisperdir+"build/bin/whisper-cli", "-f", name, "-m", whisperdir+"models/ggml-medium.bin", "-nt", "-l", "auto"]
    res=run(cmd, capture_output=True, check=True)
    txt = res.stdout.decode("utf-8").strip('\n ')
    err = res.stderr.decode("utf-8")
    langpos = err.find("auto-detected language:")
    if langpos:
        lang = err[langpos+24:langpos+26]
    else:
        lang = 'en'
    return (txt, lang)

# import time
# import sys
# from readnb import *
#
# make_nonblocking(sys.stdin)
#
#def record_to_text():
#    print("Press return to start recording, and again to stop")
#    while not nb_available(sys.stdin):
#        time.sleep(0.05)
#    res = nb_readline(sys.stdin)
#    if not len(res):
#        res = record(tempfile, nb_available, sys.stdin)
#        nb_readline(sys.stdin)
#    if res is True:
#        print("Processing")
#        return transcribe(tempfile)
#    else:
#        return (res, 'en')

def speak(txt, lang):
    run(['./piperscript', lang if lang in ['en','sv','de','fr','es'] else 'en', txt])
