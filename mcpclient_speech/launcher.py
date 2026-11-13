#!/usr/bin/env python3
"""CandyTron 4000 launcher: pick cameras/mic/robot, then start and watch the stack.

    ./launch-candytron.sh            (from the repo root)

Cameras are chosen by their LIVE PICTURE (OpenCV index 0..5), never by name:
on this Mac AVFoundation names/indices are unreliable and an idle iPhone
Continuity camera delivers black frames. Microphones are remembered by name
(sounddevice indices move when devices are plugged in).

Start runs ../start-candytron.sh in its own process group, streams its output
into the log pane and turns the known lines into a status checklist.
Stop (or closing the window) sends SIGINT to the group, then SIGTERM, then
SIGKILL. The launcher never talks to the robot beyond a TCP probe of :9090.

Test hooks: --autostart starts with the saved settings; --exit-when-ready S
stops the stack S seconds after the checklist reaches Ready and quits.
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import queue
import re
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
import urllib.request
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import cv2
import numpy as np
import sounddevice as sd
from PIL import Image, ImageTk

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "face"))
import prompts  # noqa: E402  face/prompts.py: the editable prompt texts

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
START_SCRIPT = os.path.join(ROOT, "start-candytron.sh")
CLIENT_SCRIPT = os.path.join(HERE, "mcpclient_speech_face.py")
CANDYTRON_DIR = os.path.join(ROOT, "candytron_mcp")
CONFIG_PATH = os.path.expanduser("~/.config/candytron/launcher.json")
RECAL_FLAG = os.environ.get("CANDYTRON_RECALIBRATE_FLAG", "/tmp/candytron-recalibrate")
LOG_DIR = os.environ.get("CANDYTRON_LOG_DIR", "/tmp")

OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "gemma4:latest"
VOICE_URL = "http://127.0.0.1:8880"
VOICE_PORT = 8880
DEFAULT_ROBOT_IP = "10.10.10.10"
ETHERNET_ROBOT_IP = "169.254.200.200"   # Niryo's static IP on the Ethernet port
ROBOT_MAC = "d8:3a:dd:6e:db:65"
ROBOT_PORT = 9090                       # rosbridge

N_CAMS = 6
THUMB_W, THUMB_H = 192, 108
BLACK_MEAN = 8.0

TABLE_COLOR = "#e07b00"
PEOPLE_COLOR = "#1f6feb"

DEFAULTS = {
    "table_cam": 1, "people_cam": 0, "mic_name": "", "robot_ip": DEFAULT_ROBOT_IP,
    "sim_robot": False, "sim_camera": False, "start_voice": True, "expressive": True,
    "extra_args": "", "ear": "whisper", "loose_pick": False,
}


# ---------------------------------------------------------------- helpers

def load_config() -> dict:
    cfg = dict(DEFAULTS)
    try:
        with open(CONFIG_PATH) as f:
            cfg.update({k: v for k, v in json.load(f).items() if k in DEFAULTS})
    except (OSError, ValueError):
        pass
    return cfg


def save_config(cfg: dict) -> None:
    try:
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        tmp = CONFIG_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cfg, f, indent=2)
        os.replace(tmp, CONFIG_PATH)
    except OSError as e:
        print(f"launcher: could not save {CONFIG_PATH}: {e}", file=sys.stderr)


def http_json(url: str, timeout: float = 2.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def tcp_open(host: str, port: int, timeout: float = 1.5) -> bool:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError:
        return False
    for fam, typ, proto, _cn, addr in infos:
        s = socket.socket(fam, typ, proto)
        s.settimeout(timeout)
        try:
            s.connect(addr)
            return True
        except OSError:
            pass
        finally:
            s.close()
    return False


def run(cmd: list[str], timeout: float = 5.0) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def norm_mac(mac: str) -> str:
    try:
        return ":".join(f"{int(p, 16):02x}" for p in mac.split(":"))
    except ValueError:
        return mac.lower()


def listening_pids(port: int) -> list[int]:
    out = run(["lsof", "-nP", f"-tiTCP:{port}", "-sTCP:LISTEN"])
    return [int(p) for p in out.split() if p.isdigit()]


def client_supports(flag: str) -> bool:
    try:
        with open(CLIENT_SCRIPT) as f:
            src = f.read()
        return f"'{flag}'" in src or f'"{flag}"' in src
    except OSError:
        return False


# ---------------------------------------------------------------- robot discovery

def wired_interfaces() -> list[str]:
    """Active en* interfaces that are not Wi-Fi (USB/Thunderbolt Ethernet, docks)."""
    ports = run(["networksetup", "-listallhardwareports"])
    wifi = set(re.findall(r"Hardware Port: Wi-Fi\s*\nDevice: (\S+)", ports))
    out = []
    for dev in run(["ifconfig", "-l"]).split():
        if not dev.startswith("en") or dev in wifi:
            continue
        if "status: active" in run(["ifconfig", dev]):
            out.append(dev)
    return out


def find_robot(candidates: list[str], log) -> str | None:
    """Try known addresses, then ARP/NDP for the robot's MAC. Returns an IPv4 or None."""
    seen = []
    for ip in candidates:
        if ip and ip not in seen:
            seen.append(ip)
    log(f"probing TCP {ROBOT_PORT} on {', '.join(seen)} ...")
    hits = {}
    threads = [threading.Thread(target=lambda ip=ip: hits.__setitem__(ip, tcp_open(ip, ROBOT_PORT)))
               for ip in seen]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for ip in seen:
        if hits.get(ip):
            log(f"robot answers on {ip}:{ROBOT_PORT}")
            return ip

    mac = norm_mac(ROBOT_MAC)

    def arp_lookup():
        for line in run(["arp", "-an"]).splitlines():
            m = re.search(r"\((\d+\.\d+\.\d+\.\d+)\) at ([0-9a-fA-F:]+) on (\S+)", line)
            if m and norm_mac(m.group(2)) == mac:
                return m.group(1), m.group(3)
        return None

    ifaces = wired_interfaces()
    log(f"wired interfaces: {', '.join(ifaces) or 'none active'}")
    hit = arp_lookup()
    v6 = None
    if not hit:
        for dev in ifaces:
            log(f"ping6 all-nodes on {dev} ...")
            run(["ping6", "-c", "2", f"ff02::1%{dev}"], timeout=6)
            # If we have a link-local IPv4 on that interface, wake ARP too.
            if re.search(r"inet 169\.254\.", run(["ifconfig", dev])):
                run(["ping", "-c", "2", "-t", "3", "169.254.255.255"], timeout=5)
        for line in run(["ndp", "-an"]).splitlines():
            parts = line.split()
            if len(parts) >= 3 and norm_mac(parts[1]) == mac:
                v6 = parts[0]
                break
        hit = arp_lookup()
    if hit:
        ip, dev = hit
        ok = tcp_open(ip, ROBOT_PORT)
        log(f"robot MAC found at {ip} on {dev}; port {ROBOT_PORT} "
            f"{'open' if ok else 'NOT answering (robot still booting?)'}")
        return ip
    if v6:
        ok = tcp_open(v6, ROBOT_PORT)
        log(f"robot is on the cable at IPv6 {v6} (port {ROBOT_PORT} "
            f"{'open' if ok else 'closed'}) but its IPv4 address is unknown. "
            f"Its Ethernet default is {ETHERNET_ROBOT_IP}; check the Mac's Ethernet "
            f"interface is on 169.254.x.x (DHCP/link-local), or check Niryo Studio.")
        return None
    log("robot not found. Is it powered on (takes ~1-2 min to boot) and the cable/Wi-Fi connected?")
    return None


# ---------------------------------------------------------------- cameras

class CamReader(threading.Thread):
    def __init__(self, idx: int, cap):
        super().__init__(daemon=True, name=f"cam{idx}")
        self.idx, self.cap = idx, cap
        self.stop_ev = threading.Event()
        self.lock = threading.Lock()
        self.thumb = None          # RGB uint8 THUMB_H x THUMB_W
        self.full = None           # last full BGR frame
        self.mean = None
        self.size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.frames = 0
        self.t_open = time.time()

    def run(self):
        last = 0.0
        try:
            while not self.stop_ev.is_set():
                ok, f = self.cap.read()
                if not ok or f is None:
                    time.sleep(0.05)
                    continue
                self.frames += 1
                now = time.time()
                if now - last < 0.2:
                    continue
                last = now
                h, w = f.shape[:2]
                scale = min(THUMB_W / w, THUMB_H / h)
                small = cv2.resize(f, (max(1, int(w * scale)), max(1, int(h * scale))),
                                   interpolation=cv2.INTER_AREA)
                canvas = np.zeros((THUMB_H, THUMB_W, 3), np.uint8)
                y0, x0 = (THUMB_H - small.shape[0]) // 2, (THUMB_W - small.shape[1]) // 2
                canvas[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
                with self.lock:
                    self.thumb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
                    self.full = f
                    self.mean = float(small.mean())
                    self.size = (w, h)
        finally:
            self.cap.release()

    def state(self) -> str:
        """'ok', 'black', 'waiting', 'noframes'"""
        if self.mean is None:
            return "waiting" if time.time() - self.t_open < 4 else "noframes"
        return "black" if self.mean < BLACK_MEAN else "ok"


class CameraPreviews:
    """Opens OpenCV indices 0..N_CAMS-1, one opener thread per index (so a
    camera that hangs while opening can't freeze the UI or the others), and
    keeps one reader thread per opened camera.

    Note: AVFoundation can only *ask* for camera permission from the main
    thread; the terminal that runs this must already have Camera permission
    (it does if the stack has ever run from it). If not, every index shows
    "no camera" and the note below the tiles says so.
    """

    def __init__(self):
        self.readers: dict[int, CamReader] = {}
        self.missing: set[int] = set()
        self.pending: set[int] = set()
        self.lock = threading.Lock()
        self._gen = 0

    @property
    def opening(self) -> bool:
        return bool(self.pending)

    def start(self):
        with self.lock:
            if self.pending or self.readers:
                return
            self._gen += 1
            gen = self._gen
            self.missing = set()
            self.pending = set(range(N_CAMS))
        for i in range(N_CAMS):
            threading.Thread(target=self._open, args=(i, gen, 0.15 * i), daemon=True).start()

    def _open(self, i, gen, delay):
        time.sleep(delay)
        cap = cv2.VideoCapture(i) if gen == self._gen else None
        with self.lock:
            self.pending.discard(i)
            if cap is None or gen != self._gen:      # stopped meanwhile
                if cap is not None:
                    cap.release()
                return
            if not cap.isOpened():
                cap.release()
                self.missing.add(i)
                return
            r = CamReader(i, cap)
            self.readers[i] = r
        r.start()

    def stop(self, timeout: float = 4.0) -> list[int]:
        """Release every camera. Returns indices whose reader did not stop."""
        t0 = time.time()
        with self.lock:
            self._gen += 1
            readers = list(self.readers.values())
            self.readers = {}
            self.pending = set()
        for r in readers:
            r.stop_ev.set()
        stuck = []
        for r in readers:
            r.join(max(0.1, timeout - (time.time() - t0)))
            if r.is_alive():
                stuck.append(r.idx)
        return stuck

    def get(self, idx: int) -> CamReader | None:
        with self.lock:
            return self.readers.get(idx)


# ---------------------------------------------------------------- microphone

class MicMeter:
    def __init__(self):
        self.stream = None
        self.level = 0.0     # 0..1 (dB-scaled)
        self.peak_hold = 0.0
        self.error = ""

    def open(self, device: int | None):
        self.close()
        self.error = ""
        try:
            info = sd.query_devices(device, "input")
            rate = int(info["default_samplerate"])

            def cb(indata, frames, t, status):
                rms = float(np.sqrt(np.mean(np.square(indata[:, 0], dtype=np.float64))) + 1e-9)
                db = 20 * math.log10(rms)
                self.level = max(0.0, min(1.0, (db + 60) / 60))

            self.stream = sd.InputStream(device=device, channels=1, samplerate=rate,
                                         blocksize=int(rate * 0.05), callback=cb)
            self.stream.start()
        except Exception as e:  # noqa: BLE001 — any PortAudio failure just shows
            self.stream = None
            self.error = str(e)

    def close(self):
        s, self.stream = self.stream, None
        self.level = 0.0
        if s is not None:
            try:
                s.stop()
                s.close()
            except Exception:  # noqa: BLE001
                pass


def input_devices() -> list[tuple[int, str]]:
    return [(i, d["name"]) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]


# ---------------------------------------------------------------- status checklist

STEPS = [
    ("ollama", "Ollama (LLM)"),
    ("voice", "Voice server :8880"),
    ("table", "Table camera"),
    ("calib", "Board calibration"),
    ("mcp", "MCP server :7999"),
    ("client", "Client (eye, mic, models)"),
    ("robot", "Robot arm"),
    ("ready", "READY"),
]
STATE_STYLE = {   # glyph, colour
    "pending": ("○", "#888888"),
    "progress": ("◐", "#e07b00"),
    "done": ("✓", "#1a7f37"),
    "warn": ("!", "#b58900"),
    "failed": ("✗", "#cf222e"),
    "skipped": ("–", "#888888"),
}


class Launcher:
    def __init__(self, root: tk.Tk, args):
        self.root = root
        self.args = args
        self.cfg = load_config()
        if args.sim:
            self.cfg.update(sim_robot=True, sim_camera=True)
        self.q: queue.Queue = queue.Queue()
        self.proc: subprocess.Popen | None = None
        self.stopping = False
        self.closing = False
        self.ready_at: float | None = None
        self.logfile = None
        self.health = None
        self.restart_voice_touched = False
        self.yolo_running = False

        self.cams = CameraPreviews()
        self.mic = MicMeter()
        self.mics: list[tuple[int, str]] = []

        root.title("CandyTron 4000 — launcher")
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        # The stack runs in its own session, so Ctrl-C / a closed terminal
        # would not reach it: treat those like closing the window (stop the
        # stack first). Handlers run on the next _tick (every 100 ms).
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, lambda *_: self.root.after(0, self.on_close))
        atexit.register(self._atexit_kill)
        self._build_ui()
        self.cams.start()
        self._refresh_mics(select_name=self.cfg["mic_name"])
        self.root.after(150, self._tick)
        self.root.after(500, self.check)
        self.root.after(5000, lambda: self.auto_detect(quiet_if_unsure=True))
        if args.autostart:
            self.root.after(2500, self.start)

    # ------------------------------------------------------------ UI
    def _build_ui(self):
        r = self.root
        style = ttk.Style()
        style.configure("Big.TButton", font=("Helvetica", 15, "bold"), padding=8)
        bold = ("Helvetica", 13, "bold")

        # ---- cameras
        camf = ttk.LabelFrame(r, text=" Cameras — choose by the PICTURE: "
                                       "TABLE = looks down at the candy board, PEOPLE = faces ")
        camf.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 4))
        self.table_var = tk.IntVar(value=int(self.cfg["table_cam"]))
        self.people_var = tk.IntVar(value=int(self.cfg["people_cam"]))
        self.tiles = []
        blank = ImageTk.PhotoImage(Image.new("RGB", (THUMB_W, THUMB_H), (40, 40, 40)))
        self._blank = blank
        self._last_thumb: dict = {}   # camera index -> its last good picture (RGB thumb)
        for i in range(N_CAMS):
            f = tk.Frame(camf, highlightthickness=4, highlightbackground="#dddddd", bd=0)
            f.grid(row=0, column=i, padx=4, pady=4)
            img = tk.Label(f, image=blank, bd=0)
            img.grid(row=0, column=0, columnspan=2)
            cap = tk.Label(f, text=f"#{i}", font=("Helvetica", 11), width=21)
            cap.grid(row=1, column=0, columnspan=2, sticky="ew")
            tb = tk.Radiobutton(f, text="TABLE", variable=self.table_var, value=i,
                                fg=TABLE_COLOR, command=self._cams_changed)
            pb = tk.Radiobutton(f, text="PEOPLE", variable=self.people_var, value=i,
                                fg=PEOPLE_COLOR, command=self._cams_changed)
            tb.grid(row=2, column=0)
            pb.grid(row=2, column=1)
            img.bind("<Button-1>", lambda e, i=i: self._tile_click(i))
            img.bind("<Shift-Button-1>", lambda e, i=i: self._tile_click(i, people=True))
            self.tiles.append((f, img, cap))
        self.cam_note = ttk.Label(camf, text="Click a picture = TABLE, shift-click = PEOPLE (or use the buttons). "
                                             "Black = idle iPhone / covered lens.")
        self.cam_note.configure(wraplength=1000)
        self.cam_note.grid(row=1, column=0, columnspan=N_CAMS - 1, sticky="w", padx=6)
        cbtns = ttk.Frame(camf)
        cbtns.grid(row=1, column=N_CAMS - 1, sticky="e", padx=6, pady=(0, 4))
        self.auto_btn = ttk.Button(cbtns, text="Auto-detect", command=self.auto_detect)
        self.auto_btn.pack(side="left", padx=(0, 4))
        ttk.Button(cbtns, text="Re-scan cameras", command=self.rescan_cameras).pack(side="left")
        self.auto_running = False

        # ---- left column: config
        left = ttk.Frame(r)
        left.grid(row=1, column=0, sticky="nsew", padx=10, pady=4)

        micf = ttk.LabelFrame(left, text=" Microphone — speak and watch the meter ")
        micf.grid(row=0, column=0, sticky="ew", pady=4)
        self.mic_list = tk.Listbox(micf, height=6, exportselection=False, width=44)
        self.mic_list.grid(row=0, column=0, columnspan=2, sticky="ew", padx=6, pady=4)
        self.mic_list.bind("<<ListboxSelect>>", lambda e: self._mic_selected())
        self.meter = tk.Canvas(micf, width=300, height=18, bg="#222222", highlightthickness=0)
        self.meter.grid(row=1, column=0, sticky="w", padx=6, pady=(0, 6))
        ttk.Button(micf, text="Re-scan", command=lambda: self._refresh_mics(rescan=True)).grid(
            row=1, column=1, sticky="e", padx=6)
        self.mic_err = ttk.Label(micf, text="", foreground="#cf222e")
        self.mic_err.grid(row=2, column=0, columnspan=2, sticky="w", padx=6)

        robf = ttk.LabelFrame(left, text=" Robot (Niryo Ned 2) ")
        robf.grid(row=1, column=0, sticky="ew", pady=4)
        ttk.Label(robf, text="IP:").grid(row=0, column=0, padx=6, pady=4)
        self.ip_var = tk.StringVar(value=self.cfg["robot_ip"])
        ttk.Entry(robf, textvariable=self.ip_var, width=18).grid(row=0, column=1, sticky="w")
        self.find_btn = ttk.Button(robf, text="Find robot", command=self.find_robot)
        self.find_btn.grid(row=0, column=2, padx=6)
        self.sim_robot = tk.BooleanVar(value=self.cfg["sim_robot"])
        self.sim_camera = tk.BooleanVar(value=self.cfg["sim_camera"])
        ttk.Checkbutton(robf, text="Simulate robot", variable=self.sim_robot,
                        command=self.check).grid(row=1, column=0, columnspan=2, sticky="w", padx=6)
        ttk.Checkbutton(robf, text="Simulate table camera", variable=self.sim_camera,
                        command=self.check).grid(row=1, column=2, sticky="w", padx=6)
        self.find_msg = ttk.Label(robf, text="", wraplength=430, justify="left")
        self.find_msg.grid(row=2, column=0, columnspan=3, sticky="w", padx=6, pady=(0, 4))

        vf = ttk.LabelFrame(left, text=" Voice ")
        vf.grid(row=2, column=0, sticky="ew", pady=4)
        self.start_voice = tk.BooleanVar(value=self.cfg["start_voice"])
        self.restart_voice = tk.BooleanVar(value=False)
        self.expressive = tk.BooleanVar(value=self.cfg["expressive"])
        ttk.Checkbutton(vf, text="Start voice server (Kokoro + VoxCPM2 Klara/Viktor)",
                        variable=self.start_voice).grid(row=0, column=0, sticky="w", padx=6)
        ttk.Checkbutton(vf, text="Restart voice server (stop the one on :8880 first)",
                        variable=self.restart_voice,
                        command=lambda: setattr(self, "restart_voice_touched", True)).grid(
            row=1, column=0, sticky="w", padx=6)
        ttk.Checkbutton(vf, text="Expressive voice (laughs, sighs, fillers)",
                        variable=self.expressive).grid(row=2, column=0, sticky="w", padx=6)
        ef = ttk.Frame(vf)
        ef.grid(row=3, column=0, sticky="w", padx=6, pady=4)
        ttk.Label(ef, text="Extra client args:").pack(side="left")
        self.extra_var = tk.StringVar(value=self.cfg["extra_args"])
        ttk.Entry(ef, textvariable=self.extra_var, width=26).pack(side="left", padx=4)

        pf = ttk.LabelFrame(left, text=" Pre-flight ")
        pf.grid(row=3, column=0, sticky="ew", pady=4)
        self.pre = {}
        for row, (key, name) in enumerate([("ollama", "Ollama + gemma4"), ("voice", "Voice server"),
                                           ("robot", "Robot :9090"), ("table", "Table camera"),
                                           ("people", "People camera")]):
            dot = tk.Label(pf, text="●", fg="#888888", font=("Helvetica", 14))
            dot.grid(row=row, column=0, padx=(6, 2))
            ttk.Label(pf, text=name, width=13).grid(row=row, column=1, sticky="w")
            msg = ttk.Label(pf, text="", wraplength=330, justify="left")
            msg.grid(row=row, column=2, sticky="w")
            self.pre[key] = (dot, msg)
        bf = ttk.Frame(pf)
        bf.grid(row=5, column=0, columnspan=3, sticky="w", padx=6, pady=4)
        ttk.Button(bf, text="Check", command=self.check).pack(side="left")
        self.yolo_btn = ttk.Button(bf, text="Count candies (YOLO)", command=self.yolo_check)
        self.yolo_btn.pack(side="left", padx=6)
        ttk.Button(bf, text="Prompts…", command=self.open_prompts).pack(side="left")

        btns = ttk.Frame(left)
        btns.grid(row=4, column=0, sticky="ew", pady=8)
        self.start_btn = ttk.Button(btns, text="▶  START", style="Big.TButton", command=self.start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(btns, text="■  STOP", style="Big.TButton", command=self.stop,
                                   state="disabled")
        self.stop_btn.pack(side="left", padx=8)
        self.recal_btn = ttk.Button(btns, text="Recalibrate board", command=self.recalibrate,
                                    state="disabled")
        self.recal_btn.pack(side="left")

        # ---- right column: status + log
        right = ttk.Frame(r)
        right.grid(row=1, column=1, sticky="nsew", padx=(0, 10), pady=4)
        r.columnconfigure(1, weight=1)
        r.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)

        self.banner = tk.Label(right, text="Not started", font=("Helvetica", 17, "bold"),
                               bg="#eeeeee", fg="#333333", wraplength=640, justify="left",
                               anchor="w", padx=10, pady=8)
        self.banner.grid(row=0, column=0, sticky="ew", pady=(4, 6))

        sf = ttk.LabelFrame(right, text=" Startup ")
        sf.grid(row=1, column=0, sticky="ew")
        sf.columnconfigure(2, weight=1)
        self.steps = {}
        for row, (key, name) in enumerate(STEPS):
            g = tk.Label(sf, text="○", font=("Helvetica", 16, "bold"), width=2, fg="#888888")
            g.grid(row=row, column=0)
            n = tk.Label(sf, text=name, font=bold, anchor="w", width=22)
            n.grid(row=row, column=1, sticky="w")
            d = tk.Label(sf, text="", anchor="w", justify="left", wraplength=420)
            d.grid(row=row, column=2, sticky="w")
            self.steps[key] = [g, d, "pending"]

        self.log = ScrolledText(right, height=12, font=("Menlo", 10), wrap="none")
        self.log.grid(row=2, column=0, sticky="nsew", pady=(6, 0))
        self.log.tag_configure("ct", foreground="#0550ae")
        self.log.tag_configure("err", foreground="#cf222e")
        self.log.tag_configure("warn", foreground="#9a6700")
        self.log.tag_configure("me", foreground="#6f42c1")

    # ------------------------------------------------------------ small helpers
    def say(self, text: str, tag: str = "me"):
        self.q.put(("log", f"[launcher] {text}", tag))

    def set_step(self, key, state=None, detail=None):
        s = self.steps[key]
        if state:
            s[2] = state
            glyph, col = STATE_STYLE[state]
            s[0].configure(text=glyph, fg=col)
        if detail is not None:
            s[1].configure(text=detail)

    def set_pre(self, key, ok, msg):
        col = {True: "#1a7f37", False: "#cf222e", None: "#b58900", "skip": "#888888"}[ok]
        dot, lab = self.pre[key]
        dot.configure(fg=col)
        lab.configure(text=msg)

    def set_banner(self, text, kind="info"):
        bg, fg = {"info": ("#eeeeee", "#333333"), "busy": ("#fff1d6", "#8a4b00"),
                  "calib": ("#ffd8a8", "#7a2e00"), "ok": ("#d3f5dc", "#0b5d24"),
                  "warn": ("#fff3b0", "#6b5200"), "bad": ("#ffd7d5", "#8c1018")}[kind]
        self.banner.configure(text=text, bg=bg, fg=fg)

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    # ------------------------------------------------------------ cameras
    @staticmethod
    def _frozen(thumb, label: str):
        """A dimmed copy of a camera's last picture with *label* on it."""
        out = (thumb.astype(np.float32) * 0.45).astype(np.uint8).copy()
        h, w = out.shape[:2]
        cv2.rectangle(out, (0, h - 26), (w, h), (0, 0, 0), -1)
        cv2.putText(out, label[:42], (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return out

    def _tile_click(self, i, people=False):
        if self.running():
            return
        (self.people_var if people else self.table_var).set(i)
        self._cams_changed()

    def _cams_changed(self):
        self.check(quick=True)

    def rescan_cameras(self):
        if self.running():
            return
        self.cams.stop()
        self.cams.start()

    def _update_tiles(self):
        t, p = self.table_var.get(), self.people_var.get()
        running = self.running()
        if not running and not self.cams.opening and len(self.cams.missing) == N_CAMS:
            self.cam_note.configure(foreground="#cf222e", text=(
                "NO camera opened. Plug in the cameras and press Re-scan. If they are "
                "plugged in: give this terminal app Camera access (System Settings > "
                "Privacy & Security > Camera) and restart it."))
        elif str(self.cam_note.cget("foreground")) == "#cf222e":
            self.cam_note.configure(foreground="", text=(
                "Click a picture = TABLE, shift-click = PEOPLE (or use the buttons). "
                "Black = idle iPhone / covered lens."))
        for i, (f, img, cap) in enumerate(self.tiles):
            border = TABLE_COLOR if (i == t and not self.sim_camera.get()) else \
                PEOPLE_COLOR if i == p else "#dddddd"
            f.configure(highlightbackground=border, highlightcolor=border)
            r = self.cams.get(i)
            if running or r is None:
                # No live picture (released to the stack, reopening, or the
                # camera vanished): keep showing its LAST picture, dimmed and
                # labelled, so it stays obvious which number was which.
                role = ("TABLE" if i == t and not self.sim_camera.get()
                        else "PEOPLE" if i == p else "")
                if running:
                    why = f"in use by the stack{' as ' + role if role else ''}"
                elif self.cams.opening:
                    why = "reopening…"
                else:
                    why = "GONE (unplugged / iPhone left?)" if i in self._last_thumb else "no camera"
                txt = f"#{i}  {why}" if i not in self._last_thumb else f"#{i}  last picture — {why}"
                key = (i, why)
                if i in self._last_thumb:
                    if getattr(img, "_frozen_key", None) != key:
                        ph = ImageTk.PhotoImage(Image.fromarray(self._frozen(self._last_thumb[i], why)))
                        img.configure(image=ph)
                        img._ph, img._frozen_key = ph, key
                elif img.cget("image") != str(self._blank):
                    img.configure(image=self._blank)
                    img._frozen_key = None
            else:
                st = r.state()
                with r.lock:
                    thumb = r.thumb
                if thumb is not None:
                    if st == "ok":
                        self._last_thumb[i] = thumb
                    ph = ImageTk.PhotoImage(Image.fromarray(thumb))
                    img.configure(image=ph)
                    img._ph, img._frozen_key = ph, None
                w, h = r.size
                txt = {"ok": f"#{i}  {w}x{h}", "black": f"#{i}  BLACK (idle iPhone?)",
                       "waiting": f"#{i}  waiting for picture…",
                       "noframes": f"#{i}  NO PICTURE"}[st]
            cap.configure(text=txt, fg="#cf222e" if ("BLACK" in txt or "NO PIC" in txt) else "black")

    # ------------------------------------------------------------ microphone
    def _refresh_mics(self, select_name: str | None = None, rescan: bool = False):
        if self.running():
            return
        cur = select_name if select_name is not None else self._mic_name()
        self.mic.close()
        if rescan:
            try:
                sd._terminate()
                sd._initialize()
            except Exception as e:  # noqa: BLE001
                self.say(f"mic re-scan failed: {e}", "err")
        try:
            self.mics = input_devices()
        except Exception as e:  # noqa: BLE001
            self.mics = []
            self.mic_err.configure(text=f"cannot list microphones: {e}")
        self.mic_list.delete(0, "end")
        self.mic_list.insert("end", "  System default input")
        for idx, name in self.mics:
            self.mic_list.insert("end", f"  [{idx}] {name}")
        sel = 0
        for row, (_idx, name) in enumerate(self.mics, start=1):
            if cur and name == cur:
                sel = row
        self.mic_list.selection_clear(0, "end")
        self.mic_list.selection_set(sel)
        self.mic_list.see(sel)
        self._mic_selected()

    def _mic_choice(self) -> tuple[int | None, str]:
        sel = self.mic_list.curselection()
        if not sel or sel[0] == 0 or sel[0] - 1 >= len(self.mics):
            return None, ""
        return self.mics[sel[0] - 1]

    def _mic_name(self) -> str:
        return self._mic_choice()[1]

    def _mic_selected(self):
        if self.running():
            return
        idx, _name = self._mic_choice()
        self.mic.open(idx)
        self.mic_err.configure(text=f"cannot open: {self.mic.error}" if self.mic.error else "")

    def _draw_meter(self):
        c = self.meter
        c.delete("all")
        if self.running():
            c.create_text(150, 9, text="released (in use by the stack)", fill="#aaaaaa",
                          font=("Helvetica", 10))
            return
        lv = self.mic.level
        self.mic.peak_hold = max(lv, self.mic.peak_hold - 0.02)
        w = int(300 * lv)
        col = "#2da44e" if lv < 0.75 else "#d4a72c" if lv < 0.9 else "#cf222e"
        c.create_rectangle(0, 0, w, 18, fill=col, width=0)
        x = int(300 * self.mic.peak_hold)
        c.create_line(x, 0, x, 18, fill="white")

    # ------------------------------------------------------------ pre-flight
    def check(self, quick: bool = False):
        """Non-blocking pre-flight. quick=True only re-evaluates the cameras."""
        if not quick:
            threading.Thread(target=self._check_services, daemon=True).start()
        self._check_cameras()

    def _check_services(self):
        try:
            tags = http_json(f"{OLLAMA_URL}/api/tags")
            names = [m.get("name") for m in tags.get("models", [])]
            if LLM_MODEL in names:
                self.q.put(("pre", "ollama", True, f"up, {LLM_MODEL} present"))
            else:
                self.q.put(("pre", "ollama", False, f"up but {LLM_MODEL} MISSING (ollama pull gemma4)"))
        except Exception:  # noqa: BLE001
            self.q.put(("pre", "ollama", None, "not running (Start will launch it)"))
        try:
            h = http_json(f"{VOICE_URL}/health")
            self.q.put(("health", h))
        except Exception:  # noqa: BLE001
            self.q.put(("health", None))
        if self.sim_robot.get():
            self.q.put(("pre", "robot", "skip", "simulated"))
        else:
            ip = self.ip_var.get().strip()
            ok = bool(ip) and tcp_open(ip, ROBOT_PORT)
            self.q.put(("pre", "robot", ok, f"{ip}:{ROBOT_PORT} reachable" if ok else
                        f"{ip or '(no IP)'}:{ROBOT_PORT} NOT reachable — try Find robot"))

    def _apply_health(self, h):
        if h is None:
            self.health = None
            if self.start_voice.get():
                self.set_pre("voice", None, "not running (Start will launch it; cold start takes minutes)")
            else:
                self.set_pre("voice", False, "not running and 'Start voice server' is off — Piper only")
            if not self.restart_voice_touched:
                self.restart_voice.set(False)
            return
        self.health = h
        vv = h.get("vox_voices") or []
        loaded = bool(h.get("vox_loaded"))
        parts = [f"up, default voice {h.get('default_voice') or '?'}",
                 f"VoxCPM2: {', '.join(vv) if vv else 'NONE'} ({'loaded' if loaded else 'NOT loaded'})"]
        if h.get("warm") is False:
            parts.append(f"warming {h.get('warm_progress', '')}")
        if not h.get("speaker"):
            parts.append("no voice-id")
        self.set_pre("voice", True if (vv and loaded) else None, "; ".join(parts))
        if not self.restart_voice_touched:
            self.restart_voice.set(not (vv and loaded))

    def _check_cameras(self):
        if self.running():
            return
        t, p = self.table_var.get(), self.people_var.get()
        for key, idx in (("table", t), ("people", p)):
            if key == "table" and self.sim_camera.get():
                self.set_pre("table", "skip", "simulated")
                continue
            r = self.cams.get(idx)
            if r is None:
                self.set_pre(key, None if self.cams.opening else False,
                             f"#{idx}: opening…" if self.cams.opening else f"#{idx}: no camera at this index")
                continue
            st = r.state()
            msg = {"ok": f"#{idx} delivers frames ({r.size[0]}x{r.size[1]})",
                   "black": f"#{idx} is BLACK — idle iPhone or covered lens?",
                   "waiting": f"#{idx}: waiting for picture…", "noframes": f"#{idx}: NO frames"}[st]
            if key == "people" and idx == t and not self.sim_camera.get():
                self.set_pre(key, False, f"#{idx} is also the TABLE camera — pick another")
                continue
            if key == "table" and getattr(self, "_yolo_msg", None) and self._yolo_msg[0] == idx:
                msg += "; " + self._yolo_msg[1]
            self.set_pre(key, {"ok": True, "black": False, "waiting": None, "noframes": False}[st], msg)

    # ------------------------------------------------------------ prompts
    def open_prompts(self):
        """See and edit the system prompt. Saved texts apply from the robot's
        NEXT turn: the client re-reads them before every turn."""
        if getattr(self, "_pw", None) is not None and self._pw.winfo_exists():
            self._pw.lift()
            return
        w = self._pw = tk.Toplevel(self.root)
        w.title("CandyTron — system prompt")
        w.geometry("980x720")
        ttk.Label(w, wraplength=940, text=(
            "Edit a part and press Save: it applies from the robot's NEXT turn, no restart. "
            "Reset = back to the built-in text. 'Last prompt sent' shows exactly what the model got "
            "last turn (rules + persona + scene)."), foreground="#555").pack(anchor="w", padx=10, pady=(8, 4))
        nb = ttk.Notebook(w)
        nb.pack(fill="both", expand=True, padx=10, pady=4)
        self._ptexts = {}
        for key in ("rules", "persona", "expressive", "greeting"):
            f = ttk.Frame(nb)
            nb.add(f, text=prompts.LABELS[key])
            top = ttk.Frame(f)
            top.pack(fill="x", pady=(6, 2))
            state = ttk.Label(top, foreground="#1a7f37")
            state.pack(side="left")
            ttk.Label(f, text=prompts.HELP[key], foreground="#555", wraplength=920).pack(anchor="w")
            t = ScrolledText(f, wrap="word", font=("Menlo", 12), undo=True)
            t.pack(fill="both", expand=True, pady=4)
            b = ttk.Frame(f)
            b.pack(fill="x", pady=(0, 6))
            ttk.Button(b, text="Save (live from next turn)",
                       command=lambda k=key: self._save_prompt(k)).pack(side="left")
            ttk.Button(b, text="Reset to default",
                       command=lambda k=key: self._reset_prompt(k)).pack(side="left", padx=6)
            self._ptexts[key] = (t, state)
            self._load_prompt(key)
        f = ttk.Frame(nb)
        nb.add(f, text="Last prompt sent")
        lt = ScrolledText(f, wrap="word", font=("Menlo", 11))
        lt.pack(fill="both", expand=True, pady=4)
        self._plast = lt

        def refresh_last():
            if not w.winfo_exists():
                return
            try:
                txt = prompts.LAST_PROMPT.read_text(encoding="utf-8")
            except FileNotFoundError:
                txt = "(nothing sent yet: start the robot and talk to it)"
            if txt != lt.get("1.0", "end-1c"):
                lt.configure(state="normal")
                lt.delete("1.0", "end")
                lt.insert("1.0", txt)
                lt.configure(state="disabled")
            w.after(2000, refresh_last)
        refresh_last()

    def _load_prompt(self, key):
        t, state = self._ptexts[key]
        t.delete("1.0", "end")
        text = prompts.get(key)
        if key == "persona" and not prompts.is_overridden("persona"):
            text = ""
            state.configure(text="Using the persona from candytron_mcp (see 'Last prompt sent'). "
                                 "Type here to replace it.", foreground="#555")
        else:
            state.configure(text="EDITED (override active)" if prompts.is_overridden(key)
                            else "Built-in default", foreground="#9a6700" if prompts.is_overridden(key) else "#1a7f37")
        t.insert("1.0", text)
        t.edit_reset()

    def _save_prompt(self, key):
        t, state = self._ptexts[key]
        data = prompts.overrides()
        text = t.get("1.0", "end-1c")
        data[key] = "" if text.strip() == prompts.DEFAULTS[key].strip() else text
        prompts.save(data)
        self._load_prompt(key)
        self.say(f"prompt '{prompts.LABELS[key]}' saved — applies from the next turn")

    def _reset_prompt(self, key):
        data = prompts.overrides()
        data.pop(key, None)
        prompts.save(data)
        self._load_prompt(key)
        self.say(f"prompt '{prompts.LABELS[key]}' reset to the built-in text")

    # ------------------------------------------------------------ auto-detect
    def auto_detect(self, then_start: bool = False, quiet_if_unsure: bool = False):
        """Pick the TABLE camera by candy (the board's own YOLO model) and the
        PEOPLE camera by faces (OpenCV's face detector) in the live pictures.
        Camera numbers shift whenever an iPhone comes or goes; the pictures
        don't lie."""
        if self.auto_running or self.running():
            return
        frames = {}
        for i in range(N_CAMS):
            r = self.cams.get(i)
            if r is None or r.state() != "ok":
                continue
            with r.lock:
                if r.full is not None:
                    frames[i] = r.full.copy()
        if not frames:
            if not quiet_if_unsure:
                self.say("auto-detect: no camera with a picture yet", "warn")
            if then_start:
                self.start(skip_auto=True)
            return
        self.auto_running = True
        self.auto_btn.configure(state="disabled", text="Detecting…")
        threading.Thread(target=self._auto_worker, args=(frames, then_start, quiet_if_unsure),
                         daemon=True).start()

    def _auto_worker(self, frames, then_start, quiet_if_unsure):
        # Faces: OpenCV's Haar cascade -- fast, in-process, no extra model.
        faces = {}
        try:
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            for i, f in frames.items():
                g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                found = cascade.detectMultiScale(g, scaleFactor=1.1, minNeighbors=6,
                                                 minSize=(max(40, g.shape[1] // 25),) * 2)
                faces[i] = max((w * h for (_x, _y, w, h) in found), default=0) / float(g.size)
        except Exception as e:  # noqa: BLE001
            self.say(f"auto-detect: face detector failed: {e}", "warn")
        # Candy: the table camera's YOLO, one subprocess for all pictures.
        candies = {}
        paths = {}
        for i, f in frames.items():
            paths[i] = os.path.join(tempfile.gettempdir(), f"candytron-launcher-cam{i}.jpg")
            cv2.imwrite(paths[i], f)
        code = ("import sys\n"
                "from ultralytics import YOLO\n"
                "m = YOLO('models/best-m.pt')\n"
                "for a in sys.argv[1:]:\n"
                "    i, p = a.split('=', 1)\n"
                "    print(f'CANDY {i} {len(m(p, verbose=False)[0].boxes)}', flush=True)\n")
        try:
            out = subprocess.run(["uv", "run", "--no-sync", "python", "-c", code,
                                  *[f"{i}={p}" for i, p in paths.items()]],
                                 cwd=CANDYTRON_DIR, capture_output=True, text=True, timeout=120)
            for m in re.finditer(r"^CANDY (\d+) (\d+)$", out.stdout, re.M):
                candies[int(m.group(1))] = int(m.group(2))
        except Exception as e:  # noqa: BLE001
            self.say(f"auto-detect: YOLO failed: {e}", "warn")
        self.q.put(("auto", candies, faces, then_start, quiet_if_unsure))

    def _apply_auto(self, candies, faces, then_start, quiet_if_unsure):
        self.auto_running = False
        self.auto_btn.configure(state="normal", text="Auto-detect")
        table, people = self.table_var.get(), self.people_var.get()
        notes = []
        if not self.sim_camera.get():
            best_t = max(candies, key=candies.get, default=None)
            if best_t is not None and candies[best_t] >= 2:
                table = best_t
                notes.append(f"TABLE #{table} ({candies[table]} candies)")
            else:
                notes.append(f"TABLE unsure (no camera sees 2+ candies) — kept #{table}")
        cands = {i: a for i, a in faces.items() if i != table or self.sim_camera.get()}
        best_p = max(cands, key=cands.get, default=None)
        if best_p is not None and cands[best_p] > 0:
            people = best_p
            notes.append(f"PEOPLE #{people} (face seen)")
        else:
            ok = [i for i in sorted(faces) if i != table]
            if people == table or people not in faces:
                people = ok[0] if ok else people
            notes.append(f"PEOPLE: no face in any picture — using #{people}; stand in front and press Auto-detect")
        changed = (table, people) != (self.table_var.get(), self.people_var.get())
        self.table_var.set(table)
        self.people_var.set(people)
        msg = "auto-detect: " + "; ".join(notes) + ("" if changed else " (unchanged)")
        if changed or not quiet_if_unsure:
            self.say(msg, "me")
        self._cams_changed()
        if then_start:
            self.start(skip_auto=True)

    def yolo_check(self):
        if self.yolo_running or self.running():
            return
        idx = self.table_var.get()
        r = self.cams.get(idx)
        frame = None
        if r is not None:
            with r.lock:
                frame = None if r.full is None else r.full.copy()
        if frame is None:
            self.set_pre("table", False, f"#{idx}: no frame to analyse")
            return
        path = os.path.join(tempfile.gettempdir(), "candytron-launcher-table.jpg")
        cv2.imwrite(path, frame)
        self.yolo_running = True
        self.yolo_btn.configure(state="disabled", text="Counting… (loading YOLO)")
        threading.Thread(target=self._yolo_worker, args=(idx, path), daemon=True).start()

    def _yolo_worker(self, idx, path):
        code = ("import sys\n"
                "from ultralytics import YOLO\n"
                "r = YOLO('models/best-m.pt')(sys.argv[1], verbose=False)[0]\n"
                "print('YOLO_NAMES=' + ','.join(r.names[int(c)] for c in r.boxes.cls))\n")
        try:
            out = subprocess.run(["uv", "run", "--no-sync", "python", "-c", code, path],
                                 cwd=CANDYTRON_DIR, capture_output=True, text=True, timeout=120)
            m = re.search(r"^YOLO_NAMES=(.*)$", out.stdout, re.M)
            if m is None:
                msg = "YOLO failed: " + (out.stderr.strip().splitlines() or ["?"])[-1][:120]
            else:
                names = [n for n in m.group(1).split(",") if n]
                msg = f"{len(names)} candies seen" + (
                    " — exactly 4: ready to calibrate" if len(names) == 4 else
                    " (calibration needs exactly 4, one per corner)")
        except Exception as e:  # noqa: BLE001
            msg = f"YOLO failed: {e}"
        self.q.put(("yolo", idx, msg))

    # ------------------------------------------------------------ robot
    def find_robot(self):
        self.find_btn.configure(state="disabled", text="Searching…")
        self.find_msg.configure(text="searching…")
        cands = [self.ip_var.get().strip(), self.cfg.get("robot_ip", ""),
                 os.environ.get("NIRYO_IP", ""), DEFAULT_ROBOT_IP, ETHERNET_ROBOT_IP]

        def work():
            msgs = []

            def log(m):
                msgs.append(m)
                self.say(f"find robot: {m}")
            try:
                ip = find_robot(cands, log)
            except Exception as e:  # noqa: BLE001
                ip = None
                log(f"error: {e}")
            self.q.put(("found", ip, msgs[-1] if msgs else ""))
        threading.Thread(target=work, daemon=True).start()

    # ------------------------------------------------------------ start / stop
    def _collect_cfg(self) -> dict:
        return {"table_cam": self.table_var.get(), "people_cam": self.people_var.get(),
                "mic_name": self._mic_name(), "robot_ip": self.ip_var.get().strip(),
                "sim_robot": self.sim_robot.get(), "sim_camera": self.sim_camera.get(),
                "start_voice": self.start_voice.get(), "expressive": self.expressive.get(),
                "extra_args": self.extra_var.get().strip()}

    def build_args(self) -> list[str]:
        c = self._collect_cfg()
        a = [START_SCRIPT]
        if c["sim_robot"] and c["sim_camera"]:
            a.append("--sim")
        elif c["sim_robot"]:
            a.append("--simulate-robot")
        elif c["sim_camera"]:
            a.append("--simulate-camera")
        if not c["sim_robot"] and c["robot_ip"]:
            a += ["--robot-ip", c["robot_ip"]]
        if not c["sim_camera"]:
            a += ["--robot-camera", str(c["table_cam"])]
        if not c["start_voice"]:
            a.append("--no-voice-server")
        a += ["--camera", str(c["people_cam"])]
        mic_idx, _ = self._mic_choice()
        if mic_idx is not None:
            a += ["--mic", str(mic_idx)]
        flag = "--expressive" if c["expressive"] else "--no-expressive"
        if client_supports(flag):
            a.append(flag)
        else:
            self.say(f"client has no {flag} flag (yet) — expressive comes from config.toml", "warn")
        if c["extra_args"]:
            a += shlex.split(c["extra_args"])
        return a

    def start(self, skip_auto: bool = False):
        if self.running():
            return
        if not skip_auto and not self.sim_camera.get():
            # Numbers shift when an iPhone comes or goes: if a chosen camera
            # is now black or gone, re-detect from the pictures first.
            bad = [i for i in (self.table_var.get(), self.people_var.get())
                   if self.cams.get(i) is None or self.cams.get(i).state() != "ok"]
            if bad:
                self.say(f"camera(s) {bad} no longer give a picture — re-detecting before start", "warn")
                self.auto_detect(then_start=True)
                return
        c = self._collect_cfg()
        if not c["sim_camera"] and c["table_cam"] == c["people_cam"]:
            if not self.args.autostart and not messagebox.askyesno(
                    "Same camera twice",
                    f"Camera #{c['table_cam']} is both TABLE and PEOPLE camera. Start anyway?"):
                return
        self.cfg.update(c)
        save_config(self.cfg)
        args = self.build_args()
        for key, _n in STEPS:
            self.set_step(key, "pending", "")
        self.set_banner("Starting…", "busy")
        self.ready_at = None
        self.stopping = False
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.recal_btn.configure(state="normal")
        self.log.delete("1.0", "end")
        # Release cameras and the mic: the stack opens them itself.
        self.mic.close()
        stuck = self.cams.stop()
        if stuck:
            self.say(f"cameras {stuck} did not release in time", "err")
        else:
            self.say("released all cameras and the microphone")
        threading.Thread(target=self._start_worker, args=(args,), daemon=True).start()

    def _start_worker(self, args):
        # Ollama: start it ourselves, detached, so stopping the stack (process
        # group signal) doesn't take it down with it.
        try:
            http_json(f"{OLLAMA_URL}/api/tags")
        except Exception:  # noqa: BLE001
            self.say("Ollama not running — starting it (detached)")
            self.q.put(("step", "ollama", "progress", "starting ollama serve"))
            try:
                with open(os.path.join(LOG_DIR, "candytron-ollama.log"), "ab") as lf:
                    subprocess.Popen(["ollama", "serve"], stdout=lf, stderr=lf,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
                for _ in range(30):
                    time.sleep(1)
                    try:
                        http_json(f"{OLLAMA_URL}/api/tags")
                        break
                    except Exception:  # noqa: BLE001
                        pass
            except OSError as e:
                self.say(f"could not start ollama: {e}", "err")
        if self.restart_voice.get() and self.start_voice.get():
            pids = listening_pids(VOICE_PORT)
            if pids:
                self.say(f"restarting voice server: stopping pid(s) {pids} on :{VOICE_PORT}")
                self.q.put(("step", "voice", "progress", "stopping the old voice server"))
                for pid in pids:
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except OSError:
                        pass
                for _ in range(40):
                    if not listening_pids(VOICE_PORT):
                        break
                    time.sleep(0.25)
                for pid in listening_pids(VOICE_PORT):
                    self.say(f"voice server pid {pid} ignored SIGTERM — SIGKILL", "warn")
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except OSError:
                        pass
                time.sleep(0.5)
        try:
            os.remove(RECAL_FLAG)
        except OSError:
            pass
        env = dict(os.environ, PYTHONUNBUFFERED="1", CANDYTRON_KEEP_VOICE_SERVER="1",
                   CANDYTRON_RECALIBRATE_FLAG=RECAL_FLAG)
        self.say("$ " + " ".join(shlex.quote(x) for x in args))
        try:
            self.logfile = open(os.path.join(LOG_DIR, f"candytron-launcher-{time.strftime('%Y%m%d-%H%M%S')}.log"),
                                "w", buffering=1)
            self.say(f"full log: {self.logfile.name}")
        except OSError:
            self.logfile = None
        try:
            proc = subprocess.Popen(args, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    start_new_session=True)
        except OSError as e:
            self.say(f"could not start: {e}", "err")
            self.q.put(("exited", -1))
            return
        self.proc = proc
        threading.Thread(target=self._reader, args=(proc,), daemon=True).start()
        threading.Thread(target=self._health_poller, args=(proc,), daemon=True).start()

    def _reader(self, proc):
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            # strip ANSI colour codes from uv/loggers
            line = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line)
            self.q.put(("line", line))
        proc.wait()
        self.q.put(("exited", proc.returncode))

    def _health_poller(self, proc):
        """While the voice server step is in progress, show its warm-up state."""
        while proc.poll() is None:
            time.sleep(2)
            if self.steps["voice"][2] not in ("progress",):
                continue
            try:
                h = http_json(f"{VOICE_URL}/health", timeout=1.5)
                self.q.put(("step", "voice", None,
                            f"up; VoxCPM2 {'loaded' if h.get('vox_loaded') else 'loading…'}; "
                            f"warm {h.get('warm_progress', '?')}"))
            except Exception:  # noqa: BLE001
                pass

    def stop(self):
        if not self.running() or self.stopping:
            return
        self.stopping = True
        self.set_banner("Stopping…", "busy")
        self.stop_btn.configure(state="disabled")
        pgid = self.proc.pid
        self.say("stopping: SIGINT to the process group")

        def escalate():
            for sig, wait_s in ((signal.SIGINT, 6), (signal.SIGTERM, 5), (signal.SIGKILL, 0)):
                if self.proc is None or self.proc.poll() is not None:
                    break
                if sig != signal.SIGINT:
                    self.say(f"still running — {sig.name} to the process group", "warn")
                try:
                    os.killpg(pgid, sig)
                except ProcessLookupError:
                    break
                t0 = time.time()
                while time.time() - t0 < wait_s and self.proc.poll() is None:
                    time.sleep(0.1)
            # children may outlive the script itself; sweep the group once more
            time.sleep(1.0)
            try:
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        threading.Thread(target=escalate, daemon=True).start()

    def recalibrate(self):
        if not self.running():
            return
        try:
            with open(RECAL_FLAG, "w") as f:
                f.write(str(time.time()))
            self.say("recalibration requested — clear the board, ONE candy in each corner")
            self.set_banner("Recalibration requested: clear the board and put ONE candy in each "
                            "corner (A1, A3, D1, D3)", "calib")
        except OSError as e:
            self.say(f"cannot write {RECAL_FLAG}: {e}", "err")

    def on_close(self):
        if self.running():
            self.closing = True
            self.stop()
            # hard deadline in case the stack refuses to die
            self.root.after(15000, self._really_close)
        else:
            self._really_close()

    def _atexit_kill(self):
        """Last resort if the launcher dies with the stack still running."""
        p = self.proc
        if p is not None and p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except OSError:
                pass

    def _really_close(self):
        self.mic.close()
        self.cams.stop(timeout=2)
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    # ------------------------------------------------------------ log parsing
    def parse(self, line: str):
        s = line.strip()
        st = self.set_step

        def has(*subs):
            return any(x in s for x in subs)

        # Ollama
        if has("ollama already running", "] ollama up"):
            st("ollama", "done", "running")
        elif has("Ollama not running"):
            st("ollama", "progress", "starting")
        elif has("ollama did not come up"):
            st("ollama", "failed", "did not come up")
        # Voice server
        if has("voice server already running"):
            st("voice", "done", "already running (kept)")
        elif has("starting kokoro-voice-server"):
            st("voice", "progress", "starting — cold start takes a few minutes")
        elif has("] voice server up"):
            st("voice", "done", "up")
        elif has("continuing with Piper voices only", "kokoro-voice-server not set up"):
            st("voice", "warn", "not available — Piper voices only")
        elif has("started without --speaker"):
            self.set_step("voice", None, "running, but without voice id (--speaker)")
        # Table camera
        m = re.search(r"\[candytron\] table camera (.*)$", s)
        if m:
            what = m.group(1)
            if "FAILED" in what:
                st("table", "failed", what)
                self.set_banner(f"Table camera failed: {what}", "bad")
            elif "ready" in what:
                st("table", "done", what.replace("ready", "").strip(" ()") or "ready")
            else:
                st("table", "progress", what)
        if has("Camera initialization failed"):
            st("table", "failed", s.split("failed:", 1)[-1].strip())
        if has("starting candytron_mcp"):
            st("mcp", "progress", "starting (waits for calibration)")
            if "--simulate-camera" in s:
                st("table", "progress", "simulated")
        # Calibration
        m = re.search(r"calibrating: (\d+)/4 candies seen", s)
        if m:
            n = int(m.group(1))
            st("calib", "progress", f"{n}/4 candies seen — need exactly 4, one in each corner")
            extra = ("Remove the extra candies!" if n > 4 else
                     "Clear the board completely, or put ONE candy in each corner (A1, A3, D1, D3).")
            mb = re.search(r"(\d+)/12 printed boxes seen", s)
            if n == 0 and mb:
                extra = f"Empty board: {mb.group(1)}/12 printed boxes seen (is one in shadow or out of view?)"
            self.set_banner(f"CALIBRATING: {n}/4 candies seen.  {extra}", "calib")
        elif has("calibrating: place"):
            st("calib", "progress", "waiting for 4 corner candies")
            self.set_banner("CALIBRATING: clear the board completely (the printed boxes are used), "
                            "or place ONE candy in each corner (A1, A3, D1, D3)", "calib")
            self._calib_warn = []
        elif has("calibration WARNING:"):
            w = s.split("calibration WARNING:", 1)[1].strip()
            self._calib_warn = getattr(self, "_calib_warn", []) + [w]
        elif re.search(r"\[candytron\] calibrated", s):
            warns = getattr(self, "_calib_warn", [])
            if warns:
                st("calib", "warn", "calibrated, but: " + " | ".join(warns))
                self.set_banner("Calibrated WITH WARNINGS: " + " | ".join(warns), "warn")
            else:
                st("calib", "done", s.split("calibrated", 1)[1].strip(" ()") or "calibrated")
                if self.ready_at is None:
                    self.set_banner("Board calibrated — starting the rest…", "busy")
                else:
                    self.set_banner("Board recalibrated. READY", "ok")
        elif has("recalibration requested"):
            st("calib", "progress", "recalibrating…")
        elif has("recalibration gave up"):
            st("calib", "warn", "recalibration gave up — previous calibration kept")
            self.set_banner("Recalibration gave up — the previous calibration is still in use", "warn")
        elif has("calibration cancelled"):
            st("calib", "failed", "cancelled ('q' in the camera window)")
        # MCP server
        if has("MCP server starting on"):
            st("mcp", "progress", "starting")
        elif has("candytron_mcp up"):
            st("mcp", "done", "listening on :7999")
        elif has("candytron_mcp exited before", "candytron_mcp never opened"):
            st("mcp", "failed", "exited — see log (robot/camera problem?)")
            self.set_banner("candytron_mcp exited before it was ready — see the log", "bad")
        elif has("something already listens on :7999"):
            st("mcp", "failed", ":7999 busy — run ./stop-demo.sh")
            self.set_banner("Port 7999 is already in use — an old stack is still running "
                            "(./stop-demo.sh)", "bad")
        # Client
        client_steps = [("starting mcpclient_speech_face", "starting"),
                        ("starting face agent", "starting (face agent)"),
                        ("LLM Chatbot using model", "LLM ok"),
                        ("Created the interaction window", "eye window open"),
                        ("Loading whisper model", "loading speech models…"),
                        ("Continuous listener started", "microphone listening"),
                        ("Direct-audio mode:", "direct-audio ready"),
                        ("Loading piper model", "loading Piper…"),
                        ("Face tracker started", "face tracker running")]
        for key, txt in client_steps:
            if key in s:
                # a warning (e.g. wrong people camera) stays visible
                if self.steps["client"][2] != "warn":
                    st("client", "progress", txt)
                if self.steps["mcp"][2] != "done":
                    st("mcp", "done", "listening on :7999")
        m = re.search(r"People camera: (\d+)", s)
        if m:
            got = int(m.group(1))
            want = self.cfg.get("people_cam")
            if want is not None and got != int(want):
                st("client", "warn", f"PEOPLE camera is #{got}, not #{want} — #{want} gave no picture?")
            else:
                st("client", "progress", f"people camera #{got}")
        if has("ERROR: no working camera"):
            st("client", "failed", "no working people camera")
        if has("ERROR: Cannot connect", "ERROR: Model", "Failed to load whisper",
               "Failed to load piper"):
            st("client", "failed", s[:120])
        if has("Face tracker started"):
            st("robot", "progress", "connecting (client initialises the service)…")
        # Robot
        m = re.search(r"\[candytron\] connecting to robot at (\S+)", s)
        if m:
            st("robot", "progress", f"connecting to {m.group(1)}…")
            self.set_banner(f"Connecting to the robot at {m.group(1)} (it moves to home)…", "busy")
        elif has("[candytron] robot: simulated"):
            st("robot", "progress", "simulated")
        elif re.search(r"\[candytron\] robot ready", s):
            st("robot", "done", "simulated" if "simulated" in s else "connected, at home")
        elif has("[candytron] robot FAILED"):
            st("robot", "failed", s.split("FAILED:", 1)[-1].strip()[:120])
            self.set_banner("Robot connection FAILED — check IP / Find robot", "bad")
        elif has("Ned2: offline"):
            st("robot", "failed", "offline")
        # Ready
        if has("Initialized service"):
            st("client", "done", "running")
            if self.steps["robot"][2] in ("pending", "progress"):
                st("robot", "done", "service initialised")
            st("ready", "done", "say hello!")
            self.ready_at = time.time()
            warn = self.steps["calib"][2] == "warn" or self.steps["client"][2] == "warn"
            self.set_banner("READY — CandyTron is listening" + (" (see warnings)" if warn else ""),
                            "warn" if warn else "ok")
        elif has("Failed to initialize service"):
            st("client", "failed", "service init failed")
            if self.steps["robot"][2] != "failed":
                st("robot", "failed", "service init failed (robot connection?)")
            self.set_banner("Service init FAILED — usually the robot connection", "bad")

    def _on_exit(self, rc):
        self.proc = None
        if self.logfile:
            self.logfile.close()
            self.logfile = None
        for key, _n in STEPS:
            if self.steps[key][2] == "progress":
                self.set_step(key, "failed" if not self.stopping else "skipped")
        if self.stopping:
            self.set_banner("Stopped.", "info")
        else:
            self.set_banner(f"The stack exited (code {rc}) — see the log", "bad")
            self.set_step("ready", "failed", f"exited with code {rc}")
        self.say(f"stack exited (code {rc})")
        self.stopping = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.recal_btn.configure(state="disabled")
        if self.closing or self.args.exit_when_ready is not None:
            self._really_close()
            return
        # Previews back on so the operator can fix things and restart.
        self.cams.start()
        self._refresh_mics()
        self.check()

    # ------------------------------------------------------------ main tick
    def _append_log(self, line, tag=None):
        if tag is None:
            if re.search(r"Traceback|ERROR|Error:|FAILED|failed|\[E\]", line):
                tag = "err"
            elif re.search(r"WARNING|WARN|warning", line):
                tag = "warn"
            elif line.startswith("[candytron]"):
                tag = "ct"
        at_end = self.log.yview()[1] > 0.98
        self.log.insert("end", line + "\n", tag or ())
        if int(self.log.index("end-1c").split(".")[0]) > 6000:
            self.log.delete("1.0", "1000.0")
        if at_end:
            self.log.see("end")
        if self.logfile:
            try:
                self.logfile.write(line + "\n")
            except OSError:
                pass

    def _tick(self):
        try:
            for _ in range(500):
                item = self.q.get_nowait()
                kind = item[0]
                if kind == "line":
                    self._append_log(item[1])
                    self.parse(item[1])
                elif kind == "log":
                    self._append_log(item[1], item[2])
                elif kind == "pre":
                    self.set_pre(item[1], item[2], item[3])
                elif kind == "health":
                    self._apply_health(item[1])
                elif kind == "step":
                    self.set_step(item[1], item[2], item[3])
                elif kind == "auto":
                    self._apply_auto(*item[1:])
                elif kind == "yolo":
                    self._yolo_msg = (item[1], item[2])
                    self.yolo_running = False
                    self.yolo_btn.configure(state="normal", text="Count candies (YOLO)")
                    self._check_cameras()
                elif kind == "found":
                    ip, msg = item[1], item[2]
                    self.find_btn.configure(state="normal", text="Find robot")
                    if ip:
                        self.ip_var.set(ip)
                        self.find_msg.configure(text=f"Found: {ip}.  {msg}", foreground="#1a7f37")
                        self.check()
                    else:
                        self.find_msg.configure(text=msg, foreground="#cf222e")
                elif kind == "exited":
                    self._on_exit(item[1])
        except queue.Empty:
            pass
        try:
            self._update_tiles()
            self._draw_meter()
            now = time.time()
            if not self.running() and now - getattr(self, "_last_camcheck", 0) > 1.0:
                self._last_camcheck = now
                self._check_cameras()
            if (self.args.exit_when_ready is not None and self.ready_at is not None
                    and self.running() and not self.stopping
                    and now - self.ready_at > self.args.exit_when_ready):
                self.say("test hook: READY reached — stopping")
                self.stop()
        except tk.TclError:
            return
        self.root.after(100, self._tick)


def main():
    ap = argparse.ArgumentParser(description="CandyTron 4000 launcher")
    ap.add_argument("--sim", action="store_true", help="preselect simulated robot + table camera")
    ap.add_argument("--autostart", action="store_true", help="start immediately with saved settings")
    ap.add_argument("--exit-when-ready", type=float, default=None, metavar="S",
                    help="test hook: stop the stack S seconds after READY, then quit")
    args = ap.parse_args()
    root = tk.Tk()
    Launcher(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
