#!/usr/bin/env python3
"""CandyTron 4000 control panel as a web page (replaces the Tk launcher window).

    ./launch-candytron.sh            -> http://127.0.0.1:8090 (opens in the browser)
    ./launch-candytron.sh --tk       -> the old Tk launcher

Same job as launcher.py -- pick cameras/mic/robot, start/stop the stack, the
startup checklist, prompt editing -- plus:
  * SUPERVISION: if the stack exits on its own it is restarted (at most
    MAX_RESTARTS within RESTART_WINDOW_S, so a hard failure does not loop).
  * MONITORING: Ollama, voice server, MCP server, robot, client process and
    the session log (camera errors, last turn) are checked every few seconds
    while running, not only at startup.
  * The live transcript of the current session.

Reuses launcher.py's helpers (config, robot discovery, camera previews,
auto-detect, the output parser) so both panels behave the same.
"""
from __future__ import annotations

import argparse
import atexit
import glob
import json
import os
import queue
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque

import cv2

import launcher as L
from launcher import prompts  # face/prompts.py

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Route

SESSION_LOG_DIR = os.path.join(L.HERE, "logs")
MAX_RESTARTS = 3
RESTART_WINDOW_S = 600
MCP_PORT = 7999


class Panel:
    """Headless controller. Launcher.parse() is reused as-is: it only needs
    set_step / set_banner / steps / cfg / ready_at."""

    parse = L.Launcher.parse

    def __init__(self, sim: bool = False):
        self.cfg = L.load_config()
        if sim:
            self.cfg.update(sim_robot=True, sim_camera=True)
        self.q: queue.Queue = queue.Queue()
        self.lock = threading.RLock()
        self.proc: subprocess.Popen | None = None
        self.stopping = False
        self.ready_at: float | None = None
        self.started_at: float | None = None
        self.logfile = None
        self.log: deque = deque(maxlen=1500)
        self.log_seq = 0
        self.steps = {k: [None, None, "pending", ""] for k, _n in L.STEPS}
        self.banner = ("Not running. Check the settings and press Start.", "info")
        self.pre: dict = {}
        self.monitor: dict = {}
        self.auto_restart = True
        self.restarts: list[float] = []
        self.auto_running = False
        self.find_msg = ""
        self.cams = L.CameraPreviews()
        self.cams.start()
        threading.Thread(target=self._pump, daemon=True).start()
        threading.Thread(target=self._monitor_loop, daemon=True).start()
        threading.Timer(5.0, lambda: self.auto_detect(quiet=True)).start()
        atexit.register(self._atexit_kill)

    # ---------------------------------------------------------- ui shims
    def say(self, text: str, tag: str = "me"):
        self._append(f"[panel] {text}", tag)

    def _append(self, line: str, tag: str | None = None):
        with self.lock:
            self.log_seq += 1
            self.log.append((self.log_seq, line, tag or ""))
        if self.logfile:
            try:
                self.logfile.write(line + "\n")
            except (OSError, ValueError):
                pass

    def set_step(self, key, state=None, detail=None):
        s = self.steps[key]
        if state:
            s[2] = state
        if detail is not None:
            s[3] = detail

    def set_banner(self, text, kind="info"):
        self.banner = (text, kind)

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    # ---------------------------------------------------------- queue pump
    def _pump(self):
        while True:
            item = self.q.get()
            kind = item[0]
            try:
                if kind == "line":
                    self._append(item[1])
                    self.parse(item[1])
                elif kind == "log":
                    self._append(item[1], item[2])
                elif kind == "step":
                    self.set_step(item[1], item[2], item[3])
                elif kind == "auto":
                    self._apply_auto(*item[1:])
                elif kind == "exited":
                    self._on_exit(item[1])
            except Exception as e:  # noqa: BLE001
                self._append(f"[panel] internal error: {e}", "err")

    # ---------------------------------------------------------- devices
    def mics(self):
        try:
            return L.input_devices()
        except Exception:  # noqa: BLE001
            return []

    def mic_index(self):
        name = self.cfg.get("mic_name") or ""
        for i, n in self.mics():
            if n == name:
                return i
        return None

    def cam_states(self):
        out = []
        for i in range(L.N_CAMS):
            r = self.cams.get(i)
            if r is None:
                st = "opening" if i in self.cams.pending else "none"
                size = None
            else:
                st, size = r.state(), r.size
            out.append({"idx": i, "state": st, "size": size})
        return out

    def cam_jpeg(self, idx: int):
        r = self.cams.get(idx)
        if r is None:
            return None
        with r.lock:
            f = None if r.thumb is None else r.thumb.copy()
        if f is None:
            return None
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        return buf.tobytes() if ok else None

    # ---------------------------------------------------------- auto-detect
    def auto_detect(self, quiet: bool = False, then_start: bool = False):
        if self.auto_running or self.running():
            return
        frames = {}
        for i in range(L.N_CAMS):
            r = self.cams.get(i)
            if r is None or r.state() != "ok":
                continue
            with r.lock:
                if r.full is not None:
                    frames[i] = r.full.copy()
        if not frames:
            if not quiet:
                self.say("auto-detect: no camera with a picture yet", "warn")
            if then_start:
                self.start(skip_auto=True)
            return
        self.auto_running = True
        threading.Thread(target=L.Launcher._auto_worker, args=(self, frames, then_start, quiet),
                         daemon=True).start()

    def _apply_auto(self, candies, faces, then_start, quiet):
        self.auto_running = False
        sim_cam = self.cfg.get("sim_camera")
        table, people = int(self.cfg["table_cam"]), int(self.cfg["people_cam"])
        notes = []
        if not sim_cam:
            best_t = max(candies, key=candies.get, default=None)
            if best_t is not None and candies[best_t] >= 2:
                table = best_t
                notes.append(f"TABLE #{table} ({candies[table]} candies)")
            else:
                notes.append(f"TABLE unsure (no camera sees 2+ candies) — kept #{table}")
        cands = {i: a for i, a in faces.items() if i != table or sim_cam}
        best_p = max(cands, key=cands.get, default=None)
        if best_p is not None and cands[best_p] > 0:
            people = best_p
            notes.append(f"PEOPLE #{people} (face seen)")
        else:
            ok = [i for i in sorted(faces) if i != table]
            if people == table or people not in faces:
                people = ok[0] if ok else people
            notes.append(f"PEOPLE: no face seen — using #{people}")
        changed = (table, people) != (self.cfg["table_cam"], self.cfg["people_cam"])
        self.cfg.update(table_cam=table, people_cam=people)
        L.save_config(self.cfg)
        if changed or not quiet:
            self.say("auto-detect: " + "; ".join(notes) + ("" if changed else " (unchanged)"))
        if then_start:
            self.start(skip_auto=True)

    def find_robot(self):
        cands = [self.cfg.get("robot_ip", ""), os.environ.get("NIRYO_IP", ""),
                 L.DEFAULT_ROBOT_IP, L.ETHERNET_ROBOT_IP]
        self.find_msg = "searching…"

        def work():
            msgs = []

            def log(m):
                msgs.append(m)
                self.say(f"find robot: {m}")
            try:
                ip = L.find_robot(cands, log)
            except Exception as e:  # noqa: BLE001
                ip = None
                log(f"error: {e}")
            if ip:
                self.cfg["robot_ip"] = ip
                L.save_config(self.cfg)
                self.find_msg = f"Found {ip}"
            else:
                self.find_msg = msgs[-1] if msgs else "not found"
        threading.Thread(target=work, daemon=True).start()

    # ---------------------------------------------------------- start / stop
    def build_args(self) -> list[str]:
        c = self.cfg
        a = [L.START_SCRIPT]
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
        if c.get("loose_pick"):
            a.append("--loose-pick")
        a += ["--camera", str(c["people_cam"])]
        mic = self.mic_index()
        if mic is not None:
            a += ["--mic", str(mic)]
        if c.get("ear") in ("whisper", "gemma4") and L.client_supports("--ear"):
            a += ["--ear", c["ear"]]
        flag = "--expressive" if c["expressive"] else "--no-expressive"
        if L.client_supports(flag):
            a.append(flag)
        if c["extra_args"]:
            a += shlex.split(c["extra_args"])
        return a

    def start(self, skip_auto: bool = False, reason: str = ""):
        with self.lock:
            if self.running():
                return "already running"
            if not skip_auto and not self.cfg.get("sim_camera"):
                bad = [i for i in (int(self.cfg["table_cam"]), int(self.cfg["people_cam"]))
                       if self.cams.get(i) is None or self.cams.get(i).state() != "ok"]
                if bad and not self.cams.opening:
                    self.say(f"camera(s) {bad} give no picture — re-detecting before start", "warn")
                    self.auto_detect(then_start=True)
                    return "re-detecting cameras first"
            args = self.build_args()
            for key, _n in L.STEPS:
                self.set_step(key, "pending", "")
            self.set_banner("Starting…" + (f" ({reason})" if reason else ""), "busy")
            self.ready_at = None
            self.stopping = False
            self.started_at = time.time()
            stuck = self.cams.stop()
            self.say(f"cameras {stuck} did not release in time" if stuck
                     else "released the camera previews")
            self.proc = None
        threading.Thread(target=self._start_worker, args=(args,), daemon=True).start()
        return "starting"

    def _start_worker(self, args):
        try:
            L.http_json(f"{L.OLLAMA_URL}/api/tags")
        except Exception:  # noqa: BLE001
            self.say("Ollama not running — starting it (detached)")
            try:
                with open(os.path.join(L.LOG_DIR, "candytron-ollama.log"), "ab") as lf:
                    subprocess.Popen(["ollama", "serve"], stdout=lf, stderr=lf,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
                for _ in range(30):
                    time.sleep(1)
                    try:
                        L.http_json(f"{L.OLLAMA_URL}/api/tags")
                        break
                    except Exception:  # noqa: BLE001
                        pass
            except OSError as e:
                self.say(f"could not start ollama: {e}", "err")
        try:
            os.remove(L.RECAL_FLAG)
        except OSError:
            pass
        env = dict(os.environ, PYTHONUNBUFFERED="1", CANDYTRON_KEEP_VOICE_SERVER="1",
                   CANDYTRON_RECALIBRATE_FLAG=L.RECAL_FLAG)
        self.say("$ " + " ".join(shlex.quote(x) for x in args))
        try:
            self.logfile = open(os.path.join(
                L.LOG_DIR, f"candytron-launcher-{time.strftime('%Y%m%d-%H%M%S')}.log"), "w", buffering=1)
            self.say(f"full log: {self.logfile.name}")
        except OSError:
            self.logfile = None
        try:
            proc = subprocess.Popen(args, cwd=L.ROOT, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    start_new_session=True)
        except OSError as e:
            self.say(f"could not start: {e}", "err")
            self.q.put(("exited", -1))
            return
        self.proc = proc
        threading.Thread(target=self._reader, args=(proc,), daemon=True).start()

    def _reader(self, proc):
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            line = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line)
            self.q.put(("line", line))
        proc.wait()
        self.q.put(("exited", proc.returncode))

    def stop(self, user: bool = True):
        with self.lock:
            if not self.running() or self.stopping:
                return "not running"
            self.stopping = True
            self.set_banner("Stopping…", "busy")
            proc = self.proc
        pgid = proc.pid
        self.say("stopping: SIGINT to the process group")

        def escalate():
            for sig, wait_s in ((signal.SIGINT, 6), (signal.SIGTERM, 5), (signal.SIGKILL, 0)):
                if proc.poll() is not None:
                    break
                if sig != signal.SIGINT:
                    self.say(f"still running — {sig.name} to the process group", "warn")
                try:
                    os.killpg(pgid, sig)
                except ProcessLookupError:
                    break
                t0 = time.time()
                while time.time() - t0 < wait_s and proc.poll() is None:
                    time.sleep(0.1)
            time.sleep(1.0)
            try:
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        threading.Thread(target=escalate, daemon=True).start()
        return "stopping"

    def restart(self):
        if self.running():
            self._restart_after_stop = True
            return self.stop()
        return self.start(skip_auto=True)

    def _on_exit(self, rc):
        was_stopping = self.stopping
        self.proc = None
        if self.logfile:
            self.logfile.close()
            self.logfile = None
        for key, _n in L.STEPS:
            if self.steps[key][2] == "progress":
                self.set_step(key, "failed" if not was_stopping else "skipped")
        self.say(f"stack exited (code {rc})")
        self.stopping = False
        if getattr(self, "_restart_after_stop", False):
            self._restart_after_stop = False
            self.start(skip_auto=True, reason="restart")
            return
        if was_stopping:
            self.set_banner("Stopped.", "info")
            self.cams.start()
            return
        # Exited on its own: supervise.
        self.set_step("ready", "failed", f"exited with code {rc}")
        now = time.time()
        self.restarts = [t for t in self.restarts if now - t < RESTART_WINDOW_S]
        ran_for = now - (self.started_at or now)
        if self.auto_restart and len(self.restarts) < MAX_RESTARTS and ran_for > 20:
            self.restarts.append(now)
            self.set_banner(f"The stack exited (code {rc}) — auto-restart "
                            f"{len(self.restarts)}/{MAX_RESTARTS}", "warn")
            self.say(f"auto-restart {len(self.restarts)}/{MAX_RESTARTS} in 3 s", "warn")
            threading.Timer(3.0, lambda: self.start(skip_auto=True, reason="auto-restart")).start()
        else:
            why = ("it died within 20 s of starting" if ran_for <= 20 else
                   f"{MAX_RESTARTS} restarts in {RESTART_WINDOW_S // 60} min" if self.auto_restart
                   else "auto-restart is off")
            self.set_banner(f"The stack exited (code {rc}) — NOT restarting: {why}. See the log.", "bad")
            self.cams.start()

    def recalibrate(self):
        if not self.running():
            return "not running"
        with open(L.RECAL_FLAG, "w") as f:
            f.write(str(time.time()))
        self.say("recalibration requested — clear the board, ONE candy in each corner")
        self.set_banner("Recalibration requested: ONE candy in each corner (A1, A3, D1, D3)", "calib")
        return "ok"

    def _atexit_kill(self):
        p = self.proc
        if p is not None and p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except OSError:
                pass

    # ---------------------------------------------------------- monitoring
    def _latest_session(self):
        files = sorted(glob.glob(os.path.join(SESSION_LOG_DIR, "*.jsonl")), key=os.path.getmtime)
        return files[-1] if files else None

    def _session_status(self):
        path = self._latest_session()
        if not path or (self.started_at and os.path.getmtime(path) < self.started_at - 5):
            return None, {}
        info = {"file": os.path.basename(path)}
        try:
            with open(path, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 200_000))
                lines = f.read().decode("utf-8", "replace").splitlines()[1:]
        except OSError:
            return path, info
        now = time.time()
        cam_err = cam_reopen = 0
        for ln in lines:
            try:
                e = json.loads(ln)
            except ValueError:
                continue
            t = e.get("type")
            if t == "turn":
                info["last_turn"] = e.get("ts")
            elif t in ("camera_error", "camera_reopen"):
                ts = e.get("ts", "")
                try:
                    from datetime import datetime
                    age = now - datetime.fromisoformat(ts).timestamp()
                except ValueError:
                    age = 1e9
                if age < 300:
                    cam_err += t == "camera_error"
                    cam_reopen += t == "camera_reopen"
            elif t == "session_end":
                info["ended"] = e.get("reason")
            elif t == "llm_error":
                info["last_llm_error"] = e.get("error_message", "")[:120]
        info["camera_errors_5min"] = cam_err
        info["camera_reopens_5min"] = cam_reopen
        return path, info

    def _monitor_loop(self):
        while True:
            m = {}
            try:
                tags = L.http_json(f"{L.OLLAMA_URL}/api/tags", timeout=1.5)
                names = [x.get("name") for x in tags.get("models", [])]
                m["ollama"] = ("ok", f"up, {L.LLM_MODEL} " + ("present" if L.LLM_MODEL in names else "MISSING"))
                if L.LLM_MODEL not in names:
                    m["ollama"] = ("bad", m["ollama"][1])
            except Exception:  # noqa: BLE001
                m["ollama"] = ("bad", "not reachable")
            try:
                h = L.http_json(f"{L.VOICE_URL}/health", timeout=1.5)
                vv = h.get("vox_voices") or []
                txt = f"up, voice {h.get('default_voice') or '?'}; VoxCPM2 " + (
                    "loaded" if h.get("vox_loaded") else "NOT loaded") + (f" ({', '.join(vv)})" if vv else "")
                if h.get("warm") is False:
                    txt += f"; warming {h.get('warm_progress', '')}"
                m["voice"] = ("ok" if h.get("vox_loaded") or not vv else "warn", txt)
            except Exception:  # noqa: BLE001
                m["voice"] = ("bad" if self.cfg.get("start_voice") else "warn", "not reachable")
            run = self.running()
            m["mcp"] = (("ok", "listening on :7999") if L.tcp_open("127.0.0.1", MCP_PORT, 0.5)
                        else (("bad" if run else "idle"), "not listening"))
            if self.cfg.get("sim_robot"):
                m["robot"] = ("ok", "simulated")
            else:
                ip = self.cfg.get("robot_ip", "")
                up = bool(ip) and L.tcp_open(ip, L.ROBOT_PORT, 1.0)
                m["robot"] = ("ok" if up else "bad", f"{ip}:{L.ROBOT_PORT} " + ("reachable" if up else "NOT reachable"))
            client = L.run(["pgrep", "-f", "mcpclient_speech_face.py"]).split()
            m["client"] = (("ok", f"running (pid {', '.join(client)})") if client
                           else (("bad" if run else "idle"), "not running"))
            _path, info = self._session_status()
            if info:
                ce, cr = info.get("camera_errors_5min", 0), info.get("camera_reopens_5min", 0)
                m["people_cam"] = (("warn" if ce or cr else "ok"),
                                   f"{ce} errors, {cr} reopens in 5 min" if ce or cr else "no errors")
                if info.get("ended"):
                    m["client"] = ("bad", f"session ended: {info['ended']}")
            self.monitor = {"checks": m, "session": info, "at": time.time()}
            time.sleep(3)

    # ---------------------------------------------------------- transcript
    def transcript(self, n: int = 80) -> str:
        path = self._latest_session()
        if not path:
            return ""
        txt = path[:-len(".jsonl")] + ".txt"
        try:
            with open(txt, encoding="utf-8") as f:
                return "".join(deque(f, maxlen=n))
        except OSError:
            return ""

    def state(self, since: int = 0) -> dict:
        with self.lock:
            log = [x for x in self.log if x[0] > since]
        return {
            "running": self.running(), "stopping": self.stopping, "cfg": self.cfg,
            "steps": [{"key": k, "name": n, "state": self.steps[k][2], "detail": self.steps[k][3]}
                      for k, n in L.STEPS],
            "banner": self.banner, "log": log, "log_seq": self.log_seq,
            "monitor": self.monitor, "auto_restart": self.auto_restart,
            "restarts": len([t for t in self.restarts if time.time() - t < RESTART_WINDOW_S]),
            "cams": self.cam_states(), "mics": [n for _i, n in self.mics()],
            "find_msg": self.find_msg, "auto_running": self.auto_running,
            "uptime": (time.time() - self.started_at) if self.running() and self.started_at else 0,
        }


# ---------------------------------------------------------------- web

PANEL: Panel | None = None


async def index(_r: Request):
    return HTMLResponse(PAGE)


async def api_state(r: Request):
    return JSONResponse(PANEL.state(int(r.query_params.get("since", 0))))


async def api_transcript(_r: Request):
    return JSONResponse({"text": PANEL.transcript()})


async def log_txt(_r: Request):
    with PANEL.lock:
        lines = [x[1] for x in PANEL.log]
    return Response("\n".join(lines[-2000:]), media_type="text/plain; charset=utf-8")


async def transcript_txt(_r: Request):
    return Response(PANEL.transcript(400) or "(no session yet)", media_type="text/plain; charset=utf-8")


async def api_cfg(r: Request):
    data = await r.json()
    if PANEL.running():
        return JSONResponse({"msg": "stop the stack before changing settings"}, status_code=409)
    for k, v in data.items():
        if k in L.DEFAULTS:
            PANEL.cfg[k] = type(L.DEFAULTS[k])(v) if not isinstance(L.DEFAULTS[k], bool) else bool(v)
    L.save_config(PANEL.cfg)
    return JSONResponse({"msg": "saved"})


async def api_action(r: Request):
    a = r.path_params["name"]
    p = PANEL
    acts = {"start": lambda: p.start(), "stop": lambda: p.stop(), "restart": p.restart,
            "recalibrate": p.recalibrate, "detect": lambda: p.auto_detect() or "detecting",
            "find_robot": lambda: p.find_robot() or "searching",
            "auto_restart_on": lambda: setattr(p, "auto_restart", True) or "on",
            "auto_restart_off": lambda: setattr(p, "auto_restart", False) or "off"}
    if a not in acts:
        return JSONResponse({"msg": "unknown"}, status_code=404)
    return JSONResponse({"msg": acts[a]()})


async def cam(r: Request):
    jpg = PANEL.cam_jpeg(int(r.path_params["idx"]))
    if jpg is None:
        return Response(status_code=204)
    return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


async def api_prompts(r: Request):
    if r.method == "POST":
        d = await r.json()
        key, text = d.get("key"), d.get("text", "")
        if key not in prompts.DEFAULTS:
            return JSONResponse({"msg": "unknown prompt"}, status_code=400)
        data = prompts.overrides()
        if d.get("reset") or text.strip() == prompts.DEFAULTS[key].strip():
            data.pop(key, None)
        else:
            data[key] = text
        prompts.save(data)
        PANEL.say(f"prompt '{prompts.LABELS[key]}' {'reset' if d.get('reset') else 'saved'} — applies from the next turn")
    out = {}
    for key in prompts.DEFAULTS:
        out[key] = {"label": prompts.LABELS.get(key, key), "help": prompts.HELP.get(key, ""),
                    "text": prompts.get(key) if (key != "persona" or prompts.is_overridden(key)) else "",
                    "overridden": prompts.is_overridden(key)}
    try:
        last = prompts.LAST_PROMPT.read_text(encoding="utf-8")
    except FileNotFoundError:
        last = "(nothing sent yet)"
    return JSONResponse({"prompts": out, "last": last})


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CandyTron Control</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--fg:#1d2329;--mut:#66707a;--line:#dde1e6;--ok:#1a7f37;--warn:#b58900;--bad:#cf222e;--busy:#c26a00;--acc:#1f6feb}
@media (prefers-color-scheme:dark){:root{--bg:#111418;--card:#1a1f25;--fg:#e6e8ea;--mut:#98a2ad;--line:#2c333b;--ok:#3fb950;--warn:#d29922;--bad:#f85149;--busy:#e3a03a;--acc:#58a6ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 -apple-system,system-ui,sans-serif}
header{display:flex;gap:12px;align-items:center;padding:12px 16px;border-bottom:1px solid var(--line);background:var(--card);position:sticky;top:0;z-index:2;flex-wrap:wrap}
h1{font-size:18px;margin:0 8px 0 0}button{font:inherit;padding:7px 14px;border-radius:7px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button.primary{background:var(--acc);border-color:var(--acc);color:#fff}button.danger{color:var(--bad)}button:disabled{opacity:.45;cursor:default}
#banner{margin:12px 16px;padding:10px 14px;border-radius:8px;font-weight:600}
.k-info{background:#8881;}.k-busy{background:#ffb54733}.k-calib{background:#ff8c2a44}.k-ok{background:#2ea04333}.k-warn{background:#d2992233}.k-bad{background:#f8514933}
main{display:grid;grid-template-columns:minmax(300px,420px) 1fr;gap:14px;padding:0 16px 16px}
@media (max-width:900px){main{grid-template-columns:1fr}main>div:last-child{order:-1}}
#tabbody pre{min-height:300px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:14px}
.card h2{font-size:13px;text-transform:uppercase;letter-spacing:.05em;color:var(--mut);margin:0 0 8px}
.row{display:flex;gap:8px;align-items:center;margin:6px 0;flex-wrap:wrap}label{color:var(--mut);min-width:96px}
input[type=text],select{font:inherit;padding:5px 8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg);flex:1;min-width:0}
.st{display:grid;grid-template-columns:22px 150px 1fr;gap:4px 6px;align-items:baseline}.st .d{color:var(--mut);overflow-wrap:anywhere}
.g-done,.g-ok{color:var(--ok)}.g-warn{color:var(--warn)}.g-failed,.g-bad{color:var(--bad)}.g-progress{color:var(--busy)}.g-pending,.g-skipped,.g-idle{color:var(--mut)}
.cams{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.cam{border:2px solid var(--line);border-radius:8px;padding:4px;font-size:12px;text-align:center}
.cam img{width:100%;aspect-ratio:4/3;object-fit:contain;background:#000;border-radius:5px;display:block}.cam.t{border-color:#e07b00}.cam.p{border-color:#1f6feb}
pre{margin:0;white-space:pre-wrap;font:12px/1.45 ui-monospace,Menlo,monospace;max-height:420px;overflow:auto}
#log .err{color:var(--bad)}#log .warn{color:var(--warn)}
.tabs button{border-radius:6px 6px 0 0;margin-right:2px}.tabs button.on{background:var(--acc);color:#fff;border-color:var(--acc)}
textarea{width:100%;min-height:340px;font:12px/1.45 ui-monospace,Menlo,monospace;padding:8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg)}
.pill{font-size:12px;padding:2px 8px;border-radius:99px;background:#8882}
</style></head><body>
<header><h1>🍬 CandyTron 4000</h1>
<button id="bStart" class="primary" onclick="act('start')">Start</button>
<button id="bStop" class="danger" onclick="act('stop')">Stop</button>
<button id="bRestart" onclick="act('restart')">Restart</button>
<button id="bRecal" onclick="act('recalibrate')">Recalibrate board</button>
<label style="min-width:0"><input type="checkbox" id="autoR" onchange="act(this.checked?'auto_restart_on':'auto_restart_off')"> auto-restart</label>
<span id="up" class="pill"></span></header>
<div id="banner" class="k-info"></div>
<main><div>
<div class="card"><h2>Startup</h2><div id="steps" class="st"></div></div>
<div class="card"><h2>Live monitoring</h2><div id="mon" class="st"></div></div>
<div class="card"><h2>Settings</h2>
<div class="row"><label>Robot IP</label><input type="text" id="robot_ip"><button onclick="act('find_robot')">Find</button></div>
<div class="row"><span id="findmsg" class="d" style="color:var(--mut)"></span></div>
<div class="row"><label>Table cam</label><select id="table_cam"></select><label style="min-width:0">People cam</label><select id="people_cam"></select></div>
<div class="row"><label>Microphone</label><select id="mic_name"></select></div>
<div class="row"><label><input type="checkbox" id="sim_robot"> sim robot</label><label><input type="checkbox" id="sim_camera"> sim camera</label></div>
<div class="row"><label><input type="checkbox" id="start_voice"> voice server</label><label><input type="checkbox" id="expressive"> expressive</label></div>
<div class="row"><label><input type="checkbox" id="loose_pick"> pick candy outside the boxes (experimental)</label></div>
<div class="row"><label>Ear</label><select id="ear"><option value="whisper">Whisper transcribes → gemma4 reads text</option><option value="gemma4">gemma4 hears the audio</option></select></div>
<div class="row"><label>Extra args</label><input type="text" id="extra_args"></div>
<div class="row"><button onclick="saveCfg()">Save settings</button><button onclick="act('detect')" id="bDetect">Auto-detect cameras</button><span id="cfgmsg" style="color:var(--mut)"></span></div>
</div>
<div class="card"><h2>Cameras <span style="text-transform:none;font-weight:400">(previews only while stopped; orange = table, blue = people)</span></h2><div id="cams" class="cams"></div></div>
</div><div>
<div class="card"><div class="tabs" id="tabs"></div><div id="tabbody"></div>
<div class="row" style="color:var(--mut);font-size:12px">Plain text: <a href="/log.txt" target="_blank">log.txt</a> · <a href="/transcript.txt" target="_blank">transcript.txt</a></div></div>
</div></main>
<script>
const $=id=>document.getElementById(id);let S=null,since=0,tab='transcript',P=null,editing=false;
const G={pending:'○',progress:'◐',done:'✓',warn:'!',failed:'✗',skipped:'–',ok:'●',bad:'●',idle:'○'};
async function act(a){const r=await fetch('/api/'+a,{method:'POST'});const j=await r.json();toast(j.msg)}
function toast(m){$('cfgmsg').textContent=m||'';setTimeout(()=>$('cfgmsg').textContent='',4000)}
async function saveCfg(){const c={};for(const k of ['robot_ip','extra_args','mic_name','ear'])c[k]=$(k).value;
 for(const k of ['table_cam','people_cam'])c[k]=+$(k).value;for(const k of ['sim_robot','sim_camera','start_voice','expressive','loose_pick'])c[k]=$(k).checked;
 const r=await fetch('/api/cfg',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(c)});toast((await r.json()).msg);editing=false}
document.addEventListener('input',e=>{if(e.target.closest('.card')&&!e.target.closest('#tabbody'))editing=true});
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function render(){const s=S;$('banner').textContent=s.banner[0];$('banner').className='k-'+s.banner[1];
 $('bStart').disabled=s.running;$('bStop').disabled=!s.running||s.stopping;$('bRecal').disabled=!s.running;$('autoR').checked=s.auto_restart;
 $('up').textContent=s.running?('running '+Math.round(s.uptime/60)+' min'+(s.restarts?' · '+s.restarts+' auto-restarts':'')):'stopped';
 $('steps').innerHTML=s.steps.map(x=>`<span class="g-${x.state}">${G[x.state]||''}</span><span>${x.name}</span><span class="d">${esc(x.detail)}</span>`).join('');
 const names={ollama:'Ollama (gemma4)',voice:'Voice server',mcp:'MCP server',robot:'Robot',client:'Client',people_cam:'People camera'};
 const c=(s.monitor.checks)||{};$('mon').innerHTML=Object.keys(names).filter(k=>c[k]).map(k=>`<span class="g-${c[k][0]}">${G[c[k][0]]}</span><span>${names[k]}</span><span class="d">${esc(c[k][1])}</span>`).join('')+
  (s.monitor.session&&s.monitor.session.last_turn?`<span></span><span>Last turn</span><span class="d">${new Date(s.monitor.session.last_turn).toLocaleTimeString()}</span>`:'');
 $('findmsg').textContent=s.find_msg;$('bDetect').disabled=s.running||s.auto_running;
 if(!editing){const cfg=s.cfg;$('robot_ip').value=cfg.robot_ip;$('extra_args').value=cfg.extra_args;$('ear').value=cfg.ear||'whisper';
  for(const k of ['sim_robot','sim_camera','start_voice','expressive','loose_pick'])$(k).checked=!!cfg[k];
  for(const k of ['table_cam','people_cam']){$(k).innerHTML=s.cams.map(x=>`<option value="${x.idx}">#${x.idx} ${x.state}</option>`).join('');$(k).value=cfg[k]}
  $('mic_name').innerHTML=[...new Set([cfg.mic_name,...s.mics])].map(n=>`<option>${esc(n)}</option>`).join('');$('mic_name').value=cfg.mic_name}
 const t=+s.cfg.table_cam,p=+s.cfg.people_cam,stamp=Date.now();
 $('cams').innerHTML=s.cams.map(x=>`<div class="cam ${x.idx==t?'t':''} ${x.idx==p?'p':''}">${(!s.running&&x.state=='ok'||x.state=='black')?`<img src="/cam/${x.idx}.jpg?${stamp}">`:'<div style="aspect-ratio:4/3;display:grid;place-items:center;color:var(--mut)">'+(s.running?'in use':x.state)+'</div>'}#${x.idx} ${x.idx==t?'TABLE':''}${x.idx==p?'PEOPLE':''}</div>`).join('')}
let logLines=[];
async function poll(){try{const r=await fetch('/api/state?since='+since);S=await r.json();for(const l of S.log)logLines.push(l);if(logLines.length>1500)logLines=logLines.slice(-1500);
 since=S.log_seq;render();if(tab=='log')drawLog();if(tab=='transcript')drawTranscript()}catch(e){$('banner').textContent='Panel server not reachable';$('banner').className='k-bad'}
 setTimeout(poll,1000)}
function drawLog(){const el=$('logpre');if(!el)return;const atBottom=el.scrollTop+el.clientHeight>=el.scrollHeight-20;
 el.innerHTML=logLines.slice(-600).map(l=>`<span class="${/error|traceback|failed/i.test(l[1])?'err':l[2]}">${esc(l[1])}</span>`).join('\n');if(atBottom)el.scrollTop=el.scrollHeight}
async function drawTranscript(){const el=$('trpre');if(!el)return;const j=await (await fetch('/api/transcript')).json();if(el.textContent!=j.text){el.textContent=j.text||'(no session yet)';el.scrollTop=el.scrollHeight}}
async function loadPrompts(){P=await (await fetch('/api/prompts')).json()}
function tabs(){const T=[['transcript','Transcript'],['log','Log'],['prompts','Prompts']];
 $('tabs').innerHTML=T.map(([k,n])=>`<button class="${tab==k?'on':''}" onclick="setTab('${k}')">${n}</button>`).join('');
 if(tab=='log'){$('tabbody').innerHTML='<pre id="log"><code id="logpre" style="display:block;max-height:640px;overflow:auto"></code></pre>';drawLog()}
 if(tab=='transcript'){$('tabbody').innerHTML='<pre id="trpre" style="max-height:640px"></pre>';drawTranscript()}
 if(tab=='prompts')drawPrompts()}
let pkey='rules';
async function drawPrompts(){await loadPrompts();const ks=Object.keys(P.prompts);
 $('tabbody').innerHTML='<div class="row">'+ks.map(k=>`<button class="${pkey==k?'primary':''}" onclick="pkey='${k}';drawPrompts()">${esc(P.prompts[k].label)}${P.prompts[k].overridden?' ✎':''}</button>`).join('')+
 `<button class="${pkey=='_last'?'primary':''}" onclick="pkey='_last';drawPrompts()">Last prompt sent</button></div>`+
 (pkey=='_last'?`<pre style="max-height:600px">${esc(P.last)}</pre>`:
 `<p style="color:var(--mut)">${esc(P.prompts[pkey].help)} — ${P.prompts[pkey].overridden?'<b>EDITED</b> (override active)':'built-in default'}. Saving applies from the robot's next turn.</p>
 <textarea id="ptext">${esc(P.prompts[pkey].text)}</textarea><div class="row"><button class="primary" onclick="savePrompt(false)">Save</button><button onclick="savePrompt(true)">Reset to default</button></div>`)}
async function savePrompt(reset){await fetch('/api/prompts',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({key:pkey,text:reset?'':$('ptext').value,reset})});toast(reset?'reset':'saved');drawPrompts()}
function setTab(k){tab=k;tabs()}
tabs();poll();
</script></body></html>"""


def main():
    global PANEL
    ap = argparse.ArgumentParser(description="CandyTron 4000 web control panel")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to reach it from a phone on the LAN")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--autostart", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    PANEL = Panel(sim=args.sim)
    if args.autostart:
        threading.Timer(3.0, PANEL.start).start()
    app = Starlette(routes=[
        Route("/", index),
        Route("/api/state", api_state),
        Route("/log.txt", log_txt),
        Route("/transcript.txt", transcript_txt),
        Route("/api/transcript", api_transcript),
        Route("/api/cfg", api_cfg, methods=["POST"]),
        Route("/api/prompts", api_prompts, methods=["GET", "POST"]),
        Route("/api/{name}", api_action, methods=["POST"]),
        Route("/cam/{idx:int}.jpg", cam),
    ])
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '') else args.host}:{args.port}"
    print(f"CandyTron control panel: {url}", flush=True)
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    def _bye(*_):
        if PANEL.running():
            PANEL.stop()
            for _ in range(150):
                if not PANEL.running():
                    break
                time.sleep(0.1)
        os._exit(0)
    import uvicorn
    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="warning")
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _bye)
    server.run()


if __name__ == "__main__":
    main()
