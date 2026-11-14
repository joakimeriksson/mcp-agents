"""A simulated candy table and arm, for running CandyTron without the robot.

The real table: white paper with 12 printed boxes (A1..D3) on a blue table,
seen from above by the table camera; a Niryo Ned 2 picks candy from a box and
puts it in another box or on the hand-out spot O0 in front of the visitor.

This module draws that scene -- the paper and boxes exactly where the arm's
taught poses (base-saved-poses.yaml) put them, real candy cut out of the
YOLO training photos, and a simple two-joint arm seen from above -- and acts
as both the camera and the arm:

- ``read()`` returns a camera frame, so the REAL vision pipeline runs on it:
  YOLO detection, calibration from the printed boxes, the consensus voting.
  ("direct" vision skips YOLO and reports what is really on the table.)
- ``pick_and_place(src_pose, dst_pose)`` is called by robotarm.py with the
  same poses the real arm would get. The arm moves there, grips whatever
  lies under the gripper (or nothing), carries it and lets go. The table
  really changes, so the camera sees the result. A candy left on O0 is
  taken by the "visitor" after a few seconds.

Nothing here is used with the real robot or the real camera.

Assets: the candy cut-outs live in sim_assets/ (built once from the YOLO
dataset with ``uv run sim_table.py build-assets``).
"""

from __future__ import annotations

import glob
import itertools
import logging
import math
import os
import random
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger("candytron.sim")

HERE = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(HERE, "sim_assets")
# Frames the YOLO model got wrong, with the TRUE labels (YOLO format, class
# ids of the candy dataset) -- material for the next training round.
MISSES_DIR = os.path.join(HERE, "sim_misses")
# Class ids of ~/work/candy-yolo11-dataset (data.yaml), used for the labels.
DATASET_NAMES = ["Dumle", "FazerMint", "Geisha", "Lakridskola", "Marianne", "Mars",
                 "Nutcreme", "NutcremePear", "Plopp", "Refreshers", "Riesen", "Snickers",
                 "VanillaToffee"]
DATASET_DIR = os.environ.get("CANDYTRON_SIM_DATASET",
                             os.path.expanduser("~/work/candy-yolo11-dataset"))

# --------------------------------------------------------------- geometry
# Table coordinates are the arm's (metres): x away from the robot base, y to
# the robot's left. The camera looks straight down; x grows DOWN the image and
# y to the RIGHT, so row A (furthest from the base) is at the bottom and
# column 1 on the left -- the orientation the calibration expects.
SCALE = 1600.0                 # pixels per metre
FRAME_W, FRAME_H = 1024, 960
X_TOP, Y_LEFT = -0.075, -0.22  # table point at pixel (0, 0)

BOX_M = 0.060                  # printed box, outer size
PAPER_MARGIN_M = 0.045
TRAY_R_M = 0.050
TABLE_Z = 0.095                # z of the taught poses = gripping height
APPROACH_DZ = 0.07             # the real pick_and_place goes 7 cm above
L1, L2 = 0.26, 0.24            # simulated arm links (reach 0.50 m)

# Long side of each candy (wrapper included), metres.
# (About 20 % above life size: the real camera sits closer than this one, and
# the YOLO model recognises them best at that size.)
CANDY_LEN = {"Lakridskola": 0.041, "Geisha": 0.050, "Dumle": 0.048,
             "FazerMint": 0.058, "Marianne": 0.058, "Plopp": 0.055,
             "Refreshers": 0.055, "Mars": 0.055, "Snickers": 0.055}
DEFAULT_LEN = 0.053

TAKE_AFTER_S = 4.0             # a candy on O0 is taken by the visitor
TRAVEL_M_S = 0.40              # arm speed over the table
VERTICAL_S = 0.35              # one lower or raise
GRIP_S = 0.30
GRIP_RADIUS_M = 0.022          # a candy this close to the fingers is gripped
FPS = 30


MODEL_PATH = os.path.join(HERE, "models", "best-m.pt")
# The model knows these candies under an older name too (and so does the
# server's translation table).
SAME = {"NutcremePear": "Pearnut", "Nutcreme": "Whitenut", "VanillaToffee": "VanillaFudge"}


def to_px(x: float, y: float) -> tuple[float, float]:
    return (y - Y_LEFT) * SCALE, (x - X_TOP) * SCALE


def _ipt(x: float, y: float) -> tuple[int, int]:
    px, py = to_px(x, y)
    return int(round(px)), int(round(py))


def load_poses() -> dict:
    """{name: (x, y, z)} of the taught poses (lower-case "home" included)."""
    import yaml
    with open(os.path.join(HERE, "base-saved-poses.yaml")) as f:
        raw = yaml.safe_load(f)
    return {k: tuple(float(v) for v in vals[:3]) for k, vals in raw.items()}


# ----------------------------------------------------------------- assets
def build_assets(dataset_dir: str = DATASET_DIR, out_dir: str = ASSET_DIR,
                 per_class: int = 3, max_side: int = 180) -> dict:
    """Cut candy out of the YOLO training photos: each labelled box that
    overlaps no other box, segmented from the paper with GrabCut. Writes
    <Class>_<n>.png (BGRA) and returns {class: count}."""
    import yaml
    with open(os.path.join(dataset_dir, "data.yaml")) as f:
        names = yaml.safe_load(f)["names"]
    os.makedirs(out_dir, exist_ok=True)
    model = None
    if os.path.isfile(MODEL_PATH):
        from ultralytics import YOLO
        model = YOLO(MODEL_PATH)
    candidates: dict = {n: [] for n in names}
    for lab in sorted(glob.glob(os.path.join(dataset_dir, "data", "*", "labels", "*.txt"))):
        rows = [r.split() for r in open(lab) if r.strip()]
        boxes = [(int(r[0]), *map(float, r[1:5])) for r in rows]
        img_glob = lab.replace(os.sep + "labels" + os.sep, os.sep + "images" + os.sep)[:-4] + ".*"
        imgs = glob.glob(img_glob)
        if not imgs:
            continue
        for i, (c, cx, cy, w, h) in enumerate(boxes):
            clear = all(abs(cx - b[1]) > (w + b[3]) / 2 + 0.01 or abs(cy - b[2]) > (h + b[4]) / 2 + 0.01
                        for j, b in enumerate(boxes) if j != i)
            if clear:
                # single-candy photos first: plain paper around them
                others = [b[1:] for j, b in enumerate(boxes) if j != i]
                candidates[names[c]].append((len(boxes), imgs[0], cx, cy, w, h, others))
    for old in glob.glob(os.path.join(out_dir, "*.png")):
        os.remove(old)
    counts = {}
    for name, cands in candidates.items():
        cands.sort(key=lambda t: t[0])
        cuts = []
        for k, path, cx, cy, w, h, others in cands[:40]:
            img = cv2.imread(path)
            r = _cut_out(img, cx, cy, w, h, others) if img is not None else None
            if r is not None:
                cut, paper_left, sharp = r
                cuts.append([k > 1, paper_left > 0.2, sharp, len(cuts), cut])
        if cuts:
            # Plain single-candy photos, little paper left along the edge,
            # and sharp (some crops are motion-blurred).
            top = max(c[2] for c in cuts)
            for c in cuts:
                c[2] = c[2] < 0.35 * top
            cuts.sort(key=lambda t: (t[2], t[0], t[1], t[3]))
        n = 0
        for *_q, cut in cuts:
            if n >= per_class:
                break
            f = max_side / max(cut.shape[:2])
            if f < 1:
                cut = cv2.resize(cut, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            if model is not None and not _recognised(model, cut, name):
                continue
            cv2.imwrite(os.path.join(out_dir, f"{name}_{n}.png"), cut)
            n += 1
        counts[name] = n
    return counts


def same_candy(a: str, b: str) -> bool:
    """The model knows some candies under two names."""
    return SAME.get(a, a) == SAME.get(b, b)


def _recognised(model, cut, name: str) -> bool:
    """Does the real YOLO model see *name* in this cut-out, alone on the
    paper at the size the simulated camera shows it? (Drops wrong labels in
    the dataset and cut-outs that lost too much of the wrapper.)"""
    length = CANDY_LEN.get(name, DEFAULT_LEN) * SCALE
    f = length / max(cut.shape[:2])
    sp = cv2.resize(cut, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    ok = 0
    for angle in (0, 45, 90, 135):       # it lies at any angle on the table
        h, w = sp.shape[:2]
        d = int(math.hypot(w, h)) + 2
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        M[0, 2] += (d - w) / 2
        M[1, 2] += (d - h) / 2
        rot = cv2.warpAffine(sp, M, (d, d), borderValue=(0, 0, 0, 0))
        canvas = np.full((320, 320, 3), (236, 238, 240), np.uint8)
        _paste(canvas, rot, 160, 160)
        res = model(canvas, verbose=False)[0]
        if len(res.boxes):
            best = max(res.boxes, key=lambda b: float(b.conf))
            ok += same_candy(res.names[int(best.cls)], name) and float(best.conf) > 0.4
    return ok >= 3


def _cut_out(img, cx, cy, w, h, others=()):
    H, W = img.shape[:2]
    x1, y1 = int((cx - w / 2) * W), int((cy - h / 2) * H)
    x2, y2 = int((cx + w / 2) * W), int((cy + h / 2) * H)
    pad = int(0.25 * max(x2 - x1, y2 - y1))
    X1, Y1, X2, Y2 = max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad)
    crop = img[Y1:Y2, X1:X2].copy()
    if crop.size == 0 or min(crop.shape[:2]) < 30:
        return None
    rect = (x1 - X1, y1 - Y1, max(1, x2 - x1), max(1, y2 - y1))
    # The paper around the candy: its colour, from the crop's border.
    lab_img = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate([lab_img[:4].reshape(-1, 3), lab_img[-4:].reshape(-1, 3),
                             lab_img[:, :4].reshape(-1, 3), lab_img[:, -4:].reshape(-1, 3)])
    paper = np.median(border, axis=0)
    paperlike = np.linalg.norm(lab_img - paper, axis=2) < 14
    mask = np.full(crop.shape[:2], cv2.GC_BGD, np.uint8)
    rx, ry, rw, rh = rect
    mask[ry:ry + rh, rx:rx + rw] = cv2.GC_PR_FGD
    inner = mask == cv2.GC_PR_FGD
    mask[inner & paperlike] = cv2.GC_PR_BGD
    for ox, oy, ow, oh in others:       # neighbouring candies are background
        a1, b1 = int((ox - ow / 2) * W) - X1, int((oy - oh / 2) * H) - Y1
        a2, b2 = int((ox + ow / 2) * W) - X1, int((oy + oh / 2) * H) - Y1
        mask[max(0, b1):max(0, b2), max(0, a1):max(0, a2)] = cv2.GC_BGD
    try:
        cv2.grabCut(crop, mask, None, np.zeros((1, 65)), np.zeros((1, 65)), 8,
                    cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n < 2:
        return None
    # the part under the label's centre, else the biggest
    centre = lab[ry + rh // 2, rx + rw // 2]
    big = centre if centre > 0 else 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    fg = np.where(lab == big, 255, 0).astype(np.uint8)
    frac = fg.sum() / 255 / (rect[2] * rect[3])
    if not 0.30 < frac < 1.05:
        return None
    # Quality: paper left along the cut-out's edge (lower is cleaner).
    ring = cv2.subtract(fg, cv2.erode(fg, np.ones((9, 9), np.uint8))) > 0
    paper_left = float(paperlike[ring].mean()) if ring.any() else 1.0
    x, y, bw, bh = cv2.boundingRect(fg)
    # sharpness at a common size (a blurred crop has weak edges)
    g = cv2.cvtColor(crop[y:y + bh, x:x + bw], cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (160, int(160 * bh / max(bw, 1))) if bw >= bh else (int(160 * bw / max(bh, 1)), 160))
    m = cv2.resize(fg[y:y + bh, x:x + bw], g.shape[::-1]) > 0
    sharp = float(cv2.Laplacian(g, cv2.CV_32F)[m].var()) if m.any() else 0.0
    alpha = cv2.GaussianBlur(fg, (5, 5), 0)
    out = np.dstack([crop, alpha])[y:y + bh, x:x + bw]
    return out, paper_left, sharp


def load_assets(asset_dir: str = ASSET_DIR) -> dict:
    """{class: [BGRA images]}"""
    out: dict = {}
    for p in sorted(glob.glob(os.path.join(asset_dir, "*.png"))):
        name = os.path.basename(p).rsplit("_", 1)[0]
        im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if im is not None and im.ndim == 3 and im.shape[2] == 4:
            out.setdefault(name, []).append(im)
    return out


# ------------------------------------------------------------------ world
@dataclass
class Candy:
    id: int
    name: str
    x: float
    y: float
    angle: float
    variant: int = 0
    taken_at: float | None = None      # on O0: when the visitor takes it


@dataclass
class ArmState:
    x: float
    y: float
    z: float
    grip: float = 0.0                  # 0 open .. 1 closed
    moving: bool = False
    held: Candy | None = None


class SimTable:
    """The simulated table: camera (``read``) and arm (``pick_and_place``)."""

    def __init__(self, vision: str = "yolo", speed: float = 1.0, seed: int | None = None,
                 assets: dict | None = None, misses_dir: str | None = None):
        self.poses = load_poses()
        self.boxes = {k: v for k, v in self.poses.items() if len(k) == 2 and k[0] in "ABCD"}
        self.o0 = self.poses.get("O0", (0.007, 0.353, 0.145))
        hx, hy, hz = self.poses.get("home", (0.108, -0.003, 0.156))
        self.home_xyz = (hx, hy, hz)
        self.assets = load_assets() if assets is None else assets
        if not self.assets:
            logger.warning("sim: no candy pictures in %s -- drawing labelled blobs and "
                           "reporting the table directly (run: uv run sim_table.py build-assets)",
                           ASSET_DIR)
            vision = "direct"
        if vision == "yolo" and not os.path.isfile(MODEL_PATH):
            logger.warning("sim: no YOLO weights at %s -- reporting the table directly",
                           MODEL_PATH)
            vision = "direct"
        self.vision = vision
        self.speed = max(0.1, float(speed))
        self.rng = random.Random(seed)
        self._lock = threading.RLock()
        self._ids = itertools.count(1)
        self.candies: list[Candy] = []
        self.arm = ArmState(hx, hy, hz)
        self.hidden = False            # calibrating: the board is "cleared"
        self.fail_next = False
        self.events: deque = deque(maxlen=40)
        self.vision_boxes: list = []   # [(name, conf, x1, y1, x2, y2)] from YOLO
        self.vision_at = 0.0
        self._sprites: dict = {}
        self._bg = self._background()
        self._noise = [np.random.default_rng(i).normal(0, 3.0, self._bg.shape).astype(np.int16)
                       for i in range(4)]
        self._frame_no = 0
        self.light = 1.0               # brightness of the "room" (varied by collect)
        # Missed-case capture: the labels of the last frame read() returned.
        self.misses_dir = misses_dir or None
        self.misses_saved = 0
        self._recorded = None          # (labels, arm idle?) of the last camera frame
        self._saved_sigs: set = set()
        self._last_save = 0.0

    # ------------------------------------------------------------ events
    def event(self, text: str):
        logger.warning("sim: %s", text)
        with self._lock:
            self.events.append((time.time(), text))

    def describe(self) -> str:
        n = sum(len(v) for v in self.assets.values())
        return (f"{len(self.assets)} candy kinds ({n} pictures), vision={self.vision}, "
                f"speed x{self.speed:g}")

    # ------------------------------------------------------------- table
    def kinds(self) -> list[str]:
        return sorted(self.assets) or sorted(CANDY_LEN)

    def _new(self, name: str, x: float, y: float) -> Candy:
        n = len(self.assets.get(name, [])) or 1
        return Candy(next(self._ids), name, x, y, self.rng.uniform(0, 360), self.rng.randrange(n))

    def restock(self, n: int = 11):
        """A fresh table: *n* candies in random boxes, a few kinds twice."""
        kinds = self.kinds()
        pick = self.rng.sample(kinds, min(len(kinds), max(1, n - 3)))
        while len(pick) < n:
            pick.append(self.rng.choice(pick))
        self.rng.shuffle(pick)
        tags = self.rng.sample(sorted(self.boxes), min(n, len(self.boxes)))
        with self._lock:
            self.candies = []
            for tag, name in zip(tags, pick):
                x, y, _z = self.boxes[tag]
                j = 0.004
                self.candies.append(self._new(name, x + self.rng.uniform(-j, j),
                                              y + self.rng.uniform(-j, j)))
        self.event(f"table refilled: {len(tags)} candies")

    def clear(self):
        with self._lock:
            self.candies = []
        self.event("table cleared")

    def drop_loose(self, name: str | None = None):
        """A candy dropped between the boxes (for --loose-pick)."""
        # between two neighbouring boxes in the same row
        row = self.rng.choice("ABCD")
        col = self.rng.choice([1, 2])
        a, b = self.boxes[f"{row}{col}"], self.boxes[f"{row}{col + 1}"]
        x = (a[0] + b[0]) / 2 + self.rng.uniform(-0.01, 0.01)
        y = (a[1] + b[1]) / 2 + self.rng.uniform(-0.005, 0.005)
        name = name or self.rng.choice(self.kinds())
        with self._lock:
            self.candies.append(self._new(name, x, y))
        self.event(f"a {name} lies loose between {row}{col} and {row}{col + 1}")

    def set_fail_next(self, on: bool = True):
        self.fail_next = on
        self.event("the next grip will slip" if on else "grips are reliable again")

    @contextmanager
    def cleared(self):
        """The board without candy while calibrating (boxes must be visible)."""
        self.hidden = True
        try:
            yield
        finally:
            self.hidden = False

    def _where(self, c: Candy) -> str:
        best, bd = None, 1e9
        for tag, (x, y, _z) in self.boxes.items():
            d = math.hypot(c.x - x, c.y - y)
            if d < bd:
                best, bd = tag, d
        if bd < BOX_M / 2:
            return best
        if math.hypot(c.x - self.o0[0], c.y - self.o0[1]) < TRAY_R_M:
            return "O0"
        return "loose"

    def truth(self) -> dict:
        """{position: [candy names]} -- what is REALLY on the table."""
        out: dict = {}
        with self._lock:
            for c in self.candies:
                out.setdefault(self._where(c), []).append(c.name)
            if self.arm.held:
                out.setdefault("in gripper", []).append(self.arm.held.name)
        return out

    def _tick(self):
        now = time.time()
        with self._lock:
            taken = [c for c in self.candies if c.taken_at and now >= c.taken_at]
            for c in taken:
                self.candies.remove(c)
        for c in taken:
            self.event(f"the visitor took the {c.name}")

    # --------------------------------------------------------------- arm
    def _ik(self, x: float, y: float):
        """Elbow position of the two-link arm reaching (x, y)."""
        r = max(abs(L1 - L2) + 1e-3, min(L1 + L2 - 1e-4, math.hypot(x, y)))
        th = math.atan2(y, x)
        a = math.acos((L1 * L1 + r * r - L2 * L2) / (2 * L1 * r))
        th1 = th + a                     # elbow towards +y (the open table on the right)
        return L1 * math.cos(th1), L1 * math.sin(th1)

    def _glide(self, x: float, y: float, z: float, dur: float | None = None):
        a = self.arm
        x0, y0, z0 = a.x, a.y, a.z
        if dur is None:
            dur = max(0.25, math.hypot(x - x0, y - y0) / TRAVEL_M_S)
        dur /= self.speed
        steps = max(1, int(dur * FPS))
        for i in range(1, steps + 1):
            t = i / steps
            s = t * t * (3 - 2 * t)      # ease in/out
            with self._lock:
                a.x, a.y, a.z = x0 + (x - x0) * s, y0 + (y - y0) * s, z0 + (z - z0) * s
                if a.held:
                    a.held.x, a.held.y = a.x, a.y
            time.sleep(dur / steps)

    def _set_grip(self, closed: bool):
        a = self.arm
        g0, g1 = a.grip, (1.0 if closed else 0.0)
        steps = max(1, int(GRIP_S / self.speed * FPS))
        for i in range(1, steps + 1):
            with self._lock:
                a.grip = g0 + (g1 - g0) * i / steps
            time.sleep(GRIP_S / self.speed / steps)

    def _grab(self) -> Candy | None:
        a = self.arm
        with self._lock:
            near = [(math.hypot(c.x - a.x, c.y - a.y), -k, c)
                    for k, c in enumerate(self.candies)]
            near = [t for t in near if t[0] < GRIP_RADIUS_M]
            if not near:
                return None
            c = min(near)[2]             # nearest, topmost if stacked
            self.candies.remove(c)
            a.held = c
            c.x, c.y = a.x, a.y
            return c

    def _release(self, jitter: float = 0.004) -> Candy | None:
        a = self.arm
        with self._lock:
            c, a.held = a.held, None
            if c is None:
                return None
            c.x = a.x + self.rng.uniform(-jitter, jitter)
            c.y = a.y + self.rng.uniform(-jitter, jitter)
            c.angle = (c.angle + self.rng.uniform(-25, 25)) % 360
            if math.hypot(c.x - self.o0[0], c.y - self.o0[1]) < TRAY_R_M:
                c.taken_at = time.time() + TAKE_AFTER_S
            self.candies.append(c)
            return c

    def pick_and_place(self, src, dst, label: str = "") -> bool:
        """Called from the arm worker with the poses the real arm would get
        (objects with .x .y .z). Returns True if a candy arrived at *dst*."""
        self.arm.moving = True
        try:
            sx, sy, sz = float(src.x), float(src.y), float(src.z)
            dx, dy, dz = float(dst.x), float(dst.y), float(dst.z)
            self._glide(self.arm.x, self.arm.y, max(self.arm.z, sz + APPROACH_DZ), VERTICAL_S)
            self._set_grip(False)
            self._glide(sx, sy, sz + APPROACH_DZ)
            self._glide(sx, sy, sz, VERTICAL_S)
            self._set_grip(True)
            got = self._grab()
            if got is None:
                self.event(f"{label}: the gripper closed on nothing")
            self._glide(sx, sy, sz + APPROACH_DZ, VERTICAL_S)
            if got is not None and self.fail_next:
                self.fail_next = False
                mx, my = (sx + dx) / 2, (sy + dy) / 2
                self._glide(mx, my, sz + APPROACH_DZ)
                self._set_grip(False)
                self._release(jitter=0.02)
                self.event(f"{label}: the {got.name} slipped out of the gripper on the way")
                got = None
            self._glide(dx, dy, dz + APPROACH_DZ)
            self._glide(dx, dy, dz, VERTICAL_S)
            self._set_grip(False)
            placed = self._release()
            self._glide(dx, dy, dz + APPROACH_DZ, VERTICAL_S)
            if placed is not None:
                self.event(f"{label}: put the {placed.name} down")
            return placed is not None
        finally:
            self.arm.moving = False

    def home(self):
        self.arm.moving = True
        try:
            self._glide(*self.home_xyz)
        finally:
            self.arm.moving = False

    # ------------------------------------------------------------ camera
    def read(self):
        """cv2.VideoCapture-style: (True, BGR frame)."""
        return True, self.render(record=True)

    def release(self):
        pass

    def set_vision(self, boxes: list, frame=None):
        """What YOLO found in the last frame read() returned:
        [(name, conf, x1, y1, x2, y2)]. With a *frame* and a misses_dir, a
        frame it got wrong is saved with the true labels."""
        with self._lock:
            self.vision_boxes = boxes
            self.vision_at = time.time()
        if frame is not None and self.misses_dir:
            try:
                self.save_if_wrong(boxes, frame)
            except Exception as e:  # noqa: BLE001 - never break the camera loop
                logger.warning("sim: could not save a missed case: %s", e)

    # ------------------------------------------------- missed-case capture
    def label_boxes(self, candies) -> list:
        """[(candy, x1, y1, x2, y2)]: where each candy's wrapper is in the
        frame (the visible pixels of its picture)."""
        out = []
        for c in candies:
            sp = self._sprite(c)
            ys, xs = np.nonzero(sp[..., 3] > 40)
            if not len(xs):
                continue
            px, py = to_px(c.x, c.y)
            x0 = int(round(px - sp.shape[1] / 2))
            y0 = int(round(py - sp.shape[0] / 2))
            x1, y1 = max(0, x0 + int(xs.min())), max(0, y0 + int(ys.min()))
            x2, y2 = min(FRAME_W - 1, x0 + int(xs.max())), min(FRAME_H - 1, y0 + int(ys.max()))
            if x2 > x1 and y2 > y1:
                out.append((c, float(x1), float(y1), float(x2), float(y2)))
        return out

    def save_if_wrong(self, boxes: list, frame, force: bool = False) -> list:
        """Compare YOLO's *boxes* with the truth of the last frame read()
        returned; if it got anything wrong, save the frame with the true
        labels. Live, only while the arm stands still (a moving arm hides
        candy, and the labels would not match), and each mistake on a given
        table only once. Returns the problems found."""
        with self._lock:
            rec = self._recorded
        if rec is None:
            return []
        labels, idle = rec
        if not idle and not force:
            return []
        problems = judge(labels, boxes)
        if not problems or not self.misses_dir:
            return problems
        layout = tuple(sorted((c.id, round(c.x, 3), round(c.y, 3)) for c, *_ in labels))
        sig = (layout, tuple(sorted((k, str(i)) for k, _n, i, _w in problems)))
        if not force and (sig in self._saved_sigs or time.time() - self._last_save < 2.0):
            return problems
        if len(self._saved_sigs) > 2000:
            self._saved_sigs.clear()
        self._saved_sigs.add(sig)
        self._last_save = time.time()
        save_case(self.misses_dir, frame, labels, problems, boxes)
        self.misses_saved += 1
        if not force:
            what = ", ".join(f"{k} {n}" for k, n, _i, _w in problems)
            self.event(f"vision got it wrong ({what}) -- saved with the true labels")
        return problems

    def detections(self) -> list[tuple]:
        """[(name, px, py)] of every candy lying on the table (direct vision)."""
        if self.hidden:
            return []
        with self._lock:
            return [(c.name, *to_px(c.x, c.y)) for c in self.candies]

    def annotate(self, frame, dets):
        """Boxes like YOLO's plot, for the direct-vision mode."""
        out = frame.copy()
        r = 0.026 * SCALE
        for name, px, py in dets:
            p1, p2 = (int(px - r), int(py - r)), (int(px + r), int(py + r))
            cv2.rectangle(out, p1, p2, (0, 200, 255), 2)
            cv2.putText(out, name, (p1[0], p1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 200, 255), 1, cv2.LINE_AA)
        return out

    # --------------------------------------------------------- rendering
    def _background(self):
        rng = np.random.default_rng(7)
        img = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
        img[:] = (160, 100, 35)                          # blue table (BGR)
        tex = cv2.resize(rng.normal(0, 6, (FRAME_H // 16, FRAME_W // 16, 1)).astype(np.float32),
                         (FRAME_W, FRAME_H))[..., None]
        img = np.clip(img.astype(np.float32) + tex, 0, 255).astype(np.uint8)
        xs = [p[0] for p in self.boxes.values()]
        ys = [p[1] for p in self.boxes.values()]
        m = BOX_M / 2 + PAPER_MARGIN_M
        p1, p2 = _ipt(min(xs) - m, min(ys) - m), _ipt(max(xs) + m, max(ys) + m)
        cv2.rectangle(img, (p1[0] + 4, p1[1] + 5), (p2[0] + 4, p2[1] + 5), (110, 70, 25), -1)
        cv2.rectangle(img, p1, p2, (236, 238, 240), -1)
        # light falls off towards the robot
        grad = np.linspace(0.97, 1.0, FRAME_H, dtype=np.float32)[:, None, None]
        img = np.clip(img.astype(np.float32) * grad, 0, 255).astype(np.uint8)
        half = BOX_M / 2 * SCALE
        for tag, (x, y, _z) in sorted(self.boxes.items()):
            cx, cy = to_px(x, y)
            _rounded_rect(img, (cx - half, cy - half), (cx + half, cy + half), 0.18 * half * 2,
                          (35, 35, 35), 3)
            cv2.putText(img, tag, (int(cx - half), int(cy + half + 14)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (120, 120, 120), 1, cv2.LINE_AA)
        # the hand-out tray in front of the visitor
        ox, oy = _ipt(self.o0[0], self.o0[1])
        rr = int(TRAY_R_M * SCALE)
        cv2.circle(img, (ox + 4, oy + 5), rr, (90, 55, 20), -1, cv2.LINE_AA)
        cv2.circle(img, (ox, oy), rr, (40, 130, 235), -1, cv2.LINE_AA)
        cv2.circle(img, (ox, oy), rr - 8, (60, 160, 250), -1, cv2.LINE_AA)
        cv2.putText(img, "O0", (ox - 12, oy + rr + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (230, 230, 230), 1, cv2.LINE_AA)
        # the robot's base
        bx, by = _ipt(0, 0)
        cv2.circle(img, (bx + 5, by + 6), int(0.07 * SCALE), (95, 60, 22), -1, cv2.LINE_AA)
        cv2.circle(img, (bx, by), int(0.07 * SCALE), (70, 70, 74), -1, cv2.LINE_AA)
        cv2.circle(img, (bx, by), int(0.05 * SCALE), (95, 95, 100), -1, cv2.LINE_AA)
        return img

    def _sprite(self, c: Candy, scale: float = 1.0):
        key = (c.id, c.name, c.variant, round(c.angle), round(scale, 2))
        sp = self._sprites.get(key)
        if sp is not None:
            return sp
        if len(self._sprites) > 400:
            self._sprites.clear()
        pics = self.assets.get(c.name)
        length = CANDY_LEN.get(c.name, DEFAULT_LEN) * SCALE * scale
        if pics:
            src = pics[c.variant % len(pics)]
            f = length / max(src.shape[:2])
            src = cv2.resize(src, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        else:
            src = _blob(c.name, int(length))
        h, w = src.shape[:2]
        d = int(math.ceil(math.hypot(w, h))) + 2
        M = cv2.getRotationMatrix2D((w / 2, h / 2), c.angle, 1.0)
        M[0, 2] += (d - w) / 2
        M[1, 2] += (d - h) / 2
        sp = cv2.warpAffine(src, M, (d, d), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        self._sprites[key] = sp
        return sp

    def render(self, overlay: bool = False, record: bool = False):
        """One frame. *record*: this is the camera's frame -- remember where
        every candy is in it (for save_if_wrong)."""
        self._tick()
        img = self._bg.copy()
        with self._lock:
            candies = [] if self.hidden else list(self.candies)
            a = ArmState(self.arm.x, self.arm.y, self.arm.z, self.arm.grip,
                         self.arm.moving, self.arm.held)
        if record:
            idle = not a.moving and a.held is None and not self.hidden
            labels = self.label_boxes(candies) if (idle or self.misses_dir) else []
            with self._lock:
                self._recorded = (labels, idle)
        for c in candies:
            _paste(img, self._sprite(c), *to_px(c.x, c.y), shadow=(3, 4))
        lift = 1.0 + 2.2 * max(0.0, a.z - TABLE_Z)
        if a.held is not None:
            off = int(30 * (a.z - TABLE_Z) * SCALE / 100)
            _paste(img, self._sprite(a.held, lift), *to_px(a.x, a.y), shadow=(3 + off, 4 + off))
        self._draw_arm(img, a, lift)
        self._frame_no += 1
        base = img.astype(np.int16)
        if self.light != 1.0:
            base = (base * self.light).astype(np.int16)
        noisy = np.clip(base + self._noise[self._frame_no % len(self._noise)],
                        0, 255).astype(np.uint8)
        if overlay:
            self._overlay(noisy)
        return noisy

    def _draw_arm(self, img, a: ArmState, lift: float):
        ex, ey = self._ik(a.x, a.y)
        base, elbow, tip = _ipt(0, 0), _ipt(ex, ey), _ipt(a.x, a.y)
        sh = int(8 + 60 * max(0.0, a.z - TABLE_Z))
        shadow = img.copy()
        for p, q, w in ((base, elbow, 40), (elbow, tip, 32)):
            cv2.line(shadow, (p[0] + sh, p[1] + sh), (q[0] + sh, q[1] + sh), (0, 0, 0), w, cv2.LINE_AA)
        cv2.addWeighted(shadow, 0.25, img, 0.75, 0, img)
        for p, q, w in ((base, elbow, 40), (elbow, tip, 32)):
            cv2.line(img, p, q, (55, 55, 60), w + 4, cv2.LINE_AA)
            cv2.line(img, p, q, (225, 225, 228), w, cv2.LINE_AA)
        for p, r in ((base, 26), (elbow, 22)):
            cv2.circle(img, p, r, (55, 55, 60), -1, cv2.LINE_AA)
            cv2.circle(img, p, r - 6, (0, 140, 255), -1, cv2.LINE_AA)
        # the gripper: two fingers across the forearm, further apart when open
        dx, dy = tip[0] - elbow[0], tip[1] - elbow[1]
        n = math.hypot(dx, dy) or 1.0
        ux, uy = dx / n, dy / n                          # along the forearm
        vx, vy = -uy, ux                                 # across
        gap = (0.008 + 0.018 * (1 - a.grip)) * SCALE * lift
        flen = 0.022 * SCALE * lift
        # the wrist plate the fingers slide on
        wx, wy = vx * (gap + 6), vy * (gap + 6)
        cv2.line(img, (int(tip[0] - wx), int(tip[1] - wy)), (int(tip[0] + wx), int(tip[1] + wy)),
                 (55, 55, 60), int(12 * lift), cv2.LINE_AA)
        for s in (-1, 1):
            cx, cy = tip[0] + s * vx * gap, tip[1] + s * vy * gap
            p = (int(cx - ux * flen / 2), int(cy - uy * flen / 2))
            q = (int(cx + ux * flen / 2), int(cy + uy * flen / 2))
            cv2.line(img, p, q, (30, 30, 34), int(10 * lift), cv2.LINE_AA)
            cv2.line(img, p, q, (0, 140, 255) if a.grip > 0.5 else (120, 120, 125),
                     max(2, int(4 * lift)), cv2.LINE_AA)
        cv2.circle(img, tip, int(14 * lift), (55, 55, 60), -1, cv2.LINE_AA)
        cv2.circle(img, tip, int(8 * lift), (0, 140, 255) if a.moving else (150, 150, 155),
                   -1, cv2.LINE_AA)

    def _overlay(self, img):
        with self._lock:
            boxes = list(self.vision_boxes) if time.time() - self.vision_at < 2.0 else []
        for name, conf, x1, y1, x2, y2 in boxes:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 230, 120), 2)
            label = f"{name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (int(x1), int(y1) - th - 6), (int(x1) + tw + 4, int(y1)),
                          (0, 230, 120), -1)
            cv2.putText(img, label, (int(x1) + 2, int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (20, 20, 20), 1, cv2.LINE_AA)
        txt = f"SIMULATED TABLE  vision: {self.vision}" + ("   [calibrating]" if self.hidden else "")
        cv2.rectangle(img, (0, 0), (FRAME_W, 28), (30, 30, 30), -1)
        cv2.putText(img, txt, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1,
                    cv2.LINE_AA)

    def jpeg(self, quality: int = 80) -> bytes:
        ok, buf = cv2.imencode(".jpg", self.render(overlay=True),
                               [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else b""

    def state(self) -> dict:
        with self._lock:
            ev = [(t, s) for t, s in self.events]
        return {"truth": self.truth(), "events": ev, "moving": self.arm.moving,
                "fail_next": self.fail_next, "vision": self.vision,
                "calibrating": self.hidden,
                "misses": {"dir": self.misses_dir, "saved": self.misses_saved}}


# ------------------------------------------------------- vision mistakes
def judge(labels: list, boxes: list) -> list:
    """What the detector got wrong: [(kind, candy name, candy id, detail)]
    with kind missed / wrong (detail = the name it gave) / double / phantom
    (a box on no candy; name = the name it gave). A box counts for a candy
    when its centre lies on that candy's wrapper (6 px slack)."""
    problems, used = [], set()
    for c, x1, y1, x2, y2 in labels:
        inside = [i for i, b in enumerate(boxes)
                  if x1 - 6 <= (b[2] + b[4]) / 2 <= x2 + 6 and y1 - 6 <= (b[3] + b[5]) / 2 <= y2 + 6]
        used.update(inside)
        wrong = [boxes[i][0] for i in inside if not same_candy(boxes[i][0], c.name)]
        if not inside:
            problems.append(("missed", c.name, c.id, None))
        elif wrong:
            problems.append(("wrong", c.name, c.id, wrong[0]))
        elif len(inside) > 1:
            problems.append(("double", c.name, c.id, None))
    for i, b in enumerate(boxes):
        if i not in used:
            problems.append(("phantom", b[0], None, None))
    return problems


_case_no = itertools.count(1)


def save_case(out_dir: str, frame, labels: list, problems: list, boxes: list) -> str:
    """<out_dir>/images/<stem>.jpg + labels/<stem>.txt (YOLO format with the
    candy dataset's class ids) + a line in misses.jsonl saying what the
    model got wrong. Drop the images/labels into the dataset to train on."""
    import json
    img_dir, lab_dir = os.path.join(out_dir, "images"), os.path.join(out_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)
    yaml_path = os.path.join(out_dir, "data.yaml")
    if not os.path.exists(yaml_path):
        with open(yaml_path, "w") as f:
            f.write("# Simulated-table frames the YOLO model got wrong, with the true labels.\n"
                    "path: .\ntrain: images\nval: images\n"
                    f"nc: {len(DATASET_NAMES)}\nnames:\n"
                    + "".join(f"- {n}\n" for n in DATASET_NAMES))
    stem = time.strftime("sim_%Y%m%dT%H%M%S") + f"_{os.getpid()}_{next(_case_no):05d}"
    h, w = frame.shape[:2]
    cv2.imwrite(os.path.join(img_dir, stem + ".jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    lines = []
    for c, x1, y1, x2, y2 in labels:
        if c.name in DATASET_NAMES:
            lines.append(f"{DATASET_NAMES.index(c.name)} {(x1 + x2) / 2 / w:.6f} "
                         f"{(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    with open(os.path.join(lab_dir, stem + ".txt"), "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    with open(os.path.join(out_dir, "misses.jsonl"), "a") as f:
        f.write(json.dumps({
            "image": f"images/{stem}.jpg",
            "problems": [{"kind": k, "candy": n, "detail": d} for k, n, _i, d in problems],
            "truth": [{"candy": c.name, "box": [round(v) for v in (x1, y1, x2, y2)]}
                      for c, x1, y1, x2, y2 in labels],
            "predicted": [{"name": b[0], "conf": round(b[1], 3), "box": [round(v) for v in b[2:6]]}
                          for b in boxes],
        }) + "\n")
    return stem


def collect(n: int, out_dir: str = MISSES_DIR, seed: int | None = None,
            model_path: str = MODEL_PATH) -> dict:
    """Render *n* random tables (5-12 candies, some lying between the
    boxes, varied light), run the YOLO model on each and save the frames it
    gets wrong with the true labels. Returns {kind: {candy: count}}."""
    from collections import Counter
    from ultralytics import YOLO
    model = YOLO(model_path)
    sim = SimTable(vision="direct", seed=seed, misses_dir=out_dir)
    stats: dict = {}
    saved = 0
    for _ in range(n):
        sim.restock(sim.rng.randint(5, 12))
        for _k in range(sim.rng.choice((0, 0, 1, 2))):
            sim.drop_loose()
        sim.light = sim.rng.uniform(0.8, 1.12)
        _ok, frame = sim.read()
        r = model(frame, verbose=False)[0]
        boxes = [(r.names[int(b.cls)], float(b.conf), *map(float, b.xyxy[0])) for b in r.boxes]
        problems = sim.save_if_wrong(boxes, frame, force=True)
        saved += bool(problems)
        for k, name, _i, _d in problems:
            stats.setdefault(k, Counter())[name] += 1
    stats["frames saved"] = saved
    return stats


# ---------------------------------------------------------------- drawing
def _rounded_rect(img, p1, p2, r, color, thickness):
    x1, y1 = p1
    x2, y2 = p2
    r = int(r)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    for (a, b), (c, d) in (((x1 + r, y1), (x2 - r, y1)), ((x1 + r, y2), (x2 - r, y2)),
                           ((x1, y1 + r), (x1, y2 - r)), ((x2, y1 + r), (x2, y2 - r))):
        cv2.line(img, (a, b), (c, d), color, thickness, cv2.LINE_AA)
    for (cx, cy), ang in (((x1 + r, y1 + r), 180), ((x2 - r, y1 + r), 270),
                          ((x2 - r, y2 - r), 0), ((x1 + r, y2 - r), 90)):
        cv2.ellipse(img, (cx, cy), (r, r), ang, 0, 90, color, thickness, cv2.LINE_AA)


def _paste(img, sp, cx: float, cy: float, shadow=(0, 0)):
    """Alpha-blend the BGRA sprite *sp* centred at (cx, cy), with a soft shadow."""
    h, w = sp.shape[:2]
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    alpha = sp[..., 3:4].astype(np.float32) / 255.0
    if shadow != (0, 0):
        sa = cv2.GaussianBlur(alpha[..., 0], (9, 9), 0)[..., None] * 0.35
        _blend(img, np.zeros_like(sp[..., :3]), sa, x0 + shadow[0], y0 + shadow[1])
    _blend(img, sp[..., :3], alpha, x0, y0)


def _blend(img, rgb, alpha, x0, y0):
    H, W = img.shape[:2]
    h, w = rgb.shape[:2]
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if X0 >= X1 or Y0 >= Y1:
        return
    sx, sy = X0 - x0, Y0 - y0
    a = alpha[sy:sy + Y1 - Y0, sx:sx + X1 - X0]
    roi = img[Y0:Y1, X0:X1].astype(np.float32)
    src = rgb[sy:sy + Y1 - Y0, sx:sx + X1 - X0].astype(np.float32)
    img[Y0:Y1, X0:X1] = (roi * (1 - a) + src * a).astype(np.uint8)


def _blob(name: str, length: int):
    """Stand-in when there is no picture: a coloured wrapper with the name."""
    h = max(12, int(length * 0.55))
    col = [int(v) for v in np.random.default_rng(abs(hash(name)) % 2**32).integers(40, 220, 3)]
    sp = np.zeros((h, length, 4), np.uint8)
    cv2.rectangle(sp, (0, 0), (length - 1, h - 1), (*col, 255), -1)
    cv2.putText(sp, name[:6], (2, h // 2 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                (255, 255, 255, 255), 1, cv2.LINE_AA)
    return sp


# -------------------------------------------------------------------- cli
def main():
    import argparse
    ap = argparse.ArgumentParser(description="CandyTron simulated table")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-assets", help="cut candy pictures out of the YOLO dataset")
    b.add_argument("--dataset", default=DATASET_DIR)
    b.add_argument("--per-class", type=int, default=3)
    p = sub.add_parser("preview", help="write a picture of a stocked table")
    p.add_argument("out", nargs="?", default="sim_preview.jpg")
    c = sub.add_parser("collect", help="save random tables the YOLO model gets wrong, "
                                        "with the true labels (for training)")
    c.add_argument("-n", type=int, default=200, help="tables to render (default 200)")
    c.add_argument("--out", default=MISSES_DIR)
    c.add_argument("--seed", type=int, default=None)
    c.add_argument("--model", default=MODEL_PATH)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.cmd == "collect":
        logger.setLevel(logging.ERROR)
        stats = collect(args.n, args.out, args.seed, args.model)
        print(f"{stats.pop('frames saved')} of {args.n} tables had a mistake; saved to {args.out}")
        for kind, cnt in sorted(stats.items()):
            print(f"  {kind:8s} " + ", ".join(f"{k} {v}" for k, v in cnt.most_common()))
        return
    if args.cmd == "build-assets":
        counts = build_assets(args.dataset, per_class=args.per_class)
        for k, v in sorted(counts.items()):
            print(f"{k:14s} {v}")
    else:
        sim = SimTable(seed=1)
        sim.restock()
        cv2.imwrite(args.out, sim.render(overlay=True))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
