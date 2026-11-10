import cv2
import logging
import numpy as np
import os
import threading
import time
from ultralytics import YOLO

logger = logging.getLogger("candytron.camera")

# Simulated scene data for testing without camera
_SIMULATED_DETECTIONS = [
    ('Riesen', 608.4746, 261.0399), ('Pearnut', 190.6651, 281.6597),
    ('Geisha', 324.3160, 287.9541), ('Dumle', 331.5356, 43.3107),
    ('VanillaFudge', 328.1843, 159.3306), ('Riesen', 463.4600, 432.1105),
    ('Refreshers', 461.0807, 162.0085), ('Riesen', 316.5636, 432.8765),
    ('Refreshers', 567.1455, 74.4779), ('Plopp', 197.7049, 161.3752),
    ('Refreshers', 466.4943, 293.8249),
]

_SIMULATED_CORNERS = [
    ('Refreshers', 465.0, 43.0), ('Riesen', 465.0, 432.0),
    ('Plopp', 193, 43.0), ('Pearnut', 190.6651, 432.0),
]



# ---------------------------------------------------------------- board boxes
# The board is white paper with one black-outlined rounded box per square.
# On an EMPTY board all of them are visible, and their centres give every
# square directly -- no candy needed, and no interpolation between corners.

def find_board_boxes(frame) -> list[tuple]:
    """(cx, cy, w, h) of every printed box outline found on the paper."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    paper = ((hsv[..., 2] > 140) & (hsv[..., 1] < 60)).astype(np.uint8) * 255
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    th = cv2.bitwise_and(th, paper)
    cnts, hier = cv2.findContours(th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    H, W = g.shape
    boxes = []
    if hier is None:
        return boxes
    for i, c in enumerate(cnts):
        if hier[0][i][3] < 0:          # the INSIDE of an outline is a hole
            continue
        x, y, w, h = cv2.boundingRect(c)
        a = cv2.contourArea(c)
        if not (0.002 * W * H < a < 0.03 * W * H):
            continue
        if a / float(w * h) < 0.75 or not (0.5 < w / float(h) < 1.5):
            continue
        boxes.append((x + w / 2.0, y + h / 2.0, float(w), float(h)))
    return boxes


def snap_to_boxes(positions: dict, boxes: list[tuple]) -> int:
    """Move each square to the centre of the printed box nearest to it, when
    that box is clearly the one (closer than half a box). In place; returns
    how many squares were snapped."""
    snapped, used = 0, set()
    for tag, (x, y) in list(positions.items()):
        best, bd = None, None
        for k, b in enumerate(boxes):
            d = ((b[0] - x) ** 2 + (b[1] - y) ** 2) ** 0.5
            if d < 0.5 * min(b[2], b[3]) and (bd is None or d < bd) and k not in used:
                best, bd = k, d
        if best is not None:
            used.add(best)
            positions[tag] = (boxes[best][0], boxes[best][1])
            snapped += 1
    return snapped


def grid_from_boxes(boxes: list[tuple], n: int, m: int):
    """{tag: (x, y)} if *boxes* are exactly an m-row x n-column grid, else None.
    Rows are letters (A at the BOTTOM of the image), columns numbers (1 left),
    the same orientation as the 4-candy calibration."""
    if len(boxes) != n * m:
        return None
    # The paper is rarely square to the camera: estimate the grid's angle
    # from the direction to each box's nearest neighbour (folded into
    # -45..45 deg), and sort into rows in that rotated frame.
    pts = np.array([[b[0], b[1]] for b in boxes])
    angs = []
    for k, p in enumerate(pts):
        d = np.hypot(*(pts - p).T)
        d[k] = np.inf
        v = pts[int(np.argmin(d))] - p
        a = np.degrees(np.arctan2(v[1], v[0]))
        angs.append((a + 45) % 90 - 45)
    theta = np.radians(float(np.median(angs)))
    c, s_ = np.cos(-theta), np.sin(-theta)
    rot = {id(b): (c * b[0] - s_ * b[1], s_ * b[0] + c * b[1]) for b in boxes}
    by_y = sorted(boxes, key=lambda b: rot[id(b)][1])
    rows = [sorted(by_y[r * n:(r + 1) * n], key=lambda b: rot[id(b)][0]) for r in range(m)]
    for r in range(m):
        # Perspective makes near boxes bigger than far ones, so sizes are
        # compared within a row only.
        hs = np.array([b[3] for b in rows[r]])
        hmed = float(np.median(hs))
        if np.any(np.abs(hs - hmed) > 0.3 * hmed):
            return None                  # something that is not a board box
        ys = [rot[id(b)][1] for b in rows[r]]
        if max(ys) - min(ys) > 0.6 * hmed:
            return None                  # a "row" that is not level
        if r and min(ys) - max(rot[id(b)][1] for b in rows[r - 1]) < 0.6 * min(hmed, prev_h):
            return None                  # rows not separated
        prev_h = hmed
    positions = {}
    for r, row in enumerate(rows):
        letter = chr(65 + (m - 1 - r))   # top row = last letter (D)
        for c, b in enumerate(row):
            positions[f"{letter}{c + 1}"] = (b[0], b[1])
    return positions

class CameraManager:
    _SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, camera_index: int = 0, model_path: str = 'models/best-m.pt',
                 show_window: bool = True, simulate: bool = False):
        self.loose_pick = False   # --loose-pick: report candy outside the boxes as L1, L2...
        self.loose: dict = {}     # {"L1": (x, y)} pixel position of loose candy
        self.camera_index = camera_index
        self.model_path = os.path.join(self._SOURCE_DIR, model_path)
        self.show_window = show_window
        self.simulate = simulate
        self.capture: cv2.VideoCapture | None = None
        self.yolomodel: YOLO | None = None
        self.positions: dict[str, tuple] = {}
        self.positions_thresdist: float = 0
        self._open_window_count: int = 0
        self._last_frame = None
        self._last_raw_frame = None
        self._last_frame_lock = threading.Lock()
        # Calibration diagnostics, for progress/status lines in main():
        self.last_detection_count: int = 0      # candies seen by the last calibrate attempt
        self.calibration_warnings: list[str] = []  # sanity-check findings of the last success
        self._frame_size: tuple[int, int] = (640, 480)  # (w, h) of the last frame

    @staticmethod
    def list_cameras(max_index: int = 10) -> list[dict]:
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

    @staticmethod
    def find_first_camera(max_index: int = 10) -> int | None:
        for i in range(max_index):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.release()
                return i
        return None

    def init_cam(self) -> None:
        if self.simulate:
            logger.info("Camera simulation mode enabled")
            return
        if not os.path.isfile(self.model_path):
            raise RuntimeError(f"YOLO model not found: {self.model_path}")
        logger.info("Loading YOLO model from %s", self.model_path)
        self.yolomodel = YOLO(self.model_path)
        logger.debug("Opening camera index %d", self.camera_index)
        self.capture = cv2.VideoCapture(self.camera_index)
        if not self.capture or not self.capture.isOpened():
            raise RuntimeError(f"Cannot open camera index {self.camera_index}")
        logger.info("Camera initialized successfully")

    def exit_cam(self) -> None:
        if self.has_camera() and self.show_window:
            cv2.destroyAllWindows()
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def has_camera(self) -> bool:
        return self.yolomodel is not None

    def get_last_frame(self):
        """Return a copy of the most recent annotated camera frame, or None if unavailable."""
        with self._last_frame_lock:
            if self._last_frame is None:
                return None
            return self._last_frame.copy()

    def get_last_raw_frame(self):
        """Return a copy of the most recent raw camera frame, or None if unavailable."""
        with self._last_frame_lock:
            if self._last_raw_frame is None:
                return None
            return self._last_raw_frame.copy()

    def acquire_scene_one(self, refresh: bool = False) -> list[tuple]:
        if not self.has_camera():
            return list(_SIMULATED_DETECTIONS)
        if not self.capture:
            return []
        if refresh:
            self.capture.read()
            self.capture.read()
        ret, fr = self.capture.read()
        if not ret:
            logger.error("Failed to read frame from camera")
            return []
        self._frame_size = (fr.shape[1], fr.shape[0])
        res = self.yolomodel(fr, verbose=False)[0]
        annotated_frame = res.plot()
        with self._last_frame_lock:
            self._last_frame = annotated_frame
            self._last_raw_frame = fr

        if self.show_window:
            h, w = annotated_frame.shape[:2]
            display_frame = annotated_frame
            if w > 1280:
                display_frame = cv2.resize(annotated_frame, (w // 2, h // 2))
            cv2.imshow(f'CandyTron 4000 - Table camera {self.camera_index} (YOLO)', display_frame)
            if self._open_window_count == 0:
                cv2.moveWindow('YOLO Detection', 200, 50)
            self._open_window_count += 1

        return [(res.names[int(bx.cls)],
                 (bx.xyxy[0][0] + bx.xyxy[0][2]) / 2,
                 (bx.xyxy[0][1] + bx.xyxy[0][3]) / 2) for bx in res.boxes]

    def poll_key(self, wait_ms: int = 50) -> str | None:
        """Pump the GUI event loop and pace the caller.

        Always sleeps ``wait_ms`` ms (via ``cv2.waitKey`` when a window is
        shown, otherwise ``time.sleep``). Returns the key pressed in the
        table-camera window as a one-character string, or None.
        """
        if self.has_camera() and self.show_window:
            k = cv2.waitKey(wait_ms) & 0xFF
            return None if k == 0xFF else chr(k)
        if wait_ms > 0:
            time.sleep(wait_ms / 1000.0)
        return None

    def check_event(self, wait_ms: int = 50) -> bool:
        """Like poll_key(); returns True if the user pressed 'q'."""
        return self.poll_key(wait_ms) == 'q'

    # Board corner -> tag. calibrate_positions() maps the image corners
    # (top-left, bottom-left, top-right, bottom-right) onto these board spots.
    _CORNER_TAGS = {'tl': 'D1', 'bl': 'A1', 'tr': 'D3', 'br': 'A3'}

    def calibrate_positions(self, n: int, m: int) -> bool:
        """One calibration attempt: needs EXACTLY 4 detections (the corners).

        On success the new position map replaces the old one atomically; on
        failure the previous calibration (if any) is kept untouched.
        ``last_detection_count`` records how many candies this attempt saw.
        """
        lst = self.acquire_scene_one() if self.has_camera() else list(_SIMULATED_CORNERS)
        self.last_detection_count = len(lst)
        self.last_box_count = 0
        if self.has_camera() and not lst:
            # Empty board: calibrate from the printed boxes themselves.
            raw = self.get_last_raw_frame()
            if raw is not None:
                try:
                    boxes = find_board_boxes(raw)
                    self.last_box_count = len(boxes)
                    grid = grid_from_boxes(boxes, n, m)
                except Exception as e:  # noqa: BLE001
                    logger.warning("box calibration failed: %s", e)
                    grid = None
                if grid:
                    pitch_x = (max(p[0] for p in grid.values()) - min(p[0] for p in grid.values())) / (n - 1)
                    pitch_y = (max(p[1] for p in grid.values()) - min(p[1] for p in grid.values())) / (m - 1)
                    self.positions = grid
                    self.positions_thresdist = (min(pitch_x, pitch_y) / 2) ** 2
                    self.calibration_warnings = []
                    self.calibration_method = "boxes"
                    logger.info("Calibration from the %d printed boxes: %d positions",
                                len(boxes), len(grid))
                    return True
        if len(lst) != 4:
            logger.debug("Calibration failed: expected 4 candies, found %d", len(lst))
            return False
        lst = [(t[0], float(t[1]), float(t[2])) for t in lst]
        lst.sort(key=lambda t: t[1])
        if lst[0][2] < lst[1][2]:
            tl, bl = lst[0], lst[1]
        else:
            tl, bl = lst[1], lst[0]
        if lst[2][2] < lst[3][2]:
            tr, br = lst[2], lst[3]
        else:
            tr, br = lst[3], lst[2]
        positions = {}
        for i in range(n):
            for j in range(m):
                tag = chr(j + 65) + str(i + 1)
                pos = tuple(
                    (tl[k] * (n - 1 - i) * j + bl[k] * (n - 1 - i) * (m - 1 - j) +
                     tr[k] * i * j + br[k] * i * (m - 1 - j)) / (n - 1) / (m - 1)
                    for k in [1, 2]
                )
                positions[tag] = pos
        # Snap every square whose printed box is visible (all but the 4
        # covered by the corner candies) to the box's exact centre: the
        # corner blend is off for the inner squares when the camera looks
        # at the board at an angle.
        if self.has_camera():
            raw = self.get_last_raw_frame()
            if raw is not None:
                try:
                    self.last_box_count = snap_to_boxes(positions, find_board_boxes(raw))
                except Exception as e:  # noqa: BLE001
                    logger.warning("box snapping failed: %s", e)
        self.positions = positions
        self.positions_thresdist = min(
            (tr[1] + br[1] - tl[1] - bl[1]) / (n - 1) / 4,
            (bl[2] + br[2] - tl[2] - tr[2]) / (m - 1) / 4
        ) ** 2
        self.calibration_warnings = self._sanity_check(
            {'tl': tl, 'bl': bl, 'tr': tr, 'br': br})
        for w in self.calibration_warnings:
            logger.warning("Calibration: %s", w)
        self.calibration_method = "candies"
        logger.info("Calibration successful: %d positions mapped (%d snapped to printed boxes)",
                    len(self.positions), getattr(self, "last_box_count", 0))
        return True

    def _sanity_check(self, corners: dict) -> list[str]:
        """Warn (never reject) about a suspicious corner quadrilateral."""
        warnings = []
        w, h = self._frame_size
        pts = {k: (v[1], v[2]) for k, v in corners.items()}

        def dist(a, b):
            return ((pts[a][0] - pts[b][0]) ** 2 + (pts[a][1] - pts[b][1]) ** 2) ** 0.5

        # Opposite sides of a board seen from above should be similar.
        for (a, b), (c, d), what in ((('tl', 'tr'), ('bl', 'br'), 'top/bottom'),
                                      (('tl', 'bl'), ('tr', 'br'), 'left/right')):
            s1, s2 = dist(a, b), dist(c, d)
            ratio = min(s1, s2) / max(s1, s2) if max(s1, s2) > 0 else 0.0
            if ratio < 0.6:
                warnings.append(f"board looks skewed: {what} sides differ "
                                f"({s1:.0f} vs {s2:.0f} px) - is a corner candy misplaced?")
        # Shoelace area of tl -> tr -> br -> bl.
        poly = [pts['tl'], pts['tr'], pts['br'], pts['bl']]
        area = abs(sum(poly[i][0] * poly[(i + 1) % 4][1] - poly[(i + 1) % 4][0] * poly[i][1]
                       for i in range(4))) / 2
        if area < 0.03 * w * h:
            warnings.append(f"board area is tiny ({100 * area / (w * h):.1f}% of the image) "
                            f"- are the 4 candies really in the corners?")
        mx, my = 0.04 * w, 0.04 * h
        for k, (x, y) in pts.items():
            if x < mx or x > w - mx or y < my or y > h - my:
                warnings.append(f"corner {self._CORNER_TAGS[k]} is at the image edge "
                                f"({x:.0f},{y:.0f} in {w}x{h}) - aim the table camera so "
                                f"the whole board is well inside the picture")
        return warnings

    def camera_positions(self) -> dict[str, tuple]:
        return self.positions

    def find_position(self, xy: tuple) -> str | bool:
        mindist = self.positions_thresdist
        mintag = False
        for tag in self.positions:
            dist = (self.positions[tag][0] - xy[0]) ** 2 + (self.positions[tag][1] - xy[1]) ** 2
            if dist < mindist:
                mindist = dist
                mintag = tag
        return mintag

    def grab_and_detect(self) -> dict[str, str]:
        scene_one = self.acquire_scene_one()
        result = {}
        loose = []
        for ele in scene_one:
            ptag = self.find_position((ele[1], ele[2]))
            if ptag:
                result[ptag] = ele[0]
            elif self.loose_pick and self._on_board((float(ele[1]), float(ele[2]))):
                loose.append((ele[0], float(ele[1]), float(ele[2])))
        # Candy on the board but in no box: "L1", "L2"... (left to right), so
        # the arm can pick it from its coordinate (--loose-pick).
        loose.sort(key=lambda t: (t[1], t[2]))
        self.loose = {}
        for k, (name, x, y) in enumerate(loose[:9]):
            tag = f"L{k + 1}"
            self.loose[tag] = (x, y)
            result[tag] = name
        return result

    def _on_board(self, xy: tuple) -> bool:
        """Inside the board (the hull of the square centres, widened by half
        a square). Off-board things -- a wrapper on the table -- are ignored."""
        if len(self.positions) < 4:
            return False
        pts = np.array(list(self.positions.values()), np.float32)
        hull = cv2.convexHull(pts)
        margin = 0.5 * (self.positions_thresdist ** 0.5) * 2   # ~half a square pitch
        return cv2.pointPolygonTest(hull, (float(xy[0]), float(xy[1])), True) >= -margin
