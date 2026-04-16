import cv2
import logging
import os
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


class CameraManager:
    _SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, camera_index: int = 0, model_path: str = 'models/best-m.pt',
                 show_window: bool = True, simulate: bool = False):
        self.camera_index = camera_index
        self.model_path = os.path.join(self._SOURCE_DIR, model_path)
        self.show_window = show_window
        self.simulate = simulate
        self.capture: cv2.VideoCapture | None = None
        self.yolomodel: YOLO | None = None
        self.positions: dict[str, tuple] = {}
        self.positions_thresdist: float = 0
        self._open_window_count: int = 0

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
        res = self.yolomodel(fr, verbose=False)[0]

        if self.show_window:
            annotated_frame = res.plot()
            h, w = annotated_frame.shape[:2]
            if w > 1280:
                annotated_frame = cv2.resize(annotated_frame, (w // 2, h // 2))
            cv2.imshow('YOLO Detection', annotated_frame)
            if self._open_window_count == 0:
                cv2.moveWindow('YOLO Detection', 200, 50)
            self._open_window_count += 1

        return [(res.names[int(bx.cls)],
                 (bx.xyxy[0][0] + bx.xyxy[0][2]) / 2,
                 (bx.xyxy[0][1] + bx.xyxy[0][3]) / 2) for bx in res.boxes]

    def check_event(self, wait_ms: int = 1) -> bool:
        if self.has_camera() and self.show_window:
            return cv2.waitKey(wait_ms) & 0xFF == ord('q')
        return False

    def calibrate_positions(self, n: int, m: int) -> bool:
        lst = self.acquire_scene_one() if self.has_camera() else list(_SIMULATED_CORNERS)
        if len(lst) != 4:
            logger.debug("Calibration failed: expected 4 candies, found %d", len(lst))
            return False
        lst.sort(key=lambda t: t[1])
        if lst[0][2] < lst[1][2]:
            tl, bl = lst[0], lst[1]
        else:
            tl, bl = lst[1], lst[0]
        if lst[2][2] < lst[3][2]:
            tr, br = lst[2], lst[3]
        else:
            tr, br = lst[3], lst[2]
        self.positions = {}
        for i in range(n):
            for j in range(m):
                tag = chr(j + 65) + str(i + 1)
                pos = tuple(
                    (tl[k] * (n - 1 - i) * j + bl[k] * (n - 1 - i) * (m - 1 - j) +
                     tr[k] * i * j + br[k] * i * (m - 1 - j)) / (n - 1) / (m - 1)
                    for k in [1, 2]
                )
                self.positions[tag] = pos
        self.positions_thresdist = min(
            (tr[1] + br[1] - tl[1] - bl[1]) / (n - 1) / 4,
            (bl[2] + br[2] - tl[2] - tr[2]) / (m - 1) / 4
        ) ** 2
        logger.info("Calibration successful: %d positions mapped", len(self.positions))
        return True

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
        for ele in scene_one:
            ptag = self.find_position((ele[1], ele[2]))
            if ptag:
                result[ptag] = ele[0]
        return result
