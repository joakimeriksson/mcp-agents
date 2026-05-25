import json
import logging
import os
from datetime import datetime, timezone

import cv2

logger = logging.getLogger("candytron.scene_logger")


class SceneLogger:
    """Persist scene-request snapshots: <timestamp>.jpg (annotated),
    <timestamp>_raw.jpg (raw), and <timestamp>.json (state)."""

    def __init__(self, directory: str):
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)

    def log(self, scene: dict[str, str], frame, raw_frame, lang: str, message: str) -> None:
        try:
            now = datetime.now(timezone.utc)
            prefix = now.strftime("%Y%m%dT%H%M%S_%f") + "Z"
            base = os.path.join(self.directory, prefix)

            if frame is not None:
                jpg_path = base + ".jpg"
                if not cv2.imwrite(jpg_path, frame):
                    logger.warning("cv2.imwrite returned False for %s", jpg_path)

            if raw_frame is not None:
                raw_path = base + "_raw.jpg"
                if not cv2.imwrite(raw_path, raw_frame):
                    logger.warning("cv2.imwrite returned False for %s", raw_path)

            payload = {
                "timestamp": now.isoformat(),
                "lang": lang,
                "scene": scene,
                "message": message,
            }
            with open(base + ".json", "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            logger.warning("Failed to log scene request", exc_info=True)
