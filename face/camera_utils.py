"""Camera selection helpers — pick a capture device that delivers real video.

On macOS the Continuity Camera (a nearby iPhone) is often enumerated *ahead* of
the built-in webcam, so OpenCV index 0 grabs the iPhone. When the iPhone isn't
actively presenting it streams all-black frames, which looks like a broken
camera. This probes candidate indices and selects the first that returns a
non-black frame, preferring an explicitly requested index when given.
"""

import logging
import time

import cv2

logger = logging.getLogger("camera")

BLACK_MEAN_THRESHOLD = 3.0   # mean pixel value below this == effectively black


def _delivers_image(cap, warmup_frames: int = 12,
                    black_thresh: float = BLACK_MEAN_THRESHOLD) -> bool:
    """True if the open capture yields at least one non-black frame.

    The first frames after opening are often black even on a good camera while
    AVFoundation warms up, so we read several before giving up.
    """
    for _ in range(warmup_frames):
        ret, frame = cap.read()
        if ret and frame is not None and float(frame.mean()) >= black_thresh:
            return True
        time.sleep(0.05)
    return False


def open_camera(preferred: int = -1, max_index: int = 4):
    """Open the first camera that delivers real video.

    If ``preferred`` >= 0 it is tried first; otherwise (or if it is black/fails)
    indices ``0..max_index-1`` are scanned. Returns ``(cap, index)`` for the
    chosen device, or ``(None, -1)`` if none deliver an image.
    """
    order = []
    if preferred is not None and preferred >= 0:
        order.append(preferred)
    order += [i for i in range(max_index) if i != preferred]

    skipped = []
    for idx in order:
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        if _delivers_image(cap):
            logger.info(f"Using camera index {idx} (delivers video)")
            return cap, idx
        logger.warning(f"Camera index {idx} opened but only black frames "
                       f"(likely an idle Continuity Camera) — skipping")
        skipped.append(idx)
        cap.release()

    logger.error(f"No working camera found (black/unavailable: {skipped or 'none'})")
    return None, -1
