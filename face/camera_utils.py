"""Camera selection helpers — pick a capture device that delivers real video.

On macOS the Continuity Camera (a nearby iPhone) is often enumerated *ahead* of
the built-in webcam, so OpenCV index 0 grabs the iPhone. When the iPhone isn't
actively presenting it streams all-black frames, which looks like a broken
camera. This probes candidate indices and selects the first that returns a
non-black frame, preferring an explicitly requested index when given.
"""

import logging
import re
import shutil
import subprocess
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


# ---------------------------------------------------------------------------
# Selecting a camera by NAME
# ---------------------------------------------------------------------------
#
# Indices are not stable on macOS: plugging in a USB webcam, or an iPhone
# waking up as a Continuity Camera, renumbers everything, so "--camera 2" means
# a different device from one day to the next. Names don't move, so a device
# can be asked for as "--camera macbook" / "--camera brio" instead.

_AVF_DEVICE = re.compile(r"^\[AVFoundation[^\]]*\]\s*\[(\d+)\]\s+(.*\S)\s*$")


def list_named_cameras() -> list:
    """[(index, name)] of AVFoundation video devices (macOS, via ffmpeg).

    Returns [] when the list can't be obtained (no ffmpeg, another OS) — the
    caller then falls back to plain index handling.
    """
    if not shutil.which("ffmpeg"):
        return []
    try:
        out = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True, text=True, timeout=10).stderr
    except Exception as e:                       # pragma: no cover
        logger.debug(f"camera name listing failed: {e}")
        return []
    cams, in_video = [], False
    for line in out.splitlines():
        if "AVFoundation video devices" in line:
            in_video = True
            continue
        if "AVFoundation audio devices" in line:
            break
        m = _AVF_DEVICE.match(line.strip()) if in_video else None
        if m:
            cams.append((int(m.group(1)), m.group(2)))
    return cams


def camera_name(index: int) -> str:
    """Human name for an index, or '' if unknown."""
    return next((n for i, n in list_named_cameras() if i == index), "")


def resolve_camera(spec, max_index: int = 6):
    """Turn a camera spec into a working (cap, index, name).

    *spec* may be None (auto), an int or digit string (an index), or a piece of
    a device name ("macbook", "brio"). Named matches are tried in order and
    each is verified to deliver real video, so a match that turns out to be an
    idle Continuity Camera falls through to the next candidate rather than
    handing back a black picture.
    """
    named = list_named_cameras()

    def _named(i):
        return next((n for j, n in named if j == i), "")

    order = []
    if isinstance(spec, str) and spec.strip() and not spec.strip().lstrip("-").isdigit():
        want = spec.strip().lower()
        order = [i for i, n in named if want in n.lower()]
        if not order:
            logger.warning(f"No camera matching {spec!r}; known: "
                           f"{[n for _i, n in named] or 'unknown'}")
    elif spec is not None and str(spec).strip() != "":
        try:
            idx = int(spec)
            if idx >= 0:
                order = [idx]
        except (TypeError, ValueError):
            pass

    order += [i for i in range(max_index) if i not in order]
    for idx in order:
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        if _delivers_image(cap):
            name = _named(idx)
            logger.info(f"Using camera {idx}{f' ({name})' if name else ''}")
            return cap, idx, name
        logger.warning(f"Camera {idx}{f' ({_named(idx)})' if _named(idx) else ''} "
                       f"opened but is black — skipping")
        cap.release()
    logger.error("No working camera found")
    return None, -1, ""
