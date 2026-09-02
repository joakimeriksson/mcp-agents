#----------------

from fastmcp import FastMCP
import argparse
import os
import atexit
import logging
import random
import sys
import threading

from camera import CameraManager
from robotarm import (init_ned, exit_ned, ned_move_between,
                      ned_is_busy, ned_known_positions)
from transtable import transhead, transtable
from scene_state import SceneState
from scene_logger import SceneLogger

logger = logging.getLogger("candytron")

mcp = FastMCP("Candytron 4000")

cam: CameraManager | None = None
scene_state = SceneState()
scene_logger: SceneLogger | None = None
use_robot = True
robot_ip = None  # None -> ned2.DEFAULT_ROBOT_IP

def ned2_default_ip():
    import ned2
    return ned2.DEFAULT_ROBOT_IP

@mcp.resource("url://get_service_name")
def get_service_name() -> str:
    """Return the name of the provided service"""
    return mcp.name

@mcp.resource("url://service_init")
def service_init() -> bool:
    """Initialize the service. Is called before the first tool call from the client."""
    init_ned(use_robot=use_robot, robot_ip=robot_ip)
    logger.info("Initialized robot arm: Niryo Ned 2%s", " (simulated)" if not use_robot
                else f" at {robot_ip or ned2_default_ip()}")
    return True

@mcp.resource("url://service_exit")
def service_exit() -> bool:
    """Clean up after the service. Is called when the current client is shutting down."""
    exit_ned()
    logger.info("Exiting ned2")
    return True


@mcp.prompt()
def get_service_prompt(lang: str) -> str:
    """Return the system message snippet suitable for this service."""
    # The robot's real name, in every language. (There used to be per-language
    # phonetic respellings here -- "Kandutron", "Candue Tronne" -- for the old
    # Piper voices; Kokoro has real g2p, so they only made it introduce itself
    # by the wrong name.)
    name = mcp.name
    return f"Your name is {name}. You are situated at an exhibition to demonstrate how several AI systems can be connected, such as speech recognition, a large language model, speech synthesis, computer vision, and a robot arm. You are this system. Specifically, you have a robot arm, which allows you to move different types of candy between different positions on a table. You can chat with the visitors, and they may ask about your demonstration. They may also ask you to move candy around on the table or to give them some specific candy. When you know what specific candy on the table the user wants (but not before), you hand it out to them by moving it to the special position O0. Information on the latest positions of candy and their characteristics will be regularly provided by the vision system, for you to internally look up information needed to answer questions or perform moves. However, you never give this type of lists directly to the user. Your replies are friendly, concise and as plain text with no formatting."

def scene_message(scene, lang):
    if not lang in transhead:
        lang = 'en'
    content = transhead[lang]
    if scene:
        for k in scene:
            if scene[k] in transtable:
                obj,cha = transtable[scene[k]][lang]
                content += "\n" + k + " : " + obj + ", " + cha + "."
    else:
        content += "No candies observed"
    return content

@mcp.prompt()
def get_service_augmentation(lang: str) -> str:
    """Return extra information on the current state, to insert before the user prompt"""
    scene = scene_state.get_scene()
    message = scene_message(scene, lang)
    if scene_logger is not None:
        frame = cam.get_last_frame() if cam else None
        raw_frame = cam.get_last_raw_frame() if cam else None
        scene_logger.log(scene, frame, raw_frame, lang, message)
    return message

@mcp.tool()
def show_demo_move() -> str:
    """Show off the arm by moving ONE RANDOM candy to a random free position.

    Only call this when the person explicitly asks to SEE a demonstration of
    the robot arm ("show me what you can do", "visa vad du kan"). It picks the
    candy and the destination at RANDOM and hands nothing to the person, so it
    must never be used to fetch, give or move a specific candy — use
    move_between for that.
    """
    scene = scene_state.get_scene()
    scenepos = list(scene.keys())
    emptypos = [k for k in cam.camera_positions() if k not in scenepos] if cam else []
    if len(scenepos) and len(emptypos):
        p1 = random.choice(scenepos)
        p2 = random.choice(emptypos)
        if ned_move_between(p1, p2):
            return "Successfully demonstrated a move with the robot arm from " + p1 + " to " + p2
        else:
            return "Failed to move"
    elif not len(emptypos):
        return "No empty positions"
    else:
        return "No candies observed"

@mcp.tool()
def move_between(src: str, dst: str) -> str:
    """Move an object from one position to another position, using the robot arm. The argument 'src' is the current position of the object. The argument 'dst' is the destination position of the object."""
    refusal = _refuse_move(src, dst)
    if refusal:
        logger.warning("Refused move %r -> %r: %s", src, dst, refusal)
        return refusal
    src, dst = src.strip().upper(), dst.strip().upper()
    if ned_move_between(src, dst):
        # The worker runs the move asynchronously, so this is "started", not
        # "finished" -- saying otherwise makes the robot announce a success it
        # cannot yet know about (and that a collision may still cancel).
        return f"Started moving the candy from {src} to {dst}; the arm is moving now."
    return "Failed to move"


# Positions the arm is calibrated for, as a last-resort allow-list for when the
# robot object isn't up yet (simulation, or before service_init).
_FALLBACK_POSITIONS = {f"{r}{c}" for r in "ABCD" for c in "123"} | {"O0"}


def _refuse_move(src: str, dst: str) -> str | None:
    """Reason to refuse this move, or None to allow it.

    This is the guard at the hardware boundary: every client goes through it,
    not just the one that happens to validate on its own side.
    """
    if not isinstance(src, str) or not isinstance(dst, str) or not src.strip() or not dst.strip():
        return "Refused: src and dst must both be position names, e.g. 'D2' and 'O0'."
    src_n, dst_n = src.strip().upper(), dst.strip().upper()

    known = {p.upper() for p in ned_known_positions()} or _FALLBACK_POSITIONS
    movable = sorted(p for p in known if p != "HOME")
    for label, pos in (("src", src_n), ("dst", dst_n)):
        if pos not in known or pos == "HOME":
            # Also catches coordinate strings: get_pose() would otherwise parse
            # "[0.3, 0.1, ...]" and drive the arm to an arbitrary point.
            return (f"Refused: {label}={pos!r} is not a position on the table. "
                    f"Valid positions are: {', '.join(movable)}.")
    if src_n == dst_n:
        return f"Refused: src and dst are both {src_n}; nothing to do."

    scene = scene_state.get_scene()
    if scene:                      # only trust these when the vision system sees something
        if src_n not in scene:
            occupied = ", ".join(f"{p} ({c})" for p, c in sorted(scene.items()))
            return (f"Refused: there is no candy at {src_n}, so the arm would grip "
                    f"nothing. Candy is at: {occupied}.")
        if dst_n in scene:
            return (f"Refused: {dst_n} already holds {scene[dst_n]}; moving there would "
                    f"drop one candy on top of another. Pick a free position.")
    if ned_is_busy():
        return "Refused: the arm is still finishing the previous move. Try again in a moment."
    return None

@mcp.tool()
def default_action() -> str:
    """Do nothing at all. The robot arm does NOT move.

    Only for a request that needs no physical action. If the person wants a
    candy moved, fetched or handed over, call move_between instead — calling
    this one leaves them empty-handed.
    """
    return "Did nothing: no physical action was taken, the arm did not move."


def _run_mcp_server(args):
    """Run the MCP server. Called in a daemon thread."""
    try:
        if args.transport != "stdio":
            mcp.run(transport=args.transport, host=args.host, port=args.port)
        else:
            mcp.run()
    except Exception:
        logger.exception("MCP server error")


def main():
    global use_robot, robot_ip, cam, scene_logger

    parser = argparse.ArgumentParser(description=mcp.name)
    parser.add_argument('--host', default="127.0.0.1", help='Host to bind to')
    parser.add_argument('--port', default=8000, type=int, help='Port to bind to')
    parser.add_argument('--transport', default="sse", help='Transport to use (stdio, sse or http)')
    parser.add_argument('--simulate-robot', action='store_true', help='Simulate the robot arm instead of using real hardware')
    parser.add_argument('--robot-ip', default=os.environ.get('NIRYO_IP'),
                        help='IP of the Niryo Ned 2 (default: $NIRYO_IP, else 10.10.10.10)')
    parser.add_argument('--simulate-camera', action='store_true', help='Simulate the camera instead of using real hardware')
    parser.add_argument('-l', '--list-cameras', action='store_true', help='List available cameras and exit')
    parser.add_argument('--camera', default=None,
                        help="Table-camera index, or part of its name (e.g. 'brio', 'hp'). "
                             "Names survive replugging; indices do not. Default: auto-detect.")
    parser.add_argument('--no-window', action='store_true', help='Disable the OpenCV display window')
    parser.add_argument('--log-scenes-dir', default=None, help='If set, log every scene request (annotated camera frame + returned state) to this directory')
    parser.add_argument('-v', '--verbose', action='count', default=0, help='Increase verbosity (-v for INFO, -vv for DEBUG)')
    args = parser.parse_args()

    # Configure logging
    log_level = logging.WARNING
    if args.verbose >= 2:
        log_level = logging.DEBUG
    elif args.verbose >= 1:
        log_level = logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Enable scene-request logging if requested
    if args.log_scenes_dir:
        scene_logger = SceneLogger(args.log_scenes_dir)
        logger.info("Scene-request logging enabled at %s", args.log_scenes_dir)

    # List cameras and exit
    if args.list_cameras:
        cameras = CameraManager.list_cameras()
        if cameras:
            print("Available cameras:")
            for c in cameras:
                print(f"  Index {c['index']}: {c['width']}x{c['height']} @ {c['fps']:.1f} fps ({c['backend']})")
        else:
            print("No cameras found")
        sys.exit(0)

    use_robot = not args.simulate_robot
    robot_ip = args.robot_ip
    simulate_camera = args.simulate_camera

    # Resolve camera index
    camera_index = args.camera
    if camera_index is not None and not str(camera_index).lstrip("-").isdigit():
        # A name: resolve it to an index that actually delivers video, so an
        # idle Continuity Camera can't be picked for the candy table.
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        '..', 'face'))
        from camera_utils import resolve_camera
        cap, idx, cam_name = resolve_camera(camera_index)
        if cap is None:
            logger.error("No camera matching %r", camera_index)
            sys.exit(1)
        cap.release()
        logger.info("Table camera %d (%s) for %r", idx, cam_name, camera_index)
        camera_index = idx
    elif camera_index is not None:
        camera_index = int(camera_index)
    if camera_index is None and not simulate_camera:
        camera_index = CameraManager.find_first_camera()
        if camera_index is not None:
            print(f"Auto-selected camera at index {camera_index}")
        else:
            logger.error("No cameras found. Use --simulate-camera or --camera N.")
            sys.exit(1)
    if camera_index is None:
        camera_index = 0  # fallback for simulation mode

    # Initialize camera
    cam = CameraManager(
        camera_index=camera_index,
        show_window=not args.no_window,
        simulate=simulate_camera,
    )
    try:
        cam.init_cam()
    except RuntimeError as e:
        logger.error("Camera initialization failed: %s", e)
        sys.exit(1)
    logger.info("Initialized camera and YOLO model%s", " (simulated)" if simulate_camera else "")

    # Calibration loop — retries until success or user presses 'q'
    attempt = 0
    while True:
        if cam.calibrate_positions(3, 4):
            if attempt > 0:
                print()  # newline after dots
            break
        attempt += 1
        if attempt % 50 == 0:
            print(f"\nCalibrating... ({attempt} attempts). Place 4 candies in the corners.")
        elif attempt % 10 == 0:
            print(".", end="", flush=True)
        if cam.check_event(wait_ms=200):
            logger.info("User cancelled calibration")
            cam.exit_cam()
            sys.exit(0)

    # Register cleanup for robot shutdown
    atexit.register(exit_ned)

    # Start MCP server in background thread
    mcp_thread = threading.Thread(target=_run_mcp_server, args=(args,), daemon=True)
    mcp_thread.start()
    logger.info("MCP server started on %s:%d (%s)", args.host, args.port, args.transport)

    # Main thread: continuous camera refresh loop (~5 fps)
    logger.info("Starting continuous camera refresh loop")
    try:
        while True:
            scene = cam.grab_and_detect()
            scene_state.update(scene)
            if cam.check_event(wait_ms=200):
                logger.info("User requested exit via 'q' key")
                break
    except KeyboardInterrupt:
        logger.info("Shutting down (KeyboardInterrupt)")
    finally:
        cam.exit_cam()
        exit_ned()
        logger.info("Cleanup complete")


if __name__ == "__main__":
    main()
