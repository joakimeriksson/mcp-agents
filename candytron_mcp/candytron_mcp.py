#----------------

from fastmcp import FastMCP
import argparse
import atexit
import logging
import random
import sys
import threading

from camera import CameraManager
from robotarm import init_ned, exit_ned, ned_move_between
from transtable import transhead, transtable
from scene_state import SceneState

logger = logging.getLogger("candytron")

mcp = FastMCP("Candytron 4000")

cam: CameraManager | None = None
scene_state = SceneState()
use_robot = True

@mcp.resource("url://get_service_name")
def get_service_name() -> str:
    """Return the name of the provided service"""
    return mcp.name

@mcp.resource("url://service_init")
def service_init() -> bool:
    """Initialize the service. Is called before the first tool call from the client."""
    init_ned(use_robot=use_robot)
    logger.info("Initialized robot arm: Niryo Ned 2%s", " (simulated)" if not use_robot else "")
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
    names = { "en": "Candy Tron",
              "sv": "Kandutron",
              "de": "Candy Tronn",
              "fr": "Candue Tronne",
              "es": "Candy Tron"}
    if not lang in names:
        lang = 'en'
    name = names[lang]
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
    return scene_message(scene, lang)

@mcp.tool()
def show_demo_move() -> str:
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
    if ned_move_between(src, dst):
        return f"Successfully moved from {src} to {dst}"
    else:
        return "Failed to move"

@mcp.tool()
def default_action() -> str:
    """This function can be called whenever there is no obvious other function to call."""
    return "Successfully did nothing"


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
    global use_robot, cam

    parser = argparse.ArgumentParser(description=mcp.name)
    parser.add_argument('--host', default="127.0.0.1", help='Host to bind to')
    parser.add_argument('--port', default=8000, type=int, help='Port to bind to')
    parser.add_argument('--transport', default="sse", help='Transport to use (stdio, sse or http)')
    parser.add_argument('--simulate-robot', action='store_true', help='Simulate the robot arm instead of using real hardware')
    parser.add_argument('--simulate-camera', action='store_true', help='Simulate the camera instead of using real hardware')
    parser.add_argument('-l', '--list-cameras', action='store_true', help='List available cameras and exit')
    parser.add_argument('--camera', type=int, default=None, help='Camera index to use (default: auto-detect first available)')
    parser.add_argument('--no-window', action='store_true', help='Disable the OpenCV display window')
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
    simulate_camera = args.simulate_camera

    # Resolve camera index
    camera_index = args.camera
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
