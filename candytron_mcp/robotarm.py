import logging
import queue
import threading
import time

import ned2

logger = logging.getLogger("candytron.robot")

# Moves that may be waiting or running at once. "A Dumle and a Marianne" is two
# moves in one turn; the arm used to refuse the second one while busy, and the
# robot then said "here is your Marianne" with nothing happening.
MAX_QUEUED_MOVES = 3
# A finished move still counts for this long: the vision system needs about a
# second (consensus over 5 frames) before the board shows the candy's new place.
RECENT_MOVE_SECONDS = 3.0
# Simulated robot: how long a pretend move takes, so queueing behaves as on
# the real arm.
SIM_MOVE_SECONDS = 4.0


class MoveQueue:
    """Runs arm jobs one at a time, in order.

    ``pending()`` lists the moves that are queued or running (the running one
    first) plus the ones that just finished, so the server can check a new
    move against the board as it WILL be, not as the camera sees it now.
    """

    def __init__(self, run_job):
        self._run_job = run_job
        self._q: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._moves: list = []      # [(src, dst)] queued or running, in order
        self._recent: list = []     # [(src, dst, finished_at)]
        self._outstanding = 0       # jobs submitted but not finished
        self._idle = threading.Event()
        self._idle.set()
        threading.Thread(target=self._worker, daemon=True, name="arm-queue").start()

    def submit(self, job: dict, move: tuple | None = None) -> int | None:
        """Queue *job*. For a move, returns how many moves are ahead of it
        (0 = it starts now); None when the queue is full."""
        with self._lock:
            if move is not None and len(self._moves) >= MAX_QUEUED_MOVES:
                return None
            ahead = len(self._moves)
            if move is not None:
                self._moves.append(move)
            self._outstanding += 1
            self._idle.clear()
        self._q.put((job, move))
        return ahead

    def pending(self) -> list:
        now = time.monotonic()
        with self._lock:
            self._recent = [r for r in self._recent if now - r[2] < RECENT_MOVE_SECONDS]
            return [(s, d) for s, d, _ in self._recent] + list(self._moves)

    def moves_waiting(self) -> int:
        """Moves queued or running (not counting finished ones)."""
        with self._lock:
            return len(self._moves)

    def cancel_waiting(self) -> int:
        """Drop every move that has not started (e.g. after a collision: the
        board is no longer what they were planned against)."""
        dropped = 0
        while True:
            try:
                _job, move = self._q.get_nowait()
            except queue.Empty:
                break
            dropped += 1
            with self._lock:
                if move in self._moves:
                    self._moves.remove(move)
                self._outstanding -= 1
                if self._outstanding == 0:
                    self._idle.set()
        return dropped

    def wait_until_idle(self, timeout: float | None = None) -> bool:
        return self._idle.wait(timeout)

    def is_busy(self) -> bool:
        return not self._idle.is_set()

    def _worker(self):
        while True:
            job, move = self._q.get()
            try:
                self._run_job(job)
            except Exception:
                logger.exception("arm job %s failed", job.get("op"))
            finally:
                with self._lock:
                    if move is not None and move in self._moves:
                        self._moves.remove(move)
                        self._recent.append((move[0], move[1], time.monotonic()))
                    self._outstanding -= 1
                    if self._outstanding == 0:
                        self._idle.set()


ned: ned2.Ned2 | None = None
_ned_worker: MoveQueue | None = None
_simulated = False


def _async_ned_call(args: dict) -> None:
    if not ned:
        return
    op = args.get('op')
    if op == 'home':
        ned.move_to_home_pose()
    elif op == 'move':
        src, dst = args['from'], args['to']
        logger.info("Arm: moving %s -> %s", args.get('src_name'), args.get('dst_name'))
        if _simulated:
            time.sleep(SIM_MOVE_SECONDS)
            return
        try:
            ned.pick_and_place(src, dst)
        except Exception as e:
            logger.error("Failed to move: %s", e)
            if ned.collision_detected:
                ned.clear_collision_detected()   # also goes home
            dropped = _ned_worker.cancel_waiting() if _ned_worker else 0
            if dropped:
                logger.warning("Arm: dropped %d queued move(s) after the failure", dropped)
            ned.move_to_home_pose()
            return
        # Home only when nothing else is waiting: saves a trip per candy.
        if _ned_worker is None or _ned_worker.moves_waiting() <= 1:
            ned.move_to_home_pose()
    else:
        logger.error("Unknown op: %s", op)


def ned_move_home():
    if _ned_worker:
        _ned_worker.submit({'op': 'home'})


def ned_is_busy() -> bool:
    """True while the arm has a move queued or running."""
    return bool(_ned_worker) and _ned_worker.moves_waiting() > 0


def ned_pending_moves() -> list:
    """[(src, dst)] position names of moves queued, running, or just finished
    (see RECENT_MOVE_SECONDS) -- to check a new move against the board as it
    will be."""
    return _ned_worker.pending() if _ned_worker else []


def ned_moves_waiting() -> int:
    return _ned_worker.moves_waiting() if _ned_worker else 0


def ned_known_positions() -> set:
    """Names the arm has a calibrated pose for. Anything else must be refused:
    get_pose() falls back to parsing its argument as raw coordinates, so an
    unchecked name is a path from model output to an arbitrary arm position."""
    if not ned:
        return set()
    try:
        return set(ned.get_poses().keys())
    except Exception:
        return set()


def ned_move_between(src: str, dst: str) -> int | None:
    """Queue a move. Returns how many moves are ahead of it (0 = starts now),
    or None if it could not be queued (unknown position or queue full)."""
    if not _ned_worker:
        return 0
    pos1 = ned.get_pose(src)
    pos2 = ned.get_pose(dst)
    if not pos1 or not pos2:
        return None
    return _ned_worker.submit({'op': 'move', 'from': pos1, 'to': pos2,
                               'src_name': src, 'dst_name': dst},
                              move=(src, dst))


def init_ned(use_robot=True, robot_ip=None):
    global ned, _ned_worker, _simulated
    if not ned:
        ned = ned2.Ned2(robot_ip) if robot_ip else ned2.Ned2()
        _simulated = not use_robot
        if use_robot:
            ned.open()
            # Set max gripper hold torque
            ned.set_hold_torque(100)
        # The simulated arm queues too (moves take SIM_MOVE_SECONDS), so the
        # queue and "busy" behaviour can be tested without hardware.
        _ned_worker = MoveQueue(_async_ned_call)
    ned_move_home()


def exit_ned():
    if _ned_worker:
        if not _ned_worker.wait_until_idle(timeout=90):
            logger.warning("Arm still busy after 90 s; closing anyway")
        if not _simulated:
            ned.close()
