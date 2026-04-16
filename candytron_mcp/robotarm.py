import logging
import ned2
import threading

logger = logging.getLogger("candytron.robot")


class SingleJobWorker:
    def __init__(self, target_func):
        self.target_func = target_func
        self.job_ready = threading.Event()
        self.job_done = threading.Event()
        self.lock = threading.Lock()
        self.args = None
        # Initially no job is running
        self.job_done.set()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def _worker(self):
        while True:
            self.job_ready.wait()
            self.job_ready.clear()

            if self.args is not None:
                self.target_func(self.args)
                self.args = None

            self.job_done.set()

    def run(self, args: dict) -> None:
        with self.lock:
            if not self.job_done.is_set():
                self.job_done.wait()

            self.job_done.clear()
            self.args = args
            self.job_ready.set()

    def wait_until_idle(self):
        """Block until no job is running."""
        self.job_done.wait()


ned: ned2.Ned2 | None = None
_ned_worker: SingleJobWorker | None = None


def _async_ned_call(args: dict) -> None:
    if not ned:
        return
    if not 'op' in args:
        logger.error("Unknown command: %s", args)
        return
    op = args['op']
    if op == 'home':
        ned.move_to_home_pose()
    elif op == 'move':
        src = args['from']
        dst = args['to']
        try:
            ned.pick_and_place(src, dst)
        except Exception as e:
            logger.error("Failed to move: %s", e)
            if ned.collision_detected:
                ned.clear_collision_detected()
                args['next'] = 'home'
        if 'next' in args and args['next'] == 'home':
            ned.move_to_home_pose()
    else:
        logger.error("Unknown op: %s", op)

def ned_move_home():
    if _ned_worker:
        _ned_worker.run({'op':'home'})

def ned_move_between(src: str, dst: str) -> bool:
    if _ned_worker:
        pos1 = ned.get_pose(src)
        pos2 = ned.get_pose(dst)
        if not pos1 or not pos2:
            return False
        command = {'op':'move','from':pos1,'to':pos2}
        command['next'] = 'home'
        _ned_worker.run(command)
    return True

def init_ned(use_robot=True):
    global ned, _ned_worker
    if not ned:
        ned = ned2.Ned2()
        if use_robot:
            ned.open()
            # Set max gripper hold torque
            ned.set_hold_torque(100)
            _ned_worker = SingleJobWorker(_async_ned_call)
    ned_move_home()

def exit_ned():
    if _ned_worker:
        _ned_worker.wait_until_idle()
        ned.close()
