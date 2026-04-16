import threading
import time
from collections import deque


class SceneState:
    """Thread-safe shared scene data with rolling consensus voting."""

    def __init__(self, consensus_window: int = 5):
        self._lock = threading.Lock()
        self._history: deque[dict[str, str]] = deque(maxlen=consensus_window)
        self._scene: dict[str, str] = {}
        self._last_updated: float = 0.0

    def update(self, raw_scene: dict[str, str]) -> None:
        with self._lock:
            self._history.append(raw_scene)
            self._scene = self._compute_consensus()
            self._last_updated = time.monotonic()

    def get_scene(self) -> dict[str, str]:
        with self._lock:
            return dict(self._scene)

    def _compute_consensus(self) -> dict[str, str]:
        votes: dict[str, dict[str, int]] = {}
        for frame in self._history:
            for pos, candy in frame.items():
                if pos not in votes:
                    votes[pos] = {}
                votes[pos][candy] = votes[pos].get(candy, 0) + 1
        return {
            pos: max(counts, key=lambda k: counts[k])
            for pos, counts in sorted(votes.items())
        }
