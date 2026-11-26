"""Discourse Consistency Engine - Master Clock and Policy Queue.

Provides timing synchronization for both offline (media PTS) and live
(monotonic clock with A/V offset) modes, alongside thread-safe bounded
queues enforcing overflow policies (block, drop_oldest, never_drop).
"""

from __future__ import annotations

import collections
import queue
import threading
import time
from enum import Enum
from typing import Any, Deque, Optional


from dce.schemas import ClockMode


class QueuePolicy(str, Enum):
    """Queue overflow management policy."""

    BLOCK = "block"
    DROP_OLDEST = "drop_oldest"
    NEVER_DROP = "never_drop"


class MasterClock:
    """Master timing coordinator for the Discourse Consistency Engine.

    Supports deterministic offline replay mode (tracking media PTS nanoseconds)
    and live streaming mode (tracking host monotonic time plus A/V offset).
    """

    def __init__(self, mode: str | ClockMode = ClockMode.OFFLINE, av_offset_ns: int = 0) -> None:
        """Initialize MasterClock.

        Args:
            mode: Clock mode ('offline' or 'live').
            av_offset_ns: A/V offset in nanoseconds added in live mode.
        """
        if isinstance(mode, ClockMode):
            normalized_mode = mode
        else:
            normalized_mode = ClockMode(str(mode).lower())
        self._mode: ClockMode = normalized_mode
        self._av_offset_ns: int = int(av_offset_ns)
        self._pts_ns: int = 0
        self._lock: threading.Lock = threading.Lock()

    @property
    def mode(self) -> ClockMode:
        """Return current clock mode."""
        return self._mode

    @property
    def av_offset_ns(self) -> int:
        """Return current A/V offset in nanoseconds."""
        with self._lock:
            return self._av_offset_ns

    def set_av_offset_ns(self, offset_ns: int) -> None:
        """Set A/V offset in nanoseconds."""
        with self._lock:
            self._av_offset_ns = int(offset_ns)

    def advance_to_pts(self, pts_ns: int) -> int:
        """Advance the media PTS in offline mode.

        Args:
            pts_ns: Target presentation timestamp in nanoseconds.

        Returns:
            Updated PTS in nanoseconds.
        """
        with self._lock:
            if pts_ns > self._pts_ns:
                self._pts_ns = int(pts_ns)
            return self._pts_ns

    def get_pts_ns(self) -> int:
        """Return the current media presentation timestamp in nanoseconds."""
        with self._lock:
            if self._mode == ClockMode.LIVE:
                return time.monotonic_ns() + self._av_offset_ns
            return self._pts_ns

    def get_time_ns(self) -> int:
        """Return the current engine time in nanoseconds based on mode."""
        if self._mode == ClockMode.LIVE:
            with self._lock:
                offset = self._av_offset_ns
            return time.monotonic_ns() + offset
        with self._lock:
            return self._pts_ns

    def get_time_sec(self) -> float:
        """Return current engine time in floating-point seconds."""
        return float(self.get_time_ns()) / 1_000_000_000.0

    def reset(self, start_pts_ns: int = 0) -> None:
        """Reset internal media timestamp.

        Args:
            start_pts_ns: Starting PTS value in nanoseconds.
        """
        with self._lock:
            self._pts_ns = int(start_pts_ns)


class PolicyQueue:
    """Thread-safe bounded queue enforcing configurable overflow policies.

    Policies:
      - 'block': Put operations block until space is available or timeout expires.
      - 'drop_oldest': If full, drops the oldest item and enqueues the new item without blocking.
      - 'never_drop': Always appends items, expanding queue capacity without dropping.
    """

    def __init__(self, max_size: int = 128, policy: str | QueuePolicy = QueuePolicy.BLOCK) -> None:
        """Initialize PolicyQueue.

        Args:
            max_size: Maximum capacity for bounded policies.
            policy: Overflow policy ('block', 'drop_oldest', or 'never_drop').
        """
        if isinstance(policy, QueuePolicy):
            normalized_policy = policy
        else:
            normalized_policy = QueuePolicy(str(policy).lower())
        self._max_size: int = max(0, int(max_size))
        self._policy: QueuePolicy = normalized_policy
        self._items: Deque[Any] = collections.deque()
        self._lock: threading.Lock = threading.Lock()
        self._cond: threading.Condition = threading.Condition(self._lock)
        self._dropped_count: int = 0

    @property
    def max_size(self) -> int:
        """Configured maximum capacity."""
        return self._max_size

    @property
    def policy(self) -> QueuePolicy:
        """Active overflow policy."""
        return self._policy

    @property
    def dropped_count(self) -> int:
        """Total count of items dropped by drop_oldest policy."""
        with self._lock:
            return self._dropped_count

    def put(self, item: Any, block: bool = True, timeout: Optional[float] = None) -> bool:
        """Enqueue an item according to the queue policy.

        Args:
            item: Item to put.
            block: Whether to block if policy is 'block' and queue is full.
            timeout: Maximum wait time in seconds when blocking.

        Returns:
            True if item was successfully enqueued.

        Raises:
            queue.Full: If policy is 'block' and queue remains full after timeout.
        """
        with self._cond:
            if self._policy == QueuePolicy.BLOCK:
                if self._max_size > 0:
                    if not block:
                        if len(self._items) >= self._max_size:
                            raise queue.Full("Queue is full")
                    else:
                        if timeout is not None and timeout < 0:
                            raise ValueError("'timeout' must be a non-negative number")
                        endtime = time.monotonic() + timeout if timeout is not None else None
                        while len(self._items) >= self._max_size:
                            if timeout is not None:
                                remaining = endtime - time.monotonic()
                                if remaining <= 0.0:
                                    raise queue.Full("Queue is full, timeout expired")
                                self._cond.wait(remaining)
                            else:
                                self._cond.wait()
                self._items.append(item)
                self._cond.notify()
                return True

            if self._policy == QueuePolicy.DROP_OLDEST:
                if self._max_size > 0 and len(self._items) >= self._max_size:
                    self._items.popleft()
                    self._dropped_count += 1
                self._items.append(item)
                self._cond.notify()
                return True

            if self._policy == QueuePolicy.NEVER_DROP:
                self._items.append(item)
                self._cond.notify()
                return True

            raise ValueError(f"Unknown queue policy: {self._policy}")

    def put_nowait(self, item: Any) -> bool:
        """Enqueue an item without blocking."""
        return self.put(item, block=False)

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Any:
        """Dequeue an item.

        Args:
            block: Whether to wait until an item is available.
            timeout: Maximum wait time in seconds.

        Returns:
            Dequeued item.

        Raises:
            queue.Empty: If queue is empty and block is False or timeout expired.
        """
        with self._cond:
            if not block:
                if not self._items:
                    raise queue.Empty("Queue is empty")
                item = self._items.popleft()
                self._cond.notify()
                return item

            if timeout is not None and timeout < 0:
                raise ValueError("'timeout' must be a non-negative number")
            endtime = time.monotonic() + timeout if timeout is not None else None
            while not self._items:
                if timeout is not None:
                    remaining = endtime - time.monotonic()
                    if remaining <= 0.0:
                        raise queue.Empty("Queue is empty, timeout expired")
                    self._cond.wait(remaining)
                else:
                    self._cond.wait()
            item = self._items.popleft()
            self._cond.notify()
            return item

    def get_nowait(self) -> Any:
        """Dequeue an item without blocking."""
        return self.get(block=False)

    def qsize(self) -> int:
        """Return current number of items in the queue."""
        with self._lock:
            return len(self._items)

    def empty(self) -> bool:
        """Return True if queue is empty."""
        with self._lock:
            return len(self._items) == 0

    def full(self) -> bool:
        """Return True if queue is at capacity (never full for never_drop)."""
        with self._lock:
            if self._policy == QueuePolicy.NEVER_DROP or self._max_size <= 0:
                return False
            return len(self._items) >= self._max_size

    def clear(self) -> int:
        """Clear all items from queue and return number of items cleared."""
        with self._cond:
            count = len(self._items)
            self._items.clear()
            self._cond.notify_all()
            return count
