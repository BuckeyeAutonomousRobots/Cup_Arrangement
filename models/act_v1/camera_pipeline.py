"""ROS-independent bounded worker and latest-observation primitives."""
import queue
import threading
import time


class BoundedWorker:
    """Never wait for disk in submit(); overflow/errors are explicit failures."""
    def __init__(self, handle, capacity, name):
        self.handle = handle
        self.queue = queue.Queue(maxsize=capacity)
        self.error = None
        self.closed = False
        self.submitted = self.completed = self.high_water = 0
        self.queue_wait_max_s = self.work_max_s = 0.
        self.thread = threading.Thread(target=self._run, name=name, daemon=True)
        self.thread.start()

    def check(self):
        if self.error is not None:
            raise RuntimeError('Recording worker failed') from self.error

    def submit(self, item):
        self.check()
        if self.closed:
            raise RuntimeError('Recording worker is closed')
        try:
            self.queue.put_nowait((time.monotonic(), item))
        except queue.Full as exc:
            raise RuntimeError('Recording queue full; refusing silent data loss') from exc
        self.submitted += 1
        self.high_water = max(self.high_water, self.queue.qsize())

    def _run(self):
        while True:
            entry = self.queue.get()
            try:
                if entry is None:
                    return
                enqueued, item = entry
                start = time.monotonic()
                self.queue_wait_max_s = max(self.queue_wait_max_s, start-enqueued)
                self.handle(item, enqueued, start)
                self.work_max_s = max(self.work_max_s, time.monotonic()-start)
                self.completed += 1
            except BaseException as exc:
                self.error = exc
                return
            finally:
                self.queue.task_done()

    def close(self, timeout=15):
        if self.closed:
            self.check()
            return
        self.closed = True
        deadline = time.monotonic()+timeout
        while self.thread.is_alive():
            self.check()
            try:
                self.queue.put(None, timeout=min(.05, max(0., deadline-time.monotonic())))
                break
            except queue.Full:
                if time.monotonic() >= deadline:
                    raise RuntimeError('Recording worker drain timed out')
        self.thread.join(max(0., deadline-time.monotonic()))
        self.check()
        if self.thread.is_alive():
            raise RuntimeError('Recording worker exit timed out')

    def stats(self):
        return dict(submitted=self.submitted, completed=self.completed,
                    high_water=self.high_water, queue_wait_max_s=self.queue_wait_max_s,
                    work_max_s=self.work_max_s, error=None if self.error is None else repr(self.error))


class LatestFrame:
    """One slot, never a FIFO of observations. Payload ownership is immutable."""
    def __init__(self):
        self.lock = threading.Lock()
        self.value = None
        self.received = self.overwritten_unselected = 0
        self.selected = True

    def publish(self, value):
        with self.lock:
            if not self.selected:
                self.overwritten_unselected += 1
            self.value = value
            self.received += 1
            self.selected = False

    def peek(self):
        with self.lock:
            return self.value

    def take_latest(self):
        with self.lock:
            self.selected = True
            return self.value
