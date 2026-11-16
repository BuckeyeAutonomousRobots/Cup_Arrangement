"""Bounded FIFO writer: never drops data silently, drains before closing files."""
import queue
import threading
import time


class AsyncSink:
    def __init__(self, write, capacity):
        self.write = write
        self.queue = queue.Queue(capacity)
        self.error = None
        self.closed = False
        self.submitted = self.written = self.high_water = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def check(self):
        if self.error is not None:
            raise RuntimeError('Async recording failed') from self.error

    def submit(self, item):
        self.check()
        if self.closed:
            raise RuntimeError('Writer already closed')
        try:
            self.queue.put_nowait(item)
        except queue.Full as exc:
            self.error = RuntimeError('Recording queue full; episode rejected')
            raise self.error from exc
        self.submitted += 1
        self.high_water = max(self.high_water, self.queue.qsize())

    def _run(self):
        while True:
            item = self.queue.get()
            try:
                if item is None:
                    return
                if self.error is None:
                    self.write(item)
                    self.written += 1
            except BaseException as exc:
                self.error = exc
            finally:
                self.queue.task_done()

    def close(self, timeout=30):
        self.closed = True
        end = time.monotonic() + timeout
        self.queue.put(None, timeout=timeout)
        self.thread.join(max(0, end-time.monotonic()))
        if self.thread.is_alive():
            raise RuntimeError('Recording writer did not drain')
        self.check()
        if self.written != self.submitted:
            raise RuntimeError('Recording count mismatch')
        return dict(submitted=self.submitted, written=self.written,
                    high_water=self.high_water, capacity=self.queue.maxsize)
