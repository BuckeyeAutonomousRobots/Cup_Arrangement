"""Offline tests: never import ROS, connect to a socket or publish commands."""
import ast
from pathlib import Path
import threading
import time
import tempfile
import json
from types import SimpleNamespace
import unittest
from camera_pipeline import BoundedWorker, LatestFrame
try:
    import numpy as np
    import cv2
except ImportError:
    np = cv2 = None


class PipelineTests(unittest.TestCase):
    def test_latest_only_and_snapshot_stability(self):
        slot = LatestFrame()
        self.assertIsNone(slot.peek())
        slot.publish(b'first'); held = slot.take_latest()
        for i in range(100):
            slot.publish(bytes([i]))
        self.assertEqual(slot.take_latest(), bytes([99]))
        self.assertEqual(held, b'first')
        self.assertEqual(slot.overwritten_unselected, 99)

    def test_slow_writer_does_not_block_latest_frame_or_submit(self):
        entered = threading.Event(); release = threading.Event(); done = []
        def handler(item, queued, started):
            entered.set(); release.wait(2); done.append(item)
        w = BoundedWorker(handler, 4, 'test-writer')
        try:
            w.submit(1); self.assertTrue(entered.wait(1))
            w.submit(2)
            slot = LatestFrame(); slot.publish('new')
            self.assertEqual(slot.take_latest(), 'new')
            self.assertEqual(done, [])
        finally:
            release.set(); w.close()
        self.assertEqual(done, [1, 2])
        self.assertEqual(w.stats()['completed'], 2)

    def test_queue_overflow_is_explicit(self):
        entered = threading.Event(); release = threading.Event()
        def handler(*args):
            entered.set(); release.wait(2)
        w = BoundedWorker(handler, 1, 'test-full')
        try:
            w.submit(1); self.assertTrue(entered.wait(1)); w.submit(2)
            with self.assertRaisesRegex(RuntimeError, 'queue full'):
                w.submit(3)
        finally:
            release.set(); w.close()

    def test_writer_error_propagates(self):
        def handler(*args):
            raise OSError('disk full')
        w = BoundedWorker(handler, 2, 'test-error')
        w.submit(1); w.thread.join(1)
        with self.assertRaisesRegex(RuntimeError, 'worker failed'):
            w.check()
        with self.assertRaises(RuntimeError):
            w.close()

    def test_close_drains_and_rejects_more(self):
        rows = []
        w = BoundedWorker(lambda item,*_: rows.append(item), 16, 'test-drain')
        for i in range(10):
            w.submit(i)
        w.close(); w.close()
        self.assertEqual(rows, list(range(10)))
        with self.assertRaises(RuntimeError):
            w.submit(11)

    def test_close_timeout_is_bounded(self):
        entered = threading.Event(); release = threading.Event()
        def handler(*args):
            entered.set(); release.wait(2)
        w = BoundedWorker(handler, 1, 'test-timeout')
        try:
            w.submit(1); self.assertTrue(entered.wait(1))
            with self.assertRaisesRegex(RuntimeError, 'timed out'):
                w.close(timeout=.01)
        finally:
            release.set(); w.thread.join(1)

    def test_ros_callback_has_no_encoding_disk_or_resizing(self):
        tree = ast.parse(Path(__file__).with_name('rollout_policy_ros.py').read_text())
        frame = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'frame')
        calls = [ast.unparse(n.func) for n in ast.walk(frame) if isinstance(n, ast.Call)]
        for forbidden in ('cv2.imwrite', 'cv2.resize', 'cv2.cvtColor', 'frames_log.write'):
            self.assertNotIn(forbidden, calls)
        self.assertIn('latest_slot.publish', calls)
        self.assertIn('image_writer.submit', calls)

    def test_guard_expression_and_qos(self):
        source = Path(__file__).with_name('rollout_policy_ros.py').read_text()
        self.assertIn("abs(c.sim_time-latest['stamp']) > .1 or any(abs(c.sim_time-joint_stamps[n]) > .1 for n in names)", source)
        self.assertIn('history=HistoryPolicy.KEEP_LAST, depth=1', source)
        self.assertIn('wall-submitted > .75', source)
        self.assertIn('c.sim_time-applied_sim > .22', source)

    @unittest.skipIf(cv2 is None, 'Requires numpy/OpenCV; also run in isolated offline Linux test')
    def test_real_frame_callback_png_and_metrics_without_ros(self):
        # Execute the actual nested production callbacks, not a reimplementation.
        # Only globals they need are supplied: no ROS import/init or network.
        tree = ast.parse(Path(__file__).with_name('rollout_policy_ros.py').read_text())
        functions = [next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
                     for name in ('frame', 'write_frame')]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'frames').mkdir()
            with (root/'frames.jsonl').open('w') as f, (root/'camera_latency.jsonl').open('w') as timing:
                g = dict(np=np, cv2=cv2, time=time, json=json, a=SimpleNamespace(output=root),
                         frames_log=f, latency_log=timing, accepting_records=[True],
                         latest_slot=LatestFrame(), frame_times=[], c=SimpleNamespace(sim_time=1.02),
                         clock_sample={'sim': 1.02, 'wall': time.monotonic()})
                exec(compile(ast.Module(body=functions, type_ignores=[]), '<production-callbacks>', 'exec'), g)
                g['image_writer'] = BoundedWorker(g['write_frame'], 32, 'test-png')
                # Padded RGB rows verify step handling as well as channel order.
                rgb = np.arange(6*8*3, dtype=np.uint8).reshape(6,8,3)
                padded = np.zeros((6,28), np.uint8); padded[:,:24] = rgb.reshape(6,24)
                msg = SimpleNamespace(encoding='rgb8', data=bytearray(padded.tobytes()),
                    height=6, width=8, step=28,
                    header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=0)))
                try:
                    g['frame'](msg)
                    msg.data[:] = bytes(len(msg.data))  # payload ownership must be independent
                    self.assertEqual(g['latest_slot'].peek()['data'], padded.tobytes())
                finally:
                    g['image_writer'].close()
            restored = cv2.cvtColor(cv2.imread(str(root/'frames/000000.png')), cv2.COLOR_BGR2RGB)
            np.testing.assert_array_equal(restored, rgb)
            row = json.loads((root/'camera_latency.jsonl').read_text())
            self.assertAlmostEqual(row['age_at_callback_clock_sim_s'], .02)
            self.assertGreaterEqual(row['png_encode_write_wall_s'], 0)
            self.assertGreaterEqual(row['writer_queue_wall_s'], 0)
            from analyze_camera_latency import analyze
            report = analyze(root)
            self.assertTrue(report['live_instrumentation_present'])
            self.assertEqual(report['png_encode_write_wall_s']['samples'], 1)


if __name__ == '__main__':
    unittest.main()
