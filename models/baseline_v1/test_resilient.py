import unittest
import contextlib
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import collect_resilient as runner
from collect_resilient import retryable_startup, retryable_recorder


class RetryTests(unittest.TestCase):
    def run_fake_batch(self, failures):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'runtime').mkdir()
            counter = [0]
            def command(args, log, timeout=None):
                if args[-1] == 'bringup_dual':
                    counter[0] += 1
                    if counter[0] <= failures:
                        log.write_text('Failed to activate controller : left_joint_state_broadcaster')
                        return 1
                log.write_text('ok')
                if log.name.endswith('-record.log'):
                    episode = log.parent/log.name.removesuffix('-record.log')
                    episode.mkdir()
                    (episode/'summary.json').write_text(json.dumps({'status':'controller_pass','frames':10}))
                return 0
            with patch.object(runner, 'BACKEND', root), patch.object(runner, 'idle'), \
                 patch.object(runner, 'command', command), patch.object(runner.time, 'sleep'), \
                 patch.object(runner.fcntl, 'flock'), patch('sys.argv', ['runner','--batch','test','--start-seed','21','--count','1']), \
                 contextlib.redirect_stdout(io.StringIO()):
                code = runner.main()
            return code, json.loads((root/'runtime/test/status.json').read_text())

    def test_retry_then_success_is_preserved(self):
        code, state = self.run_fake_batch(1)
        self.assertEqual(code, 0)
        self.assertEqual(state['successful'], 1)
        self.assertEqual([x['status'] for x in state['attempts']], ['bringup_failed','controller_pass'])
        self.assertNotEqual(state['attempts'][0]['directory'], state['attempts'][1]['directory'])

    def test_retries_do_not_exceed_budget(self):
        code, state = self.run_fake_batch(99)
        self.assertEqual(code, 1)
        self.assertEqual(len(state['attempts']), 3)
        self.assertEqual(state['episodes'], [])
        self.assertIn('budget exhausted', state['error'])

    def test_controller_timeout_allowed(self):
        self.assertTrue(retryable_startup('Failed to activate controller : left_joint_state_broadcaster'))
        self.assertTrue(retryable_startup('Switch controller timed out after 5.000000 seconds!'))

    def test_unknown_and_fatal_rejected(self):
        for text in ('build failed', 'unknown failure', 'No space left: Failed to activate controller',
                     'Traceback: Switch controller timed out'):
            self.assertFalse(retryable_startup(text))

    def test_readiness_only_before_motion(self):
        text = 'RuntimeError: Camera/joints/calibration/static TF readiness failed'
        self.assertTrue(retryable_recorder({'status':'startup_failed','cheat_returncode':None}, text))
        self.assertFalse(retryable_recorder({'status':'startup_failed','cheat_returncode':1}, text))
        for status in ('timeout', 'interrupted', 'controller_failed'):
            self.assertFalse(retryable_recorder({'status':status,'cheat_returncode':None}, text))
        self.assertFalse(retryable_recorder({'status':'startup_failed','cheat_returncode':None}, 'disk error'))


if __name__ == '__main__':
    unittest.main()
