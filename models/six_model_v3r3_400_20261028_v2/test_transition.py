"""Fail-closed completion and generated-code equivalence checks; no remote work."""
import copy, json, unittest, ast
from pathlib import Path
from transition_guard import assert_terminal
from audit_collection import recorder_digest

O = Path(__file__).resolve().parent

class CompletionTests(unittest.TestCase):
    def state(self):
        return {'status': 'completed', 'accepted_count': 400,
                'accepted': {str(i): {} for i in range(400)}, 'active_claims': {},
                'finished': '2026-10-28T00:00:00Z',
                'attempts': [{'cleanup_completed': True} for _ in range(408)]}

    def test_completed(self):
        assert_terminal(self.state())

    def test_incomplete_rejected(self):
        s = self.state();s['accepted_count'] = 399;s['accepted'].pop('399')
        with self.assertRaises(AssertionError): assert_terminal(s)

    def test_wrong_seed_domain_rejected(self):
        s = self.state();s['accepted']['400'] = s['accepted'].pop('399')
        with self.assertRaises(AssertionError): assert_terminal(s)

    def test_live_claim_rejected(self):
        s = self.state();s['active_claims'] = {'a': {'seed': 399}}
        with self.assertRaises(AssertionError): assert_terminal(s)

    def test_missing_final_cleanup_rejected(self):
        s = self.state();s.pop('finished')
        with self.assertRaises(AssertionError): assert_terminal(s)

    def test_cleanup_failure_rejected(self):
        s = self.state();s['cleanup_errors'] = ['failed']
        with self.assertRaises(AssertionError): assert_terminal(s)

    def test_unfinished_attempt_rejected(self):
        s = self.state();s['attempts'][0]['cleanup_completed'] = False
        with self.assertRaises(AssertionError): assert_terminal(s)

    def test_failed_state_rejected(self):
        s = self.state();s['status'] = 'blocked'
        with self.assertRaises(AssertionError): assert_terminal(s)

    def test_policy_and_loader_unchanged(self):
        old = (O/'reference220/common.py').read_text()
        new = (O/'common.py').read_text()
        self.assertEqual(old.replace('six_model_v3r3_20261026', 'six_model_v3r3_400_20261028_v2'), new)

    def test_training_math_unchanged(self):
        old = (O/'reference220/train_one.py').read_text()
        new = (O/'train_one.py').read_text()
        new = new.replace('\nfrom transition_guard import completed', '')
        new = new.replace('  completed();assert shutil.disk_usage(O).free/2**30>65',
                          "  assert (S/'control/USER_COLLECTION_PAUSED.json').exists();assert shutil.disk_usage(O).free/2**30>65")
        new = new.replace('len(train)==320 and len(val)==40', 'len(train)==176 and len(val)==22')
        new = new.replace("'splits':[320,40,40]", "'splits':[176,22,22]")
        self.assertEqual(old, new)

    def test_generated_sources_parse(self):
        for f in O.glob('*.py'): ast.parse(f.read_text(), filename=str(f))

    def test_qualification_entry_without_optional_ledger_hash(self):
        clock = {'recorder_sha256': 'recorded', 'source_snapshot_hash_verified': True}
        summary = {'recorder_sha256': 'recorded', 'source_sha256': {'recorder.py': 'recorded'}}
        self.assertEqual(recorder_digest({}, clock, summary), 'recorded')

    def test_recorder_provenance_mismatch_rejected(self):
        clock = {'recorder_sha256': 'recorded', 'source_snapshot_hash_verified': True}
        summary = {'recorder_sha256': 'recorded', 'source_sha256': {'recorder.py': 'recorded'}}
        with self.assertRaises(AssertionError): recorder_digest({'recorder_sha256': 'wrong'}, clock, summary)
        summary['recorder_sha256'] = 'wrong'
        with self.assertRaises(AssertionError): recorder_digest({}, clock, summary)

    def test_missing_episode_recorder_proof_rejected(self):
        with self.assertRaises(KeyError): recorder_digest({}, {}, {})

if __name__ == '__main__':
    unittest.main()
