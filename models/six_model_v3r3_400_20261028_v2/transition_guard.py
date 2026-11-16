"""Completion and resource contract for the separate 400-demo experiment."""
from pathlib import Path
import hashlib, json, os, shutil, subprocess, datetime

R = Path(__file__).resolve().parents[2]
S = R / 'ros_backend1.1/runtime/fixed_angle_constant_speed_position_only_v3r3_20261025'
P = S / 'control/parallel-segment-07'
O = Path(__file__).resolve().parent
OLD = R / 'models/six_model_v3r3_20261026'
OLD_MANIFEST = 'b4f98dcb8f650745d7744dabd48d4d5808bb1db90a360ccdbb2c8e0120640506'
RESERVE_GIB = 60
TRAINING_FLOOR_GIB = 65
CHECKPOINT_BUDGET_GIB = 12

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def save(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    os.replace(tmp, path)

def assert_terminal(state):
    assert state['status'] == 'completed', 'Collection has not completed successfully'
    assert state['accepted_count'] == len(state['accepted']) == 400
    assert set(map(int, state['accepted'])) == set(range(400))
    assert not state['active_claims'] and not state.get('cleanup_errors')
    assert state.get('finished'), 'Final cleanup has not been recorded'
    assert all(a.get('cleanup_completed') for a in state['attempts'])

def completed(full=False, require_audit=True):
    state = json.loads((P / 'progress.json').read_text())
    assert_terminal(state)
    assert not Path('/proc/' + str(state['pid'])).exists(), 'Collector owner is still alive'
    assert sha(OLD / 'FROZEN_MANIFEST.json') == OLD_MANIFEST
    assert shutil.disk_usage(O).free / 2**30 > TRAINING_FLOOR_GIB
    assert not (O / 'STOP').exists(), 'Experiment stop requested'
    if require_audit:
        proof = json.loads((O / 'COLLECTION_COMPLETE.json').read_text())
        assert proof['status'] == 'passed' and proof['accepted_count'] == 400
        assert sha(P / 'progress.json') == proof['source_progress_sha256']
        assert proof['all_raw_hashes_verified'] and proof['schema_compatible']
    if full:
        review = json.loads((P / 'REVIEW.json').read_text())
        for w in ['a', 'b']:
            c = json.loads(subprocess.check_output([
                'docker', 'inspect', 'cup_bench_v3r3_' + w + '_20261025'], text=True))[0]
            assert c['Id'] == review['containers'][w]['id'] and not c['State']['Running']
        inventory = json.loads((P / 'RECOVERY_INVENTORY.json').read_text())
        assert not inventory['unrecorded_claims'] and not inventory['uncommitted_receipts']
        for n, digest in review['operational_sha256'].items():
            assert sha(P / n) == digest, n
        for n, digest in review['recorder_transport_sha256'].items():
            assert sha(S / n) == digest, n
        for n, digest in review['reference_source_sha256'].items():
            assert sha(R / 'ros_backend1.1' / n) == digest, n
    return state

def gpu_idle():
    compute = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True)
    assert not compute.strip(), 'Another GPU workload is active: ' + compute

def preserve_old():
    proof = json.loads((O / 'OLD220_PROTECTED.json').read_text())
    for name, digest in proof['sha256'].items():
        assert sha(OLD / name) == digest, 'Old220 protected artifact changed: ' + name
