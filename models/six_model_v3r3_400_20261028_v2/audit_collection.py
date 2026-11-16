"""Exclusive, complete raw-file audit after collector cleanup; never edits source data."""
from transition_guard import *
import fcntl, time, collections

def recorder_digest(entry, clock, summary):
    digest = clock['recorder_sha256']
    assert clock['source_snapshot_hash_verified']
    assert summary['recorder_sha256'] == digest
    assert digest in summary['source_sha256'].values()
    if 'recorder_sha256' in entry:
        assert entry['recorder_sha256'] == digest
    return digest

def main():
    with open('/tmp/baseline_collection_batch.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = completed(full=True, require_audit=False)
        gpu_idle()
        assert not (O / 'COLLECTION_COMPLETE.json').exists()
        implementation = json.loads((O / 'reference220/IMPLEMENTATION_PROVENANCE.json').read_text())
        for name, entry in implementation.items():
            assert sha(Path(entry['file'])) == entry['sha256'], name
        save(O / 'IMPLEMENTATION_PROVENANCE.json', implementation)
        progress_hash = sha(P / 'progress.json')
        expected_schema = None
        expected_order = None
        samples = 0
        files = 0
        raw_bytes = 0
        records = []
        recorder_versions = collections.Counter()
        for seed in range(400):
            completed(require_audit=False)
            entry = state['accepted'][str(seed)]
            ep = Path(entry['path'])
            assert entry['seed'] == seed and entry['status'] == 'accepted'
            assert (S / 'demos' / f'seed-{seed:03d}').resolve() == ep.resolve()
            acceptance = json.loads((ep / 'ACCEPTANCE.json').read_text())
            assert acceptance['seed'] == seed and acceptance['status'] == 'accepted'
            assert acceptance['final_task_result']['success'] and acceptance['fixed_orientation_verified']
            assert acceptance['variant'] == 'fixed_angle_constant_speed_position_only_v3r3'
            assert acceptance['expert_sha256'] == 'e4bf67b3f3d8192680022a1333fee664e7593466378ce99383d8222a76e8583d'
            assert acceptance['bundle_manifest_sha256'] == 'f2d71512dedb9b7a14a4848896665b3340cbeef6e7203fb332da4c42666e60b8'
            assert sha(ep / 'RAW_MANIFEST.json') == entry['manifest_sha256'] == acceptance['manifest_sha256']
            raw = json.loads((ep / 'RAW_MANIFEST.json').read_text())
            assert len({f['path'] for f in raw}) == len(raw)
            for f in raw:
                rel = Path(f['path'])
                assert not rel.is_absolute() and '..' not in rel.parts
                path = ep / rel
                assert path.stat().st_size == f['bytes'] and sha(path) == f['sha256'], str(path)
                files += 1
                raw_bytes += f['bytes']
            schema = json.loads((ep / 'action_schema.json').read_text())
            validation = json.loads((ep / 'ACTION_SCHEMA_VALIDATION.json').read_text())
            views = json.loads((ep / 'DATASET_VIEWS.json').read_text())
            summary = json.loads((ep / 'summary.json').read_text())
            clock = json.loads((ep / 'CLOCK_READINESS_VALIDATION.json').read_text())
            assert schema.get('native_joint_position_target_available') is False
            assert schema['ee_frame'] == 'world' and schema['ee_link'] == 'right_tool0'
            assert schema['orientation_order'] == 'xyzw'
            if expected_schema is None:
                expected_schema = schema
                expected_order = validation['measured_state_order']
            assert schema == expected_schema and validation['measured_state_order'] == expected_order
            assert validation['status'] == views['status'] == clock['status'] == 'passed'
            assert not clock['synthetic_timestamps_used'] and views['raw_full_episode_preserved']
            assert set(views['views']) >= {'cycle_included', 'cycle_excluded'}
            invalid = validation['invalid_ticks_preserved']
            valid = validation['valid_10hz_samples']
            assert invalid / (valid + invalid) < .05 and summary['max_gap_sim_s'] <= .2
            assert summary['status'] == 'controller_pass'
            for rel, digest in views['source_hashes'].items():
                assert sha(ep / rel) == digest
            count = len(views['views']['cycle_excluded']['samples'])
            samples += count
            recorder_versions[recorder_digest(entry, clock, summary)] += 1
            records.append({'seed': seed, 'episode': str(ep), 'samples': count,
                            'acceptance_sha256': sha(ep / 'ACCEPTANCE.json'),
                            'raw_manifest_sha256': entry['manifest_sha256']})
            save(O / 'audit_status.json', {'status': 'auditing', 'utc': now(),
                 'episodes_verified': seed + 1, 'raw_files_verified': files,
                 'raw_bytes_verified': raw_bytes, 'samples': samples})
            if (seed + 1) % 10 == 0:
                print(json.dumps({'audited': seed + 1, 'files': files}), flush=True)
        cache_bytes = samples * (320 * 240 * 3 + (8 + 7 + 7 + 8) * 4 + 3 * 8) + 400 * 8 * 512
        budget = {'free_GiB': shutil.disk_usage(O).free / 2**30,
                  'estimated_cache_GiB': cache_bytes / 2**30,
                  'cache_budget_GiB': cache_bytes * 1.10 / 2**30,
                  'checkpoint_budget_GiB': CHECKPOINT_BUDGET_GIB,
                  'scratch_GiB': 5, 'reserve_GiB': RESERVE_GIB}
        budget['projected_final_free_GiB'] = budget['free_GiB'] - budget['cache_budget_GiB'] - CHECKPOINT_BUDGET_GIB - 5
        assert budget['projected_final_free_GiB'] >= RESERVE_GIB, budget
        old_files = ['FROZEN_MANIFEST.json', 'TRAINING_PINS.json', 'CACHE_VALIDATION.json',
                     'TRAINING_PROGRESS.json', 'common.py', 'train_one.py', 'train_driver.py']
        old_files += [str(f.relative_to(OLD)) for f in (OLD / 'runs').rglob('*.safetensors')]
        old_files += [str(f.relative_to(OLD)) for f in (OLD / 'runs').glob('*/normalization.npz')]
        for status_file in (OLD / 'runs').glob('*/status.json'):
            old_status = json.loads(status_file.read_text())
            assert old_status['status'] == 'completed' and old_status['checkpoint_reload_pass']
            assert sha(status_file.parent / 'best_policy/model.safetensors') == old_status['best_checkpoint_sha256']
        save(O / 'OLD220_PROTECTED.json', {'utc': now(), 'sha256': {n: sha(OLD / n) for n in old_files}})
        assert sha(P / 'progress.json') == progress_hash
        save(O / 'COLLECTION_COMPLETE.json', {'status': 'passed', 'utc': now(), 'accepted_count': 400,
             'source_progress': str(P / 'progress.json'), 'source_progress_sha256': progress_hash,
             'all_raw_hashes_verified': True, 'raw_files_verified': files, 'raw_bytes_verified': raw_bytes,
             'schema_compatible': True, 'canonical_schema': expected_schema, 'measured_state_order': expected_order,
             'recorder_version_counts': dict(recorder_versions), 'samples_cycle_excluded': samples,
             'budget': budget, 'episodes': records, 'historical_failures_preserved': len(state['failures']),
             'collector_exited': True, 'both_workers_stopped': True, 'exclusive_batch_lock_acquired': True})
        save(O / 'audit_status.json', {'status': 'completed', 'utc': now(), 'episodes_verified': 400, 'budget': budget})
        print(json.dumps(budget), flush=True)

if __name__ == '__main__':
    main()
