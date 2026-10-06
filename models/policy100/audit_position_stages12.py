"""Stage 1: immutable source verification and one complete fit-demo reconstruction."""
import bisect
import fcntl
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from position_act_contract import sha, partition, validate_targets, fit_normalization, adapter_metrics, aggregate
from train_compare import check_workloads

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'models/act_v1'))
from train_act import prepare_episode, JOINTS, ARM, GRIP

OUT = ROOT / 'models/policy100/runs/PositionStages12_audit_20261006_01'
SEED = 12


def main():
    start = time.monotonic()
    with open('/tmp/baseline_collection_batch.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_workloads()
        OUT.mkdir(exist_ok=False)
        report = dict(status='running', pid=os.getpid(), seed=SEED,
                      selection='Among fit episodes with at least one stationary command-rotation observation, minimize cached gap count then seed. Full successful episode, no phase crop. Missing ticks remain masked, never filled.',
                      script_sha256=sha(Path(__file__)), episodes=[])

        def save():
            report['elapsed_seconds'] = time.monotonic() - start
            (OUT / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

        save()
        try:
            base = ROOT / 'models/policy100/runs'
            manifest_path = base / 'ACT_Diffusion_100_20261004/dataset_manifest.json'
            prior_path = base / 'FeatureBC_calibrated_20261005/summary.json'
            target_dir = base / 'position_target_pilot_20261005'
            manifest = json.loads(manifest_path.read_text())
            prior = json.loads(prior_path.read_text())
            target_report = json.loads((target_dir / 'summary.json').read_text())
            assert sha(manifest_path) == prior['dataset_sha256'] == target_report['source_manifest_sha256']
            splits = partition(manifest, prior)
            assert SEED in [e['seed'] for e in splits['fit']]
            report['splits'] = {k: [e['seed'] for e in v] for k, v in splits.items()}
            report['dataset_manifest_sha256'] = sha(manifest_path)
            report['target_summary_sha256'] = sha(target_dir / 'summary.json')
            onset_rows = []
            for entry in manifest['episodes']:
                if time.monotonic() - start > 600:
                    raise TimeoutError('Audit exceeded 10-minute bound')
                check_workloads()
                cache = Path(entry['cache'])
                hashes = {k: sha(cache / (k + '.npy')) for k in ('images', 'states', 'actions', 'ticks')}
                assert hashes == entry['cache_sha256'], entry['seed']
                d = {k: np.load(cache / (k + '.npy'), mmap_mode='r') for k in ('states', 'actions', 'ticks')}
                side_path = target_dir / f"seed-{entry['seed']}.npz"
                side = np.load(side_path)
                assert sha(side_path) == next(e['targets_sha256'] for e in target_report['episodes'] if e['seed'] == entry['seed'])
                valid = validate_targets(d['states'], d['actions'], d['ticks'], side)
                ticks, q, a = d['ticks'], d['states'], d['actions']
                past_valid = np.r_[False, np.diff(ticks) == 1]
                past_v = np.r_[0., np.diff(q[:, 5]) / (np.diff(ticks) / 10.)]
                onset = valid & past_valid & (np.abs(past_v) <= .005) & (np.abs(a[:, 5]) > .02)
                next_v = (side['targets'][:, 5] - q[:, 5]) / .1
                for i in np.flatnonzero(onset):
                    onset_rows.append(dict(seed=entry['seed'], tick=int(ticks[i]),
                                           split=next(k for k, v in report['splits'].items() if entry['seed'] in v),
                                           previous_measured_wrist_velocity=float(past_v[i]),
                                           expert_wrist_command=float(a[i, 5]),
                                           next_measured_wrist_velocity=float(next_v[i]),
                                           requests_correct_motion=bool(next_v[i] * a[i, 5] > 0 and abs(next_v[i]) > .02)))
                report['episodes'].append(dict(seed=entry['seed'], valid=int(valid.sum()), samples=len(ticks),
                                                gaps=int((np.diff(ticks) != 1).sum()), stationary_command_rotation=int(onset.sum()),
                                                cache_sha256=hashes, target_sha256=sha(side_path)))
                if entry['seed'] % 20 == 19:
                    save(); print(json.dumps({'verified_episodes':len(report['episodes'])}), flush=True)
            report['stationary_onsets'] = onset_rows
            stats = fit_normalization(splits['fit'], target_dir)
            np.savez(OUT / 'fit64_normalization.npz', **stats)
            report['normalization'] = 'Computed only from valid rows in the 64 fixed fit episodes; no calibration/outer rows.'
            entry = next(e for e in splits['fit'] if e['seed'] == SEED)
            ep = Path(entry['episode'])
            assert sha(ep / 'summary.json') == entry['summary_sha256']
            assert sha(ep / 'robot_data.jsonl') == entry['robot_data_sha256']
            summary = json.loads((ep / 'summary.json').read_text())
            assert summary['status'] == 'controller_pass'
            # Re-run the original causal adapter on the entire saved raw episode.
            # This creates only an isolated audit cache, never replaces source caches.
            reconstructed = prepare_episode(ep, OUT / 'reconstructed_seed12_cache')
            equal = {}
            for key in ('images', 'states', 'actions', 'ticks'):
                original = np.load(Path(entry['cache']) / (key + '.npy'), mmap_mode='r')
                rebuilt = np.load(OUT / 'reconstructed_seed12_cache' / (key + '.npy'), mmap_mode='r')
                np.testing.assert_array_equal(original, rebuilt)
                equal[key] = True
            report['selected'] = dict(entry=entry, raw_summary=summary, reconstruction=reconstructed, exact_cache_equal=equal)
            cache = Path(entry['cache'])
            q, actions, ticks = [np.load(cache / (k + '.npy')) for k in ('states', 'actions', 'ticks')]
            side = np.load(target_dir / f'seed-{SEED}.npz')
            frames = [json.loads(line) for line in (ep / 'frames.jsonl').open()]
            frame_times = [f['stamp_sim_s'] for f in frames]
            arm_rows = []
            for line in (ep / 'robot_data.jsonl').open():
                row = json.loads(line)
                if row['topic'] == ARM and row['receipt_sim_s'] is not None:
                    arm_rows.append((row['receipt_sim_s'], row['message']['data'][5]))
            arm_rows.sort()
            rotation = np.abs(actions[:, 5]) > .02
            starts = np.flatnonzero(rotation & ~np.r_[False, rotation[:-1]])
            timing = []
            for i in starts:
                t = ticks[i] / 10.
                recent = [(s, v) for s, v in arm_rows if t - .3 <= s <= t and abs(v) > .02]
                receipt = min(s for s, _ in recent) if recent else None
                fidx = bisect.bisect_right(frame_times, t) - 1
                previous = bisect.bisect_left(frame_times, receipt) - 1 if receipt is not None else None
                timing.append(dict(tick=int(ticks[i]), grid_sim_s=t, selected_image_sim_s=frame_times[fidx],
                                   image_age_s=t - frame_times[fidx], recent_rotation_receipt_sim_s=receipt,
                                   image_minus_receipt_s=None if receipt is None else frame_times[fidx] - receipt,
                                   preceding_image_sim_s=frame_times[previous] if previous is not None and previous >= 0 else None,
                                   selected_frame=frames[fidx]['file'],
                                   preceding_frame=frames[previous]['file'] if previous is not None and previous >= 0 else None))
            report['selected']['rotation_timing'] = timing
            report['selected']['gap_ticks'] = [[int(ticks[i]), int(ticks[i+1])] for i in np.flatnonzero(np.diff(ticks) != 1)]
            identity = np.column_stack([q[:, :6], q[:, 6:8].mean(1)])
            reference = side['targets'].copy()
            reference[~side['valid']] = identity[~side['valid']]
            report['selected']['identity_metrics'] = adapter_metrics(identity, q, actions, side['targets'], ticks, side['valid'])
            report['selected']['measured_reference_metrics'] = adapter_metrics(reference, q, actions, side['targets'], ticks, side['valid'])
            report['timing_interpretation'] = 'Inputs are latest at/before each 10Hz tick. Existing commanded-velocity labels use receipt-clock estimates; selected onset images can follow command receipt. Position labels instead use next contiguous measured joints. Causal construction is verified, not a proof of exact actuation alignment or sufficient onset coverage.'
            report['status'] = 'passed'
            report['stage2_allowed'] = all(equal.values()) and any(r['seed'] == SEED and r['requests_correct_motion'] for r in onset_rows)
        except BaseException as exc:
            report.update(status='failed', error=repr(exc), stage2_allowed=False)
            raise
        finally:
            save()
        print(json.dumps({k:v for k,v in report.items() if k not in ('episodes','selected')}), flush=True)


if __name__ == '__main__':
    main()
