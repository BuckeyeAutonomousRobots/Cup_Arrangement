"""Offline position-pilot contracts, shared by tests and the isolated trainer."""
import hashlib
import json
from pathlib import Path

import numpy as np


CALIBRATION_SEEDS = [1, 2, 11, 20, 22, 25, 27, 32, 40, 56, 57, 61, 66, 72, 74, 80]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def partition(manifest, prior):
    entries = manifest['episodes']
    assert len(entries) == 100 and {e['seed'] for e in entries} == set(range(100))
    outer = sorted(s for s in range(100) if s % 10 in (8, 9))
    calibration = CALIBRATION_SEEDS
    fit = sorted(set(range(100)) - set(outer) - set(calibration))
    assert sorted(prior['fit_seeds']) == fit
    assert sorted(prior['calibration_seeds']) == calibration
    assert sorted(prior['outer_seeds']) == outer
    assert sorted(manifest['train_seeds']) == sorted(fit + calibration)
    assert sorted(manifest['validation_seeds']) == outer
    return {name: [e for e in entries if e['seed'] in seeds]
            for name, seeds in [('fit', fit), ('calibration', calibration), ('outer', outer)]}


def validate_targets(states, actions, ticks, side):
    assert states.shape == (len(ticks), 8) and actions.shape == (len(ticks), 7)
    assert len(ticks) > 1 and np.all(np.diff(ticks) > 0)
    assert np.isfinite(states).all() and np.isfinite(actions).all()
    np.testing.assert_array_equal(ticks, side['ticks'])
    valid = np.r_[np.diff(ticks) == 1, False]
    np.testing.assert_array_equal(valid, side['valid'])
    targets = side['targets']
    assert targets.shape == (len(ticks), 7)
    assert np.isfinite(targets[valid]).all() and np.isnan(targets[~valid]).all()
    ids = np.flatnonzero(valid)
    np.testing.assert_array_equal(targets[ids, :6], states[ids + 1, :6])
    np.testing.assert_array_equal(targets[ids, 6], actions[ids, 6])
    return valid


def fit_normalization(entries, targets_dir):
    """Caller passes only the already verified 64 fit episodes."""
    states, actions = [], []
    for entry in entries:
        side = np.load(Path(targets_dir) / f"seed-{entry['seed']}.npz")
        valid = side['valid']
        states.append(np.load(Path(entry['cache']) / 'states.npy')[valid])
        actions.append(side['targets'][valid])
    q, a = np.concatenate(states), np.concatenate(actions)
    return dict(state_mean=q.mean(0),
                state_std=np.maximum(q.std(0), np.array([.001] * 6 + [.0001] * 2, np.float32)),
                action_mean=a.mean(0),
                action_std=np.maximum(a.std(0), np.array([.001] * 6 + [.0001], np.float32)))


def adapter_metrics(position, states, commands, targets, ticks, valid):
    """Teacher-forced diagnostics; guard violations count, never disappear via clipping.

    A real adapter must reject >0.15 rad error. Hypothetically clipped actions below
    are only diagnostic when any guard fails, and are never sent to a simulator.
    """
    error = position[:, :6] - states[:, :6]
    assert np.isfinite(position).all()
    raw = np.column_stack([error / .1, position[:, 6]])
    applied = raw.copy()
    applied[:, :6] = np.clip(applied[:, :6], -.245, .245)
    applied[:, 6] = np.clip(applied[:, 6], -.01, 0)
    rot = np.abs(commands[:, 5]) > .02
    steady = np.array([not np.any(rot[np.abs(ticks - t) <= 10]) for t in ticks])
    past_valid = np.r_[False, np.diff(ticks) == 1]
    past_velocity = np.zeros(len(ticks))
    past_velocity[1:] = np.diff(states[:, 5]) / (np.diff(ticks) / 10.)
    stationary = rot & past_valid & (np.abs(past_velocity) <= .005)
    correct = (applied[:, 5] * commands[:, 5] > 0) & (np.abs(applied[:, 5]) > .02)
    mask = valid.astype(bool)
    rot, steady, stationary = [m & mask for m in (rot, steady, stationary)]
    mae = np.abs(applied - commands)
    measured_velocity = (targets[mask, :6] - states[mask, :6]) / .1
    close = np.flatnonzero(commands[:, 6] < -.004)
    approach = mask & (np.arange(len(ticks)) < (close[0] if len(close) else len(ticks)))
    return {
        'samples': int(mask.sum()),
        'position_mae': np.abs(position[mask] - targets[mask]).mean(0).tolist(),
        'command_mae': mae[mask].mean(0).tolist(),
        'measured_velocity_mae': np.abs(applied[mask, :6] - measured_velocity).mean(0).tolist(),
        'rotation_samples': int(rot.sum()), 'rotation_error_sum': float(mae[rot, 5].sum()),
        'rotation_correct': int(correct[rot].sum()),
        'steady_samples': int(steady.sum()),
        'steady_false': int((np.abs(applied[steady, 5]) > .02).sum()),
        'stationary_samples': int(stationary.sum()), 'stationary_correct': int(correct[stationary].sum()),
        'tracking_guard_violations': int((np.max(np.abs(error[mask]), axis=1) > .15).sum()),
        'adapter_saturation_samples': int(np.any(np.abs(raw[mask, :6]) > .245, axis=1).sum()),
        'jaw_out_of_bounds': int(((raw[mask, 6] < -.01) | (raw[mask, 6] > 0)).sum()),
        'max_tracking_error_rad': float(np.max(np.abs(error[mask]))),
        'predicted_approach_deg': float(np.rad2deg(np.sum(applied[approach, 5]) * .1)),
        'expert_approach_deg': float(np.rad2deg(np.sum(commands[approach, 5]) * .1)),
    }


def aggregate(entries):
    total = sum(e['samples'] for e in entries)
    result = {'samples': total}
    for key in ['position_mae', 'command_mae', 'measured_velocity_mae']:
        result[key] = (sum(np.array(e[key]) * e['samples'] for e in entries) / total).tolist()
    for key in ['rotation_samples', 'rotation_correct', 'steady_samples', 'steady_false',
                'stationary_samples', 'stationary_correct', 'tracking_guard_violations',
                'adapter_saturation_samples', 'jaw_out_of_bounds']:
        result[key] = sum(e[key] for e in entries)
    for name, numerator, denominator in [
        ('rotation_recall', 'rotation_correct', 'rotation_samples'),
        ('steady_false_fraction', 'steady_false', 'steady_samples'),
        ('stationary_recall', 'stationary_correct', 'stationary_samples'),
        ('saturation_fraction', 'adapter_saturation_samples', 'samples'),
    ]:
        result[name] = result[numerator] / result[denominator] if result[denominator] else None
    result['rotation_mae'] = sum(e['rotation_error_sum'] for e in entries) / max(result['rotation_samples'], 1)
    result['max_tracking_error_rad'] = max(e['max_tracking_error_rad'] for e in entries)
    return result
