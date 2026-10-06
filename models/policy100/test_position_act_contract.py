"""Synthetic contract tests: leakage, gaps, unit conversion and hidden guard failures."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from position_act_contract import adapter_metrics, fit_normalization, partition, validate_targets, CALIBRATION_SEEDS
from prepare_position_pilot import targets
from position_dataset import PositionChunks, chunk_labels


class ContractTests(unittest.TestCase):
    def fixture(self):
        q = np.arange(40, dtype=np.float32).reshape(5, 8) / 1000
        a = np.zeros((5, 7), np.float32)
        a[:, 6] = -.004
        ticks = np.array([0, 1, 3, 4, 5])
        y, valid = targets(q, a, ticks)
        return q, a, ticks, dict(targets=y, valid=valid, ticks=ticks)

    def test_targets_and_no_cross_gap(self):
        q, a, ticks, side = self.fixture()
        validate_targets(q, a, ticks, side)
        y, pad = chunk_labels(side['targets'], side['valid'], ticks, 0, 20)
        self.assertEqual(int((~pad).sum()), 1)
        np.testing.assert_array_equal(y[0, :6], q[1, :6])
        y, pad = chunk_labels(side['targets'], side['valid'], ticks, 2, 20)
        self.assertEqual(int((~pad).sum()), 2)
        np.testing.assert_array_equal(y[1, :6], q[4, :6])
        with self.assertRaises(ValueError):
            chunk_labels(side['targets'], side['valid'], ticks, 4, 20)

    def test_shifted_or_bridged_targets_rejected(self):
        q, a, ticks, side = self.fixture()
        side['targets'][0, :6] = q[0, :6]
        with self.assertRaises(AssertionError):
            validate_targets(q, a, ticks, side)
        q, a, ticks, side = self.fixture()
        side['valid'][1] = True
        with self.assertRaises(AssertionError):
            validate_targets(q, a, ticks, side)

    def test_fixed_split_rejects_leakage(self):
        outer = [s for s in range(100) if s % 10 in (8, 9)]
        train = [s for s in range(100) if s not in outer]
        fit = [s for s in train if s not in CALIBRATION_SEEDS]
        manifest = dict(episodes=[dict(seed=s) for s in range(100)], train_seeds=train, validation_seeds=outer)
        prior = dict(fit_seeds=fit, calibration_seeds=CALIBRATION_SEEDS, outer_seeds=outer)
        self.assertEqual([len(v) for v in partition(manifest, prior).values()], [64, 16, 20])
        prior['fit_seeds'] = fit[:-1] + [8]
        with self.assertRaises(AssertionError):
            partition(manifest, prior)

    def test_fit_only_stats_and_current_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            q, a, ticks, side = self.fixture()
            entries = []
            for seed, offset in [(0, 0.), (8, 10000.)]:
                cache = root / str(seed)
                cache.mkdir()
                current = q + offset
                y, valid = targets(current, a, ticks)
                for name, value in dict(states=current, ticks=ticks, images=np.zeros((5, 4, 4, 3), np.uint8)).items():
                    np.save(cache / (name + '.npy'), value)
                np.savez(root / f'seed-{seed}.npz', targets=y, valid=valid, ticks=ticks)
                entries.append(dict(seed=seed, cache=str(cache)))
            stats = fit_normalization(entries[:1], root)
            np.testing.assert_allclose(stats['state_mean'], q[side['valid']].mean(0))
            ds = PositionChunks(entries[:1], root, stats)
            sample = ds[0]
            recovered = sample['observation.state'].numpy() * stats['state_std'] + stats['state_mean']
            np.testing.assert_allclose(recovered, q[0], atol=1e-8)
            labels = sample['action'].numpy() * stats['action_std'] + stats['action_mean']
            np.testing.assert_allclose(labels[0, :6], q[1, :6], atol=1e-8)
            self.assertTrue(np.isfinite(sample['action'].numpy()).all())

    def test_adapter_units_and_guard_not_hidden_by_clipping(self):
        q = np.zeros((4, 8)); commands = np.zeros((4, 7)); commands[:, 5] = -.1
        predicted = np.zeros((4, 7)); predicted[:, 5] = -.01
        target = predicted.copy(); valid = np.ones(4, bool)
        report = adapter_metrics(predicted, q, commands, target, np.arange(4), valid)
        self.assertEqual(report['rotation_correct'], 4)
        self.assertEqual(report['stationary_correct'], 3)
        self.assertEqual(report['tracking_guard_violations'], 0)
        predicted[1, 5] = -.2
        report = adapter_metrics(predicted, q, commands, target, np.arange(4), valid)
        self.assertEqual(report['tracking_guard_violations'], 1)
        self.assertEqual(report['adapter_saturation_samples'], 1)


if __name__ == '__main__':
    unittest.main()
