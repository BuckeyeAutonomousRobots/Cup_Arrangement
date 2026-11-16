# Six-model, 400-demonstration research source

This is the local source snapshot for the October 28, 2026 experiment: ACT and
Diffusion Policy, each trained with velocity, joint-position-reference and
end-effector-pose-reference actions. See [results and limitations](../../docs/EXPERIMENTS_400.md).

The repository-root paths resolve from the scripts' locations. Experiment IDs,
completion guards, resource floors and source-provenance requirements are
retained. This is **not a ready-to-run training bundle**: datasets, caches,
checkpoints, completed collection ledgers, historical protected-artifact proofs
and frozen manifests are intentionally not published. A fresh clone cannot pass
those guards without the experiment inputs. Do not bypass them to run a script.

Source flow, once the required external artifacts are available:

1. `audit_collection.py`: verify completed collection, cleanup and raw hashes.
2. `prepare_data.py`: freeze the 400 accepted episodes and shared causal cache.
3. `validate_cache.py`: verify image/target parity, boundaries and episode splits.
4. `train_driver.py` / `train_one.py`: bounded sequential training and checkpoint reload checks.
5. `offline_evaluate.py`: a separate held-out report, never checkpoint selection.

`common.py` contains the shared observation preprocessing, masked action chunks,
normalization, ACT/Diffusion configurations and inference. `transition_guard.py`
enforces completion and resource requirements. `reference220/` is a source-only
test fixture for verifying that scaling the dataset retained the training math;
it is not an alternative launch directory.

The dependencies are recorded in `../../requirements/training-observed.txt`.
CUDA and the private experiment inputs are required for actual training. Run the
pure guard/provenance tests without starting ROS or loading a model:

```bash
python3 models/six_model_v3r3_400_20261028_v2/test_transition.py
```

All six models use current wrist RGB and eight measured joint positions, a
16-action horizon, and four-action execution. The shared 10 Hz dataset has
320 training, 40 validation and 40 test episodes. Invalid ticks and removed
startup prefixes cannot be crossed by chunks; tail padding is masked.

Velocity labels are native published commands. Joint-position labels are
IK-derived trajectory references, not position-actuator commands. EE labels are
internal world-frame tracking references, not native Cartesian actuator commands.
Measured state remains separate from all three action representations.
