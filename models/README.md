# Learning and experiment source

- `baseline_v1`: expert collection and initial learning utilities.
- `act_v1`: causal recording-to-cache adapter, ACT trainer and diagnostics.
- `policy100`: comparison source, position-target contracts and one-demo pilots.
- `six_model_v3r3_400_20261028_v2`: 400-demo ACT/Diffusion comparison source,
  shared causal data loader, completion guards and offline evaluation. See its
  README for the external data/provenance prerequisites.

These are research scripts, including historical bounded launchers with fixed
run IDs. They require local datasets/manifests and preserve durable claims to
prevent duplicate launches. Importing/running arbitrary launchers is not a test.
Start with the contract tests and read each experiment's required inputs.

No data, pretrained weights, fitted checkpoints or optimizer states are included.
See `../docs/DATA_AND_REPRODUCTION.md` and `../docs/EXPERIMENTS.md`.
