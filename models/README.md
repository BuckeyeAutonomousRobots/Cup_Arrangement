# Learning and experiment source

- `baseline_v1`: expert collection and initial learning utilities.
- `act_v1`: causal recording-to-cache adapter, ACT trainer and diagnostics.
- `policy100`: comparison source, position-target contracts and one-demo pilots.

These are research scripts, including historical bounded launchers with fixed
run IDs. They require local datasets/manifests and preserve durable claims to
prevent duplicate launches. Importing/running arbitrary launchers is not a test.
Start with the contract tests and read each experiment's required inputs.

No data, pretrained weights, fitted checkpoints or optimizer states are included.
See `../docs/DATA_AND_REPRODUCTION.md` and `../docs/EXPERIMENTS.md`.
