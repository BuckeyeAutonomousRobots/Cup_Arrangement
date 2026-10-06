# Learning environment

For simulator installation and rebuilding, follow the [main README](../README.md).
For a separate Linux server, use the [remote Linux README](remote/README.md).

## Offline learning environment

The recorded learning environment uses Python 3.12.3 and PyTorch 2.7.1 with CUDA
12.6. Primary package versions are recorded in
`requirements/training-observed.txt`. This is an environment manifest, not a
complete cross-platform dependency lock or a guarantee that a fresh resolver
will reproduce the environment.

Create a separate environment at `models/act_v1/.venv` before using the historical
launchers. Install a compatible PyTorch/torchvision pair and the listed research
dependencies using the appropriate platform indexes. The setup does
not install packages, download weights or allocate GPUs automatically.

With the dependencies available, contract tests are independent of ROS:

```bash
cd models/policy100
../act_v1/.venv/bin/python -m unittest -v test_position_act_contract
```

Training and recorded-trial launchers require local manifests, caches and prior
run artifacts that are not public in this repository. Do not create dummy data,
remove their preflight checks, or relaunch a run with an existing `.claim` file.
