# Docker setup

The image packages Linux, pinned versions of Python, PyTorch, NumPy and SciPy, and the project's training and evaluation scripts. Compose provides a read-only upstream source checkout and an experiment input folder. Files written to `/runs` appear in `container-runs/` on the host.

## Check the CPU runtime

Run from the repository root in PowerShell. Docker Desktop must be running in Linux container mode.

```powershell
New-Item -ItemType Directory -Force research/reference, experiments, container-runs | Out-Null
git -c core.autocrlf=false clone https://github.com/jbakermans/torch_tem.git research/reference/torch_tem
git -C research/reference/torch_tem -c core.autocrlf=false checkout --detach bf103fb32b5fdc7541ebbd95ba77a2d35d049d7c
docker compose build cpu
docker compose run --rm cpu
```

The check verifies the upstream file hashes, the 4×4 environment and a small tensor operation. It performs zero optimizer updates. Set `TEM_UPSTREAM_DIR`, `TEM_EXPERIMENTS_DIR` or `TEM_RUNS_DIR` to use different host directories. On Linux, the output folder must be writable by container UID 10001.

## Use a GPU

With NVIDIA GPU support available to Docker, run:

```powershell
docker compose --profile gpu build cuda
docker compose --profile gpu run --rm cuda
```

The CUDA check requires the GPU. For the commands below, use the CUDA service and `--device cuda` to train on it.

## Train and evaluate

These examples use the upstream 4×4 grid. Use a new output path for each run. A local environment file can be mounted via `experiments/` and passed with `--env` if it satisfies the adapter's grid contract.

```powershell
docker compose run --rm --entrypoint python cpu /app/scripts/train_tem.py --repo /upstream --env /upstream/envs/4x4.json --device cpu --iterations 2 --batch-size 1 --rollout 2 --walk-length 8 --no-eval --out /runs/example-fit
docker compose run --rm --entrypoint python cpu /app/scripts/tem_trained_adapter.py evaluate --repo /upstream --env /upstream/envs/4x4.json --checkpoint /runs/example-fit/checkpoint.pt --params /runs/example-fit/params.pt --world both --steps 8 --out /runs/example-eval
```

The example verifies that training and checkpoint evaluation work end to end. Its output belongs in the local run folder. Save the exact command, dependency pins and hardware information with each research run.
