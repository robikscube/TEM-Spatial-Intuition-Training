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

## Train and evaluate a compass grid

These examples use the upstream 4×4 grid. Use a new output path for each run. A local environment file can be mounted via `experiments/` and passed with `--env` if it satisfies the adapter's grid contract.

```powershell
docker compose run --rm --entrypoint python cpu /app/scripts/train_tem.py --repo /upstream --env /upstream/envs/4x4.json --device cpu --iterations 2 --batch-size 1 --rollout 2 --walk-length 8 --no-eval --out /runs/example-fit
docker compose run --rm --entrypoint python cpu /app/scripts/tem_trained_adapter.py evaluate --repo /upstream --env /upstream/envs/4x4.json --checkpoint /runs/example-fit/checkpoint.pt --params /runs/example-fit/params.pt --world both --steps 8 --out /runs/example-eval
```

The example verifies that training and checkpoint evaluation work end to end. Its output belongs in the local run folder. Save the exact command, dependency pins and hardware information with each research run.

## Train and evaluate a native access graph

Put your local TOML file in `experiments/`. The TOML supplies `version`, `mode`, training and evaluation settings, and graph definitions with explicit splits. The `four-layer` mode requires multiple training graphs. The `closure-mixture` mode requires an open graph and at least one closed variant in the training split. Both modes require validation and test graphs. The [graph compiler](../scripts/access_graph_worlds.py) validates the local inputs.

Prepare a fresh run and check it without changing model weights.

```powershell
docker compose run --rm --entrypoint python cpu -B /app/scripts/access_graph_tem.py prepare --repo /upstream --config /app/experiments/access.toml --out /runs/access-example
docker compose run --rm --entrypoint python cpu -B /app/scripts/access_graph_tem.py check --repo /upstream --prepared /runs/access-example --device cpu
```

After reviewing the prepared settings, use the manifest's config hash to start fitting. Set the config's training device to match the chosen service before preparing the run.

```powershell
$temConfigHash = (Get-Content -Raw container-runs/access-example/manifest.json | ConvertFrom-Json).config_sha256
docker compose run --rm --entrypoint python cpu -B /app/scripts/access_graph_tem.py train --repo /upstream --prepared /runs/access-example --approve-config-sha256 $temConfigHash
docker compose run --rm --entrypoint python cpu -B /app/scripts/access_graph_tem.py evaluate --repo /upstream --prepared /runs/access-example --device cpu --suite validation --out /runs/access-example/validation.json
```

Keep validation, test and cue-remap results in separate files. To use a private layout extension such as `architectural_grid.py`, set `$temLayoutDir` to the absolute path of its local source directory. Add `--volume "${temLayoutDir}:/app/local-layouts:ro"` before the service name on each relevant Compose run. The trainer records the extension's hash when the prepared configuration uses it. Use the original code and image for already fitted runs whose manifests bind to earlier code.

## Prepare and fit a multi-topology cohort

The multi-topology scripts generate a frozen synthetic cohort, prepare paired `single`/`diverse` arms against it, and fit a native TEM on each. Cohort generation, arm preparation, audits and route-coverage run with zero optimizer updates and no GPU; a scientific fit requires the CUDA service and the Docker runtime marker.

Generate a cohort into a fresh run folder. The generator never reads model inputs, private ledgers or sealed payloads; with no `--source` history its exclusion set is empty and disclosed.

```powershell
docker compose run --rm --entrypoint python cpu -B /app/scripts/multitopology_retrain_graphs.py prepare --out /runs/cohort
```

This writes `graphs-manifest.json`, `exclusions-private.json`, `test-sealed.json` and `prepare-report.json`. The final test families stay sealed behind a SHA-256 commitment; only the train and validation payloads are public.

Put a config TOML in `experiments/` (for example `interim.toml`) and prepare one arm. The config names the training budget, route policy and scientific stage. A scientific fit requires `scientific_mode = true`, `device = "cuda"`, full-episode BPTT (`bptt = "episode"`), the upstream schedule, at least 48 observations per episode, and the declared update count with its checkpoint list.

```powershell
docker compose --profile gpu run --rm --entrypoint python cuda -B /app/scripts/train_multitopology_retrain.py prepare --config /app/experiments/interim.toml --manifest /runs/cohort/graphs-manifest.json --arm single --repo /upstream --out /runs/interim-single
docker compose --profile gpu run --rm --entrypoint python cuda -B /app/scripts/train_multitopology_retrain.py check --prepared /runs/interim-single --repo /upstream --device cuda
```

Prepare the `diverse` arm the same way with `--arm diverse`. After reviewing the prepared manifests, start fitting with the prepared `config_sha256` as the approval token. The trainer refuses to start without it, and it never opens `test-sealed.json`.

```powershell
$temConfigHash = (Get-Content -Raw container-runs/interim-single/manifest.json | ConvertFrom-Json).config_sha256
docker compose --profile gpu run --rm --entrypoint python cuda -B /app/scripts/train_multitopology_retrain.py train --prepared /runs/interim-single --repo /upstream --approve-config-sha256 $temConfigHash
```

Audit the pairing without fitting or scoring, and measure the training-route distribution:

```powershell
docker compose run --rm --entrypoint python cpu -B /app/scripts/train_multitopology_retrain.py pair-audit --single /runs/interim-single --diverse /runs/interim-diverse
docker compose run --rm --entrypoint python cpu -B /app/scripts/train_multitopology_retrain.py route-coverage --prepared /runs/interim-single --episodes 64
```

Continue a completed interim endpoint into the standard confirmation budget. The continuation must reuse the same prepared design and only extend `iterations`/`save_updates`, and it is labelled `scientific_continuation`, never `scientific_fit`.

```powershell
docker compose --profile gpu run --rm --entrypoint python cuda -B /app/scripts/train_multitopology_retrain.py train --prepared /runs/standard-single --repo /upstream --approve-config-sha256 $temStandardHash --resume --continuation-from /runs/interim-single
```

A completed fit is a prerequisite for an independent reviewer, not evidence of acquisition or transfer. The trainer records zero optimizer updates for every prepare/audit/route-coverage step.

## Reuse the image

Start a new container and use a fresh output directory for each run. Changes to settings, checkpoints, or supported mounted inputs use the same image. Rebuild when changing packaged code, installed dependencies, or build instructions. With `--rm`, Docker removes the container after exit and the host keeps the run outputs.
