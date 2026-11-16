# TEM Spatial Intuition Training

A Python training and evaluation wrapper for the Tolman–Eichenbaum Machine. The Docker image includes a compass-grid trainer, a native access-graph trainer for labelled passages between rooms, and a generic multi-topology cohort generator with a paired single/diverse retraining trainer. The trainers can score a fitted model's cue prediction before the arrival cue is shown; the walk supplies actions. The multi-topology scripts freeze a fresh synthetic cohort whose final test families stay sealed behind a SHA-256 commitment that the trainer never opens.

Start with [Docker setup](docs/DOCKER.md) to check the pinned CPU or CUDA environment and choose a training or evaluation path. See [Git and local files](docs/GIT.md) for the publication boundary.

This repository contains the training wrappers (grid, native access-graph, and multi-topology retraining), the cohort generators, tests and build instructions. Bring a separate checkout of [torch_tem](https://github.com/jbakermans/torch_tem) at the revision in [the upstream lock](docker/upstream.lock.json). Study-specific layouts, settings, runs, checkpoints and research notes are kept locally.

No project license has been selected. The upstream implementation is maintained separately by its authors.
