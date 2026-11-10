# TEM Spatial Intuition Training

A Python training and evaluation wrapper for the Tolman–Eichenbaum Machine. The Docker image includes a compass-grid trainer and a native access-graph trainer for labelled passages between rooms. Both can score a fitted model's cue prediction before the arrival cue is shown. The walk supplies actions.

Start with [Docker setup](docs/DOCKER.md) to check the pinned CPU or CUDA environment and choose a training or evaluation path. See [Git and local files](docs/GIT.md) for the publication boundary.

This repository contains the training wrapper, tests and build instructions. Bring a separate checkout of [torch_tem](https://github.com/jbakermans/torch_tem) at the revision in [the upstream lock](docker/upstream.lock.json). Study-specific layouts, settings, runs, checkpoints and research notes are kept locally.

No project license has been selected. The upstream implementation is maintained separately by its authors.
