# TEM Spatial Intuition Training

A Python training and evaluation wrapper for the Tolman–Eichenbaum Machine. It trains on a supplied 4×4 environment and can score a saved model's prediction before the arrival cue is shown. Actions are supplied by the walk. The model predicts cues.

Start with [Docker setup](docs/DOCKER.md) to check the pinned CPU or CUDA environment, run a small training job, or evaluate a checkpoint. See [Git and local files](docs/GIT.md) for the publication boundary.

This repository contains the training wrapper, tests and build instructions. Bring a separate checkout of [torch_tem](https://github.com/jbakermans/torch_tem) at the revision in [the upstream lock](docker/upstream.lock.json). Study-specific layouts, settings, runs, checkpoints and research notes are kept locally.

No project license has been selected. The upstream implementation is maintained separately by its authors.
