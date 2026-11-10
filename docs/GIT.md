# Git and local files

Commit source, tests, public setup instructions and dependency locks. Keep experiment inputs and settings under `experiments/`, fitted runs under `container-runs/` or `artifacts/`, and unpublished notes under `notes/`. These paths are ignored, as are the upstream checkout, model weights, local credentials and generated outputs.

Before committing, review the exact files and changes staged for upload:

```sh
git status --short
git diff --cached --name-only
git diff --cached
```

Use `git check-ignore -v <path>` to check which rule excludes a local file. A Git ignore rule applies to untracked files. Review already tracked files and history before publishing an existing checkout. Use GitHub's no-reply commit address if you want to keep your personal email off public commits.

Git and Docker have separate boundaries. The Git ignore rules govern what can enter commits. The Docker allowlist governs what can enter the image build. Compose mounts local inputs at runtime and writes results outside the image.

Update the Dockerfile, Compose, the build allowlist and the relevant Python code together. Verify the supported grid and native access workflows from the exact staged files in a clean checkout before publishing. Keep private layouts and upstream source in local read-only mounts.

Record the Git commit and Docker image ID with each run. Save the exact local config and its SHA-256, the command, hashes for any mounted code, and the checkpoint fingerprint in the local run record.
