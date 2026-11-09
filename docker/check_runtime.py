"""Verify the pinned upstream checkout and the requested runtime device."""
import argparse
import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tem_trained_adapter as adapter


def check(device):
    upstream_path = Path(os.getenv("TEM_UPSTREAM", str(adapter.DEFAULT_REPO)))
    lock = json.loads((ROOT / "docker" / "upstream.lock.json").read_text(encoding="utf-8"))
    for relative_path, expected in lock["files"].items():
        actual = adapter.sha256_file(upstream_path / relative_path)
        if actual != expected:
            raise ValueError(f"Pinned upstream file differs: {relative_path}")

    upstream = adapter.load_upstream(upstream_path)
    adapter.load_4x4(upstream_path / "envs" / "4x4.json")
    torch = adapter.require_torch()
    import numpy
    import scipy

    resolved = adapter.resolve_device(device, torch)
    if (torch.tensor([1.0], device=resolved["device"]) + 1).item() != 2.0:
        raise ValueError("Tensor check failed")
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "device": resolved,
        "upstream_pin": lock["commit"],
        "model": adapter.model_interface(upstream),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    try:
        print(json.dumps(check(args.device), indent=2))
    except (OSError, ValueError, adapter.AdapterError) as error:
        parser.exit(1, f"Runtime check failed: {error}\n")


if __name__ == "__main__":
    main()
