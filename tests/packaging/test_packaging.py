"""Packaging contract: the Docker build allowlist and image contents.

Static checks only. They read the Dockerfile, the ``.dockerignore`` allowlist
and the dependency pins and assert that every script the public multi-topology
cohort preparation and single/diverse retraining commands import is shipped in
the image, and that the scripts' third-party runtime dependency (``networkx``)
is a declared, locked direct dependency rather than only a transitive one of
PyTorch. No Docker daemon, PyTorch or networkx installation is required.
"""
from __future__ import annotations

import re
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Every script that must exist under /app/scripts for the public multi-topology
# cohort preparation and retraining commands to run (the transitive sibling
# imports of train_multitopology_retrain.py and multitopology_retrain_graphs.py).
MULTITOPOLOGY_SCRIPTS = {
    "multitopology_graphs.py",
    "multitopology_retrain_graphs.py",
    "train_multitopology_retrain.py",
    "access_graph_tem.py",
    "access_graph_worlds.py",
    "tem_trained_adapter.py",
    "train_tem.py",
}

# The image must never embed private research material. Only ``scripts/`` and
# ``docker/`` (dependency pins and the runtime check) may enter the image.



def _dockerfile_copy_scripts() -> set[str]:
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    return set(re.findall(r"scripts/([\w.-]+\.py)", text))


def _dockerignore_allowed_scripts() -> set[str]:
    text = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    return {
        match.group(1)
        for line in text.splitlines()
        if (match := re.match(r"!scripts/([\w.-]+\.py)\s*$", line.strip()))
    }


class DockerAllowlistTest(unittest.TestCase):
    def test_image_copies_every_multi_topology_script(self):
        copied = _dockerfile_copy_scripts()
        self.assertTrue(
            MULTITOPOLOGY_SCRIPTS <= copied,
            f"missing from Dockerfile: {sorted(MULTITOPOLOGY_SCRIPTS - copied)}",
        )

    def test_dockerignore_allows_every_copied_script(self):
        copied = _dockerfile_copy_scripts()
        allowed = _dockerignore_allowed_scripts()
        self.assertTrue(
            copied <= allowed,
            f"copied but excluded from the build context: {sorted(copied - allowed)}",
        )

    def test_dockerfile_copies_only_public_source(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        sources: list[str] = []
        for line in dockerfile.splitlines():
            stripped = line.strip()
            if not stripped.startswith("COPY ") or "--from=" in stripped:
                continue
            tokens = stripped.split()
            sources.extend(tokens[1:-1])  # COPY <src>... <dest>
        self.assertTrue(sources, "no host-path COPY sources found")
        for source in sources:
            self.assertTrue(
                source.startswith("scripts/") or source.startswith("docker/"),
                f"Dockerfile copies a non-public path: {source}",
            )

    def test_dockerignore_allowlist_reincludes_only_public_source(self):
        dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
        for line in dockerignore.splitlines():
            stripped = line.strip()
            if not stripped.startswith("!"):
                continue
            item = stripped[1:]
            self.assertTrue(
                item in ("Dockerfile",)
                or item == "scripts/"
                or item.startswith("scripts/")
                or item == "docker/"
                or item.startswith("docker/"),
                f"allowlist re-includes a non-public path: {item}",
            )


class DependencyPinTest(unittest.TestCase):
    def test_networkx_is_a_declared_direct_dependency(self):
        pyproject = tomllib.loads(
            (ROOT / "docker" / "pyproject.toml").read_text(encoding="utf-8")
        )
        dependencies = pyproject["project"]["dependencies"]
        self.assertTrue(
            any(item.startswith("networkx==") for item in dependencies),
            f"networkx missing from project dependencies: {dependencies}",
        )

    def test_networkx_is_in_the_project_lock_not_only_transitive(self):
        lock = (ROOT / "docker" / "uv.lock").read_text(encoding="utf-8")
        block = re.search(
            r'\[\[package\]\]\nname = "tem-container-runtime".*?^\[package\.metadata\]',
            lock,
            flags=re.S | re.M,
        )
        self.assertIsNotNone(block, "tem-container-runtime package block not found in uv.lock")
        self.assertIn('{ name = "networkx" }', block.group(0))


if __name__ == "__main__":
    unittest.main()
