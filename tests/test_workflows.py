"""Repository workflow invariants."""

from __future__ import annotations

from pathlib import Path

import yaml

WORKFLOW_ROOT = Path(__file__).parents[1] / ".github" / "workflows"


def test_ci_and_package_workflows_use_the_canonical_gate_once() -> None:
    for filename in ("ci.yml", "package.yml"):
        path = WORKFLOW_ROOT / filename
        document = yaml.safe_load(path.read_text())
        assert isinstance(document, dict)
        workflow_text = path.read_text()
        assert workflow_text.count("make check-ci") == 1
        assert "name" in document


def test_release_workflow_runs_validation_before_image_publication() -> None:
    text = (WORKFLOW_ROOT / "package.yml").read_text()
    assert "run: PYTHON=python make check-ci" in text
    assert text.index("Run the repository validation gate") < text.index(
        "Build, smoke test, and publish the runtime image"
    )
    assert text.index("make check-ci") < text.index("docker build")
