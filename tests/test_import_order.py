"""Each module must import on its own, so no pytest import order can hide an import loop."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "adaptrag.ingest",
        "adaptrag.retrieval.dense",
        "adaptrag.routing.router",
        "adaptrag.strategies.registry",
        "adaptrag.routes.query",
        "adaptrag.main",
    ],
)
def test_module_imports_in_a_fresh_interpreter(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"], capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
