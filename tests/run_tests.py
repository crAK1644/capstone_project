#!/usr/bin/env python3
"""Helper script to run the SSFL pytest suite and emit `docs/test_results.md`.

Why this file exists
--------------------
`pytest` is the canonical way to run the suite. The conftest.py installed in
this folder already writes `docs/test_results.md` at session end, so the
normal usage is just:

    cd capstone_project/
    pytest tests/

This helper adds three conveniences on top:

1. It invokes pytest as a subprocess with sensible defaults (quiet, no
   warning spam) and captures its exit status.
2. If pytest is missing, it prints a clear installation hint instead of a
   generic ImportError.
3. It ensures the process's CWD is the project root, so the conftest's
   `DOCS_DIR / "test_results.md"` resolves to `<project_root>/docs/`.

Usage:
    python tests/run_tests.py             # run full suite
    python tests/run_tests.py -k vote     # run tests whose name matches 'vote'
    python tests/run_tests.py --markers   # list registered markers
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent


def _pytest_available() -> bool:
    return importlib.util.find_spec("pytest") is not None


def _install_hint() -> str:
    return (
        "pytest is not installed in this Python environment.\n"
        "Install the test toolchain with:\n\n"
        "    pip install pytest scikit-learn\n\n"
        "Or, if you're using a virtual environment, activate it first and\n"
        "then install pytest inside it.\n"
        "The main SSFL runtime dependencies (`torch`, `flwr`, `pandas`,\n"
        "`numpy`) are needed separately — check your project's README or\n"
        "requirements file."
    )


def main(argv: list[str]) -> int:
    if not _pytest_available():
        print(_install_hint(), file=sys.stderr)
        return 2

    # Make sure we run from the project root so docs/test_results.md lands there.
    os.chdir(PROJECT_ROOT)

    cmd = [sys.executable, "-m", "pytest", "tests", *argv]
    print(f"[run_tests] {' '.join(cmd)}")
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
