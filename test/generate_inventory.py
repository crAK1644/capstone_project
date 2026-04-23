#!/usr/bin/env python3
"""Generate an *inventory* Markdown report for the SSFL test suite.

When `pytest` is installed, the preferred way to get a real `test_results.md`
is simply:

    cd capstone_project/
    pytest test/

The conftest.py hook will then write a proper per-test table with pass/fail
outcomes.

In environments where pytest (or its runtime dependencies like `torch` /
`flwr` / `sklearn`) cannot be installed, this script is a non-executing
fallback: it statically parses every `test_*.py` file, lists the test cases
it finds, reports which dependencies are importable in the current
interpreter, and writes `test_results.md` at the project root.

The resulting document answers three questions the student will want to see
before committing:

1. How many tests are in the suite, grouped by file?
2. What is each test class responsible for?
3. Which of the tests can actually run in this environment, and which are
   currently gated behind a missing dependency?
"""
from __future__ import annotations

import ast
import importlib.util
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
OUT_PATH = PROJECT_ROOT / "test_results.md"


# ---------------------------------------------------------------------------
# Static parsing helpers — no test execution, just AST walks.
# ---------------------------------------------------------------------------
def _module_markers(tree: ast.Module) -> List[str]:
    """Best-effort extraction of module-level `pytestmark` values.

    Handles both `pytestmark = pytest.mark.foo` and
    `pytestmark = [pytest.mark.foo, pytest.mark.bar]`. Anything else is
    treated as no markers.
    """
    markers: List[str] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "pytestmark" not in targets:
            continue
        value = node.value
        items = value.elts if isinstance(value, (ast.List, ast.Tuple)) else [value]
        for item in items:
            # Unwrap pytest.mark.<name>
            if isinstance(item, ast.Attribute) and isinstance(item.value, ast.Attribute):
                if (
                    isinstance(item.value.value, ast.Name)
                    and item.value.value.id == "pytest"
                    and item.value.attr == "mark"
                ):
                    markers.append(item.attr)
    return markers


def _class_docstring(cls_node: ast.ClassDef) -> str:
    doc = ast.get_docstring(cls_node) or ""
    return doc.strip().splitlines()[0] if doc else ""


def _class_markers(cls_node: ast.ClassDef) -> List[str]:
    """Extract a class-level `pytestmark = pytest.mark.foo` attribute if present."""
    markers: List[str] = []
    for item in cls_node.body:
        if not isinstance(item, ast.Assign):
            continue
        targets = [t.id for t in item.targets if isinstance(t, ast.Name)]
        if "pytestmark" not in targets:
            continue
        value = item.value
        items = value.elts if isinstance(value, (ast.List, ast.Tuple)) else [value]
        for node in items:
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "pytest"
                and node.value.attr == "mark"
            ):
                markers.append(node.attr)
    return markers


def _parse_test_file(path: Path) -> List[Tuple[str, List[str], List[str], str]]:
    """Return list of `(class_or_module_name, markers, tests, docstring)`.

    Free-standing `def test_*` functions are grouped under a synthetic
    "<module>" entry so they still appear in the inventory.
    """
    tree = ast.parse(path.read_text())
    module_marks = _module_markers(tree)

    rows: List[Tuple[str, List[str], List[str], str]] = []
    loose_tests: List[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            tests = [
                n.name for n in node.body
                if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
            ]
            markers = module_marks + _class_markers(node)
            rows.append((node.name, markers, tests, _class_docstring(node)))
        elif isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            loose_tests.append(node.name)
    if loose_tests:
        rows.append(("<module-level>", module_marks, loose_tests, ""))
    return rows


# ---------------------------------------------------------------------------
# Environment probes.
# ---------------------------------------------------------------------------
DEPS = ["pytest", "numpy", "pandas", "torch", "flwr", "sklearn"]


def _probe_dependency(name: str) -> Tuple[bool, str]:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return False, "(not installed)"
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "(version unknown)")
        return True, ver
    except Exception as e:  # noqa: BLE001
        return False, f"(import failed: {e})"


# ---------------------------------------------------------------------------
# Report writer.
# ---------------------------------------------------------------------------
def main() -> int:
    # 1) Scan test files.
    test_files = sorted(HERE.glob("test_*.py"))
    total_tests = 0
    per_file: Dict[str, List[Tuple[str, List[str], List[str], str]]] = {}
    for tf in test_files:
        rows = _parse_test_file(tf)
        per_file[tf.name] = rows
        total_tests += sum(len(tests) for _, _, tests, _ in rows)

    # 2) Probe environment.
    dep_status = {name: _probe_dependency(name) for name in DEPS}
    pytest_ok = dep_status["pytest"][0]

    # 3) Compose Markdown.
    lines: List[str] = []
    lines.append("# SSFL Infrastructure — pytest Test Results")
    lines.append("")
    lines.append(f"- **Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')} (inventory mode)")
    lines.append(f"- **Mode:** static inventory (pytest was not executed)")
    lines.append(f"- **Total test cases:** {total_tests}")
    lines.append("")

    lines.append("## How to get real test outcomes")
    lines.append("")
    lines.append("This report was produced by `test/generate_inventory.py`, which only")
    lines.append("reads the test files statically. It does **not** execute anything, so")
    lines.append("there are no pass/fail results below.")
    lines.append("")
    lines.append("To replace this file with real per-test pass/fail output, run:")
    lines.append("")
    lines.append("```bash")
    lines.append("cd capstone_project/")
    lines.append("pytest test/")
    lines.append("```")
    lines.append("")
    lines.append("The `test/conftest.py` `pytest_sessionfinish` hook will overwrite this")
    lines.append("file with a proper per-test results table, summary counts, skipped-reason")
    lines.append("list, and failure tracebacks.")
    lines.append("")

    # Environment section.
    lines.append("## Environment")
    lines.append("")
    lines.append(f"- Python: `{sys.version.splitlines()[0]}`")
    lines.append("")
    lines.append("| Package | Status | Version |")
    lines.append("|---|---|---|")
    for dep, (ok, ver) in dep_status.items():
        lines.append(f"| `{dep}` | {'installed' if ok else 'missing'} | {ver} |")
    lines.append("")
    if not pytest_ok:
        lines.append(
            "> `pytest` is not installed in this interpreter, so the inventory"
            " fallback was used. Install it with `pip install pytest"
            " scikit-learn` (and make sure `torch` and `flwr` are available)"
            " before re-running."
        )
        lines.append("")

    # Per-file sections.
    lines.append("## Test inventory")
    lines.append("")
    for fname, rows in per_file.items():
        n = sum(len(tests) for _, _, tests, _ in rows)
        lines.append(f"### `{fname}` — {n} test(s)")
        lines.append("")
        lines.append("| Class | Markers | Tests | Purpose |")
        lines.append("|---|---|---|---|")
        for cls, marks, tests, doc in rows:
            marks_txt = ", ".join(f"`{m}`" for m in marks) if marks else "—"
            test_count = len(tests)
            purpose = doc if doc else "—"
            # Keep cells short — tests may be long; we just list the count.
            lines.append(
                f"| `{cls}` | {marks_txt} | {test_count} | {purpose} |"
            )
        lines.append("")
        # Nested bullet list of test names, so they're actually discoverable
        # without having to open the file.
        lines.append("<details><summary>Show test names</summary>")
        lines.append("")
        for cls, _marks, tests, _doc in rows:
            lines.append(f"- **{cls}**")
            for t in tests:
                lines.append(f"  - `{t}`")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    # Final note about markers.
    lines.append("## Markers")
    lines.append("")
    lines.append("The suite uses the following custom markers (registered in")
    lines.append("`pytest.ini`):")
    lines.append("")
    lines.append("| Marker | Effect |")
    lines.append("|---|---|")
    lines.append("| `flwr` | Test module is skipped when `flwr` isn't installed. |")
    lines.append("| `torch` | Skipped when `torch` isn't installed. |")
    lines.append("| `sklearn` | Skipped when `scikit-learn` isn't installed. |")
    lines.append("| `integration` | Uses the real `prepared_data/` files (slower). |")
    lines.append("| `cnn_dependent` | Assertions about `NotImplementedError` stubs that should flip to real checks once `model.py` ships. |")
    lines.append("")

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote inventory report -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
