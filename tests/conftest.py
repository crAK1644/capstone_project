"""Shared pytest configuration, fixtures, and the session-finish report hook.

Why this file exists
--------------------
Three responsibilities, in one place:

1. Make `ssfl_project/` importable as a flat package root (no installation needed),
   so tests can write `import data_preparation` / `import strategy` / etc. exactly
   the way production modules do.
2. Provide reusable fixtures: synthetic N-BaIoT-shaped CSV trees, a session-scoped
   tmp partition directory, access to the real `prepared_data/` files that the
   user re-added, and a couple of numpy helpers.
3. Implement `pytest_sessionfinish`, which walks every test result collected this
   session and writes a Markdown report at `<workspace>/docs/test_results.md`.

The report hook is deliberately defensive: it reads from `terminalreporter.stats`
(the canonical place pytest accumulates results per outcome) so it never touches
pytest internals that change between 7.x and 8.x.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# 1. Make `ssfl_project/` importable without packaging it.
# ---------------------------------------------------------------------------
# The source tree (classic src-layout) is:
#   capstone_project/
#       src/
#           ssfl_project/       <- flat module files, no __init__.py needed
#       tests/conftest.py       <- this file
#       docs/                   <- markdown plan + generated test_results.md
#       include/                <- shared interfaces / schema stubs
#       prepared_data/
#
# Production code does `import config`, `import strategy`, etc. directly (no
# `ssfl_project.` prefix), so we insert that folder at the front of sys.path.
TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent                              # capstone_project/
SRC_DIR = PROJECT_ROOT / "src" / "ssfl_project"             # capstone_project/src/ssfl_project/
PREPARED_DATA_DIR = PROJECT_ROOT / "prepared_data"
DOCS_DIR = PROJECT_ROOT / "docs"

for p in (str(SRC_DIR), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# 2. Fixtures.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to `capstone_project/`."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def prepared_data_dir() -> Path:
    """Absolute path to the user-provided `prepared_data/` tree."""
    return PREPARED_DATA_DIR


@pytest.fixture(scope="session")
def feature_column_names() -> List[str]:
    """The canonical 115 N-BaIoT feature column names (from config)."""
    import config  # noqa: WPS433 — late import after sys.path patch above
    return list(config.FEATURE_COLUMN_NAMES)


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic NumPy RNG so assertions about values are stable."""
    return np.random.default_rng(seed=1234)


@pytest.fixture
def synthetic_nbaiot_tree(tmp_path: Path, feature_column_names: List[str]) -> Path:
    """Build a **minimal** fake N-BaIoT directory tree on disk.

    Structure (2 devices × 2 classes × small row counts — enough to exercise
    load_device_csvs, build_mini_nbaiot, shard math in partition_scenario1):

        tmp_path/
            raw/
                Device_A/
                    benign.csv
                    gafgyt_combo.csv
                Device_B/
                    benign.csv
                    mirai_udp.csv

    Each CSV has 115 columns named exactly like the canonical feature set so
    downstream code can take the same `feature_cols` path as in production.
    """
    raw = tmp_path / "raw"
    devices = {
        "Device_A": {"benign": 120, "gafgyt_combo": 120},
        "Device_B": {"benign": 120, "mirai_udp": 120},
    }
    for dev_name, classes in devices.items():
        dev_dir = raw / dev_name
        dev_dir.mkdir(parents=True, exist_ok=True)
        for class_stem, n_rows in classes.items():
            # Each class gets its own RNG seed so the test is deterministic
            # but the three files don't all hold identical rows.
            local_rng = np.random.default_rng(
                seed=abs(hash((dev_name, class_stem))) % (2**32)
            )
            data = local_rng.standard_normal(
                size=(n_rows, len(feature_column_names))
            ).astype(np.float32)
            df = pd.DataFrame(data, columns=feature_column_names)
            df.to_csv(dev_dir / f"{class_stem}.csv", index=False)
    return raw


@pytest.fixture
def tiny_mini_df(feature_column_names: List[str]) -> pd.DataFrame:
    """A synthetic mini-N-BaIoT-shaped DataFrame for tests that don't need CSVs.

    Shape: 3 devices × 3 classes × 60 rows = 540 rows. Labels use the global
    space (0, 1, 9 — i.e. `benign`, `gafgyt_combo`, `mirai_udp`), so shard
    sorting in `partition_scenario1` produces a non-trivial ordering.
    """
    rng_ = np.random.default_rng(42)
    rows: List[Dict[str, Any]] = []
    for device_id in (0, 1, 2):
        for label in (0, 1, 9):
            for _ in range(60):
                row = {name: float(rng_.standard_normal()) for name in feature_column_names}
                row["label"] = int(label)
                row["device_id"] = int(device_id)
                rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def partitioned_tmp_dir(tmp_path: Path, tiny_mini_df: pd.DataFrame) -> Path:
    """Run the pickle-save pipeline once into a tmp dir so the three loaders can
    exercise their disk paths independently."""
    import data_preparation as dp
    private_df, open_df, test_df = dp.split_private_open_test(tiny_mini_df)
    partitions = dp.build_all_client_partitions(private_df, k_di=3)
    out = tmp_path / "partitions"
    dp.save_partitions(partitions, open_df, test_df, str(out))
    return out


# ---------------------------------------------------------------------------
# 3. Session-level state and the test_results.md hook.
# ---------------------------------------------------------------------------
# The hook is invoked exactly once per `pytest` session, after every test has
# run. We pull per-test outcomes from `terminalreporter.stats`, which is the
# same data source pytest uses for its own summary table.
#
# In addition, `pytest_warning_recorded` is used to collect warnings emitted
# during any phase and map them to a nodeid, so the report can flag exactly
# which tests produced warnings rather than just listing them at the end.
_SESSION_START: Dict[str, float] = {}
_WARNINGS_BY_NODEID: Dict[str, List[str]] = {}
_ORPHAN_WARNINGS: List[str] = []  # warnings not tied to a specific nodeid


def pytest_sessionstart(session: pytest.Session) -> None:
    _SESSION_START["t"] = time.time()
    _WARNINGS_BY_NODEID.clear()
    _ORPHAN_WARNINGS.clear()


def pytest_warning_recorded(warning_message, when, nodeid, location) -> None:
    """Called by pytest every time a warning is recorded.

    We stringify the warning into `<Category>: <message>` form so the final
    report stays compact. `nodeid` is the test's full node id, or "" for
    warnings raised during collection / at import time — those go into the
    orphan bucket and appear in the "Warnings (other)" section.
    """
    category = getattr(warning_message, "category", None)
    category_name = category.__name__ if category is not None else "Warning"
    msg_txt = str(getattr(warning_message, "message", warning_message))
    formatted = f"{category_name}: {msg_txt} [{when}]"
    if nodeid:
        _WARNINGS_BY_NODEID.setdefault(nodeid, []).append(formatted)
    else:
        _ORPHAN_WARNINGS.append(formatted)


def _fmt_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    return f"{seconds / 60:.2f} min"


def _status_badge(outcome: str) -> str:
    """Map pytest's lowercase outcome to a scan-friendly bracketed badge."""
    return {
        "passed": "[PASS]",
        "failed": "[FAIL]",
        "error": "[ERROR]",
        "skipped": "[SKIP]",
        "xfailed": "[XFAIL]",
        "xpassed": "[XPASS]",
    }.get(outcome, f"[{outcome.upper()}]")


def _split_nodeid(nodeid: str) -> Dict[str, str]:
    """Break `file.py::TestClass::test_fn[params]` into its three pieces.

    Free-standing module-level functions have no class and get the synthetic
    label `<module-level>`. Parametrised IDs keep their `[...]` tail on the
    test name so different parameter sets remain distinguishable.
    """
    parts = nodeid.split("::")
    if len(parts) >= 3:
        return {"file": parts[0], "cls": parts[1], "test": "::".join(parts[2:])}
    if len(parts) == 2:
        return {"file": parts[0], "cls": "<module-level>", "test": parts[1]}
    return {"file": parts[0], "cls": "<module-level>", "test": "(unknown)"}


def _extract_report_rows(stats: Dict[str, list]) -> List[Dict[str, Any]]:
    """Collapse pytest's per-phase reports (setup/call/teardown) into one row
    per test, using the **authoritative** `rep.outcome` field rather than the
    `stats` dict key.

    Why not iterate by dict key? Because `terminalreporter.stats` is keyed by
    outcome *category*, and reports that pytest does not put in a named bucket
    (e.g. teardown-phase reports for tests that pass cleanly) end up under the
    empty-string key `""`. Using that empty string as the canonical outcome
    produces empty cells in the report and zeroed summary counts — exactly the
    symptom we're fixing here.

    Aggregation rules (matching pytest's own outcome logic):
      * `setup` phase failed -> test is `error`
      * `teardown` phase failed -> test is `error`
      * `call` phase failed -> `failed`
      * `call` phase skipped (or no call phase, only a skipped `setup`) ->
        `skipped`
      * `wasxfail` attribute on any phase elevates passed -> `xpassed`,
        skipped/failed -> `xfailed`
      * Otherwise `passed`.
    """
    # 1) Flatten all reports, dropping any stats bucket whose key starts with
    #    an underscore (pytest uses those for internal bookkeeping).
    all_reports = []
    for key, reports in stats.items():
        if isinstance(key, str) and key.startswith("_"):
            continue
        if not isinstance(reports, list):
            continue
        for rep in reports:
            if getattr(rep, "nodeid", None):  # guard against non-TestReport entries
                all_reports.append(rep)

    # 2) Bucket per nodeid, recording outcome + longrepr for each phase.
    per_test: Dict[str, Dict[str, Any]] = {}
    for rep in all_reports:
        nodeid = rep.nodeid
        phase = getattr(rep, "when", "call")
        outcome = getattr(rep, "outcome", "") or ""  # "passed" / "failed" / "skipped"
        duration = float(getattr(rep, "duration", 0.0) or 0.0)
        longrepr = str(rep.longrepr) if getattr(rep, "longrepr", None) else ""

        info = per_test.setdefault(nodeid, {
            "nodeid": nodeid,
            "phases": {},         # phase -> {outcome, longrepr}
            "duration": 0.0,
            "wasxfail": False,
        })
        info["phases"][phase] = {"outcome": outcome, "longrepr": longrepr}
        info["duration"] += duration
        if hasattr(rep, "wasxfail"):
            info["wasxfail"] = True

    # 3) Reduce each test to a single final outcome row.
    result_rows: List[Dict[str, Any]] = []
    for nodeid, info in per_test.items():
        phases = info["phases"]
        setup = phases.get("setup", {})
        call = phases.get("call", {})
        teardown = phases.get("teardown", {})
        setup_o = setup.get("outcome", "")
        call_o = call.get("outcome", "")
        teardown_o = teardown.get("outcome", "")

        final_outcome = "passed"
        final_phase = "call"
        final_longrepr = ""

        if setup_o == "failed":
            final_outcome = "error"
            final_phase = "setup"
            final_longrepr = setup.get("longrepr", "")
        elif teardown_o == "failed":
            final_outcome = "error"
            final_phase = "teardown"
            final_longrepr = teardown.get("longrepr", "")
        elif call_o == "failed":
            final_outcome = "xfailed" if info["wasxfail"] else "failed"
            final_phase = "call"
            final_longrepr = call.get("longrepr", "")
        elif call_o == "skipped" or setup_o == "skipped":
            final_outcome = "xfailed" if info["wasxfail"] else "skipped"
            final_phase = "call" if call_o == "skipped" else "setup"
            # Skip reason lives on whichever phase reported it.
            final_longrepr = (
                call.get("longrepr", "") if call_o == "skipped"
                else setup.get("longrepr", "")
            )
        elif call_o == "passed":
            final_outcome = "xpassed" if info["wasxfail"] else "passed"
            final_phase = "call"
        else:
            # No call phase ran, nothing failed, nothing skipped — highly
            # unusual but keep the test visible with an explicit label so the
            # anomaly doesn't get silently swallowed.
            final_outcome = "unknown"
            final_phase = "call"

        result_rows.append({
            "nodeid": nodeid,
            "outcome": final_outcome,
            "phase": final_phase,
            "duration": info["duration"],
            "longrepr": final_longrepr,
        })

    return sorted(result_rows, key=lambda r: r["nodeid"])


def _short_reason(longrepr: str) -> str:
    """Extract a single-line reason from a pytest longrepr blob.

    Skip messages look like `('path', line, 'Skipped: flwr not installed ...')`
    — we take the tail after the last colon, strip quotes. Failure longreprs
    are multi-line; we keep the first non-empty line as a preview.
    """
    if not longrepr:
        return ""
    text = longrepr.strip()
    # Tuple-form skip reasons: ('file', lineno, 'Skipped: <reason>')
    if text.startswith("(") and "Skipped:" in text:
        idx = text.rfind("Skipped:")
        trailing = text[idx + len("Skipped:"):].rstrip(" )'\"")
        return trailing.strip()
    # Otherwise just the first meaningful line.
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Write `test_results.md` to `docs/`.

    The layout is:
      1. Metadata + summary-by-outcome counts.
      2. Results grouped by file -> class -> test, each row labelled with a
         PASS/FAIL/SKIP/ERROR/XFAIL/XPASS badge, its run time, and any
         short-form reason (skip reason or first line of traceback).
      3. Warnings — per-test and orphan buckets.
      4. Full failure tracebacks.
      5. Long-form skip reasons (helpful when a whole marker is skipped).

    This does **not** raise — report generation must never turn a green suite
    red. Any failure while writing is logged to stderr and swallowed.
    """
    try:
        tr = session.config.pluginmanager.get_plugin("terminalreporter")
        stats = getattr(tr, "stats", {}) if tr is not None else {}
        rows = _extract_report_rows(stats)
        total = len(rows)
        by_outcome: Dict[str, int] = {}
        for r in rows:
            by_outcome[r["outcome"]] = by_outcome.get(r["outcome"], 0) + 1
        duration = time.time() - _SESSION_START.get("t", time.time())

        total_warnings = sum(len(v) for v in _WARNINGS_BY_NODEID.values()) + len(_ORPHAN_WARNINGS)
        tests_with_warnings = len(_WARNINGS_BY_NODEID)

        # -------------------------------------------------------------------
        # Header
        # -------------------------------------------------------------------
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = DOCS_DIR / "test_results.md"
        lines: List[str] = []
        lines.append("# SSFL Infrastructure — pytest Test Results")
        lines.append("")
        lines.append(f"- **Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- **Session duration:** {_fmt_duration(duration)}")
        lines.append(
            f"- **Exit status:** {exitstatus} "
            f"({'all green' if exitstatus == 0 else 'see failures below'})"
        )
        lines.append(f"- **Total tests:** {total}")
        lines.append(
            f"- **Warnings:** {total_warnings} "
            f"(across {tests_with_warnings} test(s), plus {len(_ORPHAN_WARNINGS)} orphan)"
        )
        lines.append("")

        # -------------------------------------------------------------------
        # Summary-by-outcome table
        # -------------------------------------------------------------------
        lines.append("## Summary by outcome")
        lines.append("")
        lines.append("| Badge | Outcome | Count |")
        lines.append("|---|---|---|")
        for key in ("passed", "failed", "error", "skipped", "xfailed", "xpassed"):
            lines.append(
                f"| `{_status_badge(key)}` | {key} | {by_outcome.get(key, 0)} |"
            )
        lines.append(f"| `[WARN]` | warnings recorded | {total_warnings} |")
        lines.append("")

        # -------------------------------------------------------------------
        # Grouped results: file -> class -> test row
        # -------------------------------------------------------------------
        grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for r in rows:
            parts = _split_nodeid(r["nodeid"])
            grouped.setdefault(parts["file"], {}).setdefault(parts["cls"], []).append(
                {**r, **parts}
            )

        lines.append("## Results by file")
        lines.append("")
        lines.append(
            "Each row is one test function. The **Status** column is the final"
            " outcome; `[WARN]` is appended when pytest recorded at least one"
            " warning against that test."
        )
        lines.append("")

        for fname in sorted(grouped.keys()):
            classes = grouped[fname]
            file_rows = [row for cls_rows in classes.values() for row in cls_rows]
            counts = {k: sum(1 for r in file_rows if r["outcome"] == k)
                      for k in ("passed", "failed", "error", "skipped",
                                "xfailed", "xpassed")}
            summary_bits = [f"{counts['passed']} pass"]
            if counts["failed"]:
                summary_bits.append(f"{counts['failed']} fail")
            if counts["error"]:
                summary_bits.append(f"{counts['error']} error")
            if counts["skipped"]:
                summary_bits.append(f"{counts['skipped']} skip")
            if counts["xfailed"]:
                summary_bits.append(f"{counts['xfailed']} xfail")
            if counts["xpassed"]:
                summary_bits.append(f"{counts['xpassed']} xpass")
            lines.append(
                f"### `{fname}` — {len(file_rows)} test(s) "
                f"({', '.join(summary_bits)})"
            )
            lines.append("")

            for cls_name in sorted(classes.keys()):
                cls_rows = sorted(classes[cls_name], key=lambda r: r["test"])
                lines.append(f"#### `{cls_name}`")
                lines.append("")
                lines.append("| Status | Test | Time | Notes |")
                lines.append("|---|---|---|---|")
                for r in cls_rows:
                    badge = _status_badge(r["outcome"])
                    warn_count = len(_WARNINGS_BY_NODEID.get(r["nodeid"], []))
                    if warn_count:
                        badge = f"{badge} `[WARN x{warn_count}]`"
                    notes = ""
                    if r["outcome"] in ("failed", "error"):
                        notes = _short_reason(r["longrepr"])
                    elif r["outcome"] == "skipped":
                        notes = _short_reason(r["longrepr"])
                    elif warn_count:
                        notes = "see warnings section"
                    # Keep note cells short — table breaks otherwise.
                    if len(notes) > 120:
                        notes = notes[:117] + "..."
                    # Pipes inside cells must be escaped for Markdown tables.
                    notes = notes.replace("|", "\\|")
                    lines.append(
                        f"| **{badge}** | `{r['test']}` "
                        f"| {_fmt_duration(r['duration'])} | {notes} |"
                    )
                lines.append("")

        # -------------------------------------------------------------------
        # Warnings
        # -------------------------------------------------------------------
        if _WARNINGS_BY_NODEID or _ORPHAN_WARNINGS:
            lines.append("## Warnings")
            lines.append("")
            if _WARNINGS_BY_NODEID:
                lines.append("### Per-test warnings")
                lines.append("")
                for nodeid in sorted(_WARNINGS_BY_NODEID):
                    lines.append(f"- `{nodeid}`")
                    for w in _WARNINGS_BY_NODEID[nodeid]:
                        lines.append(f"    - {w}")
                lines.append("")
            if _ORPHAN_WARNINGS:
                lines.append("### Collection / import-time warnings")
                lines.append("")
                for w in _ORPHAN_WARNINGS:
                    lines.append(f"- {w}")
                lines.append("")

        # -------------------------------------------------------------------
        # Failure tracebacks
        # -------------------------------------------------------------------
        failed_rows = [r for r in rows if r["outcome"] in ("failed", "error")]
        if failed_rows:
            lines.append("## Failures — full tracebacks")
            lines.append("")
            for r in failed_rows:
                lines.append(f"### `{r['nodeid']}` ({r['phase']})")
                lines.append("")
                lines.append("```text")
                longrepr = r["longrepr"] or "(no traceback captured)"
                if len(longrepr) > 4000:
                    longrepr = longrepr[:4000] + "\n... [truncated]"
                lines.append(longrepr)
                lines.append("```")
                lines.append("")

        # -------------------------------------------------------------------
        # Skipped reasons (detailed)
        # -------------------------------------------------------------------
        skipped_rows = [r for r in rows if r["outcome"] == "skipped"]
        if skipped_rows:
            lines.append("## Skipped tests — full reasons")
            lines.append("")
            for r in skipped_rows:
                reason = _short_reason(r["longrepr"]) or "(no reason recorded)"
                lines.append(f"- `{r['nodeid']}` — {reason}")
            lines.append("")

        out_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"\n[conftest] test_results.md written to {out_path}", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001 — never break the session
        print(f"[conftest] failed to write test_results.md: {exc!r}", file=sys.stderr)


# ---------------------------------------------------------------------------
# 4. Auto-skip markers for missing third-party deps.
# ---------------------------------------------------------------------------
# Lets us tag an entire test module with `pytestmark = pytest.mark.flwr` and
# have it skipped cleanly on machines without flwr installed — no import-time
# explosions, no red tests that are really env problems.
def pytest_collection_modifyitems(
    config: pytest.Config, items: List[pytest.Item]
) -> None:
    missing: Dict[str, str] = {}
    for pkg, marker in (("flwr", "flwr"), ("torch", "torch"),
                        ("sklearn", "sklearn")):
        try:
            __import__(pkg)
        except ImportError:
            missing[marker] = f"{pkg} not installed in this environment"
    if not missing:
        return
    for item in items:
        for marker_name, reason in missing.items():
            if marker_name in item.keywords:
                item.add_marker(pytest.mark.skip(reason=reason))
