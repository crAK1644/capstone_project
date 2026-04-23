"""Tests for `utils.py`.

`utils.py` owns four public responsibilities:

1. `compute_metrics` — standardized accuracy / F1 / precision / recall /
   confusion matrix computation. Depends on scikit-learn; the test module is
   tagged with the `sklearn` marker so missing installs skip gracefully.
2. `_json_safe` — private helper that coerces numpy scalars / arrays into
   JSON-serializable Python types. Used by `save_round_metrics` before dumping
   per-round metrics to disk.
3. `save_round_metrics` — append-on-disk JSON logger. Must tolerate a missing
   file, an existing file, and corrupted JSON (silently resets on JSONDecodeError).
4. `get_feature_column_names` — thin wrapper that returns a **copy** of the
   canonical 115 feature names; callers rely on the list being list-typed and
   exactly the right length.

`setup_logging` is intentionally left mostly untested (stdlib side effects).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

import utils


# ---------------------------------------------------------------------------
# compute_metrics (requires sklearn)
# ---------------------------------------------------------------------------
pytestmark_sklearn = pytest.mark.sklearn


class TestComputeMetrics:
    """Behavioural tests for utils.compute_metrics."""

    pytestmark = pytestmark_sklearn  # skipped if sklearn not installed

    def test_perfect_prediction_gives_all_ones(self) -> None:
        # When y_pred == y_true, all four averaged metrics should be 1.0 and
        # the confusion matrix should be diagonal.
        y_true = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
        y_pred = y_true.copy()
        out = utils.compute_metrics(y_true, y_pred, num_classes=3)
        assert out["accuracy"] == pytest.approx(1.0)
        assert out["f1_macro"] == pytest.approx(1.0)
        assert out["precision_macro"] == pytest.approx(1.0)
        assert out["recall_macro"] == pytest.approx(1.0)
        cm = out["confusion_matrix"]
        assert cm.shape == (3, 3)
        # Off-diagonal entries must be zero for a perfect classifier.
        assert np.trace(cm) == len(y_true)
        np.fill_diagonal(cm, 0)
        assert cm.sum() == 0

    def test_all_wrong_gives_zero_accuracy(self) -> None:
        # Deliberately flip every prediction; accuracy must be 0.
        y_true = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        y_pred = np.array([1, 1, 1, 0, 0, 0], dtype=np.int64)
        out = utils.compute_metrics(y_true, y_pred, num_classes=2)
        assert out["accuracy"] == pytest.approx(0.0)
        # macro F1 for fully-flipped binary predictions is also 0.0
        assert out["f1_macro"] == pytest.approx(0.0)

    def test_confusion_matrix_labels_cover_all_classes(self) -> None:
        # If no sample has label 2 and no prediction has label 2, we still
        # want a 3x3 confusion matrix rather than 2x2. The `labels=` argument
        # inside compute_metrics guarantees this.
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        cm = utils.compute_metrics(y_true, y_pred, num_classes=3)["confusion_matrix"]
        assert cm.shape == (3, 3)


# ---------------------------------------------------------------------------
# _json_safe
# ---------------------------------------------------------------------------
class TestJsonSafe:
    """Private coercion helper — important because save_round_metrics depends
    on it to dump numpy objects into JSON. A silent regression here means
    training logs stop being written with no visible error other than
    `TypeError: Object of type ndarray is not JSON serializable`.
    """

    def test_coerces_numpy_scalar_int(self) -> None:
        val = utils._json_safe(np.int64(7))
        assert isinstance(val, int)
        assert val == 7

    def test_coerces_numpy_scalar_float(self) -> None:
        val = utils._json_safe(np.float32(3.5))
        assert isinstance(val, float)
        assert val == pytest.approx(3.5)

    def test_coerces_numpy_array_to_list(self) -> None:
        val = utils._json_safe(np.arange(3, dtype=np.int64))
        # Must be an actual Python list of Python ints — json can't handle
        # numpy scalars or numpy arrays.
        assert isinstance(val, list)
        assert val == [0, 1, 2]

    def test_recurses_into_dicts_and_lists(self) -> None:
        src: Dict[str, Any] = {
            "a": np.int32(1),
            "b": [np.float64(2.0), np.arange(2)],
            "c": {"nested": np.int64(3)},
        }
        out = utils._json_safe(src)
        # The result must serialize cleanly with the stdlib json module —
        # that's the whole point of this helper.
        dumped = json.dumps(out)
        assert "1" in dumped and "2.0" in dumped
        assert out["c"]["nested"] == 3

    def test_passthrough_for_builtins(self) -> None:
        # Plain Python types should be returned unchanged.
        assert utils._json_safe("hello") == "hello"
        assert utils._json_safe(True) is True
        assert utils._json_safe(None) is None


# ---------------------------------------------------------------------------
# save_round_metrics
# ---------------------------------------------------------------------------
class TestSaveRoundMetrics:
    def test_creates_file_and_writes_single_entry(self, tmp_path: Path) -> None:
        out = tmp_path / "log.json"
        utils.save_round_metrics(
            round_num=1, metrics={"loss": 0.25}, output_path=str(out)
        )
        data = json.loads(out.read_text())
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["round"] == 1
        assert data[0]["loss"] == pytest.approx(0.25)

    def test_appends_to_existing_file(self, tmp_path: Path) -> None:
        # Two back-to-back calls should yield a 2-element list on disk. This
        # is exactly the logging pattern main.run_server uses per round.
        out = tmp_path / "log.json"
        utils.save_round_metrics(1, {"loss": 0.3}, str(out))
        utils.save_round_metrics(2, {"loss": 0.2}, str(out))
        data = json.loads(out.read_text())
        assert len(data) == 2
        assert [e["round"] for e in data] == [1, 2]

    def test_recovers_from_corrupted_json(self, tmp_path: Path) -> None:
        # The implementation catches JSONDecodeError and starts from [].
        out = tmp_path / "log.json"
        out.write_text("{not valid json")
        utils.save_round_metrics(5, {"loss": 0.1}, str(out))
        data = json.loads(out.read_text())
        assert data == [{"round": 5, "loss": pytest.approx(0.1)}]

    def test_creates_parent_directory_if_missing(self, tmp_path: Path) -> None:
        # This matches run_server's `logs/history.json` pattern.
        out = tmp_path / "nested" / "deeper" / "log.json"
        utils.save_round_metrics(9, {"x": 1}, str(out))
        assert out.exists()

    def test_coerces_numpy_metric_values(self, tmp_path: Path) -> None:
        # save_round_metrics must tolerate numpy scalars — otherwise the
        # real system breaks every time a loss lands in the dict.
        out = tmp_path / "log.json"
        utils.save_round_metrics(
            round_num=3,
            metrics={"loss": np.float32(0.125), "cm": np.arange(4)},
            output_path=str(out),
        )
        data = json.loads(out.read_text())
        assert data[0]["loss"] == pytest.approx(0.125)
        assert data[0]["cm"] == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# get_feature_column_names
# ---------------------------------------------------------------------------
class TestGetFeatureColumnNames:
    def test_returns_exactly_115_names(self) -> None:
        names = utils.get_feature_column_names()
        assert len(names) == 115

    def test_returns_a_fresh_list(self) -> None:
        # Mutating the returned list must not corrupt the module-level source.
        a = utils.get_feature_column_names()
        b = utils.get_feature_column_names()
        a.append("injected")
        assert "injected" not in b

    def test_names_are_nonempty_strings(self) -> None:
        for n in utils.get_feature_column_names():
            assert isinstance(n, str) and len(n) > 0


# ---------------------------------------------------------------------------
# setup_logging — extremely light check; we just ensure it doesn't raise and
# installs at least one handler on the root logger.
# ---------------------------------------------------------------------------
class TestSetupLogging:
    def test_installs_handler_and_sets_level(self) -> None:
        utils.setup_logging(level=logging.DEBUG)
        # logging.basicConfig is idempotent when handlers exist, so checking
        # for >=1 handler suffices.
        assert len(logging.getLogger().handlers) >= 1
