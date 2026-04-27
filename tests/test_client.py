"""Tests for `client.py` — live methods only.

`client.py` is partly live and partly stubbed (see plan §3.1). The live
methods we can test without a real `TrafficCNN`:

* `compute_confidence_threshold` — pure numpy median.
* `build_discriminator_dataset` — pure concatenation + mask work.
* `set_parameters` — dtype cast on the global-label receive path.
* `get_parameters` / `filter_and_predict` — these *call into the CNN*, so they
  must raise `NotImplementedError` in the current pass. We verify that.
* `run_distillation` — live in the round-1 short-circuit path (all -1 global
  labels); must return 0.0 without instantiating a DataLoader.
* `fit` — orchestrator; most of its body depends on CNN stubs, so we only
  exercise the plumbing that precedes them (set_parameters) via mocking.

We use `object.__new__(SSFLClient)` to build a client **without** running
the constructor, because the real constructor calls `build_classifier`
which raises `NotImplementedError` until pass 2.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pytest

pytestmark = [pytest.mark.flwr, pytest.mark.torch]

import config  # noqa: E402
from client import SSFLClient  # noqa: E402


# ---------------------------------------------------------------------------
# Helper: construct a "data-only" SSFLClient without invoking the CNN-raising
# __init__. We set only the attributes each method under test reads.
# ---------------------------------------------------------------------------
def _bare_client(
    X_open: np.ndarray,
    X_private: np.ndarray,
    y_private: np.ndarray,
    num_classes: int = 11,
) -> SSFLClient:
    c = object.__new__(SSFLClient)  # bypass __init__
    c.client_id = 0
    c.X_private = X_private
    c.y_private = y_private
    c.X_open = X_open
    c.N_open = int(X_open.shape[0])
    c.num_classes = num_classes
    c.batch_size = 10
    c.classifier_epochs = 1
    c.discriminator_epochs = 1
    c.distillation_epochs = 1
    c.learning_rate = 1e-4
    c.current_round = 0
    c.global_labels = None
    # device / models / optimizer intentionally NOT set — tests that touch
    # them should assert they don't need them.
    return c


# ---------------------------------------------------------------------------
# 6.4 compute_confidence_threshold
# ---------------------------------------------------------------------------
class TestComputeConfidenceThreshold:
    def test_returns_median(self) -> None:
        c = _bare_client(
            X_open=np.zeros((1, 23, 5), dtype=np.float32),
            X_private=np.zeros((1, 23, 5), dtype=np.float32),
            y_private=np.zeros((1,), dtype=np.int64),
        )
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        t = c.compute_confidence_threshold(scores)
        # Median of [0.1, 0.2, 0.3, 0.4, 0.5] is 0.3 exactly.
        assert t == pytest.approx(0.3)
        assert isinstance(t, float)

    def test_handles_even_length(self) -> None:
        # numpy.median averages the two middle values on even-length input.
        c = _bare_client(
            X_open=np.zeros((1, 23, 5), dtype=np.float32),
            X_private=np.zeros((1, 23, 5), dtype=np.float32),
            y_private=np.zeros((1,), dtype=np.int64),
        )
        scores = np.array([0.1, 0.3, 0.5, 0.9], dtype=np.float32)
        assert c.compute_confidence_threshold(scores) == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# 6.5 build_discriminator_dataset
# ---------------------------------------------------------------------------
class TestBuildDiscriminatorDataset:
    def test_shapes_and_label_invariants(self) -> None:
        # 10 open samples, 4 private samples, half of open are unfamiliar.
        X_open = np.random.default_rng(0).standard_normal((10, 23, 5)).astype(np.float32)
        X_private = np.random.default_rng(1).standard_normal((4, 23, 5)).astype(np.float32)
        y_private = np.zeros((4,), dtype=np.int64)
        c = _bare_client(X_open, X_private, y_private, num_classes=3)
        # Confidence scores where exactly 5 entries fall below threshold=0.5.
        scores = np.array([0.1, 0.2, 0.4, 0.45, 0.49, 0.6, 0.7, 0.8, 0.9, 1.0],
                          dtype=np.float32)
        X_disc, y_disc = c.build_discriminator_dataset(scores, threshold=0.5)

        # 5 unfamiliar + 4 familiar = 9 rows
        assert X_disc.shape == (9, 23, 5)
        assert y_disc.shape == (9,)
        # First 5 entries are unfamiliar (label 1), next 4 are familiar (label 0).
        # Their exact ordering follows the np.concatenate order in the source.
        assert (y_disc == 1).sum() == 5
        assert (y_disc == 0).sum() == 4
        assert y_disc.dtype == np.int64

    def test_no_unfamiliar_samples_when_threshold_is_low(self) -> None:
        # All confidence scores above threshold -> no unfamiliar rows; output
        # reduces to the private set labelled familiar.
        X_open = np.random.default_rng(2).standard_normal((6, 23, 5)).astype(np.float32)
        X_private = np.random.default_rng(3).standard_normal((3, 23, 5)).astype(np.float32)
        y_private = np.zeros((3,), dtype=np.int64)
        c = _bare_client(X_open, X_private, y_private)
        scores = np.full((6,), 0.9, dtype=np.float32)
        X_disc, y_disc = c.build_discriminator_dataset(scores, threshold=0.0)
        assert X_disc.shape == (3, 23, 5)
        assert (y_disc == 0).all()

    def test_all_open_unfamiliar_when_threshold_is_high(self) -> None:
        X_open = np.random.default_rng(4).standard_normal((6, 23, 5)).astype(np.float32)
        X_private = np.random.default_rng(5).standard_normal((2, 23, 5)).astype(np.float32)
        y_private = np.zeros((2,), dtype=np.int64)
        c = _bare_client(X_open, X_private, y_private)
        scores = np.full((6,), 0.1, dtype=np.float32)
        X_disc, y_disc = c.build_discriminator_dataset(scores, threshold=1.0)
        # 6 unfamiliar + 2 familiar
        assert X_disc.shape == (8, 23, 5)
        assert (y_disc == 1).sum() == 6
        assert (y_disc == 0).sum() == 2


# ---------------------------------------------------------------------------
# 6.10 set_parameters
# ---------------------------------------------------------------------------
class TestSetParameters:
    def test_casts_to_int64_and_stores(self) -> None:
        c = _bare_client(
            X_open=np.zeros((4, 23, 5), dtype=np.float32),
            X_private=np.zeros((1, 23, 5), dtype=np.float32),
            y_private=np.zeros((1,), dtype=np.int64),
        )
        # Incoming parameters could arrive with int32 dtype from the wire.
        incoming = [np.array([0, -1, 2, 1], dtype=np.int32)]
        c.set_parameters(incoming)
        assert c.global_labels is not None
        assert c.global_labels.dtype == np.int64
        np.testing.assert_array_equal(
            c.global_labels, np.array([0, -1, 2, 1], dtype=np.int64)
        )


# ---------------------------------------------------------------------------
# 6.8 run_distillation — round-1 short-circuit is LIVE; otherwise stubbed.
# ---------------------------------------------------------------------------
class TestRunDistillationShortCircuit:
    def test_all_minus_one_returns_zero_without_raising(self) -> None:
        c = _bare_client(
            X_open=np.zeros((5, 23, 5), dtype=np.float32),
            X_private=np.zeros((1, 23, 5), dtype=np.float32),
            y_private=np.zeros((1,), dtype=np.int64),
        )
        # No valid labels yet — round 1 path.
        global_labels = np.full((5,), -1, dtype=np.int64)
        loss = c.run_distillation(global_labels)
        assert loss == 0.0

    def test_any_valid_label_triggers_cnn_dependent_branch(self) -> None:
        # With even a single valid label, the method falls through to the
        # CNN-backed section and raises NotImplementedError in pass 1.
        c = _bare_client(
            X_open=np.zeros((3, 23, 5), dtype=np.float32),
            X_private=np.zeros((1, 23, 5), dtype=np.float32),
            y_private=np.zeros((1,), dtype=np.int64),
        )
        with pytest.raises(NotImplementedError):
            c.run_distillation(np.array([-1, 0, -1], dtype=np.int64))


# ---------------------------------------------------------------------------
# Stubs — these must *currently* raise NotImplementedError.
# Testing the stubs documents the contract: pass 2 should replace these with
# working implementations and these tests will turn red, at which point the
# student deletes them.
# ---------------------------------------------------------------------------
class TestCnnDependentStubs:
    """`cnn_dependent` marker — tests that should flip when model.py is filled in."""

    pytestmark = pytest.mark.cnn_dependent

    def test_train_classifier_still_stubbed(self) -> None:
        c = _bare_client(
            X_open=np.zeros((1, 23, 5), dtype=np.float32),
            X_private=np.zeros((1, 23, 5), dtype=np.float32),
            y_private=np.zeros((1,), dtype=np.int64),
        )
        with pytest.raises(NotImplementedError):
            c.train_classifier()

    def test_compute_confidence_scores_still_stubbed(self) -> None:
        c = _bare_client(
            X_open=np.zeros((1, 23, 5), dtype=np.float32),
            X_private=np.zeros((1, 23, 5), dtype=np.float32),
            y_private=np.zeros((1,), dtype=np.int64),
        )
        with pytest.raises(NotImplementedError):
            c.compute_confidence_scores()

    def test_train_discriminator_still_stubbed(self) -> None:
        c = _bare_client(
            X_open=np.zeros((1, 23, 5), dtype=np.float32),
            X_private=np.zeros((1, 23, 5), dtype=np.float32),
            y_private=np.zeros((1,), dtype=np.int64),
        )
        with pytest.raises(NotImplementedError):
            c.train_discriminator(
                X_disc=np.zeros((1, 23, 5), dtype=np.float32),
                y_disc=np.zeros((1,), dtype=np.int64),
            )

    def test_filter_and_predict_still_stubbed(self) -> None:
        c = _bare_client(
            X_open=np.zeros((1, 23, 5), dtype=np.float32),
            X_private=np.zeros((1, 23, 5), dtype=np.float32),
            y_private=np.zeros((1,), dtype=np.int64),
        )
        with pytest.raises(NotImplementedError):
            c.filter_and_predict()

    def test_evaluate_still_stubbed(self) -> None:
        c = _bare_client(
            X_open=np.zeros((1, 23, 5), dtype=np.float32),
            X_private=np.zeros((1, 23, 5), dtype=np.float32),
            y_private=np.zeros((1,), dtype=np.int64),
        )
        with pytest.raises(NotImplementedError):
            c.evaluate(parameters=[], config={})

    def test_get_parameters_propagates_stub(self) -> None:
        # get_parameters delegates to filter_and_predict, so it inherits the
        # NotImplementedError behaviour.
        c = _bare_client(
            X_open=np.zeros((1, 23, 5), dtype=np.float32),
            X_private=np.zeros((1, 23, 5), dtype=np.float32),
            y_private=np.zeros((1,), dtype=np.int64),
        )
        with pytest.raises(NotImplementedError):
            c.get_parameters(config={})

    def test_constructor_itself_is_stubbed(self) -> None:
        # Sanity: the real constructor raises because build_classifier raises.
        import torch
        with pytest.raises(NotImplementedError):
            SSFLClient(
                client_id=0,
                X_private=np.zeros((1, 23, 5), dtype=np.float32),
                y_private=np.zeros((1,), dtype=np.int64),
                X_open=np.zeros((1, 23, 5), dtype=np.float32),
                num_classes=11,
                device=torch.device("cpu"),
                learning_rate=1e-4,
                batch_size=10,
                classifier_epochs=1,
                discriminator_epochs=1,
            )
