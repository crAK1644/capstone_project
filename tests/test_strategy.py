"""Tests for `strategy.py` — the SSFL server strategy.

`strategy.py` is the **core server-side algorithm** of SSFL. It is completely
live (no CNN dependency), which makes it one of the most important modules to
cover. We focus on:

1. `__init__` — initial state shape and typing.
2. `initialize_parameters` — round-trips the initial all-(-1) global labels
   through Flower's Parameters ↔ ndarrays converter.
3. `vote_mechanism` — the majority-vote core. We parametrize a handful of
   curated scenarios: unanimous vote, mixed non-(-1)/(-1), tie-break by
   argmax (numpy argmax returns the smallest index on ties, so "first class
   wins" is the documented contract), all-unfamiliar, and the defensive path
   for out-of-range / wrong-shape inputs.
4. `aggregate_fit` — uses a small in-test `DummyFitRes` double to avoid
   instantiating Flower's real gRPC objects; we only care about the parsing
   and vote invocation path.
5. `aggregate_evaluate` — weighted averaging across `EvaluateRes` doubles.
6. `evaluate` — always returns None in our strategy (delegated to eval_fn).
7. `configure_fit` / `configure_evaluate` — broadcast the current global
   labels + correct config dict to every sampled client.

The whole module is marked `flwr` because strategy.py imports flwr at the top.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

pytestmark = pytest.mark.flwr

# Imports that require flwr/torch — guarded by the module-level marker above.
import config  # noqa: E402
from strategy import SSFLStrategy  # noqa: E402


# ---------------------------------------------------------------------------
# Minimal doubles for Flower types.
# ---------------------------------------------------------------------------
# Instead of spinning up a real ClientManager (which wants gRPC plumbing),
# we hand-roll a tiny stand-in that satisfies only the parts of the API
# strategy.py actually touches: `sample(num_clients, min_num_clients)`.
class FakeClientProxy:
    """Stand-in for `flwr.server.client_proxy.ClientProxy`.

    SSFLStrategy only treats these as opaque tokens (they get passed back into
    the `(client_proxy, FitRes)` tuple). No methods are called.
    """

    def __init__(self, cid: int) -> None:
        self.cid = cid

    def __repr__(self) -> str:  # pragma: no cover — debugging aid only
        return f"FakeClientProxy(cid={self.cid})"


class FakeClientManager:
    """Stand-in for `flwr.server.client_manager.ClientManager`."""

    def __init__(self, n: int) -> None:
        self._clients = [FakeClientProxy(i) for i in range(n)]

    def sample(self, num_clients: int, min_num_clients: int) -> List[FakeClientProxy]:
        assert len(self._clients) >= min_num_clients
        return self._clients[:num_clients]


@dataclass
class FakeFitRes:
    """Stand-in for `flwr.common.FitRes` — only needs .parameters and .metrics."""
    parameters: Any
    num_examples: int = 1
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeEvaluateRes:
    """Stand-in for `flwr.common.EvaluateRes`."""
    loss: float
    num_examples: int
    metrics: Dict[str, Any] = field(default_factory=dict)


def _make_strategy(n_open: int = 10, num_classes: int = 3,
                   num_clients: int = 3) -> SSFLStrategy:
    """Convenience constructor used by multiple test classes."""
    return SSFLStrategy(
        num_clients=num_clients,
        num_classes=num_classes,
        n_open_samples=n_open,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        eval_fn=None,
    )


# ---------------------------------------------------------------------------
# 7.1 __init__
# ---------------------------------------------------------------------------
class TestStrategyInit:
    def test_initial_global_labels_are_all_minus_one(self) -> None:
        s = _make_strategy(n_open=4, num_classes=3)
        assert s.global_labels.shape == (4,)
        assert s.global_labels.dtype == np.int64
        assert np.all(s.global_labels == -1)

    def test_round_metrics_starts_empty(self) -> None:
        s = _make_strategy()
        assert s.round_metrics == []

    def test_stores_constructor_args(self) -> None:
        s = SSFLStrategy(
            num_clients=27, num_classes=11, n_open_samples=100,
            min_fit_clients=27, min_available_clients=27, eval_fn=None,
            classifier_epochs=7,
        )
        assert s.num_clients == 27
        assert s.num_classes == 11
        assert s.n_open_samples == 100
        assert s.classifier_epochs == 7


# ---------------------------------------------------------------------------
# 7.2 initialize_parameters
# ---------------------------------------------------------------------------
class TestInitializeParameters:
    def test_round_trips_initial_global_labels(self) -> None:
        from flwr.common import parameters_to_ndarrays
        s = _make_strategy(n_open=6, num_classes=3)
        params = s.initialize_parameters(client_manager=None)  # type: ignore[arg-type]
        # Flower Parameters wrap a list of ndarrays — single-element by our
        # contract (the global labels).
        arrs = parameters_to_ndarrays(params)
        assert len(arrs) == 1
        np.testing.assert_array_equal(arrs[0], np.full(6, -1, dtype=np.int64))


# ---------------------------------------------------------------------------
# 7.3 configure_fit
# ---------------------------------------------------------------------------
class TestConfigureFit:
    def test_broadcasts_current_labels_and_config_to_all_clients(self) -> None:
        from flwr.common import parameters_to_ndarrays
        s = _make_strategy(n_open=8, num_classes=3, num_clients=3)
        # Mutate server state so we can tell the broadcast picks up the fresh
        # labels (not the initial all-(-1) snapshot).
        s.global_labels = np.array([0, 1, 2, -1, 0, 1, 2, -1], dtype=np.int64)
        fcm = FakeClientManager(n=3)
        fit_tuples = s.configure_fit(server_round=5, parameters=None, client_manager=fcm)  # type: ignore[arg-type]
        assert len(fit_tuples) == 3
        for client_proxy, fit_ins in fit_tuples:
            assert isinstance(client_proxy, FakeClientProxy)
            # Config must carry the correct round and classifier_epochs.
            assert int(fit_ins.config["round"]) == 5
            assert int(fit_ins.config["classifier_epochs"]) == s.classifier_epochs
            # Parameters must equal self.global_labels.
            arrs = parameters_to_ndarrays(fit_ins.parameters)
            np.testing.assert_array_equal(arrs[0], s.global_labels)


# ---------------------------------------------------------------------------
# 7.5 vote_mechanism — the most important test in this file.
# ---------------------------------------------------------------------------
class TestVoteMechanism:
    """Majority-vote invariants.

    vote_mechanism uses `np.argmax` on the per-sample vote histogram. numpy
    argmax returns the **smallest** index among ties — that's the contract
    we assert in `test_tie_break_favors_lowest_class_index`.
    """

    def test_unanimous_vote_wins(self) -> None:
        s = _make_strategy(n_open=3, num_classes=2)
        uploads = [
            np.array([0, 0, 1], dtype=np.int64),
            np.array([0, 0, 1], dtype=np.int64),
            np.array([0, 0, 1], dtype=np.int64),
        ]
        out = s.vote_mechanism(uploads)
        np.testing.assert_array_equal(out, np.array([0, 0, 1], dtype=np.int64))

    def test_simple_majority(self) -> None:
        s = _make_strategy(n_open=2, num_classes=3)
        # Position 0: class 1 has 2 votes, class 2 has 1 vote -> winner = 1.
        # Position 1: class 0 has 2 votes, class 2 has 1 vote -> winner = 0.
        uploads = [
            np.array([1, 0], dtype=np.int64),
            np.array([1, 0], dtype=np.int64),
            np.array([2, 2], dtype=np.int64),
        ]
        out = s.vote_mechanism(uploads)
        np.testing.assert_array_equal(out, np.array([1, 0], dtype=np.int64))

    def test_minus1_uploads_are_ignored(self) -> None:
        # Any client emitting -1 for a sample abstains — vote is computed
        # only from the clients that upload a valid class.
        s = _make_strategy(n_open=2, num_classes=2)
        uploads = [
            np.array([0, -1], dtype=np.int64),
            np.array([-1, 1], dtype=np.int64),
            np.array([0, -1], dtype=np.int64),
        ]
        out = s.vote_mechanism(uploads)
        np.testing.assert_array_equal(out, np.array([0, 1], dtype=np.int64))

    def test_all_abstain_keeps_minus1(self) -> None:
        s = _make_strategy(n_open=3, num_classes=2)
        uploads = [
            np.array([-1, -1, -1], dtype=np.int64),
            np.array([-1, -1, -1], dtype=np.int64),
        ]
        out = s.vote_mechanism(uploads)
        np.testing.assert_array_equal(out, np.array([-1, -1, -1], dtype=np.int64))

    def test_tie_break_favors_lowest_class_index(self) -> None:
        # One vote for class 0, one vote for class 1, at position 0. argmax
        # returns the first occurrence of the max -> class 0 wins the tie.
        s = _make_strategy(n_open=1, num_classes=3)
        uploads = [
            np.array([0], dtype=np.int64),
            np.array([1], dtype=np.int64),
        ]
        out = s.vote_mechanism(uploads)
        assert out[0] == 0

    def test_wrong_length_upload_is_skipped_silently(self) -> None:
        # The strategy warns but must not crash when a rogue client uploads
        # a mismatched array. The surviving client's vote should prevail.
        s = _make_strategy(n_open=3, num_classes=2)
        uploads = [
            np.array([1, 0, 0], dtype=np.int64),
            np.array([999, 999], dtype=np.int64),  # wrong length, skipped
        ]
        out = s.vote_mechanism(uploads)
        np.testing.assert_array_equal(out, np.array([1, 0, 0], dtype=np.int64))

    def test_out_of_range_class_is_clipped(self) -> None:
        # num_classes=2 but a client (malformed) uploads class 7. The clip
        # to [0, num_classes-1] means it becomes class 1.
        s = _make_strategy(n_open=1, num_classes=2)
        uploads = [np.array([7], dtype=np.int64)]
        out = s.vote_mechanism(uploads)
        assert out[0] == 1

    def test_scales_to_twenty_seven_clients(self) -> None:
        # The paper's Scenario 1 size — make sure the vectorized path works
        # at the real scale we'll see in production.
        K = 27
        N = 50
        C = 11
        rng = np.random.default_rng(0)
        uploads = [rng.integers(low=-1, high=C, size=N).astype(np.int64)
                   for _ in range(K)]
        s = _make_strategy(n_open=N, num_classes=C, num_clients=K)
        out = s.vote_mechanism(uploads)
        assert out.shape == (N,)
        # Every output should either be -1 (all abstained) or a valid class.
        assert np.all((out == -1) | ((0 <= out) & (out < C)))


# ---------------------------------------------------------------------------
# 7.4 aggregate_fit
# ---------------------------------------------------------------------------
class TestAggregateFit:
    def _make_fit_res(self, hard_labels: np.ndarray, metrics: Dict[str, Any]) -> FakeFitRes:
        from flwr.common import ndarrays_to_parameters
        return FakeFitRes(
            parameters=ndarrays_to_parameters([hard_labels.astype(np.int64)]),
            num_examples=len(hard_labels),
            metrics=metrics,
        )

    def test_votes_and_updates_global_labels(self) -> None:
        from flwr.common import parameters_to_ndarrays
        s = _make_strategy(n_open=3, num_classes=2)
        results: List[Tuple[Any, FakeFitRes]] = [
            (FakeClientProxy(0), self._make_fit_res(
                np.array([0, 1, -1]),
                {"classifier_loss": 0.4, "n_familiar": 2, "n_unfamiliar": 1}
            )),
            (FakeClientProxy(1), self._make_fit_res(
                np.array([0, 1, 0]),
                {"classifier_loss": 0.2, "n_familiar": 3, "n_unfamiliar": 0}
            )),
        ]
        params, metrics = s.aggregate_fit(server_round=1, results=results, failures=[])
        # Global labels must reflect the vote (unanimous at positions 0, 1;
        # single vote at position 2 -> class 0).
        assert np.array_equal(s.global_labels, np.array([0, 1, 0], dtype=np.int64))
        # Returned parameters must match.
        rtn_arr = parameters_to_ndarrays(params)[0]
        np.testing.assert_array_equal(rtn_arr, s.global_labels)
        # Aggregated metrics include averaged losses and totals.
        assert metrics["avg_classifier_loss"] == pytest.approx(0.3)
        assert metrics["total_familiar"] == 5
        assert metrics["total_unfamiliar"] == 1
        assert metrics["valid_global_labels"] == 3
        assert metrics["round"] == 1
        assert len(s.round_metrics) == 1

    def test_no_results_returns_prior_labels_untouched(self) -> None:
        from flwr.common import parameters_to_ndarrays
        s = _make_strategy(n_open=3, num_classes=2)
        before = s.global_labels.copy()
        params, metrics = s.aggregate_fit(
            server_round=1, results=[], failures=[]
        )
        # State must not regress.
        np.testing.assert_array_equal(s.global_labels, before)
        np.testing.assert_array_equal(
            parameters_to_ndarrays(params)[0], before
        )
        # Metrics dict is empty in this branch.
        assert metrics == {}

    def test_eval_fn_notimplemented_is_caught(self) -> None:
        # In the starting infrastructure, server.build_eval_fn's closure
        # raises NotImplementedError. aggregate_fit must swallow it.
        def raising_eval(_round, _labels, _X):
            raise NotImplementedError("stub")
        s = SSFLStrategy(
            num_clients=1, num_classes=2, n_open_samples=2,
            min_fit_clients=1, min_available_clients=1,
            eval_fn=raising_eval,
        )
        result = s.aggregate_fit(
            server_round=1,
            results=[(FakeClientProxy(0),
                      self._make_fit_res(np.array([0, 1]), {}))],
            failures=[],
        )
        # No crash; strategy returned a valid (params, metrics) pair.
        assert result is not None


# ---------------------------------------------------------------------------
# 7.6 configure_evaluate
# ---------------------------------------------------------------------------
class TestConfigureEvaluate:
    def test_sends_to_all_clients_with_round_config(self) -> None:
        s = _make_strategy(n_open=4, num_classes=2, num_clients=3)
        fcm = FakeClientManager(n=3)
        tuples = s.configure_evaluate(
            server_round=7, parameters=None, client_manager=fcm  # type: ignore[arg-type]
        )
        assert len(tuples) == 3
        for _client, ins in tuples:
            assert int(ins.config["round"]) == 7


# ---------------------------------------------------------------------------
# 7.7 aggregate_evaluate
# ---------------------------------------------------------------------------
class TestAggregateEvaluate:
    def test_weighted_mean_of_loss_and_accuracy(self) -> None:
        s = _make_strategy()
        # Client A: loss=0.1, acc=1.0, 10 samples
        # Client B: loss=0.3, acc=0.0, 30 samples
        # Expected: loss = (0.1*10 + 0.3*30) / 40 = 0.25
        # Expected: acc  = (1.0*10 + 0.0*30) / 40 = 0.25
        results = [
            (FakeClientProxy(0), FakeEvaluateRes(
                loss=0.1, num_examples=10, metrics={"accuracy": 1.0}
            )),
            (FakeClientProxy(1), FakeEvaluateRes(
                loss=0.3, num_examples=30, metrics={"accuracy": 0.0}
            )),
        ]
        loss, metrics = s.aggregate_evaluate(server_round=1, results=results, failures=[])
        assert loss == pytest.approx(0.25)
        assert metrics["accuracy"] == pytest.approx(0.25)
        assert metrics["round"] == 1

    def test_empty_results_returns_none(self) -> None:
        s = _make_strategy()
        loss, metrics = s.aggregate_evaluate(server_round=1, results=[], failures=[])
        assert loss is None
        assert metrics == {}

    def test_zero_total_samples_returns_none(self) -> None:
        s = _make_strategy()
        results = [
            (FakeClientProxy(0), FakeEvaluateRes(loss=0.0, num_examples=0, metrics={})),
        ]
        loss, metrics = s.aggregate_evaluate(server_round=1, results=results, failures=[])
        assert loss is None
        assert metrics == {}


# ---------------------------------------------------------------------------
# Centralised evaluate — always None in our strategy (folded into eval_fn).
# ---------------------------------------------------------------------------
class TestStrategyEvaluate:
    def test_returns_none(self) -> None:
        s = _make_strategy()
        assert s.evaluate(server_round=1, parameters=None) is None  # type: ignore[arg-type]
