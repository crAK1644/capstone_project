"""Tests for `server.py`.

Two live entry points:

* `build_eval_fn` — factory that returns a closure. The factory itself is
  live (even in pass 1), so it should succeed and produce a callable. The
  closure body is stubbed (NotImplementedError) until the CNN arrives.
* `start_server` — actually launches a gRPC server. We don't test it
  directly (it would block on a port); we document that as a known gap.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.flwr, pytest.mark.torch]

import torch  # noqa: E402
from server import build_eval_fn  # noqa: E402


class TestBuildEvalFn:
    def test_factory_returns_callable(self) -> None:
        # The factory must succeed independent of the CNN. Its return value
        # is what the strategy tries to call after every round.
        X_test = np.zeros((3, 23, 5), dtype=np.float32)
        y_test = np.array([0, 1, 2], dtype=np.int64)
        fn = build_eval_fn(X_test, y_test, num_classes=3, device=torch.device("cpu"))
        assert callable(fn)

    def test_closure_body_is_stubbed(self) -> None:
        # Invoking the closure in pass 1 should raise NotImplementedError.
        # The strategy is designed to catch this, so it must stay a raise.
        X_test = np.zeros((2, 23, 5), dtype=np.float32)
        y_test = np.array([0, 1], dtype=np.int64)
        fn = build_eval_fn(X_test, y_test, num_classes=2, device=torch.device("cpu"))
        with pytest.raises(NotImplementedError):
            fn(server_round=1, global_labels=np.array([-1, -1]), X_open=None)
