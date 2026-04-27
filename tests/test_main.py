"""Tests for `main.py`.

`main.py` has three live entry points worth testing:

* `parse_arguments` — CLI contract; defaults must match `config.py` so the
  two sources don't drift.
* `setup_environment` — seeds numpy / torch / stdlib random; must be
  reproducible across invocations.
* `main` itself is just routing — covered indirectly by the above.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

pytestmark = [pytest.mark.flwr, pytest.mark.torch]

import config  # noqa: E402
import main as ssfl_main  # noqa: E402


# ---------------------------------------------------------------------------
# parse_arguments
# ---------------------------------------------------------------------------
class TestParseArguments:
    def test_server_mode_defaults_match_config(self, monkeypatch) -> None:
        # Simulate `python main.py --mode server` and verify every default
        # lines up with `config.py`. Otherwise CLI and library silently
        # diverge and experiments become irreproducible.
        monkeypatch.setattr("sys.argv", ["main.py", "--mode", "server"])
        args = ssfl_main.parse_arguments()
        assert args.mode == "server"
        assert args.server_address == config.DEFAULT_SERVER_ADDRESS
        assert args.num_rounds == config.NUM_ROUNDS
        assert args.num_clients == config.NUM_CLIENTS
        assert args.num_classes == config.NUM_CLASSES
        assert args.learning_rate == config.LEARNING_RATE
        assert args.batch_size == config.BATCH_SIZE
        assert args.classifier_epochs == config.CLASSIFIER_EPOCHS
        assert args.discriminator_epochs == config.DISCRIMINATOR_EPOCHS
        assert args.partition_dir == config.PARTITION_DIR
        assert args.seed == config.RANDOM_SEED

    def test_client_mode_accepts_client_id(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "sys.argv", ["main.py", "--mode", "client", "--client_id", "17"]
        )
        args = ssfl_main.parse_arguments()
        assert args.mode == "client"
        assert args.client_id == 17

    def test_mode_is_required(self, monkeypatch) -> None:
        monkeypatch.setattr("sys.argv", ["main.py"])
        # argparse exits on missing required argument — SystemExit is expected.
        with pytest.raises(SystemExit):
            ssfl_main.parse_arguments()

    def test_invalid_mode_rejected(self, monkeypatch) -> None:
        monkeypatch.setattr("sys.argv", ["main.py", "--mode", "bogus"])
        with pytest.raises(SystemExit):
            ssfl_main.parse_arguments()


# ---------------------------------------------------------------------------
# setup_environment
# ---------------------------------------------------------------------------
class TestSetupEnvironment:
    def test_seeds_are_reproducible(self) -> None:
        # Call twice with the same seed -> identical RNG outputs from numpy,
        # torch, and stdlib random. If this breaks, the federated run is not
        # reproducible.
        import torch

        ssfl_main.setup_environment(seed=123)
        a_np = np.random.rand(3)
        a_py = random.random()
        a_torch = torch.rand(3).tolist()

        ssfl_main.setup_environment(seed=123)
        b_np = np.random.rand(3)
        b_py = random.random()
        b_torch = torch.rand(3).tolist()

        np.testing.assert_allclose(a_np, b_np)
        assert a_py == b_py
        assert a_torch == b_torch

    def test_different_seeds_diverge(self) -> None:
        # Regression guard: setup_environment must actually *use* the seed
        # (nothing hidden is overriding it). Different seeds -> different
        # numpy draws.
        ssfl_main.setup_environment(seed=1)
        a = np.random.rand(3)
        ssfl_main.setup_environment(seed=2)
        b = np.random.rand(3)
        assert not np.allclose(a, b)
