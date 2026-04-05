"""
Shared runtime configuration for SSFL.

This module holds the runtime config dict that is set by ``run_simulation.py``
before Flower's simulation engine starts. Both ``server_app.py`` and
``client_app.py`` read from this dict in their factory functions.

This is necessary because Flower 1.27's ``run_simulation()`` does not accept
a ``run_config`` parameter — config is normally read from ``pyproject.toml``.
We use this module-level dict to pass CLI arguments through instead.
"""

from __future__ import annotations

# Defaults match the paper's Section V-C
RUNTIME_CONFIG: dict[str, str | int | float] = {
    "scenario": 1,
    "num-rounds": 200,
    "learning-rate": 0.0001,
    "batch-size": 100,
    "local-epochs": 5,
    "seed": 42,
}
