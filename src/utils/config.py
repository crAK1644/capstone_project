from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class RunConfig:
    scenario: int = 1
    seed: int = 42
    rounds: int = 10
    local_epochs: int = 1
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_ratio: float = 0.1
    fraction_fit: float = 0.3
    fraction_evaluate: float = 0.3
    min_fit_clients: int = 5
    min_evaluate_clients: int = 5
    min_available_clients: int = 10
    num_cpus_per_client: float = 1.0
    num_gpus_per_client: float = 0.0
    ssfl_enabled: bool = False
    ssfl_lambda: float = 0.2
    ssfl_confidence_threshold: float = 0.9
    base_data_dir: Path | None = None

    @property
    def prepared_data_dir(self) -> Path | None:
        return self.base_data_dir
