# Capstone Project: Federated IoT Intrusion Detection with Flower

This project implements a modular federated learning pipeline for IoT intrusion detection on mini-N-BaIoT style data.

The codebase contains:
1. A complete dataset preparation pipeline.
2. A Flower-based federated baseline (FedAvg).
3. An SSFL-ready extension path using open-set pseudo-label consistency.

## Repository Purpose

The main goal is to run and compare federated experiments across three non-IID scenarios using a consistent model, data contract, and evaluation process.

## High-Level Workflow

1. Build prepared dataset artifacts from raw CSVs.
2. Select a scenario (1, 2, or 3).
3. Start Flower simulation.
4. Each client trains locally on private data.
5. Server aggregates updates with FedAvg.
6. Server evaluates on global test split each round.
7. Optionally include SSFL consistency update using open split.

## Project Structure

```text
.
├── main.py
├── prepare_dataset.py
├── DATASET_README.md
├── ROADMAP_FLOWER.md
├── pyproject.toml
├── prepared_data/
└── src/
	├── __init__.py
	├── data/
	│   ├── __init__.py
	│   └── loaders.py
	├── experiments/
	│   ├── __init__.py
	│   ├── run_baseline.py
	│   └── run_ssfl.py
	├── fl/
	│   ├── __init__.py
	│   ├── client.py
	│   └── server.py
	├── models/
	│   ├── __init__.py
	│   └── cnn.py
	└── utils/
		├── __init__.py
		├── config.py
		├── metrics.py
		└── training.py
```

## Module Guide

### Entry Points

1. [main.py](main.py)
Purpose: Lightweight launcher that dispatches to baseline or ssfl mode.

2. [prepare_dataset.py](prepare_dataset.py)
Purpose: End-to-end dataset preparation from raw CSV to federated-ready splits and client partitions.

### Dataset and Preparation

1. [prepare_dataset.py](prepare_dataset.py)
Purpose: Core data pipeline.
What it does:
- Samples mini-N-BaIoT (1000 samples per device-class pair).
- Splits into private/open/test.
- Applies min-max normalization fit on private data.
- Reshapes flat 115 features to CNN format (N, 1, 23, 5).
- Builds scenario-specific client partitions.
- Saves all artifacts under prepared_data.

2. [DATASET_README.md](DATASET_README.md)
Purpose: Dataset contract, assumptions, and usage reference.

### Federated Data Adapters

1. [src/data/loaders.py](src/data/loaders.py)
Purpose: Bridge prepared_data artifacts to PyTorch DataLoader objects.
Key functions:
- get_client_ids: reads client IDs from scenario summary.
- make_client_loaders: creates train and validation loaders for a client.
- make_test_loader: global test loader used by server evaluation.
- make_open_loader: open-split loader used for SSFL consistency path.

2. [src/data/__init__.py](src/data/__init__.py)
Purpose: Public import surface for data loader helpers.

### Model

1. [src/models/cnn.py](src/models/cnn.py)
Purpose: Compact CNN classifier for intrusion category prediction.
Input: (N, 1, 23, 5)
Output: logits over 11 classes.

2. [src/models/__init__.py](src/models/__init__.py)
Purpose: Exposes model class for clean imports.

### Flower Client and Server

1. [src/fl/client.py](src/fl/client.py)
Purpose: Defines Flower client behavior.
What it does:
- Implements parameter get/set utilities.
- Builds local model instance.
- Runs supervised local training in fit.
- Runs optional SSFL consistency step on open split.
- Returns training metrics and updated parameters.

2. [src/fl/server.py](src/fl/server.py)
Purpose: Defines server strategy and centralized evaluation.
What it does:
- Configures FedAvg strategy.
- Sets client sampling parameters.
- Evaluates global model on test set each round.

3. [src/fl/__init__.py](src/fl/__init__.py)
Purpose: Exposes client and server builder APIs.

### Experiment Runners

1. [src/experiments/run_baseline.py](src/experiments/run_baseline.py)
Purpose: Starts supervised FedAvg experiment.

2. [src/experiments/run_ssfl.py](src/experiments/run_ssfl.py)
Purpose: Starts FedAvg experiment with SSFL consistency enabled.

3. [src/experiments/__init__.py](src/experiments/__init__.py)
Purpose: Experiment package marker.

### Utilities

1. [src/utils/config.py](src/utils/config.py)
Purpose: Central run configuration dataclass used by clients and server.

2. [src/utils/training.py](src/utils/training.py)
Purpose: Training and evaluation routines.
Key functions:
- train_local: supervised local epochs.
- train_open_set_consistency: pseudo-label consistency update for SSFL.
- evaluate_classifier: computes loss and metrics on a loader.

3. [src/utils/metrics.py](src/utils/metrics.py)
Purpose: Core metric helpers.
Key functions:
- confusion_matrix
- macro_f1_from_confusion

4. [src/utils/__init__.py](src/utils/__init__.py)
Purpose: Utility package marker.

### Planning and Documentation

1. [ROADMAP_FLOWER.md](ROADMAP_FLOWER.md)
Purpose: Follow-up roadmap for expanding baseline to stronger SSFL and production-ready experimentation.

2. [README.md](README.md)
Purpose: Comprehensive project guide.

## How to Run

## 1. Environment Setup

Use one environment consistently. Recommended: project venv with uv.

```bash
uv sync
```

If needed, run through local venv Python directly:

```bash
.venv/bin/python -m pip install -e .
```

## 2. Prepare Data (if not already prepared)

```bash
uv run python prepare_dataset.py
```

Expected output: prepared_data directory with private/open/test and scenario_1..3 client folders.

## 3. Run Baseline Federated Learning

### Direct experiment module

```bash
uv run python -m src.experiments.run_baseline --scenario 1 --rounds 10
```

### Via main launcher

```bash
uv run python main.py baseline --scenario 1 --rounds 10
```

## 4. Run SSFL Variant

### Direct experiment module

```bash
uv run python -m src.experiments.run_ssfl --scenario 1 --rounds 10 --ssfl-lambda 0.2 --ssfl-threshold 0.9
```

### Via main launcher

```bash
uv run python main.py ssfl --scenario 1 --rounds 10 --ssfl-lambda 0.2 --ssfl-threshold 0.9
```

## Command Reference

### Common baseline arguments

- --scenario: 1, 2, or 3
- --rounds: number of federated rounds
- --local-epochs: local epochs per selected client
- --batch-size: local batch size
- --lr: learning rate
- --fraction-fit: fraction of clients sampled for fit each round
- --fraction-evaluate: fraction of clients sampled for evaluate
- --min-fit-clients: lower bound for fit client count
- --min-evaluate-clients: lower bound for evaluate client count
- --seed: run seed

### SSFL-only extra arguments

- --ssfl-lambda: weight of open-set consistency loss
- --ssfl-threshold: confidence threshold for pseudo-label acceptance

## Scenario Matrix

1. Scenario 1
- 27 clients
- K=3 per device (shard split)
- Good for initial debugging and fast baseline checks

2. Scenario 2
- 89 clients
- K=L per device (shard split)
- Stronger heterogeneity than scenario 1

3. Scenario 3
- 89 clients
- K=L per device (Dirichlet alpha=0.1)
- Most non-IID and generally hardest

## What Happens During Training

1. Server initializes global model.
2. Flower samples a subset of clients.
3. Each sampled client:
- Loads its local private data.
- Trains model locally.
- Optionally runs SSFL consistency update on open data.
- Returns updated parameters and metrics.
4. Server aggregates updates with FedAvg.
5. Server evaluates global model on test split.
6. Process repeats for configured rounds.

## Interpreting Output

Current runs print Flower history including:
- Centralized test loss per round.
- Centralized test accuracy per round.
- Centralized test macro F1 per round.

Typical first-round behavior:
- low accuracy and macro F1 at initialization.
- gradual improvement over early rounds.

## Current Limitations

1. Flower deprecation warnings may appear for simulation API and client_fn signature. Current implementation is functional, migration to newer Flower app/context API is planned.
2. Metrics are printed to console; persistent experiment logging is planned.
3. SSFL implementation is a practical baseline consistency method, not yet teacher-student distilled variant from full paper reproduction.

## Troubleshooting

1. Import errors for torch or flwr
- Ensure VS Code is using the workspace venv interpreter.
- Run uv sync again.

2. Missing prepared_data artifacts
- Run prepare_dataset script first.

3. Slow or memory-heavy run
- Lower fraction-fit.
- Lower batch-size.
- Use scenario 1 for smoke tests.

4. No metric improvement
- Increase rounds.
- Increase local-epochs cautiously.
- Verify scenario selection and dataset availability.

## Suggested First Runs

1. Smoke test

```bash
uv run python -m src.experiments.run_baseline --scenario 1 --rounds 1 --min-fit-clients 2 --min-evaluate-clients 2 --fraction-fit 0.2 --fraction-evaluate 0.2
```

2. Baseline sanity run

```bash
uv run python -m src.experiments.run_baseline --scenario 1 --rounds 10
```

3. SSFL comparison run

```bash
uv run python -m src.experiments.run_ssfl --scenario 1 --rounds 10 --ssfl-lambda 0.2 --ssfl-threshold 0.9
```