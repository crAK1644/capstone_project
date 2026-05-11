# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

SSFL — a Flower-based reimplementation of Zhao et al. (2023) *"Semisupervised Federated-Learning-Based Intrusion Detection Method for Internet of Things"* (IEEE IoT Journal Vol. 10 No. 10). Trains a CNN intrusion detector across 27 simulated IoT clients (Scenario 1: 9 devices × K=3 clients) on the **N-BaIoT** dataset. **Scenario 1 only** — Scenarios 2 and 3 are out of scope on this branch.

The defining design choice: clients exchange **hard labels** (one int per open-set sample) instead of model weights. The Flower `Parameters` object is repurposed as a label carrier in both directions; the server runs majority voting, not FedAvg. See `docs/SSFL_FLOWER_INFRASTRUCTURE_PLAN.md` for the full plan.

## Commands

```bash
# Build partitions from raw N-BaIoT (writes data/partitions/*.pkl from data/raw/*.csv)
python src/ssfl_project/data_preparation.py --raw_dir data/raw --output_dir data/partitions

# Run everything locally: 1 server + N client subprocesses, logs under logs/
python src/ssfl_project/launch.py --num_clients 27 --num_rounds 150 --device cpu

# Run a single role manually (useful for debugging one client in a terminal)
python src/ssfl_project/main.py --mode server --num_clients 27 --num_rounds 150
python src/ssfl_project/main.py --mode client --client_id 0

# Tests — note `testpaths = .` in tests/pytest.ini, so cd into tests/
cd tests && pytest                              # full suite, also writes docs/test_results.md
cd tests && pytest -k vote                      # filter by name
cd tests && pytest test_strategy.py             # single file
cd tests && pytest test_strategy.py::test_majority_vote_basic   # single test
python tests/run_tests.py                       # wrapper that pins CWD to repo root
```

Markers registered in `tests/pytest.ini`: `integration`, `cnn_dependent` (skipped until `model.py` ships a real `TrafficCNN`), `flwr`, `torch`, `sklearn`. Use `pytest -m "not cnn_dependent"` to skip model-dependent tests.

There is no `pyproject.toml`, `requirements.txt`, or `uv.lock` on this branch. Runtime deps are `torch`, `flwr`, `pandas`, `numpy`; tests additionally need `pytest` and `scikit-learn`. Install in whichever environment you use.

## Architecture

`src/ssfl_project/` is a **flat package with implicit imports** — modules use bare `import config`, `from client import ...` rather than `from ssfl_project import ...`. To make these resolve, scripts are run *as files* (not `-m`), and `src/ssfl_project/` ends up on `sys.path` via the launching script's location. Don't rewrite these as package-relative imports without also fixing every entry point.

`config.py` is the **single source of truth** for hyperparameters, paths, and dataset constants. `main.parse_arguments()` wires every CLI default back to a `config.*` constant. Change a number in one place, not both.

Round flow (mirrors paper Algorithm 1):

| Step | Where | What |
|---|---|---|
| 1. Train classifier | `client.py` (called from `SSFLClient.fit`) | CE on private labeled shards |
| 2. Train discriminator | `client.py` | 2-class CNN: familiar vs unfamiliar open-set samples |
| 3. Filter & upload | `client.py` → `Parameters` | per-open-sample hard label (-1 = "unfamiliar / abstain") |
| 4. **Vote & broadcast** | `strategy.py::SSFLStrategy.aggregate_fit` | per-sample majority across clients → global labels |
| 5. Distillation | `client.py` | classifier fine-tuned on open data with voted labels as teacher |

`SSFLStrategy` is the heart of the server. It has no CNN dependency and can be unit-tested in isolation; tests under `tests/test_strategy.py` exercise the voting math directly. It also maintains a `CommCostLedger` (paper Table IV byte accounting) and a `round_metrics` list that `main.run_server` persists at shutdown.

`server.py::build_eval_fn` produces the closure that scores the global model on `X_test` each round. It is **stubbed** until `model.py` has a real CNN — strategy/voting code is fully wired but end-to-end accuracy numbers won't be meaningful until then. `cnn_dependent`-marked tests will start passing at that point.

`main.run_server` writes four artefacts under `--metrics_dir` (default `metrics/`) plus Flower's native `history.json` under `--logs_dir` (default `logs/`): `per_round.json`, `per_round.csv`, `summary.json` (Tables II–IV cells, see `metrics.build_summary_report`), and `confusion_matrix_final.json`. The summary protocol is documented in `docs/SSFL_FLOWER_INFRASTRUCTURE_PLAN.md` §14 and the constants live in `config.SNAPSHOT_ROUNDS`, `config.TARGET_ACCURACIES`.

## Data Layout — two parallel trees

There are two on-disk dataset representations and they are **not** interchangeable:

- `data/raw/*.csv` → `data/partitions/*.pkl`: written and consumed by `data_preparation.py`. This is the path the runtime (`main.py`, `client.py`, `server.py`) actually uses via `config.PARTITION_DIR`. Build it before running anything.
- `prepared_data/{private,open,test,scenario_1}/*.npy`: a pre-built `.npy` snapshot tracked in git from the earlier branch. The current code does **not** read this directory. Treat it as reference material or a fallback source for `data/raw/` regeneration, not a runtime input.

The open set's labels exist in both trees but must **never** be used during training — they are the held-out signal that voting reconstructs.

## Repo Hygiene Gotchas

- No `.gitignore` is committed. `__pycache__/*.pyc` files are tracked in both `src/ssfl_project/` and `tests/`. Don't add to that — and if you touch a Python file, watch that you don't accidentally commit a refreshed `.pyc`. Suggest adding a `.gitignore` if the user is doing cleanup work.
- `tests/pytest.ini` uses `testpaths = .`, so pytest must be invoked from inside `tests/` (or via `tests/run_tests.py`, which handles CWD). Running `pytest` from the repo root will not discover anything.
- `tests/conftest.py` writes `docs/test_results.md` at session end. Expect that file to change after every test run; don't treat its diff as a code change.
