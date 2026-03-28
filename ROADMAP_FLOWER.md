# Flower Implementation Roadmap

## Current Baseline (Done)

- Modular package structure under src/
- CNN model for (1, 23, 5) input
- Flower FedAvg client/server wiring
- Scenario-based experiment entrypoints
- Optional SSFL consistency hook using open split

## Next Milestones

1. Experiment tracking
   - Persist round metrics to JSON/CSV and TensorBoard
   - Add per-device and per-class reporting

2. Stronger SSFL
   - Add EMA teacher model and confidence calibration
   - Support consistency across augmentations

3. Better robustness for non-IID
   - Add FedProx strategy variant
   - Add client sampling schedules by scenario

4. Reproducibility and testing
   - Add unit tests for loaders, metrics, and parameter exchange
   - Add small integration smoke test for 2 rounds

5. Checkpoint and resume
   - Save global model each round
   - Resume from checkpoint for long runs
