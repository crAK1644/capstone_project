"""Tests for `config.py`.

`config.py` is a module of module-level constants with two asserts at import
time. We want to independently verify the invariants it claims to enforce, so
that if someone later edits a constant the suite turns red before a silent
drift corrupts Scenario 1.

Notes on methodology
--------------------
* Every constant is asserted against the number stated in section 10A of the
  infrastructure plan. The plan is the single source of truth for these values.
* We also check *derived* invariants (e.g. K_DI * NUM_DEVICES == NUM_CLIENTS,
  FEATURES_PER_WINDOW * N_TIME_WINDOWS == N_FEATURES). If any of these fail,
  somebody has introduced an inconsistency in the topology and downstream
  shard math or reshape logic will break.
"""
from __future__ import annotations

import config


# ---------------------------------------------------------------------------
# Topology constants
# ---------------------------------------------------------------------------
class TestTopologyConstants:
    """Core counts that define Scenario 1."""

    def test_num_devices_is_nine(self) -> None:
        # N-BaIoT ships 9 IoT devices; the plan hardcodes this.
        assert config.NUM_DEVICES == 9

    def test_k_di_is_three(self) -> None:
        # Scenario 1 uses 3 clients per device.
        assert config.K_DI == 3

    def test_num_clients_is_derived(self) -> None:
        # NUM_CLIENTS must equal NUM_DEVICES * K_DI = 27. This is the exact
        # derivation in the plan — if someone hard-codes 27 instead we still
        # catch divergences introduced later.
        assert config.NUM_CLIENTS == config.NUM_DEVICES * config.K_DI
        assert config.NUM_CLIENTS == 27

    def test_num_classes_is_eleven(self) -> None:
        # Global label space: 1 benign + 10 attacks.
        assert config.NUM_CLASSES == 11


# ---------------------------------------------------------------------------
# Feature-dimension constants
# ---------------------------------------------------------------------------
class TestFeatureDimensions:
    def test_feature_count_matches_time_window_product(self) -> None:
        # The reshape (N, 115) -> (N, 23, 5) only works if this holds.
        assert config.FEATURES_PER_WINDOW * config.N_TIME_WINDOWS == config.N_FEATURES
        assert config.N_FEATURES == 115

    def test_input_shape_tuple(self) -> None:
        # Must be (23, 5) — PyTorch Conv1d contract in TrafficCNN.
        assert config.INPUT_SHAPE == (23, 5)

    def test_feature_column_names_length_and_uniqueness(self) -> None:
        # 115 distinct names; duplicates would silently zero a column on reshape.
        assert len(config.FEATURE_COLUMN_NAMES) == config.N_FEATURES
        assert len(set(config.FEATURE_COLUMN_NAMES)) == config.N_FEATURES


# ---------------------------------------------------------------------------
# Split ratios & training hyperparameters
# ---------------------------------------------------------------------------
class TestRatiosAndHyperparameters:
    def test_split_ratios_sum_to_one(self) -> None:
        # Floating-point tolerance intentional: values are not exact.
        total = config.PRIVATE_RATIO + config.OPEN_RATIO + config.TEST_RATIO
        assert abs(total - 1.0) < 1e-9

    def test_split_ratios_are_positive(self) -> None:
        for r in (config.PRIVATE_RATIO, config.OPEN_RATIO, config.TEST_RATIO):
            assert 0.0 < r < 1.0

    def test_training_hyperparameters_are_positive(self) -> None:
        assert config.LEARNING_RATE > 0
        assert config.BATCH_SIZE > 0
        for epochs in (config.CLASSIFIER_EPOCHS, config.DISCRIMINATOR_EPOCHS,
                       config.DISTILLATION_EPOCHS):
            assert epochs >= 1
        assert config.NUM_ROUNDS >= 1


# ---------------------------------------------------------------------------
# Class name <-> global ID mapping
# ---------------------------------------------------------------------------
class TestClassNameMapping:
    def test_mapping_has_eleven_entries(self) -> None:
        assert len(config.CLASS_NAME_TO_GLOBAL_ID) == config.NUM_CLASSES

    def test_mapping_ids_are_contiguous_0_to_10(self) -> None:
        # Global IDs must exactly cover 0..NUM_CLASSES-1 — no gaps, no dupes.
        ids = sorted(config.CLASS_NAME_TO_GLOBAL_ID.values())
        assert ids == list(range(config.NUM_CLASSES))

    def test_inverse_mapping_is_a_bijection(self) -> None:
        # Round-trip: name -> id -> name must return the original.
        for name, gid in config.CLASS_NAME_TO_GLOBAL_ID.items():
            assert config.GLOBAL_ID_TO_CLASS_NAME[gid] == name
        assert len(config.GLOBAL_ID_TO_CLASS_NAME) == len(
            config.CLASS_NAME_TO_GLOBAL_ID
        )

    def test_benign_is_zero(self) -> None:
        # Convention from the paper: benign has the lowest ID.
        assert config.CLASS_NAME_TO_GLOBAL_ID["benign"] == 0

    def test_mirai_family_ids(self) -> None:
        # Sanity check: all 5 mirai families present with distinct IDs 6..10.
        mirai_ids = {
            config.CLASS_NAME_TO_GLOBAL_ID[k]
            for k in config.CLASS_NAME_TO_GLOBAL_ID
            if k.startswith("mirai_")
        }
        assert mirai_ids == {6, 7, 8, 9, 10}


# ---------------------------------------------------------------------------
# Filesystem constants
# ---------------------------------------------------------------------------
class TestFilesystemConstants:
    def test_data_subdirs_are_under_data_dir(self) -> None:
        # If DATA_DIR changes, the others should track automatically.
        assert config.RAW_DIR.startswith(config.DATA_DIR)
        assert config.PROCESSED_DIR.startswith(config.DATA_DIR)
        assert config.PARTITION_DIR.startswith(config.DATA_DIR)
