"""Tests for `data_preparation.py`.

This module is the fully-live data pipeline — CSV ingestion, min-max
normalization, (23, 5) reshape, stratified split, Scenario 1 shard
partitioning, pickle I/O, and the PyTorch DataLoader factory. Every function
is a concrete target with testable invariants.

We split tests into three buckets:

* **Pure-function tests** that work on in-memory numpy/pandas objects
  (`reshape_sample_to_2d`, `normalize_features`, `apply_normalization`,
  `partition_scenario1`, etc.). These are the fastest and don't touch the
  filesystem.
* **Tree-based tests** that build a synthetic N-BaIoT CSV directory via the
  `synthetic_nbaiot_tree` fixture, then drive the CSV-level functions
  (`load_device_csvs`, `build_mini_nbaiot`).
* **Pickle round-trip tests** that use the `partitioned_tmp_dir` fixture to
  exercise `save_partitions` + the three loaders on real pickle files.

The `torch` marker is applied to the whole module because `data_preparation`
imports torch at the top for `DataLoader` — without torch the module itself
is unimportable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.torch

import config
import data_preparation as dp


# ---------------------------------------------------------------------------
# 4.0 _extract_class_name (helper; worth covering because its normalization
#     rules silently affect every CSV loaded from disk)
# ---------------------------------------------------------------------------
class TestExtractClassName:
    @pytest.mark.parametrize(
        "csv_path,expected",
        [
            ("data/raw/Dev_A/benign.csv", "benign"),
            ("data/raw/Dev_A/mirai.udp.csv", "mirai_udp"),
            ("data/raw/Dev_A/gafgyt-combo.csv", "gafgyt_combo"),
            ("data/raw/Dev_A/mirai_udp_1.csv", "mirai_udp"),   # trailing int stripped
            ("data/raw/Dev_A/MIRAI_UDP.csv", "mirai_udp"),     # case-insensitive
            ("data/raw/Dev_A/__gafgyt_tcp__.csv", "gafgyt_tcp"),  # trimmed underscores
        ],
    )
    def test_normalization_rules(self, csv_path: str, expected: str) -> None:
        assert dp._extract_class_name(csv_path) == expected


# ---------------------------------------------------------------------------
# 4.1 load_device_csvs
# ---------------------------------------------------------------------------
class TestLoadDeviceCsvs:
    def test_loads_and_labels_every_row(self, synthetic_nbaiot_tree: Path) -> None:
        # Synthetic tree contains 2 classes × 120 rows in Device_A.
        dev_a = str(synthetic_nbaiot_tree / "Device_A")
        df = dp.load_device_csvs(dev_a)
        assert len(df) == 240
        # Labels must come from the config map — benign=0, gafgyt_combo=1.
        assert set(df["label"].unique()) == {0, 1}
        # Feature columns should be preserved exactly.
        for col in config.FEATURE_COLUMN_NAMES[:5]:  # spot-check a few
            assert col in df.columns

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        # Guardrail: empty folder means misconfigured path, not silent empty df.
        empty = tmp_path / "empty_dev"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            dp.load_device_csvs(str(empty))

    def test_unknown_class_filename_raises_keyerror(self, tmp_path: Path) -> None:
        # If a CSV filename doesn't map into CLASS_NAME_TO_GLOBAL_ID, we must
        # surface a loud error — silent mislabeling would poison training.
        dev = tmp_path / "Dev"
        dev.mkdir()
        # 115 columns of zeros in one row
        pd.DataFrame(
            [[0.0] * 115], columns=config.FEATURE_COLUMN_NAMES
        ).to_csv(dev / "definitely_not_a_class.csv", index=False)
        with pytest.raises(KeyError):
            dp.load_device_csvs(str(dev))


# ---------------------------------------------------------------------------
# 4.2 build_mini_nbaiot
# ---------------------------------------------------------------------------
class TestBuildMiniNbaiot:
    def test_samples_per_class_are_respected(
        self, synthetic_nbaiot_tree: Path
    ) -> None:
        # With 2 devices × 2 classes each and samples_per_class=50, we expect
        # 2*2*50 = 200 rows in the mini set.
        mini = dp.build_mini_nbaiot(
            raw_data_dir=str(synthetic_nbaiot_tree), samples_per_class=50
        )
        assert len(mini) == 200
        # device_id column should be dense integer 0..1
        assert set(mini["device_id"].unique()) == {0, 1}
        # Each (device, label) group has exactly 50 rows.
        counts = mini.groupby(["device_id", "label"]).size()
        assert (counts == 50).all()

    def test_empty_raw_dir_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty_raw"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            dp.build_mini_nbaiot(raw_data_dir=str(empty))

    def test_sample_with_replacement_when_source_is_small(
        self, synthetic_nbaiot_tree: Path
    ) -> None:
        # Ask for more rows than a single class provides (each class has 120
        # rows synthetically) — sampling must fall back to replace=True.
        mini = dp.build_mini_nbaiot(
            raw_data_dir=str(synthetic_nbaiot_tree), samples_per_class=500
        )
        # 2 devices * 2 classes * 500 = 2000
        assert len(mini) == 2000


# ---------------------------------------------------------------------------
# 4.3 normalize_features & apply_normalization
# ---------------------------------------------------------------------------
class TestNormalization:
    def _toy_df(self, feature_col_names: List[str]) -> pd.DataFrame:
        # Keep it tiny but cover edge cases: a constant column (all zeros) to
        # exercise the divide-by-zero guard.
        n_rows = 10
        n_cols = len(feature_col_names)
        data = np.arange(n_rows * n_cols, dtype=np.float32).reshape(n_rows, n_cols)
        df = pd.DataFrame(data, columns=feature_col_names)
        # Force first column to be constant to hit the zero-range branch.
        df[feature_col_names[0]] = 7.5
        return df

    def test_output_is_in_unit_interval(
        self, feature_column_names: List[str]
    ) -> None:
        df = self._toy_df(feature_column_names)
        norm_df, params = dp.normalize_features(df, feature_column_names)
        vals = norm_df[feature_column_names].to_numpy()
        assert np.all(vals >= 0.0)
        assert np.all(vals <= 1.0 + 1e-6)
        assert set(params.keys()) == {"min", "max"}

    def test_zero_range_column_is_not_nan(
        self, feature_column_names: List[str]
    ) -> None:
        # Constant columns should become 0 after normalization, not NaN.
        df = self._toy_df(feature_column_names)
        norm_df, _ = dp.normalize_features(df, feature_column_names)
        assert not norm_df[feature_column_names].isna().any().any()

    def test_apply_normalization_matches_original_min_max(
        self, feature_column_names: List[str]
    ) -> None:
        # Running apply_normalization with the params returned by
        # normalize_features on the *same* data must reproduce the normalized
        # DataFrame exactly.
        df = self._toy_df(feature_column_names)
        norm_df, params = dp.normalize_features(df, feature_column_names)
        reapplied = dp.apply_normalization(df.copy(), feature_column_names, params)
        np.testing.assert_allclose(
            norm_df[feature_column_names].to_numpy(),
            reapplied[feature_column_names].to_numpy(),
            rtol=1e-6,
        )

    def test_input_df_is_not_mutated(
        self, feature_column_names: List[str]
    ) -> None:
        # normalize_features creates a deep copy — the caller should keep the
        # raw values. If this breaks, test/train pipelines would silently
        # share mutated buffers.
        df = self._toy_df(feature_column_names)
        snapshot = df.copy()
        dp.normalize_features(df, feature_column_names)
        pd.testing.assert_frame_equal(df, snapshot)


# ---------------------------------------------------------------------------
# 4.4 reshape_sample_to_2d
# ---------------------------------------------------------------------------
class TestReshapeSampleTo2D:
    def test_output_shape_is_23_by_5(self) -> None:
        flat = np.arange(115, dtype=np.float32)
        m = dp.reshape_sample_to_2d(flat)
        assert m.shape == (23, 5)
        assert m.dtype == np.float32

    def test_index_mapping_matches_plan(self) -> None:
        # Plan §4.4: flat index j*23+k -> matrix[k, j]. Construct a flat vector
        # where each element is j*23+k and verify that matrix[k, j] == j*23+k.
        flat = np.arange(115, dtype=np.float32)
        m = dp.reshape_sample_to_2d(flat)
        for j in range(5):
            for k in range(23):
                assert m[k, j] == j * 23 + k


# ---------------------------------------------------------------------------
# 4.5 apply_2d_reshape_to_dataset
# ---------------------------------------------------------------------------
class TestApply2DReshapeToDataset:
    def test_returns_correct_shapes_with_labels(
        self, feature_column_names: List[str]
    ) -> None:
        df = pd.DataFrame(
            np.random.default_rng(0).standard_normal((7, 115)),
            columns=feature_column_names,
        )
        df["label"] = np.arange(7, dtype=np.int64) % 3
        X, y = dp.apply_2d_reshape_to_dataset(df, feature_column_names)
        assert X.shape == (7, 23, 5)
        assert X.dtype == np.float32
        assert y.shape == (7,)
        assert y.dtype == np.int64
        np.testing.assert_array_equal(y, df["label"].to_numpy(dtype=np.int64))

    def test_returns_minus1_labels_when_label_missing(
        self, feature_column_names: List[str]
    ) -> None:
        # Open data has no label column — per the plan, y must be all -1.
        df = pd.DataFrame(
            np.random.default_rng(1).standard_normal((3, 115)),
            columns=feature_column_names,
        )
        X, y = dp.apply_2d_reshape_to_dataset(df, feature_column_names)
        assert X.shape == (3, 23, 5)
        assert np.all(y == -1)


# ---------------------------------------------------------------------------
# 4.6 split_private_open_test
# ---------------------------------------------------------------------------
class TestSplitPrivateOpenTest:
    def test_ratios_sum_and_disjoint(self, tiny_mini_df: pd.DataFrame) -> None:
        # Splits must cover all rows and be disjoint in their original index.
        before = len(tiny_mini_df)
        priv, openn, testn = dp.split_private_open_test(tiny_mini_df)
        assert len(priv) + len(openn) + len(testn) == before

    def test_open_split_drops_label_column(
        self, tiny_mini_df: pd.DataFrame
    ) -> None:
        # By design — open data is unlabeled.
        _priv, openn, _test = dp.split_private_open_test(tiny_mini_df)
        assert "label" not in openn.columns

    def test_stratification_preserves_every_group(
        self, tiny_mini_df: pd.DataFrame
    ) -> None:
        # Every (device_id, label) pair must appear in at least the private
        # split (the largest). Otherwise shard partitioning will starve a
        # client of that class later.
        priv, _openn, _test = dp.split_private_open_test(tiny_mini_df)
        src_groups = set(
            map(tuple, tiny_mini_df[["device_id", "label"]].drop_duplicates().to_numpy())
        )
        priv_groups = set(
            map(tuple, priv[["device_id", "label"]].drop_duplicates().to_numpy())
        )
        assert src_groups == priv_groups

    def test_bad_ratios_raise(self, tiny_mini_df: pd.DataFrame) -> None:
        # Guard against silent logic errors.
        with pytest.raises(ValueError):
            dp.split_private_open_test(
                tiny_mini_df, private_ratio=0.5, open_ratio=0.2, test_ratio=0.4
            )

    def test_determinism_with_same_seed(
        self, tiny_mini_df: pd.DataFrame
    ) -> None:
        # Reproducibility: two calls with the same seed must return identical
        # private splits. If not, training runs become non-reproducible.
        p1, _, _ = dp.split_private_open_test(tiny_mini_df, random_seed=42)
        p2, _, _ = dp.split_private_open_test(tiny_mini_df, random_seed=42)
        pd.testing.assert_frame_equal(
            p1.reset_index(drop=True), p2.reset_index(drop=True)
        )


# ---------------------------------------------------------------------------
# 4.7 partition_scenario1
# ---------------------------------------------------------------------------
class TestPartitionScenario1:
    def test_produces_exactly_k_di_clients(
        self, tiny_mini_df: pd.DataFrame
    ) -> None:
        priv, _, _ = dp.split_private_open_test(tiny_mini_df)
        clients = dp.partition_scenario1(priv, target_device_id=0, k_di=3)
        assert len(clients) == 3

    def test_shards_are_equal_sized(self, tiny_mini_df: pd.DataFrame) -> None:
        # Each client gets 2 shards, so all 3 clients should have the same
        # row count (any remainder is discarded by design).
        priv, _, _ = dp.split_private_open_test(tiny_mini_df)
        clients = dp.partition_scenario1(priv, target_device_id=0, k_di=3)
        sizes = {len(c) for c in clients}
        assert len(sizes) == 1

    def test_adjacent_label_shards(self, tiny_mini_df: pd.DataFrame) -> None:
        # Each client should carry a small, **contiguous** run of labels after
        # the sort-then-shard partitioning — the whole point of Scenario 1.
        priv, _, _ = dp.split_private_open_test(tiny_mini_df)
        clients = dp.partition_scenario1(priv, target_device_id=0, k_di=3)
        for c in clients:
            labels = sorted(c["label"].unique().tolist())
            # labels must be a contiguous run in the global space — no gaps
            # that would require non-adjacent shards.
            assert labels == sorted(set(labels))
            assert labels[-1] - labels[0] <= len(labels) - 1 + 1  # generous

    def test_insufficient_rows_raises(self) -> None:
        # A device with fewer rows than shards must not silently produce zero-
        # sized shards.
        tiny = pd.DataFrame(
            {
                "label": [0, 1],
                "device_id": [0, 0],
                **{col: [0.0, 0.0] for col in config.FEATURE_COLUMN_NAMES},
            }
        )
        with pytest.raises(ValueError):
            dp.partition_scenario1(tiny, target_device_id=0, k_di=3)


# ---------------------------------------------------------------------------
# 4.8 build_all_client_partitions
# ---------------------------------------------------------------------------
class TestBuildAllClientPartitions:
    def test_keys_cover_all_global_ids(self, tiny_mini_df: pd.DataFrame) -> None:
        # 3 devices × 3 clients = 9 partitions, keyed 0..8.
        priv, _, _ = dp.split_private_open_test(tiny_mini_df)
        all_parts = dp.build_all_client_partitions(priv, k_di=3)
        assert set(all_parts.keys()) == set(range(9))

    def test_global_id_formula(self, tiny_mini_df: pd.DataFrame) -> None:
        # global_id = device_id * k_di + local_client_id. Check via the data
        # each partition actually carries (every row must share the same
        # device_id by construction).
        priv, _, _ = dp.split_private_open_test(tiny_mini_df)
        all_parts = dp.build_all_client_partitions(priv, k_di=3)
        for gid, df in all_parts.items():
            device_ids = set(df["device_id"].unique().tolist())
            assert len(device_ids) == 1
            (dev,) = tuple(device_ids)
            assert gid // 3 == int(dev)


# ---------------------------------------------------------------------------
# 4.9 save_partitions + 4.10/4.11/4.12 loaders
# ---------------------------------------------------------------------------
class TestSaveAndLoadRoundTrip:
    def test_all_expected_files_written(self, partitioned_tmp_dir: Path) -> None:
        # Every client must have a private pickle; open and test must each
        # have exactly one pickle.
        clients = list(partitioned_tmp_dir.glob("client_*_private.pkl"))
        assert len(clients) == 9  # 3 devices × 3 clients in the synthetic tree
        assert (partitioned_tmp_dir / "open_data.pkl").exists()
        assert (partitioned_tmp_dir / "test_data.pkl").exists()

    def test_load_client_partition_returns_aligned_arrays(
        self, partitioned_tmp_dir: Path
    ) -> None:
        # X and y must have the same leading dimension; X must be (N, 23, 5).
        X, y = dp.load_client_partition(
            client_id=0, partition_dir=str(partitioned_tmp_dir)
        )
        assert X.ndim == 3 and X.shape[1:] == (23, 5)
        assert y.ndim == 1 and y.shape[0] == X.shape[0]

    def test_load_open_data_has_no_labels(
        self, partitioned_tmp_dir: Path
    ) -> None:
        X_open = dp.load_open_data(str(partitioned_tmp_dir))
        # Shape sanity
        assert X_open.ndim == 3 and X_open.shape[1:] == (23, 5)

    def test_load_test_data_has_labels(
        self, partitioned_tmp_dir: Path
    ) -> None:
        X_test, y_test = dp.load_test_data(str(partitioned_tmp_dir))
        assert X_test.shape[0] == y_test.shape[0]


# ---------------------------------------------------------------------------
# 4.12 create_torch_dataloader
# ---------------------------------------------------------------------------
class TestCreateTorchDataLoader:
    def test_yields_expected_tensor_shapes(self) -> None:
        import torch  # local import — module already torch-marked
        X = np.random.default_rng(7).standard_normal((50, 23, 5)).astype(np.float32)
        y = np.arange(50, dtype=np.int64) % 3
        loader = dp.create_torch_dataloader(X, y, batch_size=10, shuffle=False)
        # Each batch: (10, 23, 5) features + (10,) labels (last batch matches
        # the remainder; here 50/10 is exact so every batch is full).
        total = 0
        for xb, yb in loader:
            assert isinstance(xb, torch.Tensor) and isinstance(yb, torch.Tensor)
            assert xb.dtype == torch.float32
            assert yb.dtype == torch.long
            assert xb.shape == (10, 23, 5)
            assert yb.shape == (10,)
            total += xb.shape[0]
        assert total == 50

    def test_shuffle_false_preserves_order(self) -> None:
        # When shuffle=False the loader must yield samples in input order.
        X = np.arange(20 * 23 * 5, dtype=np.float32).reshape(20, 23, 5)
        y = np.arange(20, dtype=np.int64)
        loader = dp.create_torch_dataloader(X, y, batch_size=5, shuffle=False)
        seen_labels: List[int] = []
        for _xb, yb in loader:
            seen_labels.extend(yb.tolist())
        assert seen_labels == list(range(20))


# ---------------------------------------------------------------------------
# Integration sanity checks against the user's prepared_data/ tree.
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestPreparedDataShapes:
    """Not strictly a function-level test, but since the user 'readded the
    prepared_data', we want an integration check that the on-disk shapes are
    consistent with what the partitioning pipeline expects.

    The bundled files are `.npy` (produced by a different preprocessing
    script) rather than the `.pkl` files `data_preparation.save_partitions`
    writes — these tests just confirm the npys are sane.
    """

    def _skip_if_missing(self, prepared_data_dir: Path, path: Path) -> None:
        if not path.exists():
            pytest.skip(f"prepared_data file missing: {path}")

    def test_scenario1_summary_lists_27_clients(
        self, prepared_data_dir: Path
    ) -> None:
        import json
        summary = prepared_data_dir / "scenario_1" / "summary.json"
        self._skip_if_missing(prepared_data_dir, summary)
        data = json.loads(summary.read_text())
        assert len(data) == 27
        assert {row["client_id"] for row in data} == set(range(27))

    def test_open_test_private_have_115_feature_columns(
        self, prepared_data_dir: Path
    ) -> None:
        for sub in ("open", "test", "private"):
            X = prepared_data_dir / sub / "X.npy"
            self._skip_if_missing(prepared_data_dir, X)
            arr = np.load(X, mmap_mode="r")
            assert arr.shape[-1] == config.N_FEATURES, (
                f"{sub}/X.npy last dim must equal N_FEATURES=115"
            )

    def test_label_map_matches_config_keys_conceptually(
        self, prepared_data_dir: Path
    ) -> None:
        # The shipped label_map.json uses dotted names like 'gafgyt.combo',
        # while config uses underscores 'gafgyt_combo'. The *ids* must still
        # line up, which is all that matters for the aggregation logic.
        import json
        lm_path = prepared_data_dir / "label_map.json"
        self._skip_if_missing(prepared_data_dir, lm_path)
        lm = json.loads(lm_path.read_text())
        normalized = {k.replace(".", "_"): v for k, v in lm.items()}
        assert normalized == config.CLASS_NAME_TO_GLOBAL_ID
