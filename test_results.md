# SSFL Infrastructure — pytest Test Results

- **Generated:** 2026-04-23 12:10:33
- **Session duration:** 3.84 s
- **Exit status:** 0 (all green)
- **Total tests:** 117
- **Warnings:** 5 (across 3 test(s), plus 0 orphan)

> **Note (2026-04-23 post-metrics-refactor):** `utils.compute_metrics` is now a **facade** over `metrics.compute_classification_metrics` (see `SSFL_FLOWER_INFRASTRUCTURE_PLAN.md` §10.1, §14). The 3 existing `TestComputeMetrics` tests in `test_utils.py` remain valid under the facade because the legacy `confusion_matrix` shape (`np.ndarray`) is preserved. A new `test_metrics.py` file needs to land next to cover the §14 surface — see the bottom "Pending §14 test coverage" section.

## Summary by outcome

| Badge | Outcome | Count |
|---|---|---|
| `[PASS]` | passed | 114 |
| `[FAIL]` | failed | 0 |
| `[ERROR]` | error | 0 |
| `[SKIP]` | skipped | 0 |
| `[XFAIL]` | xfailed | 0 |
| `[XPASS]` | xpassed | 0 |
| `[WARN]` | warnings recorded | 5 |

## Results by file

Each row is one test function. The **Status** column is the final outcome; `[WARN]` is appended when pytest recorded at least one warning against that test.

### `test_client.py` — 15 test(s) (15 pass)

#### `TestBuildDiscriminatorDataset`

| Status | Test | Time | Notes |
|---|---|---|---|
| **[PASS]** | `test_all_open_unfamiliar_when_threshold_is_high` | 0.7 ms |  |
| **[PASS]** | `test_no_unfamiliar_samples_when_threshold_is_low` | 0.6 ms |  |
| **[PASS]** | `test_shapes_and_label_invariants` | 0.6 ms |  |

#### `TestCnnDependentStubs`

| Status | Test | Time | Notes |
|---|---|---|---|
| **[PASS]** | `test_compute_confidence_scores_still_stubbed` | 0.4 ms |  |
| **[PASS]** | `test_constructor_itself_is_stubbed` | 0.4 ms |  |
| **[PASS]** | `test_evaluate_still_stubbed` | 0.5 ms |  |
| **[PASS]** | `test_filter_and_predict_still_stubbed` | 0.4 ms |  |
| **[PASS]** | `test_get_parameters_propagates_stub` | 0.5 ms |  |
| **[PASS]** | `test_train_classifier_still_stubbed` | 0.3 ms |  |
| **[PASS]** | `test_train_discriminator_still_stubbed` | 0.3 ms |  |

#### `TestComputeConfidenceThreshold`

| Status | Test | Time | Notes |
|---|---|---|---|
| **[PASS]** | `test_handles_even_length` | 0.5 ms |  |
| **[PASS]** | `test_returns_median` | 0.9 ms |  |

#### `TestRunDistillationShortCircuit`

| Status | Test | Time | Notes |
|---|---|---|---|
| **[PASS]** | `test_all_minus_one_returns_zero_without_raising` | 0.4 ms |  |
| **[PASS]** | `test_any_valid_label_triggers_cnn_dependent_branch` | 0.3 ms |  |

#### `TestSetParameters`

| Status | Test | Time | Notes |
|---|---|---|---|
| **[PASS]** | `test_casts_to_int64_and_stores` | 0.5 ms |  |

### `test_config.py` — 16 test(s) (16 pass)

#### `TestClassNameMapping`

| Status | Test | Time | Notes |
|---|---|---|---|
| **[PASS]** | `test_benign_is_zero` | 0.4 ms |  |
| **[PASS]** | `test_inverse_mapping_is_a_bijection` | 0.4 ms |  |
| **[PASS]** | `test_mapping_has_eleven_entries` | 0.3 ms |  |
| **[PASS]** | `test_mapping_ids_are_contiguous_0_to_10` | 0.4 ms |  |
| **[PASS]** | `test_mirai_family_ids` | 0.3 ms |  |

#### `TestFeatureDimensions`

| Status | Test | Time | Notes |
|---|---|---|---|
| **[PASS]** | `test_feature_column_names_length_and_uniqueness` | 0.4 ms |  |
| **[PASS]** | `test_feature_count_matches_time_window_product` | 0.4 ms |  |
| **[PASS]** | `test_input_shape_tuple` | 0.3 ms |  |

#### `TestFilesystemConstants`

| Status | Test | Time | Notes |
|---|---|---|---|
| **[PASS]** | `test_data_subdirs_are_under_data_dir` | 0.3 ms |  |

#### `TestRatiosAndHyperparameters`

| Status | Test | Time | Notes |
|---|---|---|---|
| **[PASS]** | `test_split_ratios_are_positive` | 0.3 ms |  |
| **[PASS]** | `test_split_ratios_sum_to_one` | 0.4 ms |  |
| **[PASS]** | `test_training_hyperparameters_are_positive` | 0.4 ms |  |

#### `TestTopologyConstants`

| Status | Test | Time | Notes |
|---|---|---|---|
| **[PASS]** | `test_k_di_is_three` | 0.3 ms |  |
| **[PASS]** | `test_num_classes_is_eleven` | 0.3 ms |  |
| **[PASS]** | `test_num_clients_is_derived` | 0.3 ms |  |
| **[PASS]** | `test_num_devices_is_nine` | 0.4 ms |  |

### `test_data_preparation.py` — 40 test(s) (37 pass)

#### `TestApply2DReshapeToDataset`

| Status | Test | Time | Notes |
|---|---|---|---|
| **[PASS]** | `test_returns_correct_shapes_with_labels` | 1.3 ms |  |
| **[PASS]** | `test_returns_minus1_labels_when_label_missing` | 0.8 ms |  |

#### `TestBuildAllClientPartitions`

| Status | Test | Time | Notes |
|---|---|---|---|
| **[PASS]** | `test_global_id_formula` | 29.5 ms |  |
| **[PASS]** | `test_keys_cover_all_global_ids` | 28.6 ms |  |

#### `TestBuildMiniNbaiot`

| Status | Test | Time | Notes |
|---|---|---|---|
| **[PASS]** | `test_empty_raw_dir_raises` | 2.0 ms |  |
| **[UNKNOWN] `[WARN x2]`** | `test_sample_with_replacement_when_source_is_small` | 100.5 ms | see warnings section |
| **[UNKNOWN] `[WARN x2]`** | `test_samples_per_class_are_respected` | 100.1 ms | see warnings section |

#### `TestCreateTorchDataLoader`

| Status | Test | Time | Notes |
|---|---|---|---|
| **[PASS]** | `test_shuffle_false_preserves_order` | 0.7 ms |  |
| **[PASS]** | `test_yields_expected_tensor_shapes` | 2.2 ms |  |

#### `TestExtractClassName`

| Status | Test | Time | Notes |
|---|---|---|---|
| **[PASS]** | `test_normalization_rules[data/raw/Dev_A/MIRAI_UDP.csv-mirai_udp]` | 0.4 ms |  |
| **[PASS]** | `test_normalization_rules[data/raw/Dev_A/__gafgyt_tcp__.csv-gafgyt_tcp]` | 0.4 ms |  |
| **[PASS]** | `test_normalization_rules[data/raw/Dev_A/benign.csv-benign]` | 0.7 ms |  |
| **[PASS]** | `test_normalization_rules[data/raw/Dev_A/gafgyt-combo.csv-gafgyt_combo]` | 0.4 ms |  |
| **[PASS]** | `test_normalization_rules[data/raw/Dev_A/mirai.udp.csv-mirai_udp]` | 0.4 ms |  |
| **[PASS]** | `test_normalization_rules[data/raw/Dev_A/mirai_udp_1.csv-mirai_udp]` | 0.4 ms |  |

#### `TestLoadDeviceCsvs`

| Status | Test | Time | Notes |
|---|---|---|---|
| **[PASS]** | `test_empty_directory_raises` | 2.5 ms |  |
| **[UNKNOWN] `[WARN x1]`** | `test_loads_and_labels_every_row` | 66.5 ms | see warnings section |
| **[PASS]** | `test_unknown_class_filename_raises_keyerror` | 3.6 ms |  |

#### `TestNormalization`

| Status | Test | Time | Notes |
|---|---|---|---|
| **[PASS]** | `test_apply_normalization_matches_original_min_max` | 12.7 ms |  |
| **[PASS]** | `test_input_df_is_not_mutated` | 10.2 ms |  |
| **[PASS]** | `test_output_is_in_unit_interval` | 7.6 ms |  |
| **[PASS]** | `test_zero_range_column_is_not_nan` | 8.6 ms |  |

#### `TestPartitionScenario1`

| Status | Test | Time | Notes |
|---|---|---|---|
| **[PASS]** | `test_adjacent_label_shards` | 27.0 ms |  |
| **[PASS]** | `test_insufficient_rows_raises` | 1.7 ms |  |
| **[PASS]** | `test_produces_exactly_k_di_clients` | 27.8 ms |  |
| **[PASS]** | `test_shards_are_equal_sized` | 29.0 ms |  |

#### `TestPreparedDataShapes`

| Status | Test | Time | Notes |
|---|---|---|---|
| **[PASS]** | `test_label_map_matches_config_keys_conceptually` | 0.7 ms |  |
| **[PASS]** | `test_open_test_private_have_115_feature_columns` | 2.3 ms |  |
| **[PASS]** | `test_scenario1_summary_lists_27_clients` | 0.9 ms |  |

#### `TestReshapeSampleTo2D`

| Status | Test | Time | Notes |
|---|---|---|---|
| **[PASS]** | `test_index_mapping_matches_plan` | 0.5 ms |  |
| **[PASS]** | `test_output_shape_is_23_by_5` | 0.4 ms |  |

#### `TestSaveAndLoadRoundTrip`

| Status | Test | Time | Notes |
|---|---|---|---|
| **[PASS]** | `test_all_expected_files_written` | 35.2 ms |  |
| **[PASS]** | `test_load_client_partition_returns_aligned_arrays` | 50.5 ms |  |
| **[PASS]** | `test_load_open_data_has_no_labels` | 50.1 ms |  |
| **[PASS]** | `test_load_test_data_has_labels` | 43.7 ms |  |

#### `TestSplitPrivateOpenTest`

| Status | Test | Time | Notes |
|---|---|---|---|
| **[PASS]** | `test_bad_ratios_raise` | 24.9 ms |  |
| **[PASS]** | `test_determinism_with_same_seed` | 32.3 ms |  |
| **[PASS]** | `test_open_split_drops_label_column` | 25.8 ms |  |
| **[PASS]** | `test_ratios_sum_and_disjoint` | 26.4 ms |  |
| **[PASS]** | `test_stratification_preserves_every_group` | 26.4 ms |  |

### `test_main.py` — 6 test(s) (6 pass)

#### `TestParseArguments`

| Status | Test | Time | Notes |
|---|---|---|---|
| **[PASS]** | `test_client_mode_accepts_client_id` | 0.9 ms |  |
| **[PASS]** | `test_invalid_mode_rejected` | 1.6 ms |  |
| **[PASS]** | `test_mode_is_required` | 1.4 ms |  |
| **[PASS]** | `test_server_mode_defaults_match_config` | 1.2 ms |  |

#### `TestSetupEnvironment`

| Status | Test | Time | Notes |
|---|---|---|---|
| **[PASS]** | `test_different_seeds_diverge` | 4.1 ms |  |
| **[PASS]** | `test_seeds_are_reproducible` | 6.2 ms |  |

### `test_server.py` — 2 test(s) (2 pass)

#### `TestBuildEvalFn`

| Status | Test | Time | Notes |
|---|---|---|---|
| **[PASS]** | `test_closure_body_is_stubbed` | 0.5 ms |  |
| **[PASS]** | `test_factory_returns_callable` | 0.5 ms |  |

### `test_strategy.py` — 21 test(s) (21 pass)

#### `TestAggregateEvaluate`

| Status | Test | Time | Notes |
|---|---|---|---|
| **[PASS]** | `test_empty_results_returns_none` | 0.3 ms |  |
| **[PASS]** | `test_weighted_mean_of_loss_and_accuracy` | 0.4 ms |  |
| **[PASS]** | `test_zero_total_samples_returns_none` | 0.3 ms |  |

#### `TestAggregateFit`

| Status | Test | Time | Notes |
|---|---|---|---|
| **[PASS]** | `test_eval_fn_notimplemented_is_caught` | 0.7 ms |  |
| **[PASS]** | `test_no_results_returns_prior_labels_untouched` | 1.1 ms |  |
| **[PASS]** | `test_votes_and_updates_global_labels` | 0.8 ms |  |

#### `TestConfigureEvaluate`

| Status | Test | Time | Notes |
|---|---|---|---|
| **[PASS]** | `test_sends_to_all_clients_with_round_config` | 0.3 ms |  |

#### `TestConfigureFit`

| Status | Test | Time | Notes |
|---|---|---|---|
| **[PASS]** | `test_broadcasts_current_labels_and_config_to_all_clients` | 0.8 ms |  |

#### `TestInitializeParameters`

| Status | Test | Time | Notes |
|---|---|---|---|
| **[PASS]** | `test_round_trips_initial_global_labels` | 0.8 ms |  |

#### `TestStrategyEvaluate`

| Status | Test | Time | Notes |
|---|---|---|---|
| **[PASS]** | `test_returns_none` | 0.4 ms |  |

#### `TestStrategyInit`

| Status | Test | Time | Notes |
|---|---|---|---|
| **[PASS]** | `test_initial_global_labels_are_all_minus_one` | 0.4 ms |  |
| **[PASS]** | `test_round_metrics_starts_empty` | 0.4 ms |  |
| **[PASS]** | `test_stores_constructor_args` | 0.3 ms |  |

#### `TestVoteMechanism`

| Status | Test | Time | Notes |
|---|---|---|---|
| **[PASS]** | `test_all_abstain_keeps_minus1` | 0.5 ms |  |
| **[PASS]** | `test_minus1_uploads_are_ignored` | 0.4 ms |  |
| **[PASS]** | `test_out_of_range_class_is_clipped` | 0.4 ms |  |
| **[PASS]** | `test_scales_to_twenty_seven_clients` | 0.8 ms |  |
| **[PASS]** | `test_simple_majority` | 0.6 ms |  |
| **[PASS]** | `test_tie_break_favors_lowest_class_index` | 0.4 ms |  |
| **[PASS]** | `test_unanimous_vote_wins` | 0.6 ms |  |
| **[PASS]** | `test_wrong_length_upload_is_skipped_silently` | 0.5 ms |  |

### `test_utils.py` — 17 test(s) (17 pass)

#### `TestComputeMetrics`

| Status | Test | Time | Notes |
|---|---|---|---|
| **[PASS]** | `test_all_wrong_gives_zero_accuracy` | 3.4 ms |  |
| **[PASS]** | `test_confusion_matrix_labels_cover_all_classes` | 3.6 ms |  |
| **[PASS]** | `test_perfect_prediction_gives_all_ones` | 34.5 ms |  |

#### `TestGetFeatureColumnNames`

| Status | Test | Time | Notes |
|---|---|---|---|
| **[PASS]** | `test_names_are_nonempty_strings` | 0.4 ms |  |
| **[PASS]** | `test_returns_a_fresh_list` | 0.4 ms |  |
| **[PASS]** | `test_returns_exactly_115_names` | 0.4 ms |  |

#### `TestJsonSafe`

| Status | Test | Time | Notes |
|---|---|---|---|
| **[PASS]** | `test_coerces_numpy_array_to_list` | 0.4 ms |  |
| **[PASS]** | `test_coerces_numpy_scalar_float` | 0.3 ms |  |
| **[PASS]** | `test_coerces_numpy_scalar_int` | 0.4 ms |  |
| **[PASS]** | `test_passthrough_for_builtins` | 0.3 ms |  |
| **[PASS]** | `test_recurses_into_dicts_and_lists` | 0.4 ms |  |

#### `TestSaveRoundMetrics`

| Status | Test | Time | Notes |
|---|---|---|---|
| **[PASS]** | `test_appends_to_existing_file` | 4.8 ms |  |
| **[PASS]** | `test_coerces_numpy_metric_values` | 4.0 ms |  |
| **[PASS]** | `test_creates_file_and_writes_single_entry` | 3.3 ms |  |
| **[PASS]** | `test_creates_parent_directory_if_missing` | 3.0 ms |  |
| **[PASS]** | `test_recovers_from_corrupted_json` | 4.2 ms |  |

#### `TestSetupLogging`

| Status | Test | Time | Notes |
|---|---|---|---|
| **[PASS]** | `test_installs_handler_and_sets_level` | 0.5 ms |  |

## Warnings

### Per-test warnings

- `test_data_preparation.py::TestBuildMiniNbaiot::test_sample_with_replacement_when_source_is_small`
    - PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()` [runtest]
    - PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()` [runtest]
- `test_data_preparation.py::TestBuildMiniNbaiot::test_samples_per_class_are_respected`
    - PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()` [runtest]
    - PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()` [runtest]
- `test_data_preparation.py::TestLoadDeviceCsvs::test_loads_and_labels_every_row`
    - PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()` [runtest]

---

## Pending §14 test coverage — new `test_metrics.py`

The new `metrics.py` module (§14 of the infrastructure plan) adds live math that is not yet covered by pytest. Ad-hoc smoke testing has verified every code path on 2026-04-23 (see log in this session — `payload_bytes_wire`, `payload_bytes_packed`, `open_dataset_distribution_bytes`, `fl_baseline_upload_bytes`, `bytes_to_mb`, `CommCostLedger.record / cumulative_bytes_at / cumulative_mb_at / cumulative_series / to_dict`, `extract_top_acc`, `extract_comm_cost_at_accuracy`, `extract_comm_cost_at_top_acc`, `extract_accuracy_snapshots`, `build_summary_report`, `save_metrics_json / save_metrics_csv / save_summary_json / save_confusion_matrix_json` all pass), but the following pytest tests still need to be written before the next CI pass:

### Payload sizing (classification-metric math can share tests with `test_utils.py::TestComputeMetrics`)
- `test_payload_bytes_wire_matches_nbytes` — for a range of dtypes and shapes
- `test_payload_bytes_packed_is_size_times_one`
- `test_open_dataset_distribution_bytes_product_of_three_ints`
- `test_fl_baseline_upload_bytes_default_matches_config_estimate`
- `test_bytes_to_mb_uses_decimal_mb`

### CommCostLedger
- `test_record_appends_entry_in_order`
- `test_cumulative_bytes_at_wire_vs_packed`
- `test_cumulative_mb_at_matches_bytes_to_mb`
- `test_cumulative_series_length_matches_entries`
- `test_non_monotonic_round_warns_but_still_records` (regression: we only log, not raise)
- `test_to_dict_round_trip_is_json_serialisable`

### Summary extractors
- `test_extract_top_acc_returns_none_when_no_accuracy_key`
- `test_extract_top_acc_picks_first_tie_winner`
- `test_extract_comm_cost_at_accuracy_target_never_reached_returns_none`
- `test_extract_comm_cost_at_accuracy_uses_first_crossing`
- `test_extract_comm_cost_at_top_acc_matches_cumulative_at_top_round`
- `test_extract_accuracy_snapshots_missing_round_is_none`
- `test_build_summary_report_contains_all_expected_keys`
- `test_build_summary_report_target_accs_stringified_with_two_decimals`

### I/O helpers
- `test_save_metrics_json_round_trips`
- `test_save_metrics_csv_has_round_first_column`
- `test_save_metrics_csv_drops_list_and_confusion_matrix_fields`
- `test_save_summary_json_is_json_loadable`
- `test_save_confusion_matrix_json_preserves_per_class_lists`

### Strategy instrumentation (in `test_strategy.py`, extending `TestAggregateFit`)
- `test_aggregate_fit_records_comm_cost_ledger_entry_per_round`
- `test_aggregate_fit_charges_open_dataset_only_on_round_1_by_default`
- `test_aggregate_fit_cumulative_mb_wire_is_monotonic_non_decreasing`
- `test_aggregate_fit_broadcast_fanout_is_bytes_times_num_results`
- `test_aggregate_fit_avg_confidence_threshold_is_mean_of_client_thresholds`
- `test_aggregate_fit_wall_clock_fields_are_floats`

### Client instrumentation (in `test_client.py`, extending `TestRunDistillationShortCircuit` / adding `TestFitMetricsPayload`)
- `test_fit_metrics_include_bytes_upload_wire_and_packed` — requires a mocked client where `filter_and_predict` returns a synthetic array
- `test_fit_metrics_include_confidence_threshold_and_fit_wall_clock_sec`

### Main / CLI
- `test_parse_arguments_accepts_metrics_dir_and_logs_dir`
- `test_parse_arguments_accepts_target_accs_and_snapshot_rounds`
- `test_parse_float_tuple_and_parse_int_tuple_helpers`

All of the above are **additive** — nothing in the existing 117-test suite needs to change beyond ensuring `TestComputeMetrics` still asserts the legacy `confusion_matrix: np.ndarray` shape (which it already does, via the `utils.compute_metrics` facade).
