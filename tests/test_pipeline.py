import csv
import json

import numpy as np
import pytest

from privfim.config import DataConfig, ExperimentConfig, ProtocolConfig
from privfim.pipeline import run_experiment
from privfim.types import (
    FM_FULL_ESTIMATOR,
    FM_OTHER_ESTIMATOR,
    LOCAL_ITEMSET_ALPHA,
    LOCAL_TOP_ITEMSET_ALPHA,
    MIXED_BINNED_ITEMSET_ALPHA,
    MIXED_ITEMSET_ALPHA,
    MIXED_PROTECTED_BINNED_ITEMSET_ALPHA,
    SINGLETON_ALPHA,
)


def test_end_to_end_comparison(tmp_path):
    rng = np.random.default_rng(12)
    rows = []
    for _ in range(300):
        gender = int(rng.random() < 0.65)
        age = gender if rng.random() < 0.85 else 1 - gender
        disease = age if rng.random() < 0.8 else 1 - age
        home = disease if rng.random() < 0.75 else 1 - disease
        rows.append([gender, age, disease, home])

    csv_path = tmp_path / "synthetic.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([0, 1, 2, 3])
        writer.writerows(rows)

    config = ExperimentConfig(
        data=DataConfig(csv_path=str(csv_path), num_clients=2),
        protocol=ProtocolConfig(
            k=3,
            candidate_multiplier=2,
            min_itemset_size=1,
            max_itemset_size=3,
            epsilon=20.0,
            phase1_ratio=0.5,
            delta=1e-5,
            m=32,
            seed=5,
            modes=(
                SINGLETON_ALPHA,
                LOCAL_ITEMSET_ALPHA,
                LOCAL_TOP_ITEMSET_ALPHA,
                MIXED_ITEMSET_ALPHA,
            ),
            local_projection_limit=1,
        ),
        output_dir=str(tmp_path / "results"),
    )
    summary = run_experiment(config)

    baseline = summary["modes"][SINGLETON_ALPHA]
    enhanced = summary["modes"][LOCAL_ITEMSET_ALPHA]
    local_top = summary["modes"][LOCAL_TOP_ITEMSET_ALPHA]
    mixed = summary["modes"][MIXED_ITEMSET_ALPHA]
    assert summary["implementation_version"] == "positive-alpha-map-v22-plan-budget"
    assert summary["privacy"]["noisy_n_epsilon_target"] == pytest.approx(1.0)
    assert summary["privacy"]["noisy_n_client_epsilon"] == pytest.approx(0.5)
    assert summary["privacy"]["statistics_epsilon_target"] == pytest.approx(19.0)
    assert summary["privacy"]["phase1_epsilon_target"] == pytest.approx(9.0)
    assert summary["privacy"]["phase2_epsilon_target"] == pytest.approx(10.0)
    assert len(summary["privacy"]["noisy_n_reports"]) == 2
    reported_noisy_n = [
        report["noisy_n"] for report in summary["privacy"]["noisy_n_reports"]
    ]
    assert all(value is not None for value in reported_noisy_n)
    assert summary["privacy"]["noisy_n_mean_raw"] == pytest.approx(
        np.mean(reported_noisy_n)
    )
    assert summary["dataset"]["noisy_rows"] == pytest.approx(
        max(1.0, summary["privacy"]["noisy_n_mean_raw"])
    )
    assert summary["privacy"]["noisy_n_mean"] == pytest.approx(
        summary["dataset"]["noisy_rows"]
    )
    assert len(summary["frequent_items"]) == 3
    assert summary["candidate_count"] == 6
    assert summary["privacy"]["formal_end_to_end_dp"] is False
    assert summary["privacy"]["theoretical_phase2_dp_under_secret_prf"] is True
    assert summary["privacy"][
        "all_modes_individually_within_total_budget_under_secret_prf"
    ] is True
    assert summary["privacy"]["simultaneous_mode_release_accounted"] is False
    assert summary["privacy"]["round1_uses_public_complete_domain"] is True
    assert baseline["scheme_label"] == "MAP-S"
    assert enhanced["scheme_label"] == "MAP-L"
    assert local_top["scheme_label"] == "MAP-L-Top"
    assert mixed["scheme_label"] == "MAP-M"
    assert enhanced["report_key_limit"] is None
    assert enhanced["report_key_limit_respected"] is None
    assert enhanced["estimated_candidate_count"] == enhanced["candidate_count"]
    assert local_top["estimated_candidate_count"] == local_top["candidate_count"]
    assert mixed["report_key_limit"] == 3
    assert max(mixed["report_key_count"].values()) <= 3
    assert mixed["report_key_limit_respected"] is True
    assert mixed["budget_allocation"] == "measurement_group_parallel"
    assert mixed["per_key_budget_role"].startswith("shared_measurement_group")
    assert mixed["measurement_group_budgets"] is not None
    assert all(
        group["key_count"] >= 1 and group["epsilon"] > 0.0
        for groups in mixed["measurement_group_budgets"].values()
        for group in groups
    )
    assert all(
        mixed["public_overlap_bound"][client_id]
        == mixed["measurement_group_count"][client_id]
        for client_id in mixed["report_keys"]
    )
    assert all(
        len(keys) <= 3 for keys in mixed["report_keys"].values()
    )
    assert enhanced["total_report_keys"] >= baseline["total_report_keys"]
    assert local_top["total_report_keys"] >= baseline["total_report_keys"]
    assert enhanced["max_public_overlap_bound"] >= baseline["max_public_overlap_bound"]
    assert baseline["overlap_rdp_within_phase2_budget"] is True
    assert enhanced["overlap_rdp_within_phase2_budget"] is True
    assert baseline["overlap_rdp_epsilon"] > 0
    assert enhanced["overlap_rdp_epsilon"] > 0
    assert baseline["certified_total_epsilon"] == pytest.approx(
        summary["privacy"]["noisy_n_epsilon_target"]
        + summary["privacy"]["phase1_epsilon_target"]
        + baseline["overlap_rdp_epsilon"]
    )
    assert baseline["theoretical_total_within_budget_under_secret_prf"] is True
    assert 0.0 <= baseline["metrics"]["f1"] <= 1.0
    assert 0.0 <= enhanced["metrics"]["f1"] <= 1.0
    assert "1" in baseline["metrics_by_itemset_size"]
    assert "1" in enhanced["metrics_by_itemset_size"]
    assert (tmp_path / "results" / "summary.json").exists()
    json.loads((tmp_path / "results" / "summary.json").read_text(encoding="utf-8"))


def test_end_to_end_fm_protocols_use_distinct_bucket_counts(tmp_path):
    rows = [
        [
            0 if index < 100 else 1 if index < 110 else 2,
            0 if index < 90 else 1,
            0 if index < 80 else 1,
        ]
        for index in range(120)
    ]
    csv_path = tmp_path / "fm.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([0, 1, 2])
        writer.writerows(rows)

    key_counts = {}
    for estimator in (FM_FULL_ESTIMATOR, FM_OTHER_ESTIMATOR):
        config = ExperimentConfig(
            data=DataConfig(csv_path=str(csv_path), num_clients=2),
            protocol=ProtocolConfig(
                k=3,
                candidate_multiplier=2,
                max_itemset_size=3,
                epsilon=10.0,
                m=16,
                seed=7,
                estimator=estimator,
                modes=(SINGLETON_ALPHA,),
            ),
            output_dir=str(tmp_path / estimator),
        )
        summary = run_experiment(config)
        result = summary["modes"][SINGLETON_ALPHA]
        key_counts[estimator] = result["total_report_keys"]
        assert result["scheme_label"] in {"FM-Full", "FM-Other"}
        assert 0.0 <= result["metrics"]["f1"] <= 1.0

    # Top-k 单项分别来自三个属性，Top-2k 候选覆盖它们。FM-Full 上传
    # 3+2+2 个公开域桶；FM-Other 每个属性上传目标值和一个 OTHER 桶。
    assert key_counts[FM_FULL_ESTIMATOR] == 7
    assert key_counts[FM_OTHER_ESTIMATOR] == 6


def test_map_m_bin_is_an_explicit_binned_mode(tmp_path):
    rows = [
        [index % 4, (index // 2) % 4, (index // 3) % 4]
        for index in range(96)
    ]
    csv_path = tmp_path / "bin.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([0, 1, 2])
        writer.writerows(rows)

    summary = run_experiment(
        ExperimentConfig(
            data=DataConfig(csv_path=str(csv_path), num_clients=2),
            protocol=ProtocolConfig(
                k=3,
                candidate_multiplier=1,
                max_itemset_size=3,
                epsilon=20.0,
                m=16,
                seed=17,
                map_m_bin_count=2,
                modes=(
                    MIXED_ITEMSET_ALPHA,
                    MIXED_BINNED_ITEMSET_ALPHA,
                    MIXED_PROTECTED_BINNED_ITEMSET_ALPHA,
                ),
            ),
            output_dir=str(tmp_path / "bin-results"),
        )
    )

    exact = summary["modes"][MIXED_ITEMSET_ALPHA]
    binned = summary["modes"][MIXED_BINNED_ITEMSET_ALPHA]
    protected = summary["modes"][MIXED_PROTECTED_BINNED_ITEMSET_ALPHA]
    assert exact["scheme_label"] == "MAP-M"
    assert exact["report_binning"]["enabled"] is False
    assert binned["scheme_label"] == "MAP-M-Bin2"
    assert binned["report_binning"]["enabled"] is True
    assert binned["report_binning"]["strategy"] == "first_round_dp_equal_mass_contiguous"
    assert binned["report_binning"]["dp_bins"] is not None
    assert binned["report_binning"]["recovery"] == "first_round_noisy_within_bin_product"
    assert binned["budget_allocation"] == "binned_measurement_group_parallel"
    assert binned["communication"]["binning_downlink_bytes"] > 0
    assert exact["communication"]["binning_downlink_bytes"] == 0
    assert binned["estimated_candidate_count"] == binned["candidate_count"]
    assert protected["scheme_label"] == "MAP-M-BinP2"
    assert protected["report_binning"]["enabled"] is True
    assert protected["report_binning"]["strategy"] == "first_round_dp_protected_value_quantile_tail"
    assert protected["report_binning"]["dp_bins"] is not None
    assert protected["report_binning"]["recovery"] == "svsm_candidate_prior_within_bucket_normalization"
    assert protected["communication"]["binning_downlink_bytes"] > 0
