"""Upload caps are independent of output k and broadcast S; defaults remain unchanged."""
import csv
from dataclasses import replace

import numpy as np
import pytest

from privfim.client import VerticalClient
from privfim.config import DataConfig, ExperimentConfig, ProtocolConfig
from privfim.pipeline import run_experiment
from privfim.types import Candidate, LOCAL_ITEMSET_ALPHA, MIXED_ITEMSET_ALPHA, SINGLETON_ALPHA


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_upload_cap_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="second_stage_upload_limit"):
        ExperimentConfig(protocol=ProtocolConfig(second_stage_upload_limit=value)).validate()


def test_upload_cap_ranks_singletons_and_joint_keys_together():
    client = VerticalClient("client_0", (0, 1), np.array([[1, 1], [0, 0]]))
    candidates = [Candidate(((0, 1), (1, 1)), 1.0, 40.0)]
    counts = {(0, 1): 80.0, (1, 1): 60.0}
    kwargs = dict(noisy_singleton_counts=counts, normalization_n=100.0)
    small = client.report_keys(candidates, MIXED_ITEMSET_ALPHA, report_key_limit=1, **kwargs)
    all_keys = client.report_keys(candidates, MIXED_ITEMSET_ALPHA, report_key_limit=0, **kwargs)
    large = client.report_keys(candidates, MIXED_ITEMSET_ALPHA, report_key_limit=9, **kwargs)
    assert small == (((0, 1),),)
    assert set(all_keys) == {((0, 1),), ((1, 1),), ((0, 1), (1, 1))}
    assert large == all_keys
    # Do not pad to the cap; unlimited mode uses the same public key-generation rule.
    assert len(large) == 3


def test_upload_cap_changes_only_mixed_mode_and_preserves_default(tmp_path):
    path = tmp_path / "toy.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(range(4))
        writer.writerows([[i % 2, i % 2, (i // 3) % 2, (i // 4) % 2] for i in range(120)])
    config = ExperimentConfig(
        data=DataConfig(csv_path=str(path), num_clients=2),
        protocol=ProtocolConfig(
            k=3, m=8, epsilon=20.0, max_itemset_size=3,
            modes=(SINGLETON_ALPHA, LOCAL_ITEMSET_ALPHA, MIXED_ITEMSET_ALPHA),
        ),
    )
    summaries = {}
    for cap in (None, 3, 1, 0, 99):
        summaries[cap] = run_experiment(replace(
            config, protocol=replace(config.protocol, second_stage_upload_limit=cap),
            output_dir=str(tmp_path / f"cap_{cap}"),
        ))
    for cap, summary in summaries.items():
        result = summary["modes"][MIXED_ITEMSET_ALPHA]
        available = result["available_report_key_count"]
        limit = 3 if cap is None else cap
        assert result["report_key_count"] == {
            client: min(n, limit) if limit else n for client, n in available.items()
        }
        assert result["report_key_limit_respected"]
        assert result["report_key_limit"] == (limit or None)
        assert summary["candidate_count"] == 6
        assert summary["frequent_items"] == summaries[None]["frequent_items"]
        for other in (SINGLETON_ALPHA, LOCAL_ITEMSET_ALPHA):
            assert summary["modes"][other]["report_key_limit"] is None
            assert summary["modes"][other]["report_keys"] == summaries[None]["modes"][other]["report_keys"]
            assert summary["modes"][other]["metrics"] == summaries[None]["modes"][other]["metrics"]
    for field in ("report_keys", "metrics", "top_k", "epsilon_per_key"):
        assert summaries[None]["modes"][MIXED_ITEMSET_ALPHA][field] == summaries[3]["modes"][MIXED_ITEMSET_ALPHA][field]
        assert summaries[0]["modes"][MIXED_ITEMSET_ALPHA][field] == summaries[99]["modes"][MIXED_ITEMSET_ALPHA][field]
