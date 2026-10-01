"""Publication entry points must run without historical files or directories."""
import itertools
import json
from dataclasses import replace

import numpy as np
import pytest

from experiments.global_topk_search import exact_global_topk
from privfim.config import ProtocolConfig
from privfim.data import VerticalDataset, load_vertical_csv
from workflows.common import digest
from workflows.crypto import run as run_crypto
from workflows.methods import one_round
from workflows.prepare import prepare
from workflows.runner import EXPERIMENTS, case_config, run_case


def test_prepare_is_deterministic_and_checks_data(tmp_path):
    source = prepare("Toy", tmp_path)
    checksum = digest(source)
    assert digest(prepare("Toy", tmp_path)) == checksum
    source.write_text(source.read_text() + "0,0,0,0,0,0,0,0\n")
    with pytest.raises(ValueError, match="Modified data"):
        prepare("Toy", tmp_path)


def test_global_truth_matches_exhaustive_enumeration_after_transforms(tmp_path):
    source = prepare("Toy", tmp_path)
    ds = load_vertical_csv(source, 2, max_rows=21, feature_ratio=.5, domain_ratio=.5, transform_seed=2026)
    items = [(a, v) for a in ds.attributes for v in ds.domains[a]]
    for minimum in (1, 2):
        supports = {}
        for size in range(minimum, 4):
            for key in itertools.combinations(items, size):
                if len({a for a, _ in key}) == size:
                    supports[key] = ds.support(key)
        expected = dict(sorted(supports.items(), key=lambda entry: (-entry[1], entry[0]))[:15])
        actual, _ = exact_global_topk(ds, 15, 3, min_size=minimum)
        assert actual == expected


@pytest.mark.parametrize("number", range(1, 12))
def test_all_experiment_configs_are_valid(tmp_path, number):
    source = prepare("Toy", tmp_path)
    axis, values = EXPERIMENTS[number]
    for value in values:
        config = case_config(source, axis=axis, value=value, m=16)
        config.validate()
        assert config.protocol.first_stage_report_limit >= 1
        assert config.protocol.second_stage_upload_limit >= 1


@pytest.mark.parametrize("method", ("MAP-M", "MAP-S", "MAP-L", "IE-Full", "IE-Other", "FO", "First-round", "Second-round", "Items-only"))
def test_audited_method_and_resume(tmp_path, method):
    source = prepare("Toy", tmp_path / "data")
    config = case_config(source, k=5, m=16, max_rows=60)
    output = tmp_path / "run"
    row = run_case(config, method, output)
    assert row["evaluation_scope"] == "global_exact"
    assert 0 <= row["f1"] <= 1 and 0 <= row["ncr"] <= 1
    assert row["total_seconds"] >= 0 and row["communication_bytes"] > 0
    assert row == run_case(config, method, output)
    changed = replace(config, protocol=replace(config.protocol, epsilon=2.))
    with pytest.raises(ValueError, match="changed"):
        run_case(changed, method, output)


def test_second_round_reports_all_public_local_triples(tmp_path):
    # Two owners with four binary attributes each: C(4,1)*2 + C(4,2)*4 + C(4,3)*8 = 64 keys per owner.
    source = prepare("Toy", tmp_path)
    ds = load_vertical_csv(source, 2, max_rows=40)
    _, info = one_round(ds, ProtocolConfig(k=5, m=16, first_stage_report_limit=5), "Second-round")
    assert info["report_keys"] == 128


def test_crypto_dataset_output_has_no_timeout_or_assumed_perfect_score(tmp_path, monkeypatch):
    from argparse import Namespace
    from workflows import crypto
    prepare("Toy", tmp_path / "data")
    seen = {}
    binary = tmp_path / "fake_binary"
    binary.write_bytes(b"test-only")
    monkeypatch.setattr(crypto, "BINARY", binary)
    def backend(ds, candidates, threshold, output, **kwargs):
        seen.update(kwargs)
        return {key: ds.support(key) for key in candidates}, {"protocol_seconds": .01}
    monkeypatch.setattr(crypto, "run_nipp", backend)
    args = Namespace(output=tmp_path / "crypto", datasets=["Toy"], data_dir=tmp_path / "data",
                     clients=2, max_rows=8, seed=2026, k=3, epsilon=1., paillier_bits=2048,
                     candidate_source="two-round", candidate_limit=10000, methods=["NIPP-FIM"])
    run_crypto(args)
    assert seen["timeout"] is None
    result = json.loads(next(args.output.rglob("result.json")).read_text())
    assert result["row"]["evaluation_scope"] == "global_exact"
    assert result["row"]["rows"] == 8
