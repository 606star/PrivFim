from __future__ import annotations

from collections import defaultdict

import numpy as np

from .binning import BinningPlan, binned_itemset, within_bin_probability
from .candidate import (
    construct_direct_candidates,
    construct_itemset_first_candidates,
    construct_svsm_candidates,
    select_frequent_singletons,
)
from .dpfm import (
    fm_category_intersection_estimate,
    intersection_bounds,
    map_intersection_estimate,
    map_nested_pair_cardinality_estimate,
    private_cardinality_map_estimate,
    union_complement_fm_estimate,
)
from .types import (
    DIRECT_UNION_COMPLEMENT,
    FM_FULL_ESTIMATOR,
    FM_INVERSE_ESTIMATOR,
    FM_OTHER_ESTIMATOR,
    LOCAL_TOP_SINGLETON_ALPHA,
    LOCAL_TOP_ITEMSET_COMPONENT_ALPHA,
    MAP_BOUNDS_ONLY_ESTIMATOR,
    MAP_ESTIMATOR,
    MAP_NO_FRECHET_ESTIMATOR,
    MIXED_PROTECTED_BINNED_ITEMSET_ALPHA,
    SINGLETON_ALPHA,
    AlphaReport,
    Candidate,
    CandidateEstimate,
    Item,
    Itemset,
    NoisyCountReport,
    canonical_itemset,
)


class PrivFimServer:
    def __init__(
        self,
        n_rows: float,
        partitions: tuple[tuple[int, ...], ...],
        domains: tuple[tuple[int, ...], ...] | None = None,
    ) -> None:
        self.n_rows = n_rows
        self.partitions = partitions
        self.domains = domains
        self.owner_by_attribute = {
            attr: owner
            for owner, attrs in enumerate(partitions)
            for attr in attrs
        }

    def aggregate_round1(self, reports: list[NoisyCountReport]) -> dict[Item, float]:
        aggregated: dict[Item, float] = defaultdict(float)
        for report in reports:
            for item, count in report.counts.items():
                aggregated[item] += count
        return dict(aggregated)

    @staticmethod
    def aggregate_noisy_n(reports: list[NoisyCountReport]) -> float:
        """Average the independently noised aligned-universe sizes."""
        if not reports or any(report.noisy_n is None for report in reports):
            raise ValueError("每个客户端的第一轮报告都必须包含 noisy N")
        values = [float(report.noisy_n) for report in reports]
        return float(np.mean(np.asarray(values, dtype=np.float64)))

    def build_candidates(
        self,
        noisy_counts: dict[Item, float],
        frequent_singleton_count: int,
        candidate_count: int | None,
        min_itemset_size: int,
        max_itemset_size: int,
        first_stage_mode: str = "singleton",
    ) -> tuple[list[tuple[Item, float]], list[Candidate], list[Candidate]]:
        frequent_items = select_frequent_singletons(
            noisy_counts=noisy_counts,
            limit=frequent_singleton_count,
        )
        if first_stage_mode == "itemset":
            candidates = construct_itemset_first_candidates(
                frequent_items=frequent_items,
                n_rows=self.n_rows,
                candidate_count=candidate_count,
                min_size=min_itemset_size,
                max_size=max_itemset_size,
            )
        else:
            candidates = construct_svsm_candidates(
                frequent_items=frequent_items,
                n_rows=self.n_rows,
                candidate_count=candidate_count,
                min_size=min_itemset_size,
                max_size=max_itemset_size,
            )
        direct_candidates = construct_direct_candidates(
            frequent_items=frequent_items,
            n_rows=self.n_rows,
            min_size=min_itemset_size,
            max_size=max_itemset_size,
        )
        return frequent_items, candidates, direct_candidates

    def _blocks_for_candidate(
        self,
        candidate: Itemset,
        mode: str,
        available_report_keys: set[tuple[str, Itemset]] | None = None,
        map_m_bin_count: int | None = None,
        map_m_binning: BinningPlan | None = None,
    ) -> tuple[Itemset, ...] | None:
        if map_m_bin_count is not None:
            if self.domains is None:
                raise ValueError("MAP-M-Bin 需要公开完整值域")
            by_owner: dict[int, list[Item]] = defaultdict(list)
            for item in candidate:
                by_owner[self.owner_by_attribute[item[0]]].append(item)
            blocks = []
            for owner in sorted(by_owner):
                projection = canonical_itemset(by_owner[owner])
                binned_projection = binned_itemset(
                    projection,
                    self.domains,
                    map_m_bin_count,
                    map_m_binning,
                )
                client_id = f"client_{owner}"
                if available_report_keys is None or (
                    client_id, binned_projection
                ) in available_report_keys:
                    blocks.append(binned_projection)
                    continue
                # If an exact joint key misses top-k but all binned component items exist,
                # fall back to the lower-dimensional binned MAP-S estimation path.
                singleton_blocks = tuple(
                    binned_itemset(
                        canonical_itemset([item]),
                        self.domains,
                        map_m_bin_count,
                        map_m_binning,
                    )
                    for item in projection
                )
                if len(projection) == 1 or any(
                    (client_id, block) not in available_report_keys
                    for block in singleton_blocks
                ):
                    return None
                blocks.extend(singleton_blocks)
            return tuple(blocks)

        if mode in (
            DIRECT_UNION_COMPLEMENT,
            SINGLETON_ALPHA,
            LOCAL_TOP_SINGLETON_ALPHA,
            LOCAL_TOP_ITEMSET_COMPONENT_ALPHA,
        ):
            blocks = tuple(canonical_itemset([item]) for item in candidate)
            if available_report_keys is not None:
                for block in blocks:
                    owner = self.owner_by_attribute[block[0][0]]
                    if (f"client_{owner}", block) not in available_report_keys:
                        return None
            return blocks

        by_owner: dict[int, list[Item]] = defaultdict(list)
        for item in candidate:
            by_owner[self.owner_by_attribute[item[0]]].append(item)
        blocks = []
        for owner in sorted(by_owner):
            projection = canonical_itemset(by_owner[owner])
            client_id = f"client_{owner}"
            if available_report_keys is None or (
                client_id, projection
            ) in available_report_keys:
                blocks.append(projection)
            else:
                singleton_blocks = tuple(
                    canonical_itemset([item]) for item in projection
                )
                if len(projection) == 1 or any(
                    (client_id, block) not in available_report_keys
                    for block in singleton_blocks
                ):
                    return None
                # Fall back to reported component items when the joint block misses local top-k.
                blocks.extend(singleton_blocks)
        return tuple(blocks)

    def estimate_candidates(
        self,
        candidates: list[Candidate],
        reports: list[AlphaReport],
        mode: str,
        gamma: float,
        map_step: int,
        max_map_points: int,
        estimator: str = "map",
        map_m_bin_count: int | None = None,
        map_m_binning: BinningPlan | None = None,
        round1_noisy_counts: dict[Item, float] | None = None,
    ) -> list[CandidateEstimate]:
        binned_map_m = map_m_bin_count is not None
        if binned_map_m and (self.domains is None or round1_noisy_counts is None):
            raise ValueError("MAP-M-Bin 需要公开值域和第一轮私有直方图")
        report_by_client_key = {
            (report.client_id, report.key): report for report in reports
            if report.used_in_estimation
        }
        category_fm = estimator in {FM_FULL_ESTIMATOR, FM_OTHER_ESTIMATOR}
        marginal_count_cache: dict[tuple[str, Itemset], float] = {}
        blocks_by_candidate: dict[Itemset, tuple[Itemset, ...]] = {}
        candidate_prior_by_itemset: dict[Itemset, float] = {}
        if (
            binned_map_m
            and mode == MIXED_PROTECTED_BINNED_ITEMSET_ALPHA
            and self.domains is not None
            and map_m_bin_count is not None
        ):
            # BinP uses first-round SVSM guesses as priors within each binned candidate key.
            # Normalize them to avoid counting the same bin-intersection mass multiple times.
            # Priors use only the server's existing DP transcript and are post-processing.
            grouped_candidates: dict[Itemset, list[Candidate]] = defaultdict(list)
            for candidate in candidates:
                bucket_key = binned_itemset(
                    candidate.itemset,
                    self.domains,
                    map_m_bin_count,
                    map_m_binning,
                )
                grouped_candidates[bucket_key].append(candidate)
            for bucket_candidates in grouped_candidates.values():
                weights = np.asarray(
                    [max(float(candidate.guessed_count), 0.0) for candidate in bucket_candidates],
                    dtype=np.float64,
                )
                total_weight = float(weights.sum())
                if total_weight <= 0.0:
                    weights.fill(1.0 / len(bucket_candidates))
                else:
                    weights /= total_weight
                for candidate, weight in zip(bucket_candidates, weights, strict=True):
                    candidate_prior_by_itemset[candidate.itemset] = float(weight)
        if category_fm:
            estimable_candidates = candidates
        else:
            available_report_keys = set(report_by_client_key)
            for candidate in candidates:
                blocks = self._blocks_for_candidate(
                    candidate.itemset,
                    mode,
                    available_report_keys=available_report_keys,
                    map_m_bin_count=map_m_bin_count if binned_map_m else None,
                    map_m_binning=map_m_binning if binned_map_m else None,
                )
                if blocks is not None:
                    blocks_by_candidate[candidate.itemset] = blocks
            estimable_candidates = [
                candidate
                for candidate in candidates
                if candidate.itemset in blocks_by_candidate
            ]

        map_estimators = {
            MAP_ESTIMATOR,
            MAP_NO_FRECHET_ESTIMATOR,
            MAP_BOUNDS_ONLY_ESTIMATOR,
        }
        if estimator in map_estimators:
            # Candidates reuse many singleton reports; build read-only caches before parallel estimation.
            for report_id, report in report_by_client_key.items():
                marginal_count_cache[report_id] = private_cardinality_map_estimate(
                    report=report,
                    n_rows=self.n_rows,
                    gamma=gamma,
                    map_step=map_step,
                    max_map_points=max_map_points,
                )

        def estimate_one(candidate: Candidate) -> CandidateEstimate:
            if category_fm:
                block_reports = []
                blocks_list = []
                for attr, target_value in candidate.itemset:
                    client_id = f"client_{self.owner_by_attribute[attr]}"
                    matching = sorted(
                        (
                            key,
                            report,
                        )
                        for (report_client, key), report in report_by_client_key.items()
                        if report_client == client_id
                        and len(key) == 1
                        and key[0][0] == attr
                        and key[0][1] != target_value
                        and not report.is_complement
                    )
                    blocks_list.extend(key for key, _ in matching)
                    block_reports.extend(report for _, report in matching)
                blocks = tuple(blocks_list)
            else:
                blocks = blocks_by_candidate[candidate.itemset]
                block_reports = []
                for block in blocks:
                    owner = self.owner_by_attribute[block[0][0]]
                    client_id = f"client_{owner}"
                    try:
                        block_reports.append(report_by_client_key[(client_id, block)])
                    except KeyError as exc:
                        raise KeyError(
                            f"缺少 {client_id} 对 {block} 的 Alpha 汇报"
                        ) from exc

            if estimator in map_estimators:
                block_counts = [
                    marginal_count_cache[(report.client_id, report.key)]
                    for report in block_reports
                ]
                if estimator == MAP_ESTIMATOR:
                    # For cross-owner candidates, retain both joint and component Alpha.
                    # Refine the local block's marginal cardinality with exact nested MAP,
                    # then use it as a marginal constraint for direct intersection MAP.
                    # Joint keys thus supplement rather than replace singleton observations.
                    for index, (block, report) in enumerate(
                        zip(blocks, block_reports, strict=True)
                    ):
                        if len(block) != 2:
                            continue
                        owner = self.owner_by_attribute[block[0][0]]
                        client_id = f"client_{owner}"
                        left_key = canonical_itemset([block[0]])
                        right_key = canonical_itemset([block[1]])
                        left_report = report_by_client_key.get((client_id, left_key))
                        right_report = report_by_client_key.get((client_id, right_key))
                        if left_report is None or right_report is None:
                            continue
                        block_counts[index] = map_nested_pair_cardinality_estimate(
                            left_report=left_report,
                            right_report=right_report,
                            joint_report=report,
                            n_rows=self.n_rows,
                            gamma=gamma,
                            map_step=map_step,
                            max_map_points=max_map_points,
                            left_count=marginal_count_cache[(left_report.client_id, left_report.key)],
                            right_count=marginal_count_cache[(right_report.client_id, right_report.key)],
                        )
                nested_pair = None
                # With fallback singletons and a local pair, MAP-L-Top observes three
                # nested sets with shared user ranks. Combining them uses more information
                # than the joint key alone, with an exact distribution for this local structure.
                if (
                    estimator == MAP_ESTIMATOR
                    and len(candidate.itemset) == 2
                    and len(blocks) == 1
                    and len(blocks[0]) == 2
                ):
                    owner = self.owner_by_attribute[blocks[0][0][0]]
                    client_id = f"client_{owner}"
                    left_key = canonical_itemset([candidate.itemset[0]])
                    right_key = canonical_itemset([candidate.itemset[1]])
                    left_report = report_by_client_key.get((client_id, left_key))
                    right_report = report_by_client_key.get((client_id, right_key))
                    if left_report is not None and right_report is not None:
                        nested_pair = (left_report, right_report, block_reports[0])

                if nested_pair is not None:
                    left_report, right_report, joint_report = nested_pair
                    estimated_count = map_nested_pair_cardinality_estimate(
                        left_report=left_report,
                        right_report=right_report,
                        joint_report=joint_report,
                        n_rows=self.n_rows,
                        gamma=gamma,
                        map_step=map_step,
                        max_map_points=max_map_points,
                        left_count=marginal_count_cache[(left_report.client_id, left_report.key)],
                        right_count=marginal_count_cache[(right_report.client_id, right_report.key)],
                    )
                elif estimator == MAP_BOUNDS_ONLY_ESTIMATOR:
                    lower, upper = intersection_bounds(
                        np.asarray(block_counts, dtype=np.float64), self.n_rows
                    )
                    # Use only marginal MAP and set-theoretic bounds, without the joint Alpha likelihood.
                    estimated_count = float((lower + upper) / 2.0)
                else:
                    estimated_count = map_intersection_estimate(
                        reports=block_reports,
                        n_rows=self.n_rows,
                        gamma=gamma,
                        map_step=map_step,
                        max_map_points=max_map_points,
                        block_counts=block_counts,
                        use_frechet_lower_bound=(
                            estimator != MAP_NO_FRECHET_ESTIMATOR
                        ),
                    )
            elif estimator in {"fm", FM_INVERSE_ESTIMATOR}:
                estimated_count = union_complement_fm_estimate(
                    reports=block_reports,
                    n_rows=self.n_rows,
                    gamma=gamma,
                )
            elif category_fm:
                estimated_count = fm_category_intersection_estimate(
                    reports=block_reports,
                    n_rows=self.n_rows,
                    gamma=gamma,
                )
            else:
                raise ValueError(f"未知估计器: {estimator}")
            if binned_map_m:
                if mode == MIXED_PROTECTED_BINNED_ITEMSET_ALPHA:
                    # BinP allocates bin-intersection mass to exact candidates by SVSM priors,
                    # rather than assuming within-bin attribute independence. Protected
                    # high-value bins usually contain a single candidate with weight 1.
                    estimated_count *= candidate_prior_by_itemset.get(
                        candidate.itemset, 0.0
                    )
                else:
                    # Standard Bin retains VertiMRF-style conditionally independent within-bin recovery.
                    within_bin_mass = float(
                        np.prod(
                            [
                                within_bin_probability(
                                    attr,
                                    value,
                                    round1_noisy_counts,
                                    self.domains,
                                    map_m_bin_count,
                                    map_m_binning,
                                )
                                for attr, value in candidate.itemset
                            ],
                            dtype=np.float64,
                        )
                    )
                    estimated_count *= within_bin_mass
            return CandidateEstimate(
                itemset=candidate.itemset,
                estimated_count=estimated_count,
                guessed_count=candidate.guessed_count,
                local_blocks=blocks,
            )

        estimates = [estimate_one(candidate) for candidate in estimable_candidates]

        estimates.sort(key=lambda result: (-result.estimated_count, result.itemset))
        return estimates
