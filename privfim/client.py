from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .binning import BinningPlan, bin_membership_values, binned_itemset
from .dpfm import RankOracle, stable_seed
from .privacy import (
    dpfm_coordinate_epsilon,
    split_budget,
    split_grouped_budget,
    split_weighted_budget,
)
from .types import (
    FM_FULL_ESTIMATOR,
    FM_INVERSE_ESTIMATOR,
    FM_OTHER_ESTIMATOR,
    FM_OTHER_VALUE,
    LOCAL_ITEMSET_ALPHA,
    LOCAL_TOP_ITEMSET_ALPHA,
    LOCAL_TOP_SINGLETON_ALPHA,
    LOCAL_TOP_ITEMSET_COMPONENT_ALPHA,
    MIXED_BINNED_ITEMSET_ALPHA,
    MIXED_PROTECTED_BINNED_ITEMSET_ALPHA,
    MAP_ESTIMATOR,
    MIXED_COVER_ITEMSET_ALPHA,
    MIXED_DUMMY_ITEMSET_ALPHA,
    MIXED_ITEMSET_ALPHA,
    SINGLETON_ALPHA,
    AlphaReport,
    Candidate,
    Item,
    Itemset,
    NoisyCountReport,
    canonical_itemset,
)


@dataclass
class VerticalClient:
    client_id: str
    attributes: tuple[int, ...]
    local_data: np.ndarray
    domains: tuple[tuple[int, ...], ...] | None = None

    def __post_init__(self) -> None:
        if self.local_data.shape[1] != len(self.attributes):
            raise ValueError("客户端本地数据列与属性列表不匹配")
        if self.domains is not None and len(self.domains) <= max(self.attributes):
            raise ValueError("公开值域与属性编号不匹配")
        self._local_index = {attr: index for index, attr in enumerate(self.attributes)}
        if np.any(self.local_data == FM_OTHER_VALUE):
            raise ValueError("本地数据取值与 FM-Other 保留值冲突")

    @property
    def n_rows(self) -> int:
        return int(self.local_data.shape[0])

    def round1_report(
        self,
        epsilon: float,
        seed: int,
        n_epsilon: float | None = None,
        report_limit: int | None = None,
    ) -> NoisyCountReport:
        """Release the noisy first-round histogram and, in-protocol, noisy N."""
        epsilon_per_attribute = epsilon / len(self.attributes)
        rng = np.random.default_rng(stable_seed(seed, self.client_id, "round1"))
        counts: dict[tuple[int, int], float] = {}

        for local_index, attr in enumerate(self.attributes):
            column = self.local_data[:, local_index]
            domain = (
                self.domains[attr]
                if self.domains is not None
                else tuple(int(value) for value in np.unique(column))
            )
            for value in domain:
                frequency = int(np.count_nonzero(column == value))
                noisy = frequency + rng.laplace(0.0, 1.0 / epsilon_per_attribute)
                counts[(attr, int(value))] = max(0.0, float(noisy))

        if report_limit is not None:
            counts = dict(
                sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[
                    :report_limit
                ]
            )

        if n_epsilon is not None:
            if n_epsilon <= 0.0:
                raise ValueError("n_epsilon 必须大于 0")
            n_rng = np.random.default_rng(
                stable_seed(seed, self.client_id, "round1_noisy_n")
            )
            noisy_n = float(self.n_rows + n_rng.laplace(0.0, 1.0 / n_epsilon))
        else:
            noisy_n = None

        return NoisyCountReport(
            client_id=self.client_id,
            counts=counts,
            epsilon=epsilon,
            noisy_n=noisy_n,
            n_epsilon=0.0 if n_epsilon is None else n_epsilon,
        )

    def report_keys(
        self,
        candidates: list[Candidate],
        mode: str,
        local_key_policy: str = "all",
        local_projection_limit: int | None = None,
        report_key_limit: int | None = None,
        estimator: str = MAP_ESTIMATOR,
        local_joint_budget_weight: float = 1.0,
        noisy_singleton_counts: dict[Item, float] | None = None,
        normalization_n: float | None = None,
    ) -> tuple[Itemset, ...]:
        if estimator in {FM_FULL_ESTIMATOR, FM_OTHER_ESTIMATOR}:
            return self._fm_report_keys(candidates, estimator)

        local_attrs = set(self.attributes)
        singleton_keys: set[Itemset] = set()
        local_projection_scores: dict[Itemset, float] = {}
        local_projection_utility: dict[Itemset, float] = {}
        required_keys: set[Itemset] = set()
        fallback_frequencies: dict[Itemset, float] = {}
        singleton_frequencies = {
            candidate.itemset[0]: candidate.guessed_count
            for candidate in candidates
            if len(candidate.itemset) == 1
        }

        for candidate in candidates:
            projection = canonical_itemset(
                item for item in candidate.itemset if item[0] in local_attrs
            )
            if not projection:
                continue
            required_keys.add(projection)
            component_keys = {
                canonical_itemset([item]) for item in projection
            }
            singleton_keys.update(component_keys)
            for key in (*component_keys, projection):
                fallback_frequencies[key] = max(
                    candidate.guessed_count,
                    fallback_frequencies.get(key, float("-inf")),
                )
            if len(projection) > 1:
                local_projection_scores[projection] = max(
                    candidate.score,
                    local_projection_scores.get(projection, float("-inf")),
                )
                # Sum public candidate frequencies instead of taking the maximum score.
                # A joint key serving multiple candidates benefits from their combined mass.
                local_projection_utility[projection] = (
                    local_projection_utility.get(projection, 0.0)
                    + max(float(candidate.guessed_count), 0.0)
                )

        if mode not in {
            LOCAL_ITEMSET_ALPHA,
            LOCAL_TOP_ITEMSET_ALPHA,
            LOCAL_TOP_SINGLETON_ALPHA,
            LOCAL_TOP_ITEMSET_COMPONENT_ALPHA,
            MIXED_ITEMSET_ALPHA,
            MIXED_BINNED_ITEMSET_ALPHA,
            MIXED_PROTECTED_BINNED_ITEMSET_ALPHA,
            MIXED_COVER_ITEMSET_ALPHA,
            MIXED_DUMMY_ITEMSET_ALPHA,
        }:
            keys = singleton_keys
        elif mode in {LOCAL_ITEMSET_ALPHA, LOCAL_TOP_ITEMSET_ALPHA}:
            if mode == LOCAL_TOP_ITEMSET_ALPHA:
                if local_projection_limit is None or local_projection_limit < 0:
                    raise ValueError(
                        "MAP-L-Top 必须设置非负的 local_projection_limit"
                    )
                # Select joint keys from public candidates, then retain component items
                # only for uncovered candidates. Joint keys replace existing keys instead
                # of always extending MAP-S's key set and diluting its budget.
                ranked_projections = sorted(
                    local_projection_scores,
                    key=lambda key: (
                        -(
                            max(local_projection_utility.get(key, 0.0), 0.0)
                            * max(len(key) - 1, 1)
                        ),
                        -max(local_projection_scores[key], 0.0),
                        key,
                    ),
                )
                selected_projections = set(
                    ranked_projections[:local_projection_limit]
                )
                fallback_singletons: set[Itemset] = set()
                for candidate in candidates:
                    projection = canonical_itemset(
                        item for item in candidate.itemset if item[0] in local_attrs
                    )
                    if not projection:
                        continue
                    if len(projection) == 1 or projection not in selected_projections:
                        fallback_singletons.update(
                            canonical_itemset([item]) for item in projection
                        )
                keys = selected_projections | fallback_singletons
            else:
                # MAP-L reports unique nonempty local projections of candidates in S.
                # Report component items separately only when another candidate needs them.
                keys = required_keys
        elif mode == LOCAL_TOP_SINGLETON_ALPHA:
            # No-guessing ablation: select at most k candidate items by raw local counts.
            # This experimental comparison leaves the MAP post-processor unchanged.
            if report_key_limit is None or report_key_limit < 1:
                raise ValueError("MAP-NoGuess-Items 必须设置正的 report_key_limit")
            ranked = sorted(
                singleton_keys,
                key=lambda key: (-float(np.count_nonzero(self._membership(key))), key),
            )
            keys = set(ranked[:report_key_limit])
        elif mode == LOCAL_TOP_ITEMSET_COMPONENT_ALPHA:
            # No-guessing ablation: rank projections by local support, then report their
            # component singleton Alpha sketches. The report count can be up to k*|X|.
            if report_key_limit is None or report_key_limit < 1:
                raise ValueError(
                    "MAP-NoGuess-ItemsetComponents 必须设置正的 report_key_limit"
                )
            ranked_projections = sorted(
                required_keys,
                key=lambda key: (-float(np.count_nonzero(self._membership(key))), key),
            )
            selected = ranked_projections[:report_key_limit]
            keys = {
                canonical_itemset([item])
                for projection in selected
                for item in projection
            }
        elif mode in {
            MIXED_ITEMSET_ALPHA,
            MIXED_BINNED_ITEMSET_ALPHA,
            MIXED_PROTECTED_BINNED_ITEMSET_ALPHA,
            MIXED_COVER_ITEMSET_ALPHA,
            MIXED_DUMMY_ITEMSET_ALPHA,
        }:
            if report_key_limit is None or report_key_limit < 0:
                raise ValueError("MAP-M 类模式必须设置非负的 report_key_limit")
            # Zero removes the upload cap while retaining the same public scoring rule.
            if report_key_limit == 0:
                report_key_limit = len(singleton_keys | required_keys)

            # Reconstruct SVSM guesses from released first-round Laplace item counts.
            # Never use raw local supports to choose the reported key set.
            reference_n = float(
                self.n_rows if normalization_n is None else normalization_n
            )
            if reference_n <= 0.0:
                raise ValueError("normalization_n 必须大于 0")
            public_singleton_counts = (
                noisy_singleton_counts
                if noisy_singleton_counts is not None
                else singleton_frequencies
            )
            key_frequencies = {}
            for key in singleton_keys | required_keys:
                if all(item in public_singleton_counts for item in key):
                    frequency = reference_n
                    for item in key:
                        frequency *= float(
                            np.clip(
                                public_singleton_counts[item], 0.0, reference_n
                            )
                        ) / reference_n
                else:
                    frequency = fallback_frequencies[key]
                key_frequencies[key] = frequency
            if mode in {
                MIXED_ITEMSET_ALPHA,
                MIXED_BINNED_ITEMSET_ALPHA,
                MIXED_PROTECTED_BINNED_ITEMSET_ALPHA,
            }:
                # Singleton and joint keys compete for k slots on the same count scale.
                # Singletons use noisy counts; r-item keys use their product / N^(r-1).
                # Deterministic ranking is post-processing of the first-round DP output.
                ranked_keys = sorted(
                    singleton_keys | required_keys,
                    key=lambda key: (-key_frequencies[key], key),
                )
                keys = set(ranked_keys[:report_key_limit])
            else:
                # MAP-M-Cover requires only keys whose local projections are singletons.
                # Prefer joint keys that combine several singleton blocks into one local block.
                # Scores use public first-round candidates, never raw local supports.
                singleton_utility = {key: 0.0 for key in singleton_keys}
                projection_utility = {key: 0.0 for key in local_projection_scores}
                mandatory_singletons: set[Itemset] = set()
                for candidate in candidates:
                    weight = max(float(candidate.guessed_count), 0.0) / max(
                        reference_n, 1.0
                    )
                    projection = canonical_itemset(
                        item for item in candidate.itemset if item[0] in local_attrs
                    )
                    if not projection:
                        continue
                    if len(projection) == 1:
                        mandatory_singletons.add(projection)
                        singleton_utility[projection] += weight
                        continue
                    for item in projection:
                        singleton_utility[canonical_itemset([item])] += (
                            weight / len(projection)
                        )
                    if projection in projection_utility:
                        # One joint key serves this candidate and removes len(projection)-1 blocks.
                        projection_utility[projection] += weight * (
                            len(projection) - 1
                        )

                ranked_mandatory = sorted(
                    mandatory_singletons,
                    key=lambda key: (-singleton_utility[key], -key_frequencies[key], key),
                )
                keys = set(ranked_mandatory[:report_key_limit])
                remaining = max(report_key_limit - len(keys), 0)
                if remaining:
                    ranked_projections = sorted(
                        projection_utility,
                        key=lambda key: (
                            -projection_utility[key],
                            -key_frequencies[key],
                            key,
                        ),
                    )
                    keys.update(ranked_projections[:remaining])
                remaining = max(report_key_limit - len(keys), 0)
                if remaining:
                    ranked_singletons = sorted(
                        singleton_keys - keys,
                        key=lambda key: (
                            -singleton_utility[key],
                            -key_frequencies[key],
                            key,
                        ),
                    )
                    keys.update(ranked_singletons[:remaining])
        else:
            keys = set(singleton_keys)
            ranked_projections = sorted(
                local_projection_scores,
                key=lambda key: (-local_projection_scores[key], key),
            )
            if local_key_policy == "capped":
                ranked_projections = ranked_projections[:local_projection_limit]
            keys.update(ranked_projections)
        return tuple(sorted(keys))

    def _target_values(self, candidates: list[Candidate]) -> dict[int, set[int]]:
        target_values: dict[int, set[int]] = {}
        local_attrs = set(self.attributes)
        for candidate in candidates:
            for attr, value in candidate.itemset:
                if attr in local_attrs:
                    target_values.setdefault(attr, set()).add(value)
        return target_values

    def _fm_report_keys(
        self, candidates: list[Candidate], estimator: str
    ) -> tuple[Itemset, ...]:
        """Build positive category bins for full-domain or frequency-compressed FM."""
        target_values = self._target_values(candidates)
        keys: set[Itemset] = set()
        for attr, targets in target_values.items():
            domain = (
                set(self.domains[attr])
                if self.domains is not None
                else {
                    int(value)
                    for value in np.unique(
                        self.local_data[:, self._local_index[attr]]
                    )
                }
            )
            if estimator == FM_FULL_ESTIMATOR:
                values = domain
            else:
                # Frequency-aware FM always reserves an OTHER bin per relevant attribute.
                # Budget it even when targets cover the domain and the bin is empty.
                values = {*(targets & domain), FM_OTHER_VALUE}
            keys.update(canonical_itemset([(attr, value)]) for value in values)
        return tuple(sorted(keys))

    def _membership(self, key: Itemset) -> np.ndarray:
        mask = np.ones(self.n_rows, dtype=bool)
        for attr, value in key:
            mask &= self.local_data[:, self._local_index[attr]] == value
        return mask

    def _fm_membership(
        self, key: Itemset, target_values: dict[int, set[int]]
    ) -> np.ndarray:
        if len(key) != 1:
            raise ValueError("FM 类别桶必须是单属性单值键")
        attr, value = key[0]
        column = self.local_data[:, self._local_index[attr]]
        if value == FM_OTHER_VALUE:
            targets = np.asarray(sorted(target_values[attr]), dtype=np.int64)
            return ~np.isin(column, targets)
        return column == value

    def round2_reports(
        self,
        candidates: list[Candidate],
        mode: str,
        epsilon: float,
        delta: float,
        oracle: RankOracle,
        seed: int,
        local_key_policy: str = "all",
        local_projection_limit: int | None = None,
        report_key_limit: int | None = None,
        estimator: str = MAP_ESTIMATOR,
        local_joint_budget_weight: float = 1.0,
        map_m_bin_count: int | None = None,
        map_m_binning: BinningPlan | None = None,
        noisy_singleton_counts: dict[Item, float] | None = None,
        normalization_n: float | None = None,
    ) -> list[AlphaReport]:
        keys = self.report_keys(
            candidates,
            mode,
            local_key_policy=local_key_policy,
            local_projection_limit=local_projection_limit,
            report_key_limit=report_key_limit,
            estimator=estimator,
            noisy_singleton_counts=noisy_singleton_counts,
            normalization_n=normalization_n,
        )
        if not keys:
            return []
        binned_map_m = (
            map_m_bin_count is not None
            and mode in {
                MIXED_BINNED_ITEMSET_ALPHA,
                MIXED_PROTECTED_BINNED_ITEMSET_ALPHA,
            }
            and estimator == MAP_ESTIMATOR
        )
        if binned_map_m:
            if self.domains is None:
                raise ValueError("MAP-M-Bin 需要公开完整值域")
            # Select top-k keys from exact public first-round candidates, then merge keys
            # in the same bin during encoding to avoid generating redundant Alpha sketches.
            keys = tuple(
                sorted(
                    {
                        binned_itemset(
                            key,
                            self.domains,
                            map_m_bin_count,
                            map_m_binning,
                        )
                        for key in keys
                    }
                )
            )
        grouped_map_m = (
            mode
            in {
                MIXED_ITEMSET_ALPHA,
                MIXED_BINNED_ITEMSET_ALPHA,
                MIXED_PROTECTED_BINNED_ITEMSET_ALPHA,
                MIXED_COVER_ITEMSET_ALPHA,
                MIXED_DUMMY_ITEMSET_ALPHA,
            }
            and estimator not in {FM_FULL_ESTIMATOR, FM_OTHER_ESTIMATOR}
            and local_joint_budget_weight > 0
        )
        if grouped_map_m:
            # Group MAP-M singleton/joint keys by attribute projection. For example,
            # {male}/{female} reuse the gender-group budget instead of splitting it by bin.
            grouped_budget = split_grouped_budget(epsilon, delta, keys)
            key_budgets = [grouped_budget.budget_by_key[key] for key in keys]
        elif (
            mode
            in {
                MIXED_ITEMSET_ALPHA,
                MIXED_BINNED_ITEMSET_ALPHA,
                MIXED_PROTECTED_BINNED_ITEMSET_ALPHA,
                MIXED_COVER_ITEMSET_ALPHA,
                MIXED_DUMMY_ITEMSET_ALPHA,
            }
            and estimator not in {FM_FULL_ESTIMATOR, FM_OTHER_ESTIMATOR}
            and local_joint_budget_weight == 0
        ):
            # Disable MAP-M group budgets for this ablation: every reported key receives
            # the same budget, whether it represents a singleton or a joint itemset.
            per_key_budget = split_budget(epsilon, delta, len(keys))
            key_budgets = [per_key_budget] * len(keys)
        else:
            if local_joint_budget_weight <= 0:
                raise ValueError("联合键预算权重必须大于 0")
            key_budgets = split_weighted_budget(
                epsilon=epsilon,
                delta=delta,
                weights=[
                    local_joint_budget_weight
                    if estimator not in {FM_FULL_ESTIMATOR, FM_OTHER_ESTIMATOR}
                    and len(key) > 1
                    else 1.0
                    for key in keys
                ],
            )
        reports = []
        target_values = self._target_values(candidates)
        category_fm = estimator in {FM_FULL_ESTIMATOR, FM_OTHER_ESTIMATOR}
        inverse_fm = estimator in {"fm", FM_INVERSE_ESTIMATOR}

        # Generate Alpha with phantom elements even when local support is zero.
        # Reporting decisions must not expose private data through the communication pattern.
        for key, budget in zip(keys, key_budgets, strict=True):
            coordinate_epsilon = dpfm_coordinate_epsilon(
                budget.epsilon,
                budget.delta,
                oracle.m,
            )
            if category_fm:
                membership = self._fm_membership(key, target_values)
            elif inverse_fm:
                membership = ~self._membership(key)
            elif binned_map_m:
                membership = self._binned_membership(
                    key, map_m_bin_count, map_m_binning
                )
            else:
                # MAP-S/MAP-L privatize only the positive sets required by candidates.
                # Other values of the same attribute generate no Alpha and receive no budget.
                membership = self._membership(key)
            # MAP-L-Top fallback items share MAP-S's experimental randomness to isolate
            # the effect of additional joint keys. Deployments release only one protocol output.
            # M should reduce exactly to MAP-S when the key cap is inactive. Even with a cap,
            # component items reuse MAP-S randomness so independent noise does not dominate comparisons.
            randomness_mode = (
                f"{mode}_bin_{map_m_bin_count}"
                if binned_map_m
                else
                SINGLETON_ALPHA
                if len(key) == 1
                and mode
                in {
                    LOCAL_TOP_ITEMSET_ALPHA,
                    MIXED_ITEMSET_ALPHA,
                    MIXED_BINNED_ITEMSET_ALPHA,
                    MIXED_PROTECTED_BINNED_ITEMSET_ALPHA,
                    MIXED_COVER_ITEMSET_ALPHA,
                }
                else MIXED_ITEMSET_ALPHA
                if mode in {
                    MIXED_COVER_ITEMSET_ALPHA,
                    MIXED_DUMMY_ITEMSET_ALPHA,
                }
                else mode
            )
            alpha, phantom_count = oracle.private_alpha(
                membership=membership,
                epsilon=budget.epsilon,
                delta=budget.delta,
                # Cover and legacy MAP-M share experimental randomness for common keys.
                # This isolates key selection from DP noise; deployments release one mode at a time.
                random_seed=stable_seed(
                    seed,
                    self.client_id,
                    estimator,
                    randomness_mode,
                    key,
                ),
            )
            reports.append(
                AlphaReport(
                    client_id=self.client_id,
                    key=key,
                    alpha=alpha,
                    epsilon=budget.epsilon,
                    delta=budget.delta,
                    phantom_count=phantom_count,
                    is_complement=inverse_fm,
                    coordinate_epsilon=coordinate_epsilon,
                    # Dummy excludes only joint keys; singleton Alpha remains available for fallback.
                    used_in_estimation=not (
                        mode == MIXED_DUMMY_ITEMSET_ALPHA and len(key) > 1
                    ),
                )
            )
        return reports

    def _binned_membership(
        self,
        bin_key: Itemset,
        bin_count: int,
        binning: BinningPlan | None = None,
    ) -> np.ndarray:
        """Build positive-set membership for a public bin key."""
        if self.domains is None:
            raise ValueError("MAP-M-Bin 需要公开完整值域")
        mask = np.ones(self.n_rows, dtype=bool)
        for attr, bin_index in bin_key:
            bucket_values = bin_membership_values(
                attr, bin_index, self.domains, bin_count, binning
            )
            column = self.local_data[:, self._local_index[attr]]
            mask &= np.isin(column, bucket_values)
        return mask
