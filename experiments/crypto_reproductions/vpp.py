"""VPP-OFIM reproduction and explicitly identified top-k adaptations.

Implements paper Sections 5.1--5.3 using real Paillier arithmetic. The comparison
in Section 6.1 does not specify equality/overflow conventions. Here a nonzero
odd difference and bounded masks give exact integer comparisons. This is a
documented completion of the paper sketch, NOT a new proof of its security.
The separate roles execute in one process. No network latency is simulated.
"""
from __future__ import annotations

import hashlib
import hmac
import itertools
import math
import secrets
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

from experiments.crypto_protocols import generate_paillier_keypair, gmpy2


def modpow(value, exponent, modulus):
    if exponent < 0:
        value = pow(value, -1, modulus)
        exponent = -exponent
    return int(gmpy2.powmod(value, exponent, modulus)) if gmpy2 else pow(value, exponent, modulus)


class BlindedEvaluator:
    """Evaluator only opens the randomized comparison message, not its operands."""

    def __init__(self, private):
        self._private = private
        self.public = private.public
        self.calls = 0

    def encrypted_sign(self, masked_ciphertext):
        self.calls += 1
        signed_residue = self._private.decrypt(masked_ciphertext)
        return self.public.encrypt(int(signed_residue <= self.public.n // 2))


class BlindedComparison:
    """Exact > on bounded nonnegative integers using the paper's masked-sign idea.

    This class has no private key. Its result is encrypted. Random masks and a
    hidden random orientation prevent direct opening of x, y, or x-y, but the
    full simulation-security argument of the paper is not re-established here.
    """

    def __init__(self, public, evaluator, bound, mask_bits=128):
        if bound < 1 or mask_bits < 32:
            raise ValueError("invalid comparison bounds")
        if (2 * bound + 2) * (1 << mask_bits) >= public.n // 2:
            raise ValueError("comparison can overflow the signed Paillier plaintext range")
        self.public, self.evaluator = public, evaluator
        self.bound, self.mask_bits = bound, mask_bits

    def greater(self, encrypted_x, encrypted_y):
        pk = self.public
        difference = pk.add(encrypted_x, modpow(encrypted_y, -1, pk.n2))
        # 2(x-y)-1 在相等时为 -1，正负号始终对应严格大于关系。
        odd_difference = pk.add(modpow(difference, 2, pk.n2), pk.encrypt(pk.n - 1))
        flipped = secrets.randbits(1)
        if flipped:
            odd_difference = modpow(odd_difference, -1, pk.n2)
        multiplier = (1 << (self.mask_bits - 1)) + secrets.randbits(self.mask_bits - 1)
        offset = secrets.randbelow(multiplier)
        message = pk.add(modpow(odd_difference, multiplier, pk.n2), pk.encrypt(offset))
        reply = self.evaluator.encrypted_sign(message)
        return pk.add(pk.encrypt(1), modpow(reply, -1, pk.n2)) if flipped else reply


def subsets(items):
    ordered = sorted(items)
    return {frozenset(x) for size in range(1, len(ordered) + 1)
            for x in itertools.combinations(ordered, size)}


def verification_pattern(owner, threshold, width=3):
    """One valid sampled negative-border set H and all its immediate subsets.

    Uses the paper's construction with one H. A complete conflict graph is
    colored into width rows, then repeated until every AFI exceeds threshold.
    No inconsistent AII constraints are discarded.
    """
    if threshold < 0 or not 2 <= width <= 4:
        raise ValueError("threshold >= 0 and 2 <= verification width <= 4 required")
    alphabet = tuple(f"verification:{owner}:{secrets.token_hex(16)}" for _ in range(width))
    h = frozenset(alphabet)
    afi = {h - {item} for item in h}
    aii = {h}
    rows = [set(key) for key in sorted(afi, key=lambda x: tuple(sorted(x)))
            for _ in range(threshold + 1)]
    if any(sum(key <= row for row in rows) <= threshold for key in afi):
        raise AssertionError("AFI generation failed")
    if any(sum(key <= row for row in rows) > threshold for key in aii):
        raise AssertionError("AII generation failed")
    return afi, aii, rows


def frequency_padding(rows, universe, anonymity):
    """Exact frequency grouping, followed by packed deficit transactions.

    Implements the frequency-hiding invariant specified by VPP §5.2. This is a
    deterministic grouping implementation, not a claim to reproduce every
    optimization in the separately cited Giannotti implementation.
    """
    if anonymity < 2 or len(universe) < anonymity:
        raise ValueError("frequency anonymity must be between 2 and the domain size")
    frequencies = Counter(item for row in rows for item in row)
    order = sorted(universe, key=lambda item: (-frequencies[item], item))
    groups = [order[i:i + anonymity] for i in range(0, len(order), anonymity)]
    if len(groups) > 1 and len(groups[-1]) < anonymity:
        groups[-2].extend(groups.pop())
    deficit = {}
    for group in groups:
        target = max(frequencies[item] for item in group)
        deficit.update({item: target - frequencies[item] for item in group})
    padding = [{item for item, missing in deficit.items() if missing > level}
               for level in range(max(deficit.values(), default=0))]
    final_counts = frequencies + Counter(item for row in padding for item in row)
    classes = Counter(final_counts[item] for item in universe)
    if min(classes.values()) < anonymity:
        raise AssertionError("frequency hiding invariant failed")
    return padding, dict(classes)


def verify_border(afi, aii, returned_frequent):
    required = set().union(*(subsets(key) for key in afi))
    missing = required - returned_frequent
    inserted = aii & returned_frequent
    return {"passed": not missing and not inserted,
            "required_frequent_patterns": len(required),
            "missing_patterns": len(missing), "inserted_infrequent_patterns": len(inserted)}


@dataclass(frozen=True)
class CloudRow:
    tid: bytes
    labels: frozenset[str]
    encrypted_real: int


def _prepare_owner(ds, owner, pk, tid_secret, threshold, anonymity, verification_width):
    attributes = ds.partitions[owner]
    universe = {f"item:{a}:{int(v)}" for a in attributes for v in ds.domains[a]}
    real = [{f"item:{a}:{int(row[a])}" for a in attributes} for row in ds.data]
    afi, aii, artificial = verification_pattern(owner, threshold, verification_width)
    universe.update(itertools.chain.from_iterable(artificial))
    padding, classes = frequency_padding(real + artificial, universe, anonymity)
    # 替代密码字典保留在数据方。云端仅看到随机标签和它们的归属方。
    substitution = {item: secrets.token_hex(16) for item in sorted(universe)}
    reverse = {label: item for item, label in substitution.items()}
    records = []
    for kind, transactions, flag in (("real", real, 1), ("verification", artificial, 1),
                                      ("fictitious", padding, 0)):
        for index, transaction in enumerate(transactions):
            # 真用户 TID 跨方一致，新增记录使用不冲突的命名空间。
            tid = f"real:{index}" if kind == "real" else f"{kind}:{owner}:{index}"
            records.append(CloudRow(hmac.digest(tid_secret, tid.encode(), "sha256"),
                                    frozenset(substitution[x] for x in transaction), pk.encrypt(flag)))
    encoded_afi = {frozenset(substitution[x] for x in key) for key in afi}
    encoded_aii = {frozenset(substitution[x] for x in key) for key in aii}
    return records, reverse, encoded_afi, encoded_aii, {
        "owner": owner, "real_rows": len(real), "verification_rows": len(artificial),
        "fictitious_rows": len(padding), "domain_size_with_verification": len(universe),
        "frequency_class_sizes": sorted(classes.values()),
        "frequency_anonymity_verified": min(classes.values()) >= anonymity,
        "verification_erv": 1, "fictitious_erv": 0,
    }


def eclat(label_tidsets, max_size, candidate_limit):
    """Enumerate every nonempty apparent support, without an oracle threshold."""
    count = 0

    def expand(prefix, suffix):
        nonlocal count
        for i, (label, tids) in enumerate(suffix):
            key = prefix + (label,)
            count += 1
            if count > candidate_limit:
                raise RuntimeError("Eclat candidate limit reached; no truncated result will be returned")
            yield key, tids
            if len(key) < max_size:
                following = [(other, tids & other_tids) for other, other_tids in suffix[i + 1:]]
                yield from expand(key, [(other, joined) for other, joined in following if joined])

    yield from expand((), [(key, tids) for key, tids in sorted(label_tidsets.items()) if tids])


def _cloud_mine(owner_records, pk, compare, encrypted_threshold, max_size, candidate_limit,
                selected_labels=None, public_label_owner=None, progress=None):
    """Only disguised labels, encrypted RVs, and the blinded SC interface enter."""
    tids, encrypted_flags, label_owner = {}, {}, dict(public_label_owner or {})
    label_tidsets = defaultdict(set)
    for owner, records in enumerate(owner_records):
        for row in records:
            index = tids.setdefault(row.tid, len(tids))
            encrypted_flags[owner, index] = row.encrypted_real
            for label in row.labels:
                label_owner[label] = owner
                label_tidsets[label].add(index)
    replies = []
    indicator_calls = 0
    if selected_labels is None:
        queries = eclat(label_tidsets, max_size, candidate_limit)
    else:
        if len(selected_labels) > candidate_limit:
            raise RuntimeError("selected candidate limit reached")
        queries = ((labels, set.intersection(*(label_tidsets[label] for label in labels)))
                   for labels in sorted(selected_labels))
    for labels, matching in queries:
        involved = {label_owner[label] for label in labels}
        required_minus_one = pk.encrypt(len(involved) - 1)
        encrypted_support = pk.encrypt(0)
        for index in sorted(matching):
            total = pk.encrypt(0)
            for owner in sorted(involved):
                total = pk.add(total, encrypted_flags[owner, index])
            # RV 都是 0/1，因此 sum > |D(S)|-1 等价于所有相关分片均为真。
            indicator = compare.greater(total, required_minus_one)
            encrypted_support = pk.add(encrypted_support, indicator)
            indicator_calls += 1
        encrypted_frequent = compare.greater(encrypted_support, encrypted_threshold)
        replies.append((labels, encrypted_support, encrypted_frequent))
        if progress is not None:
            progress("cloud_candidates", len(replies), None if selected_labels is None else len(selected_labels))
    return replies, {"apparent_candidates": len(replies), "indicator_comparisons": indicator_calls,
                     "threshold_comparisons": len(replies), "joined_rows": len(tids)}


def run_vpp(ds, *, threshold=0, max_size=4, anonymity=2, verification_width=3,
            key_bits=2048, candidate_limit=100_000, candidates=None, progress=None):
    if not 0 <= threshold <= ds.n_rows or max_size < 1:
        raise ValueError("invalid mining parameters")
    started = time.perf_counter()
    phase = time.perf_counter()
    public, private = generate_paillier_keypair(key_bits)
    keygen = time.perf_counter() - phase
    if progress is not None:
        progress("keygen", 1, 1)
    phase = time.perf_counter()
    tid_secret = secrets.token_bytes(32)
    owners = []
    for owner in range(len(ds.partitions)):
        owners.append(_prepare_owner(ds, owner, public, tid_secret, threshold, anonymity, verification_width))
        if progress is not None:
            progress("owner_prepare_encrypt", owner + 1, len(ds.partitions))
    selected_labels = None
    if candidates is not None:
        substitutions = [{item: label for label, item in owner[1].items()} for owner in owners]
        selected_labels = set()
        for key in candidates:
            if len({attribute for attribute, _ in key}) != len(key) or len(key) > max_size:
                raise ValueError(f"invalid selected itemset: {key}")
            selected_labels.add(tuple(sorted(substitutions[ds.owner_by_attribute[attribute]][
                f"item:{attribute}:{value}"] for attribute, value in key)))
        for owner in owners:
            selected_labels.update(tuple(sorted(key)) for key in owner[2] | owner[3])
            selected_labels.update(tuple(sorted(key)) for pattern in owner[2]
                                   for key in subsets(pattern))
    encrypted_threshold = public.encrypt(threshold)
    bound = max(sum(len(owner[0]) for owner in owners), len(owners), threshold, 1)
    evaluator = BlindedEvaluator(private)
    compare = BlindedComparison(public, evaluator, bound, mask_bits=min(128, key_bits // 4))
    preparation = time.perf_counter() - phase
    phase = time.perf_counter()
    public_label_owner = {label: index for index, owner in enumerate(owners)
                          for label in owner[1]}
    replies, counts = _cloud_mine([owner[0] for owner in owners], public, compare,
                                 encrypted_threshold, max(max_size, verification_width), candidate_limit,
                                 selected_labels, public_label_owner, progress)
    cloud = time.perf_counter() - phase
    phase = time.perf_counter()
    # 数据方协作解密、还原标签与排序是支持 top-k 的显式扩展。
    reverse = {label: item for owner in owners for label, item in owner[1].items()}
    returned_frequent, supports, flags = set(), {}, {}
    for labels, encrypted_count, encrypted_flag in replies:
        count, frequent = private.decrypt(encrypted_count), private.decrypt(encrypted_flag)
        if frequent != int(count > threshold):
            raise AssertionError("encrypted threshold comparison disagreement")
        if frequent:
            returned_frequent.add(frozenset(labels))
        plain = [reverse[label] for label in labels]
        if any(not item.startswith("item:") for item in plain) or len(plain) > max_size:
            continue
        key = tuple(sorted(tuple(map(int, item.split(":")[1:])) for item in plain))
        if len({a for a, _ in key}) != len(key):
            continue
        supports[key], flags[key] = count, frequent
    checks = [verify_border(owner[2], owner[3], returned_frequent) for owner in owners]
    if not all(check["passed"] for check in checks):
        raise RuntimeError(f"VPP verification rejected cloud output: {checks}")
    if progress is not None:
        progress("decrypt_verify", len(replies), len(replies))
    finish = time.perf_counter() - phase
    total = time.perf_counter() - started
    return supports, {
        "backend": f"VPP-Paillier-{key_bits}-blinded-comparison", "key_bits": key_bits,
        "native_crypto_measured": True, "publication_eligible": False,
        "paper_alignment_verified": False, "workflow_implemented": True,
        "timing_scope": "protocol_end_to_end", "protocol_seconds": total,
        "keygen_seconds": keygen, "owner_prepare_encrypt_seconds": preparation,
        "cloud_seconds": cloud, "owner_decrypt_verify_seconds": finish,
        "candidate_source": "disguised_database_complete_eclat" if candidates is None else "shared_round1_candidates",
        "selected_candidate_count": None if candidates is None else len(candidates),
        "discovery_public_threshold": 0, "encrypted_support_threshold": threshold,
        "frequency_anonymity": anonymity, "verification_width": verification_width,
        "owners": [owner[4] for owner in owners], "border_checks": checks,
        "comparison_calls": evaluator.calls, **counts,
        "adaptations": ["conservative Eclat threshold 0, independent of private supports",
                        "owners collaboratively recover counts for top-k ranking",
                        "bounded odd-difference completion of the paper comparison sketch",
                        "frequency groups use exact greedy padding",
                        "one sampled negative-border challenge per owner"],
        "limits": ["in-process role separation, without network or key-distribution latency",
                   "comparison simulation security has not been independently proved",
                   "frequency anonymity is not a proof against all table-domain side information",
                   "border verification is sampled and does not detect every possible malicious omission"],
    }
