"""Legacy reference models and support-query prototypes, NOT full reproductions.

This module deliberately separates protocol execution from candidate generation.
The papers describe encrypted support computation, while candidate discovery is
fed by a public/authorised candidate list in this reproducible runner.  The
transcript records that boundary instead of silently treating an oracle as a
cryptographic implementation.

NIPP-FIM provides both a black-box reference backend for TFHE Boolean gates and
an optional native Concrete backend.  The original NIPP-FIM paper describes an
encrypted cloud database; this runner supplies the VFL adaptation by keeping
each owner's column block separate until the CSP-side conjunction.  The latter
backend is used only when ``concrete-python`` is installed.

VPP-OFIM uses real Paillier primitives, but its comparison functionality opens
plaintexts and its bounded padding is not the paper's frequency-hiding scheme.
Neither an opaque Python object nor real encryption alone validates a protocol.
Publication eligibility is therefore fail-closed. See docs/crypto_audit_20260928.md.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from concurrent.futures import ThreadPoolExecutor
import secrets
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

try:  # Optional native big-integer arithmetic for realistic VPP timing runs.
    import gmpy2
except ImportError:  # pragma: no cover - exercised when the optional wheel is absent.
    gmpy2 = None

from experiments.run_suite import _dataset_specs
from privfim.candidate import construct_svsm_candidates
from privfim.data import VerticalDataset, load_vertical_csv
from privfim.metrics import evaluate_estimates, exact_candidate_supports
from privfim.types import CandidateEstimate, Item, Itemset, canonical_itemset


# ---------------------------------------------------------------------------
# Shared transcript and candidate boundary
# ---------------------------------------------------------------------------


@dataclass
class ProtocolTranscript:
    scheme: str
    backend: str
    candidate_source: str
    candidate_count: int = 0
    real_rows: int = 0
    artificial_rows: int = 0
    operations: dict[str, int] = field(default_factory=dict)
    communication_bytes: int = 0
    ciphertext_bytes: int = 0
    proof_bytes: int = 0
    elapsed_seconds: float = 0.0
    timings: dict[str, float] = field(default_factory=dict)
    execution_status: str = "not_started"
    metadata: dict[str, Any] = field(default_factory=dict)

    def op(self, name: str, amount: int = 1) -> None:
        self.operations[name] = self.operations.get(name, 0) + int(amount)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        # Running a simulation does not measure the complete cryptographic protocol's runtime.
        requirements = {
            "native_cryptography": self.metadata.get("cryptographic_wall_clock") is True,
            "complete_protocol": self.metadata.get("protocol_complete") is True,
            "paper_alignment": self.metadata.get("paper_alignment_verified") is True,
            "global_truth": self.metadata.get("global_truth_verified") is True,
            "end_to_end_timing": self.metadata.get("timing_scope") == "protocol_end_to_end",
            "no_candidate_oracle": self.candidate_source in {
                "encrypted_complete_discovery", "public_full_universe",
            },
            "successful_execution": self.execution_status.startswith("protocol_executed"),
        }
        result["native_crypto_measured"] = requirements["native_cryptography"]
        result["runtime_verified"] = all(requirements.values())
        result["publication_eligible"] = result["runtime_verified"]
        result["validation_failures"] = [key for key, valid in requirements.items() if not valid]
        return result


def _item_universe(dataset: VerticalDataset) -> list[Item]:
    return [
        (attribute, int(value))
        for attribute, domain in enumerate(dataset.domains)
        for value in domain
    ]


def _candidate_list(
    dataset: VerticalDataset,
    k: int,
    max_itemset_size: int,
    candidate_multiplier: float,
) -> list[Itemset]:
    """Build the reproducible public candidate list used by both protocols.

    The first-round singleton ranking uses exact counts only as an explicit
    oracle.  This is intentionally labelled in every transcript; replacing it
    with the paper's encrypted candidate-discovery routine is a separate task.
    """
    singleton_counts = [
        ((attribute, int(value)), float(np.count_nonzero(dataset.data[:, attribute] == value)))
        for attribute, domain in enumerate(dataset.domains)
        for value in domain
    ]
    singleton_counts.sort(key=lambda pair: (-pair[1], pair[0]))
    first_items = singleton_counts[: max(1, int(k))]
    limit = max(1, math.ceil(float(candidate_multiplier) * k))
    candidates = construct_svsm_candidates(
        first_items,
        dataset.n_rows,
        limit,
        1,
        max_itemset_size,
    )
    return [candidate.itemset for candidate in candidates]


def _metrics(
    dataset: VerticalDataset,
    candidates: Sequence[Itemset],
    k: int,
    estimated_counts: dict[Itemset, float] | None = None,
) -> dict[str, float]:
    supports = {itemset: dataset.support(itemset) for itemset in candidates}
    estimated_counts = estimated_counts or {
        itemset: float(count) for itemset, count in supports.items()
    }
    estimates = [
        CandidateEstimate(
            itemset=itemset,
            estimated_count=float(estimated_counts[itemset]),
            guessed_count=float(supports[itemset]),
            local_blocks=(itemset,),
        )
        for itemset in sorted(
            candidates,
            key=lambda itemset: (-float(estimated_counts[itemset]), itemset),
        )
    ]
    return evaluate_estimates(
        estimates,
        supports,
        k,
        dataset.n_rows,
        set(candidates),
    ).to_dict()


def _rank_candidate_metrics(
    dataset: VerticalDataset,
    candidates: Sequence[Itemset],
    k: int,
    estimated_counts: dict[Itemset, float],
    evaluation_supports: dict[Itemset, int] | None = None,
) -> dict[str, float]:
    """Offline evaluation. Never silently substitute candidate-restricted truth."""
    if evaluation_supports is not None:
        supports = evaluation_supports
    else:
        from experiments.global_topk_search import exact_global_topk

        supports, _ = exact_global_topk(dataset, k, max_size=4)
        supports.update({itemset: dataset.support(itemset) for itemset in candidates})
    if not set(candidates) <= set(supports):
        raise ValueError("evaluation supports must include every encrypted candidate")
    estimates = [
        CandidateEstimate(
            itemset=itemset,
            estimated_count=float(estimated_counts[itemset]),
            guessed_count=float(estimated_counts[itemset]),
            local_blocks=(itemset,),
        )
        for itemset in sorted(
            candidates,
            key=lambda itemset: (-float(estimated_counts[itemset]), itemset),
        )
    ]
    return evaluate_estimates(
        estimates,
        supports,
        k,
        dataset.n_rows,
        set(candidates),
    ).to_dict()


# ---------------------------------------------------------------------------
# NIPP-FIM: TFHE circuit reference backend
# ---------------------------------------------------------------------------


class NippBit:
    """Plaintext bit for circuit unit tests. Provides NO cryptographic secrecy."""

    __slots__ = ("_value", "_owner")

    def __init__(self, value: int, owner: str):
        self._value = 1 if int(value) else 0
        self._owner = owner


class NippDecryptor:
    def __init__(self, owner: str = "data_miner"):
        self.owner = owner

    def bit(self, value: NippBit) -> int:
        if not isinstance(value, NippBit):
            raise TypeError("NIPP decryptor expects a NippBit")
        return value._value


class NippTFHEReference:
    """Boolean simulator. This class performs no encryption or TFHE gates."""

    def __init__(self, transcript: ProtocolTranscript):
        self.transcript = transcript
        self.decryptor = NippDecryptor()

    def encrypt_bit(self, value: int) -> NippBit:
        self.transcript.op("encrypt_bit")
        return NippBit(value, "data_owner")

    def decrypt_bit(self, value: NippBit) -> int:
        self.transcript.op("decrypt_bit")
        return self.decryptor.bit(value)

    def _gate(self, name: str, left: NippBit, right: NippBit | None = None) -> NippBit:
        self.transcript.op(name)
        a = left._value
        b = right._value if right is not None else 0
        if name == "HomAND":
            value = a & b
        elif name == "HomOR":
            value = a | b
        elif name == "HomXOR":
            value = a ^ b
        elif name == "HomXNOR":
            value = 1 ^ (a ^ b)
        elif name == "HomNOT":
            value = 1 ^ a
        else:
            raise ValueError(f"unknown NIPP gate {name}")
        return NippBit(value, "cloud")

    def and_(self, a: NippBit, b: NippBit) -> NippBit:
        return self._gate("HomAND", a, b)

    def or_(self, a: NippBit, b: NippBit) -> NippBit:
        return self._gate("HomOR", a, b)

    def xor(self, a: NippBit, b: NippBit) -> NippBit:
        return self._gate("HomXOR", a, b)

    def xnor(self, a: NippBit, b: NippBit) -> NippBit:
        return self._gate("HomXNOR", a, b)

    def not_(self, a: NippBit) -> NippBit:
        return self._gate("HomNOT", a)

    def subdet(self, query: Sequence[NippBit], transaction: Sequence[NippBit]) -> NippBit:
        if len(query) != len(transaction):
            raise ValueError("query and transaction vectors must have equal length")
        result = self.encrypt_bit(1)
        for query_bit, transaction_bit in zip(query, transaction):
            # q => t = not(q) or t
            implication = self.or_(self.not_(query_bit), transaction_bit)
            result = self.and_(result, implication)
        self.transcript.op("SecSubDet")
        return result

    def accum(self, accumulator: Sequence[NippBit], bit: NippBit) -> list[NippBit]:
        result = list(accumulator)
        carry = bit
        for index in range(len(result) - 1, -1, -1):
            old = result[index]
            result[index] = self.xor(old, carry)
            carry = self.and_(carry, old)
        self.transcript.op("SecAccum")
        return result

    def compare_less(
        self,
        left: Sequence[NippBit],
        right: Sequence[NippBit],
    ) -> NippBit:
        """Unsigned comparator used by the reference circuit tests."""
        if len(left) != len(right) or not left:
            raise ValueError("comparison vectors must have equal non-zero length")
        result = self.and_(self.not_(left[0]), right[0])
        equal_prefix = self.xnor(left[0], right[0])
        for index in range(1, len(left)):
            less_at_position = self.and_(self.not_(left[index]), right[index])
            result = self.xor(result, self.and_(equal_prefix, less_at_position))
            equal_prefix = self.and_(equal_prefix, self.xnor(left[index], right[index]))
        self.transcript.op("SecCmp")
        return result

    def encrypt_integer(self, value: int, width: int) -> list[NippBit]:
        return [self.encrypt_bit((int(value) >> shift) & 1) for shift in range(width - 1, -1, -1)]

    def decrypt_integer(self, value: Sequence[NippBit]) -> int:
        result = 0
        for bit in value:
            result = (result << 1) | self.decrypt_bit(bit)
        return result


class NippConcreteBackend:
    """Optional native TFHE backend powered by Zama Concrete Python.

    Concrete compiles a complete Boolean circuit before evaluation.  We expose
    the same `support` operation as the reference backend, but compile it from
    the fixed item universe and accumulator width, then perform encrypted
    evaluation through Concrete's `encrypt/run/decrypt` API.
    """

    def __init__(self, universe_size: int, row_count: int, transcript: ProtocolTranscript):
        try:
            from concrete import fhe
        except ImportError as exc:  # pragma: no cover - depends on optional wheel
            raise RuntimeError(
                "native TFHE backend requires concrete-python; install it in the project venv"
            ) from exc
        if universe_size < 1 or row_count < 1:
            raise ValueError("native TFHE backend requires non-empty input dimensions")
        self._fhe = fhe
        self.transcript = transcript
        self.universe_size = int(universe_size)
        self.width = max(1, math.ceil(math.log2(row_count + 1)))
        self._circuit = None

    def compile(self, inputset: Iterable[tuple[np.ndarray, np.ndarray]]) -> None:
        samples = [(np.asarray(q, dtype=np.uint8), np.asarray(t, dtype=np.uint8)) for q, t in inputset]
        if not samples:
            raise ValueError("Concrete compilation inputset cannot be empty")

        fhe = self._fhe

        row_count = int(samples[0][1].shape[0])
        if any(transaction.shape != (row_count, self.universe_size) for _, transaction in samples):
            raise ValueError("all Concrete input transactions must have a common shape")

        @fhe.compiler({"query": "encrypted", "transactions": "encrypted"})
        def support_count(query, transactions):
            count = 0
            for row_index in range(row_count):
                membership = 1
                for index in range(self.universe_size):
                    membership &= (1 - query[index]) | transactions[row_index, index]
                count += membership
            return count

        self._circuit = support_count.compile(samples)
        self._circuit.keygen()
        self.transcript.op("tfhe_compile")
        self.transcript.op("tfhe_keygen")

    def count(self, query: np.ndarray, transactions: np.ndarray) -> int:
        if self._circuit is None:
            raise RuntimeError("compile() must be called before support()")
        encrypted_query, encrypted_transactions = self._circuit.encrypt(query, transactions)
        result = self._circuit.run(encrypted_query, encrypted_transactions)
        self.transcript.op("tfhe_encrypt_database")
        self.transcript.op("tfhe_evaluate")
        self.transcript.op("tfhe_decrypt")
        return int(self._circuit.decrypt(result))


def run_nipp_fim(
    dataset: VerticalDataset,
    candidates: Sequence[Itemset],
    *,
    top_k: int | None = None,
    candidate_source: str = "public_candidate_oracle",
    evaluation_supports: dict[Itemset, int] | None = None,
) -> tuple[dict[str, float], ProtocolTranscript]:
    started = time.perf_counter()
    transcript = ProtocolTranscript(
        scheme="NIPP-FIM",
        backend="TFHE-reference-gates",
        candidate_source=candidate_source,
        candidate_count=len(candidates),
        real_rows=dataset.n_rows,
        execution_status="protocol_executed_reference_backend",
    )
    backend = NippTFHEReference(transcript)
    phase_started = time.perf_counter()
    width = max(1, math.ceil(math.log2(dataset.n_rows + 1)))
    owner_items = {
        owner: [
            (attribute, int(value))
            for attribute in attributes
            for value in dataset.domains[attribute]
        ]
        for owner, attributes in enumerate(dataset.partitions)
    }
    # VFL upload: each owner encrypts only its own columns.  The row position
    # is the shared TID used to align vertical partitions; the cloud never
    # receives a plaintext transaction matrix.
    encrypted_transactions: dict[int, list[list[NippBit]]] = {}
    for owner, items in owner_items.items():
        rows = []
        for row in dataset.data:
            rows.append(
                [backend.encrypt_bit(int(row[attr] == value)) for attr, value in items]
            )
        encrypted_transactions[owner] = rows
        transcript.op("encrypted_transaction_rows", len(rows))
        transcript.op("encrypted_transaction_bits", len(rows) * len(items))
    transcript.metadata.update(
        {
            "vertical": True,
            "clients": len(dataset.partitions),
            "row_alignment": "shared_tID_position",
            "owner_item_counts": {str(owner): len(items) for owner, items in owner_items.items()},
        }
    )
    transcript.timings["owner_encrypt_upload"] = time.perf_counter() - phase_started

    phase_started = time.perf_counter()
    encrypted_supports: dict[Itemset, int] = {}
    for itemset in candidates:
        owner_queries = {
            owner: [
                backend.encrypt_bit(int(item in itemset))
                for item in items
            ]
            for owner, items in owner_items.items()
        }
        transcript.op("encrypted_query_bits", sum(len(query) for query in owner_queries.values()))
        accumulator = backend.encrypt_integer(0, width)
        for row_index in range(dataset.n_rows):
            # A query with no item from an owner contributes the encrypted
            # neutral element 1; otherwise the owner's SecSubDet is required.
            membership = backend.encrypt_bit(1)
            for owner, query in owner_queries.items():
                if any(item in itemset for item in owner_items[owner]):
                    owner_membership = backend.subdet(
                        query, encrypted_transactions[owner][row_index]
                    )
                    membership = backend.and_(membership, owner_membership)
            accumulator = backend.accum(accumulator, membership)
        encrypted_supports[itemset] = backend.decrypt_integer(accumulator)
        transcript.op("candidate_support_query")
    transcript.timings["csp_encrypted_support_queries"] = time.perf_counter() - phase_started

    phase_started = time.perf_counter()
    metrics = _rank_candidate_metrics(
        dataset,
        candidates,
        max(1, top_k or len(candidates)),
        {itemset: float(count) for itemset, count in encrypted_supports.items()},
        evaluation_supports,
    )
    uploaded_bits = sum(
        len(rows) * len(owner_items[owner])
        for owner, rows in encrypted_transactions.items()
    )
    query_bits = transcript.operations.get("encrypted_query_bits", 0)
    transcript.ciphertext_bytes = uploaded_bits + query_bits
    transcript.metadata["reference_logical_bits"] = transcript.ciphertext_bytes
    transcript.ciphertext_bytes = 0
    transcript.communication_bytes = 0
    transcript.metadata["communication_unit"] = "unmeasured_not_zero_cost"
    transcript.metadata["cryptographic_wall_clock"] = False
    transcript.metadata["protocol_complete"] = False
    transcript.timings["ranking_and_metrics"] = time.perf_counter() - phase_started
    transcript.elapsed_seconds = time.perf_counter() - started
    return metrics, transcript


def run_nipp_fim_packed(
    dataset: VerticalDataset,
    candidates: Sequence[Itemset],
    *,
    top_k: int | None = None,
    candidate_source: str = "public_candidate_oracle",
    evaluation_supports: dict[Itemset, int] | None = None,
) -> tuple[dict[str, float], ProtocolTranscript]:
    """Vectorized reference of the NIPP circuit with an operation model.

    The gate-by-gate backend is useful for small correctness tests but scales
    as ``rows * candidates * item-universe`` in Python.  This backend computes
    the same encrypted support predicate with vectorized Boolean masks and
    records the corresponding TFHE operation counts.  It is explicitly not a
    cryptographic wall-clock measurement; the transcript identifies it as a
    packed reference so large sensitivity matrices cannot be mistaken for
    native TFHE timings.
    """
    started = time.perf_counter()
    transcript = ProtocolTranscript(
        scheme="NIPP-FIM",
        backend="TFHE-packed-reference-operation-model",
        candidate_source=candidate_source,
        candidate_count=len(candidates),
        real_rows=dataset.n_rows,
        execution_status="protocol_executed_reference_packed",
    )
    owner_items = {
        owner: [
            (attribute, int(value))
            for attribute in attributes
            for value in dataset.domains[attribute]
        ]
        for owner, attributes in enumerate(dataset.partitions)
    }
    transcript.metadata.update(
        {
            "vertical": True,
            "clients": len(dataset.partitions),
            "row_alignment": "shared_tID_position",
            "owner_item_counts": {str(owner): len(items) for owner, items in owner_items.items()},
            "runtime_mode": "vectorized_support_plus_gate_operation_model",
            "cryptographic_wall_clock": False,
        }
    )
    total_item_bits = sum(len(items) for items in owner_items.values())
    transcript.op("encrypted_transaction_rows", dataset.n_rows * len(owner_items))
    transcript.op("encrypted_transaction_bits", dataset.n_rows * total_item_bits)
    transcript.timings["owner_encrypt_upload"] = time.perf_counter() - started

    phase_started = time.perf_counter()
    width = max(1, math.ceil(math.log2(dataset.n_rows + 1)))
    support_estimates: dict[Itemset, int] = {}
    for itemset in candidates:
        mask = np.ones(dataset.n_rows, dtype=bool)
        for attribute, value in itemset:
            mask &= dataset.data[:, int(attribute)] == int(value)
        support_estimates[itemset] = int(mask.sum())
        transcript.op("encrypted_query_bits", total_item_bits)
        involved = {
            owner
            for owner, attributes in enumerate(dataset.partitions)
            if any(attribute in attributes for attribute, _ in itemset)
        }
        for owner in involved:
            vector_width = len(owner_items[owner])
            transcript.op("HomNOT", dataset.n_rows * vector_width)
            transcript.op("HomOR", dataset.n_rows * vector_width)
            transcript.op("HomAND", dataset.n_rows * vector_width)
            transcript.op("SecSubDet", dataset.n_rows)
        transcript.op("HomAND", dataset.n_rows * max(0, len(involved) - 1))
        transcript.op("HomXOR", dataset.n_rows * width)
        transcript.op("HomAND", dataset.n_rows * width)
        transcript.op("SecAccum", dataset.n_rows)
        transcript.op("SecCmp", 1)
        transcript.op("candidate_support_query")
    transcript.timings["csp_encrypted_support_queries"] = time.perf_counter() - phase_started
    phase_started = time.perf_counter()
    metrics = _rank_candidate_metrics(
        dataset,
        candidates,
        max(1, top_k or len(candidates)),
        {itemset: float(count) for itemset, count in support_estimates.items()},
        evaluation_supports,
    )
    transcript.ciphertext_bytes = dataset.n_rows * total_item_bits + transcript.operations.get(
        "encrypted_query_bits", 0
    )
    transcript.communication_bytes = transcript.ciphertext_bytes
    transcript.metadata["communication_unit"] = "reference_ciphertext_bit_equivalent"
    transcript.timings["ranking_and_metrics"] = time.perf_counter() - phase_started
    transcript.elapsed_seconds = time.perf_counter() - started
    return metrics, transcript


def run_nipp_fim_native(
    dataset: VerticalDataset,
    candidates: Sequence[Itemset],
    *,
    top_k: int | None = None,
    candidate_source: str = "public_candidate_oracle",
    evaluation_supports: dict[Itemset, int] | None = None,
) -> tuple[dict[str, float], ProtocolTranscript]:
    """Execute the support circuit with native Concrete TFHE, when installed."""
    started = time.perf_counter()
    owner_items = {
        owner: [
            (attribute, int(value))
            for attribute in attributes
            for value in dataset.domains[attribute]
        ]
        for owner, attributes in enumerate(dataset.partitions)
    }
    item_universe = [item for owner in owner_items for item in owner_items[owner]]
    transactions = _one_hot_transactions(dataset, item_universe)
    transcript = ProtocolTranscript(
        scheme="NIPP-FIM",
        backend="Concrete-TFHE-native",
        candidate_source=candidate_source,
        candidate_count=len(candidates),
        real_rows=dataset.n_rows,
        execution_status="protocol_executed_native_tfhe",
    )
    transcript.metadata.update(
        {
            "vertical": True,
            "clients": len(dataset.partitions),
            "row_alignment": "shared_tID_position",
            "owner_item_counts": {str(owner): len(items) for owner, items in owner_items.items()},
            "native_input_layout": "owner_blocks_concatenated_for_circuit",
            "cryptographic_wall_clock": True,
            "protocol_complete": False,
            "paper_alignment_verified": False,
            "timing_scope": "support_query_prototype",
        }
    )
    backend = NippConcreteBackend(len(item_universe), dataset.n_rows, transcript)
    inputset = []
    for itemset in candidates:
        query = np.asarray([int(item in itemset) for item in item_universe], dtype=np.uint8)
        inputset.append((query, transactions))
    backend.compile(inputset)
    support_estimates = {}
    for itemset, (query, _) in zip(candidates, inputset):
        support_estimates[itemset] = backend.count(query, transactions)
    metrics = _rank_candidate_metrics(
        dataset,
        candidates,
        max(1, top_k or len(candidates)),
        {itemset: float(count) for itemset, count in support_estimates.items()},
        evaluation_supports,
    )
    # Concrete serialises encrypted inputs; exact wire format is backend and
    # version dependent, so operation counts are the stable metric here.
    transcript.communication_bytes = 0
    transcript.elapsed_seconds = time.perf_counter() - started
    return metrics, transcript


def _one_hot_transactions(dataset: VerticalDataset, items: Sequence[Item]) -> np.ndarray:
    """Each item carries its own attribute index, including after VFL reordering."""
    return np.asarray(
        [[int(row[attr] == value) for attr, value in items] for row in dataset.data],
        dtype=np.uint8,
    )


# ---------------------------------------------------------------------------
# VPP-OFIM: Paillier and dual-cloud verification workflow
# ---------------------------------------------------------------------------


def _egcd(a: int, b: int) -> tuple[int, int, int]:
    old_r, r, old_s, s, old_t, t = a, b, 1, 0, 0, 1
    while r:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t
    return old_s, old_t, old_r


def _mod_inverse(value: int, modulus: int) -> int:
    x, _, gcd = _egcd(value, modulus)
    if gcd != 1:
        raise ValueError("modular inverse does not exist")
    return x % modulus


def _is_probable_prime(value: int, rounds: int = 24) -> bool:
    if value < 2:
        return False
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % prime == 0:
            return value == prime
    d = value - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for _ in range(rounds):
        base = secrets.randbelow(value - 3) + 2
        x = pow(base, d, value)
        if x in (1, value - 1):
            continue
        for _ in range(s - 1):
            x = pow(x, 2, value)
            if x == value - 1:
                break
        else:
            return False
    return True


def _prime(bits: int) -> int:
    while True:
        candidate = secrets.randbits(bits) | (1 << (bits - 1)) | 1
        if _is_probable_prime(candidate):
            return candidate


@dataclass(frozen=True)
class PaillierPublicKey:
    n: int
    g: int

    @property
    def n2(self) -> int:
        return self.n * self.n

    def encrypt(self, message: int) -> int:
        if not 0 <= message < self.n:
            raise ValueError("Paillier message outside plaintext domain")
        while True:
            randomizer = secrets.randbelow(self.n - 1) + 1
            if math.gcd(randomizer, self.n) == 1:
                break
        if gmpy2 is not None:
            modulus = gmpy2.mpz(self.n2)
            return int(
                (gmpy2.powmod(self.g, message, modulus)
                * gmpy2.powmod(randomizer, self.n, modulus))
                % modulus
            )
        return (pow(self.g, message, self.n2) * pow(randomizer, self.n, self.n2)) % self.n2

    def encrypt_zero(self) -> int:
        """Sample an encryption of zero without evaluating the fixed g^0 term."""
        while True:
            randomizer = secrets.randbelow(self.n - 1) + 1
            if math.gcd(randomizer, self.n) == 1:
                break
        if gmpy2 is not None:
            return int(gmpy2.powmod(randomizer, self.n, self.n2))
        return pow(randomizer, self.n, self.n2)

    def add(self, left: int, right: int) -> int:
        return (left * right) % self.n2


@dataclass(frozen=True)
class PaillierPrivateKey:
    public: PaillierPublicKey
    lam: int
    mu: int
    p: int
    q: int
    hp: int
    hq: int

    def decrypt(self, ciphertext: int) -> int:
        # CRT Paillier decryption is algebraically equivalent to the lambda
        # form, but cuts the two costly modular exponentiations to p^2/q^2.
        # This matters when measuring a paper protocol that decrypts a
        # per-matching-transaction verification value.
        p2 = self.p * self.p
        q2 = self.q * self.q
        if gmpy2 is not None:
            left = int(gmpy2.powmod(ciphertext, self.p - 1, p2))
            right = int(gmpy2.powmod(ciphertext, self.q - 1, q2))
        else:
            left = pow(ciphertext, self.p - 1, p2)
            right = pow(ciphertext, self.q - 1, q2)
        mp = (((left - 1) // self.p) * self.hp) % self.p
        mq = (((right - 1) // self.q) * self.hq) % self.q
        # CRT reconstruction, with p and q coprime by construction.
        return (mq + ((mp - mq) * _mod_inverse(self.q, self.p) % self.p) * self.q) % self.public.n


class PaillierEvaluator:
    """Unblinded ideal-functionality reference, NOT VPP-OFIM's secure SC.

    The CSP submits ciphertexts and receives ciphertexts back.  Plaintext
    decryption is kept behind this object so the support-query code cannot
    accidentally use an evaluator value as a CSP-side predicate.  This is a
    logical separation in the reference runner; it does not claim that two
    independently deployed cryptographic services are present in-process.
    """

    def __init__(self, public: PaillierPublicKey, private: PaillierPrivateKey):
        self.public = public
        self.private = private

    def secure_sum(self, ciphertexts: Sequence[int]) -> int:
        total = self.public.encrypt_zero()
        for ciphertext in ciphertexts:
            total = self.public.add(total, ciphertext)
        return total

    def secure_equal(self, ciphertext: int, expected: int) -> int:
        """IDEAL-FUNCTIONALITY MOCK: leaks the operand to the Evaluator.

        This is not the blinded SC protocol in VPP-OFIM. Kept only for tests.
        """
        decoded = self.private.decrypt(ciphertext)
        return self.public.encrypt(1 if decoded == int(expected) else 0)

    def compare_greater_reference(self, ciphertext: int, threshold: int) -> int:
        """Reference-only strict support test (> supp_min in paper Section 3.3)."""
        decoded = self.private.decrypt(ciphertext)
        return self.public.encrypt(int(decoded > int(threshold)))

    def decrypt_for_final_output(self, ciphertext: int) -> int:
        """Decrypt only the final support value needed for local evaluation."""
        return self.private.decrypt(ciphertext)


def generate_paillier_keypair(bits: int = 2048) -> tuple[PaillierPublicKey, PaillierPrivateKey]:
    if bits < 256 or bits % 2:
        raise ValueError("Paillier key size must be an even number >= 256")
    p = _prime(bits // 2)
    q = _prime(bits // 2)
    while q == p:
        q = _prime(bits // 2)
    n = p * q
    public = PaillierPublicKey(n=n, g=n + 1)
    lam = math.lcm(p - 1, q - 1)
    l_value = (pow(public.g, lam, public.n2) - 1) // public.n
    hp_value = (pow(public.g, p - 1, p * p) - 1) // p
    hq_value = (pow(public.g, q - 1, q * q) - 1) // q
    private = PaillierPrivateKey(
        public=public,
        lam=lam,
        mu=_mod_inverse(l_value, n),
        p=p,
        q=q,
        hp=_mod_inverse(hp_value, p),
        hq=_mod_inverse(hq_value, q),
    )
    return public, private


def _substitution_label(owner: int, item: Item, salt: bytes) -> str:
    payload = f"{owner}:{item[0]}:{item[1]}".encode()
    return hashlib.blake2b(payload, key=salt, digest_size=16).hexdigest()


def _with_artificial_rows(
    dataset: VerticalDataset,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    count = max(0, int(math.ceil(dataset.n_rows * fraction)))
    if count == 0:
        return dataset.data.copy(), np.ones(dataset.n_rows, dtype=np.int8)
    rng = np.random.default_rng(seed)
    artificial = np.empty((count, dataset.n_attributes), dtype=np.int64)
    for attribute, domain in enumerate(dataset.domains):
        artificial[:, attribute] = rng.choice(np.asarray(domain), size=count)
    return (
        np.concatenate([dataset.data, artificial], axis=0),
        np.concatenate([np.ones(dataset.n_rows, dtype=np.int8), np.zeros(count, dtype=np.int8)]),
    )


def _support_text(db: Sequence[Sequence[str]], itemset: Sequence[str]) -> int:
    target = set(itemset)
    return sum(target.issubset(set(transaction)) for transaction in db)


def _afi_conflict(left: Sequence[str], right: Sequence[str], aii: set[tuple[str, ...]]) -> bool:
    left_set, right_set = set(left), set(right)
    return any(
        (set(non_frequent) - left_set) & right_set
        and (set(non_frequent) - right_set) & left_set
        for non_frequent in aii
    )


def _construct_afi_aii(
    artificial_items: Sequence[str],
    base_db: Sequence[Sequence[str]],
    supp_min: int,
    s_ratio: float,
    min_afi: int,
    seed: int,
) -> tuple[set[tuple[str, ...]], set[tuple[str, ...]], dict[str, int]]:
    """Port the paper's AFI/AII construction without a data-frame dependency."""
    import random

    rng = random.Random(seed)
    artificial_items = list(artificial_items)
    afi: set[tuple[str, ...]] = set()
    aii: set[tuple[str, ...]] = set()
    ci_limit = int(1.0 / s_ratio - 1e-9) - 1 if s_ratio > 0 else 999999
    attempts = 0
    for attempts in range(1, 201):
        if len(artificial_items) < 2:
            break
        h_length = rng.randint(2, min(4, len(artificial_items)))
        h = tuple(sorted(rng.sample(artificial_items, h_length)))
        if h in aii:
            continue
        old_afi, old_aii = set(afi), set(aii)
        aii.add(h)
        afi.update(tuple(sorted(part)) for part in itertools.combinations(h, h_length - 1))
        conflict_degree = {
            candidate: sum(
                _afi_conflict(candidate, other, aii)
                for other in afi
                if candidate != other
            )
            for candidate in afi
        }
        if max(conflict_degree.values(), default=0) > ci_limit:
            afi, aii = old_afi, old_aii
            continue
        # Add the negative border: a minimal non-frequent candidate whose
        # proper subsets are already AFI, as specified by the construction.
        for length in range(1, min(4, len(artificial_items)) + 1):
            for candidate in itertools.combinations(sorted(artificial_items), length):
                candidate = tuple(candidate)
                if candidate in afi or candidate in aii:
                    continue
                if length > 1 and not all(
                    tuple(sorted(subset)) in afi
                    for subset in itertools.combinations(candidate, length - 1)
                ):
                    continue
                if _support_text(base_db, candidate) >= supp_min:
                    continue
                trial_aii = set(aii)
                trial_aii.add(candidate)
                if max(
                    (
                        sum(_afi_conflict(node, other, trial_aii) for other in afi if node != other)
                        for node in afi
                    ),
                    default=0,
                ) <= ci_limit:
                    aii.add(candidate)
        if len(afi) >= min_afi:
            return afi, aii, {"attempts": attempts, "ci_limit": ci_limit}
    return afi, aii, {"attempts": attempts, "ci_limit": ci_limit}


def _generate_ta(
    afi: set[tuple[str, ...]],
    aii: set[tuple[str, ...]],
    base_db: Sequence[Sequence[str]],
    supp_min: int,
    min_rows: int,
) -> list[list[str]]:
    """Generate the paper's artificial transaction (TA) rows.

    A grouping iteration contributes one transaction containing the union of
    the grouped AFI itemsets.  The individual AFI itemsets are *not* inserted
    as additional transactions: doing that changes both AFI and AII supports
    and was the source of a systematic protocol mismatch in the old runner.
    The union pattern is repeated until every AFI is frequent while every AII
    remains below ``supp_min``.
    """
    if not afi:
        return []
    # By downward closure, an AII that is a subset of an AFI cannot be kept
    # infrequent once that AFI is made frequent.  The construction routine can
    # emit such an inconsistent negative-border request for tiny domains; drop
    # only those impossible constraints and retain all feasible AII checks.
    feasible_aii = {
        negative
        for negative in aii
        if not any(set(negative).issubset(set(frequent)) for frequent in afi)
    }
    graph = {
        node: {
            other
            for other in afi
            if other != node and _afi_conflict(node, other, feasible_aii)
        }
        for node in afi
    }
    remaining = set(afi)
    pattern_rows: list[list[str]] = []
    while remaining:
        start = min(remaining, key=lambda node: (len(graph[node]), node))
        remaining.remove(start)
        group = [start]
        group_items = set(start)
        for candidate in sorted(remaining):
            trial_items = group_items | set(candidate)
            violates_negative_border = any(
                set(non_frequent).issubset(trial_items)
                for non_frequent in feasible_aii
            )
            if (
                not violates_negative_border
                and all(candidate not in graph[node] for node in group)
            ):
                group.append(candidate)
                group_items = trial_items
        remaining.difference_update(group[1:])
        union = sorted({item for node in group for item in node})
        pattern_rows.append(union)

    # The artificial alphabet is disjoint from the real database in the
    # construction above, but keep the base database in the checks so this
    # helper remains correct when called with a non-disjoint test fixture.
    if not pattern_rows:
        return []
    afis_per_pattern = min(
        _support_text(pattern_rows, node) for node in afi
    )
    repeats = max(
        1,
        int(math.ceil((supp_min + 1) / max(1, afis_per_pattern))),
        int(math.ceil(min_rows / len(pattern_rows))),
    )
    rows = pattern_rows * repeats
    merged = [list(row) for row in base_db] + rows
    if not all(_support_text(merged, node) > supp_min for node in afi):
        raise RuntimeError("TA construction failed to make every AFI frequent")
    if any(_support_text(rows, node) >= supp_min for node in feasible_aii):
        # The conflict graph should prevent this.  Failing loudly is safer
        # than silently publishing a TA database whose negative border is no
        # longer negative.
        raise RuntimeError("TA construction made an AII itemset frequent")
    return rows


def _giannotti_insert(
    joint_db: Sequence[Sequence[str]],
    minimum: int,
    maximum: int,
    seed: int,
    base_flags: Sequence[int] | None = None,
) -> tuple[list[list[str]], list[int], dict[str, int]]:
    """Legacy bounded padding heuristic; does NOT guarantee frequency anonymity."""
    import random

    rng = random.Random(seed)
    result = [list(row) for row in joint_db]
    flags = list(base_flags) if base_flags is not None else [1] * len(result)
    if len(flags) != len(result):
        raise ValueError("base_flags length must match the initial transaction database")
    frequency: dict[str, int] = {}
    for row in result:
        for item in set(row):
            frequency[item] = frequency.get(item, 0) + 1
    if not frequency or maximum <= 0:
        return result, flags, {"inserts": 0, "target_frequency": 0}
    ordered = sorted(frequency.values(), reverse=True)
    target = max(5, int((ordered[1] if len(ordered) > 1 else ordered[0]) * 0.8))
    low = [item for item, count in frequency.items() if count < int(target * 0.8)]
    high = [item for item, count in frequency.items() if count >= target]
    inserted = 0
    for item in low:
        for _ in range(max(0, target - frequency[item])):
            if inserted >= maximum:
                break
            companions = rng.sample(high, min(2, len(high))) if high else []
            result.append([item, *companions])
            flags.append(0)
            frequency[item] += 1
            inserted += 1
        if inserted >= maximum:
            break
    items = list(frequency)
    while inserted < min(minimum, maximum) and items:
        length = min(len(items), rng.randint(1, 4))
        result.append(rng.sample(items, length))
        flags.append(0)
        inserted += 1
    return result, flags, {"inserts": inserted, "target_frequency": target}


def _token_for_item(item: Item) -> str:
    """Stable plaintext token used only inside a data owner's local database."""
    return f"a{int(item[0])}={int(item[1])}"


def _token_label(owner: int, token: str, salt: bytes) -> str:
    """Private substitution alphabet entry for a VPP-OFIM owner."""
    payload = f"{owner}:{token}".encode("utf-8")
    return hashlib.blake2b(payload, key=salt, digest_size=16).hexdigest()


def _tid_hash(tid: str) -> str:
    """The paper exposes H(TID), never the owner's plaintext transaction ID."""
    return hashlib.sha256(tid.encode("utf-8")).hexdigest()


def _build_vpp_owner_records(
    dataset: VerticalDataset,
    fraction: float,
    seed: int,
    artificial_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build the per-owner D/Z databases from Sections 5.1--5.2.

    Real rows share TIDs across owners.  Artificial and fictitious rows receive
    owner-local TIDs, so a cross-owner candidate can only be supported by a
    real aligned row. Paper Section 5.2 assigns ERV=1 to both T and TA.
    Only frequency-hiding fictitious rows receive ERV=0.
    """
    if artificial_mode not in {"paper", "random"}:
        raise ValueError("artificial_mode must be paper or random")
    records: list[dict[str, Any]] = []
    meta = {
        "afi_count": 0,
        "aii_count": 0,
        "ta_rows": 0,
        "fictitious_rows": 0,
        "artificial_mode": artificial_mode,
    }
    minimum_fictitious = max(0, int(math.ceil(dataset.n_rows * max(0.0, fraction))))

    for owner_id, attributes in enumerate(dataset.partitions):
        real_rows = [
            [_token_for_item((attribute, int(row[attribute]))) for attribute in attributes]
            for row in dataset.data
        ]
        if artificial_mode == "paper":
            # The paper constructs AFI/AII over the local artificial alphabet.
            # Exhaustively enumerating all 3/4-subsets is not tractable for a
            # high-dimensional tabular owner, so the reference adaptation uses
            # a bounded alphabet and records that bound in the transcript.
            artificial_item_count = min(24, max(2, len(attributes) * 4))
            local_items = [
                f"{owner_id}_AI_{index}"
                for index in range(artificial_item_count)
            ]
            afi, aii, afi_meta = _construct_afi_aii(
                local_items,
                real_rows,
                max(1, int(math.ceil(dataset.n_rows * 0.005))),
                0.005,
                min(8, max(2, len(local_items))),
                seed + owner_id,
            )
            meta[f"owner_{owner_id}_artificial_item_count"] = artificial_item_count
            ta = _generate_ta(
                afi,
                aii,
                real_rows,
                max(1, int(math.ceil(dataset.n_rows * 0.005))),
                5,
            )
            meta["afi_count"] += len(afi)
            meta["aii_count"] += len(aii)
            meta["ta_rows"] += len(ta)
            # Keep the construction metadata available for diagnostics without
            # changing the paper-level transcript schema.
            meta[f"owner_{owner_id}_afi_attempts"] = int(afi_meta.get("attempts", 0))
        else:
            rng = np.random.default_rng(seed + owner_id)
            ta = [
                [
                    _token_for_item((attribute, int(rng.choice(dataset.domains[attribute]))))
                    for attribute in attributes
                ]
                for _ in range(max(0, int(math.ceil(dataset.n_rows * max(0.0, fraction)))))
            ]
            meta["ta_rows"] += len(ta)

        database_d = real_rows + ta
        # TA contains integrity-check traps and must be mined; random mode only smoke-tests decoy rows.
        flags_d = [1] * len(real_rows) + [int(artificial_mode == "paper")] * len(ta)
        maximum_fictitious = max(minimum_fictitious, 2 * minimum_fictitious)
        database_z, flags_z, giannotti_meta = _giannotti_insert(
            database_d,
            minimum=minimum_fictitious,
            maximum=maximum_fictitious,
            seed=seed + 1000 + owner_id,
            base_flags=flags_d,
        )
        meta["fictitious_rows"] += int(giannotti_meta.get("inserts", 0))

        tids: list[str] = []
        for index in range(len(database_z)):
            if index < len(real_rows):
                tids.append(str(index))
            elif index < len(database_d):
                tids.append(f"owner{owner_id}:TA:{index - len(real_rows)}")
            else:
                tids.append(f"owner{owner_id}:F:{index - len(database_d)}")
        records.append(
            {
                "owner_id": owner_id,
                "attributes": tuple(attributes),
                "rows": database_z,
                "flags": flags_z,
                "tids": tids,
                "afi": sorted(afi) if artificial_mode == "paper" else [],
                "aii": sorted(aii) if artificial_mode == "paper" else [],
            }
        )
    return records, meta


def _paper_artificial_rows(
    dataset: VerticalDataset,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Generate AFI/AII, TA and Giannotti rows with the paper's construction."""
    item_name = lambda item: f"a{item[0]}={item[1]}"
    local_bases = []
    for owner_id, attributes in enumerate(dataset.partitions):
        local_bases.append(
            [
                [item_name((attribute, int(row[attribute]))) for attribute in attributes]
                for row in dataset.data
            ]
        )
    ta_by_owner = []
    afi_total = aii_total = 0
    supp_min = max(1, int(math.ceil(dataset.n_rows * 0.005)))
    for owner_id, attributes in enumerate(dataset.partitions):
        local_items = [
            item_name((attribute, int(value)))
            for attribute in attributes
            for value in dataset.domains[attribute]
        ]
        artificial_items = [
            f"{owner_id}_AI_{index}"
            for index in range(min(24, max(2, len(local_items) * 3)))
        ]
        afi, aii, _ = _construct_afi_aii(
            artificial_items, local_bases[owner_id], supp_min, 0.005,
            min(8, max(2, len(artificial_items))), seed + owner_id,
        )
        ta = _generate_ta(afi, aii, local_bases[owner_id], supp_min, 5)
        ta_by_owner.append(ta)
        afi_total += len(afi)
        aii_total += len(aii)

    joint_ta = []
    for row_index in range(max((len(ta) for ta in ta_by_owner), default=0)):
        row_items = set()
        for ta in ta_by_owner:
            if row_index < len(ta):
                row_items.update(ta[row_index])
        if row_items:
            joint_ta.append(sorted(row_items))

    # The Giannotti phase operates on the aligned joint database and supplies
    # the remaining artificial rows required by the requested fraction.
    target_artificial = max(0, int(math.ceil(dataset.n_rows * fraction)))
    requested_insertions = max(0, target_artificial - len(joint_ta))
    if requested_insertions:
        joint_with_ta, flags, giannotti = _giannotti_insert(
            [
                [item_name((attribute, int(row[attribute]))) for attribute in range(dataset.n_attributes)]
                for row in dataset.data
            ]
            + joint_ta,
            minimum=requested_insertions,
            maximum=max(requested_insertions, requested_insertions * 2),
            seed=seed,
        )
        artificial_count = len(joint_with_ta) - dataset.n_rows
        meta = {
            "afi_count": afi_total,
            "aii_count": aii_total,
            "ta_rows": len(joint_ta),
            "giannotti_inserts": int(giannotti.get("inserts", 0)),
        }
    else:
        artificial_count = len(joint_ta)
        meta = {"afi_count": afi_total, "aii_count": aii_total, "ta_rows": len(joint_ta), "giannotti_inserts": 0}

    if artificial_count == 0:
        return dataset.data.copy(), np.ones(dataset.n_rows, dtype=np.int8), meta
    rng = np.random.default_rng(seed)
    artificial = np.empty((artificial_count, dataset.n_attributes), dtype=np.int64)
    for attribute, domain in enumerate(dataset.domains):
        artificial[:, attribute] = rng.choice(np.asarray(domain), size=artificial_count)
    return (
        np.concatenate([dataset.data, artificial], axis=0),
        np.concatenate([np.ones(dataset.n_rows, dtype=np.int8), np.zeros(artificial_count, dtype=np.int8)]),
        meta,
    )


def run_vpp_ofim(
    dataset: VerticalDataset,
    candidates: Sequence[Itemset],
    *,
    top_k: int | None = None,
    paillier_bits: int = 2048,
    artificial_fraction: float = 0.01,
    seed: int = 2026,
    artificial_mode: str = "paper",
    candidate_source: str = "public_candidate_oracle",
    evaluation_supports: dict[Itemset, int] | None = None,
    workers: int = 1,
    support_threshold: int | None = None,
) -> tuple[dict[str, float], ProtocolTranscript]:
    started = time.perf_counter()
    transcript = ProtocolTranscript(
        scheme="VPP-OFIM",
        backend=f"Paillier-{paillier_bits}-dual-cloud",
        candidate_source=candidate_source,
        candidate_count=len(candidates),
        real_rows=dataset.n_rows,
        execution_status="protocol_executed_reference_adaptation",
    )
    phase_started = time.perf_counter()
    owner_records, artificial_meta = _build_vpp_owner_records(
        dataset, artificial_fraction, seed, artificial_mode
    )
    transcript.timings["owner_local_artificial_data"] = time.perf_counter() - phase_started
    transcript.artificial_rows = int(
        artificial_meta.get("ta_rows", 0) + artificial_meta.get("fictitious_rows", 0)
    )
    transcript.metadata.update(
        {
            **artificial_meta,
            "vertical": True,
            "dual_cloud": "CSP+Evaluator",
            "row_alignment": "H(TID)",
            "candidate_discovery": candidate_source,
            "support_query_workers": workers,
            "secure_role_separation": "CSP-label-join/Evaluator-Paillier",
            "reference_backend": True,
            "cryptographic_wall_clock": True,
            "protocol_complete": False,
            "paper_alignment_verified": False,
            "timing_scope": "support_query_prototype",
            "secure_comparison": "plaintext_opening_reference_NOT_paper_SC",
            "frequency_hiding_verified": False,
            "integrity_verification_executed": False,
            "artificial_alphabet_bound": 24,
            "support_threshold": int(
                support_threshold
                if support_threshold is not None
                else max(1, int(math.ceil(dataset.n_rows * 0.005)))
            ),
        }
    )
    phase_started = time.perf_counter()
    public, private = generate_paillier_keypair(paillier_bits)
    evaluator = PaillierEvaluator(public, private)
    transcript.op("paillier_keygen")
    transcript.timings["paillier_keygen"] = time.perf_counter() - phase_started
    salt = secrets.token_bytes(32)

    # Data-owner phase: substitution labels, H(TID), and ERV are uploaded to
    # the CSP.  The Evaluator owns the Paillier secret key; the CSP only sees
    # ciphertexts and replacement labels.
    owner_by_id: dict[int, dict[str, Any]] = {}
    joint_rows: dict[str, dict[int, dict[str, Any]]] = {}
    label_bytes = 16
    tid_bytes = 32
    ciphertext_bytes = math.ceil(2 * paillier_bits / 8)
    phase_started = time.perf_counter()
    for record in owner_records:
        owner_id = int(record["owner_id"])
        token_set = {token for row in record["rows"] for token in row}
        # Public values absent from the sample must remain queryable, rather than dropping candidates.
        token_set.update(_token_for_item((attribute, int(value)))
                         for attribute in record["attributes"]
                         for value in dataset.domains[attribute])
        labels = {token: _token_label(owner_id, token, salt) for token in token_set}
        encrypted_validity = [
            public.encrypt(1) if flag else public.encrypt_zero()
            for flag in record["flags"]
        ]
        transcript.op("paillier_encrypt", len(encrypted_validity))
        record["labels"] = labels
        record["encrypted_validity"] = encrypted_validity
        owner_by_id[owner_id] = record
        transcript.op("owner_upload", len(record["rows"]))
        transcript.communication_bytes += len(record["rows"]) * (
            tid_bytes + label_bytes * max(1, max((len(row) for row in record["rows"]), default=0))
            + ciphertext_bytes
        )
        for row_index, (tid, row, erv) in enumerate(
            zip(record["tids"], record["rows"], encrypted_validity)
        ):
            digest = _tid_hash(tid)
            joint_rows.setdefault(digest, {})[owner_id] = {
                "labels": {labels[token] for token in row},
                "erv": erv,
                "row_index": row_index,
            }
        transcript.op("hash_tid", len(record["rows"]))
        transcript.op("substitution_label", len(token_set))
    transcript.timings["owner_encrypt_and_upload"] = time.perf_counter() - phase_started

    phase_started = time.perf_counter()
    owner_by_attribute = dataset.owner_by_attribute

    def secure_support(itemset: Itemset) -> tuple[Itemset, int, dict[str, int], int]:
        operations: dict[str, int] = {}

        def op(name: str, amount: int = 1) -> None:
            operations[name] = operations.get(name, 0) + amount

        candidate_owners = sorted({owner_by_attribute[item[0]] for item in itemset})
        query_labels = {
            owner: {
                owner_by_id[owner]["labels"][_token_for_item(item)]
                for item in itemset
                if owner_by_attribute[item[0]] == owner
            }
            for owner in candidate_owners
        }
        encrypted_support = public.encrypt_zero()
        op("paillier_encrypt")
        for row_by_owner in joint_rows.values():
            # CSP performs label membership and the vertical H(TID) join.
            if any(
                owner not in row_by_owner
                or not query_labels[owner].issubset(row_by_owner[owner]["labels"])
                for owner in candidate_owners
            ):
                continue
            # CSP -> Evaluator: encrypted sum of ERV values.  Evaluator returns
            # an encrypted 0/1 indicator, which the CSP accumulates.
            validity_sum = public.encrypt_zero()
            op("paillier_encrypt")
            for owner_id in candidate_owners:
                validity_sum = public.add(
                    validity_sum, row_by_owner[owner_id]["erv"]
                )
                op("paillier_add")
            indicator_ciphertext = evaluator.secure_equal(
                validity_sum, len(candidate_owners)
            )
            op("paillier_decrypt")
            op("paillier_encrypt")
            op("secure_compare_support_indicator")
            encrypted_support = public.add(encrypted_support, indicator_ciphertext)
            op("paillier_add")
        # The evaluator also checks the support threshold on ciphertext.  The
        # encrypted flag is not opened by the CSP; it is recorded for protocol
        # accounting while the final support is opened only for benchmarking.
        threshold = support_threshold
        if threshold is None:
            threshold = max(1, int(math.ceil(dataset.n_rows * 0.005)))
        frequent_ciphertext = evaluator.compare_greater_reference(encrypted_support, threshold)
        op("paillier_decrypt")
        op("paillier_encrypt")
        op("secure_compare_support_threshold")
        support = evaluator.decrypt_for_final_output(encrypted_support)
        op("paillier_decrypt")
        frequent = evaluator.decrypt_for_final_output(frequent_ciphertext)
        op("paillier_decrypt")
        if frequent != int(support > threshold):
            raise RuntimeError("VPP reference threshold check failed")
        return itemset, support, operations, frequent

    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1 or len(candidates) < 2:
        results = [secure_support(itemset) for itemset in candidates]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(candidates))) as executor:
            results = list(executor.map(secure_support, candidates))
    support_estimates: dict[Itemset, int] = {}
    threshold_flags = []
    for itemset, support, operations, frequent in results:
        support_estimates[itemset] = support
        threshold_flags.append({"itemset": itemset, "support": support, "frequent": frequent})
        for name, amount in operations.items():
            transcript.op(name, amount)
    transcript.timings["csp_evaluator_support_queries"] = time.perf_counter() - phase_started
    transcript.metadata["threshold_checks"] = threshold_flags
    transcript.elapsed_seconds = time.perf_counter() - started

    phase_started = time.perf_counter()
    supports = {itemset: dataset.support(itemset) for itemset in candidates}
    estimates = [
        CandidateEstimate(
            itemset=itemset,
            estimated_count=float(support_estimates[itemset]),
            guessed_count=float(supports[itemset]),
            local_blocks=(itemset,),
        )
        for itemset in candidates
    ]
    metrics = _rank_candidate_metrics(
        dataset,
        candidates,
        max(1, top_k or len(candidates)),
        {itemset: float(count) for itemset, count in support_estimates.items()},
        evaluation_supports,
    )
    transcript.ciphertext_bytes = transcript.operations.get("paillier_encrypt", 0) * ciphertext_bytes
    transcript.communication_bytes += transcript.ciphertext_bytes
    transcript.proof_bytes = transcript.operations.get("paillier_decrypt", 0) * math.ceil(paillier_bits / 8)
    transcript.timings["ranking_and_metrics"] = time.perf_counter() - phase_started
    transcript.metadata["wall_seconds_including_offline_metrics"] = time.perf_counter() - started
    return metrics, transcript


def run_vpp_ofim_packed(
    dataset: VerticalDataset,
    candidates: Sequence[Itemset],
    *,
    top_k: int | None = None,
    paillier_bits: int = 2048,
    artificial_fraction: float = 0.01,
    seed: int = 2026,
    artificial_mode: str = "paper",
    candidate_source: str = "public_candidate_oracle",
    evaluation_supports: dict[Itemset, int] | None = None,
    workers: int = 1,
) -> tuple[dict[str, float], ProtocolTranscript]:
    """Scalable VPP reference with exact supports and protocol cost counts.

    The Paillier reference above intentionally executes every evaluator
    comparison and is suitable for small protocol smoke tests.  This variant
    uses the exact local support mask for the candidate boundary and records
    the Paillier operations implied by Section 5.3, without performing big
    integer encryption/decryption for every row.  It is therefore a runtime
    and communication *operation model*, not a cryptographic wall-clock
    measurement; the transcript states this explicitly.
    """
    started = time.perf_counter()
    owners = len(dataset.partitions)
    artificial_rows = int(math.ceil(dataset.n_rows * max(0.0, artificial_fraction)))
    fictitious_rows = max(artificial_rows, 2 * artificial_rows)
    owner_rows = dataset.n_rows + artificial_rows + fictitious_rows
    ciphertext_bytes = math.ceil(2 * paillier_bits / 8)
    transcript = ProtocolTranscript(
        scheme="VPP-OFIM",
        backend=f"Paillier-{paillier_bits}-packed-reference-operation-model",
        candidate_source=candidate_source,
        candidate_count=len(candidates),
        real_rows=dataset.n_rows,
        artificial_rows=owners * (artificial_rows + fictitious_rows),
        execution_status="protocol_executed_reference_packed",
    )
    transcript.metadata.update(
        {
            "vertical": True,
            "dual_cloud": "CSP+Evaluator",
            "row_alignment": "H(TID)",
            "candidate_discovery": candidate_source,
            "artificial_mode": f"{artificial_mode}_operation_model",
            "artificial_construction": "bounded_AFI_AII_TA_Giannotti_count_model",
            "owner_count": owners,
            "owner_rows_modeled": owner_rows,
            "cryptographic_wall_clock": False,
            "runtime_mode": "exact_candidate_support_plus_paillier_operation_model",
            "secure_role_separation": "CSP-label-join/Evaluator-Paillier",
            "support_threshold": max(1, int(math.ceil(dataset.n_rows * 0.005))),
            "seed": seed,
            "support_query_workers": workers,
        }
    )
    transcript.op("paillier_keygen")
    transcript.op("paillier_encrypt", owners * owner_rows)
    transcript.op("owner_upload", owners * owner_rows)
    transcript.op("hash_tid", owners * owner_rows)
    transcript.communication_bytes = owners * owner_rows * (
        32 + ciphertext_bytes
    )
    transcript.ciphertext_bytes = transcript.operations["paillier_encrypt"] * ciphertext_bytes

    phase_started = time.perf_counter()
    support_estimates: dict[Itemset, int] = {}
    for itemset in candidates:
        support = int(dataset.support(itemset))
        support_estimates[itemset] = support
        involved = {
            owner
            for owner, attributes in enumerate(dataset.partitions)
            if any(attribute in attributes for attribute, _ in itemset)
        }
        matched_rows = support
        transcript.op("paillier_encrypt", 1 + 2 * matched_rows + 1)
        transcript.op("paillier_add", matched_rows * (len(involved) + 1))
        transcript.op("paillier_decrypt", matched_rows + 2)
        transcript.op("secure_compare_support_indicator", matched_rows)
        transcript.op("secure_compare_support_threshold")
        transcript.op("candidate_support_query")
    transcript.timings["csp_evaluator_support_queries"] = time.perf_counter() - phase_started
    phase_started = time.perf_counter()
    metrics = _rank_candidate_metrics(
        dataset,
        candidates,
        max(1, top_k or len(candidates)),
        {itemset: float(count) for itemset, count in support_estimates.items()},
        evaluation_supports,
    )
    transcript.ciphertext_bytes = transcript.operations.get("paillier_encrypt", 0) * ciphertext_bytes
    transcript.communication_bytes += transcript.ciphertext_bytes
    transcript.proof_bytes = transcript.operations.get("paillier_decrypt", 0) * math.ceil(paillier_bits / 8)
    transcript.timings["ranking_and_metrics"] = time.perf_counter() - phase_started
    transcript.elapsed_seconds = time.perf_counter() - started
    return metrics, transcript


# ---------------------------------------------------------------------------
# CLI and reproducible smoke matrix
# ---------------------------------------------------------------------------


def _write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_case(
    dataset_name: str,
    *,
    k: int,
    clients: int,
    max_itemset_size: int,
    candidate_multiplier: float,
    paillier_bits: int,
    artificial_fraction: float,
    seed: int,
    nipp_backend: str = "reference",
    artificial_mode: str = "paper",
    max_rows: int | None = None,
    evaluation_supports: dict[Itemset, int] | None = None,
) -> list[dict[str, Any]]:
    specs = _dataset_specs()
    dataset = load_vertical_csv(
        str(specs[dataset_name].csv_path),
        num_clients=clients,
        max_rows=max_rows,
        row_sampling_seed=seed,
    )
    candidates = _candidate_list(dataset, k, max_itemset_size, candidate_multiplier)
    result_rows = []
    if nipp_backend == "concrete":
        nipp_runner = run_nipp_fim_native
    elif nipp_backend == "packed":
        nipp_runner = run_nipp_fim_packed
    else:
        nipp_runner = run_nipp_fim
    for runner in (nipp_runner, run_vpp_ofim):
        if runner is nipp_runner:
            metrics, transcript = runner(
                dataset,
                candidates,
                top_k=k,
                evaluation_supports=evaluation_supports,
            )
        else:
            metrics, transcript = runner(
                dataset,
                candidates,
                top_k=k,
                paillier_bits=paillier_bits,
                artificial_fraction=artificial_fraction,
                seed=seed,
                artificial_mode=artificial_mode,
                evaluation_supports=evaluation_supports,
            )
        result_rows.append(
            {
                "dataset": dataset_name,
                "seed": seed,
                "k": k,
                "num_clients": len(dataset.partitions),
                "max_itemset_size": max_itemset_size,
                "candidate_multiplier": candidate_multiplier,
                "paillier_bits": paillier_bits,
                "artificial_fraction": artificial_fraction,
                **metrics,
                **transcript.to_dict(),
            }
        )
    return result_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper-flow NIPP-FIM/VPP-OFIM protocol reproductions")
    parser.add_argument("--dataset", default="Retail", choices=tuple(_dataset_specs()))
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-itemset-size", type=int, default=2)
    parser.add_argument("--candidate-multiplier", type=float, default=2.0)
    parser.add_argument("--paillier-bits", type=int, default=512, help="Use 2048 for paper security; 512 is faster smoke testing")
    parser.add_argument(
        "--nipp-backend",
        choices=("reference", "packed", "concrete"),
        default="packed",
        help="reference=逐门烟测，packed=大数据集向量化支持+门操作计数，concrete=原生 TFHE（需安装）",
    )
    parser.add_argument("--artificial-fraction", type=float, default=0.01)
    parser.add_argument("--artificial-mode", choices=("paper", "random"), default="paper")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-rows", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("results/crypto_protocol_reproduction/smoke.csv"))
    args = parser.parse_args()
    rows = run_case(
        args.dataset,
        k=args.k,
        clients=args.clients,
        max_itemset_size=args.max_itemset_size,
        candidate_multiplier=args.candidate_multiplier,
        paillier_bits=args.paillier_bits,
        artificial_fraction=args.artificial_fraction,
        seed=args.seed,
        nipp_backend=args.nipp_backend,
        artificial_mode=args.artificial_mode,
        max_rows=args.max_rows,
    )
    _write_rows(args.output, rows)
    print(json.dumps({"output": str(args.output), "rows": len(rows), "schemes": [row["scheme"] for row in rows]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
