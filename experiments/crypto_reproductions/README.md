# Native cryptographic backends

The supported dataset-level entry point is `privfim crypto`. See
[the reproducibility guide](../../docs/CRYPTO.md) for setup, timing scope and limitations.

- `nipp_upstream.cpp`, `CMakeLists.txt`: adapter to the pinned upstream TFHE implementation.
- `vpp.py`: Paillier ERV, hashed transaction IDs, blinded comparisons and verification patterns.
- `two_round.py`: shared first-round DP candidate construction.
- `run.py`: native adapter, standalone public-universe queries and support verification.
- `preflight.py`: exact public candidate counts and input ciphertext size estimates.
- `export_k_sweep.py`: support-reuse diagnostic, not independent runtime measurements.

Upstream sources are fetched by `python scripts/setup_crypto.py`; they are not copied into this repository.
No private keys or ciphertexts are committed. `run.py` and `two_round.py` have no default time limit.
The primary workflow evaluates datasets independently and writes actual timings only.
