# PrivFim

A reproducible implementation of two-round, vertically federated frequent itemset mining.
Rows represent aligned users, and columns are assigned to data owners. The first round
releases noisy item counts and noisy N to select candidate itemsets. The second round
releases positive-set DP-FM Alpha sketches. The server uses MAP to estimate intersection
support counts and returns the top-k itemsets.

End-to-end workflow: **prepare integer data → partition attributes → run the protocol →
verify against global ground truth → aggregate metrics → generate PDF figures**.
The repository runs independently of the original project, historical results, or paper
directory. It uses the CPU by default and does not require a GPU.

## Installation and quick start

Linux and Python 3.11 or later are recommended. Run the following from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,data,crypto]'
privfim prepare --datasets Toy
privfim run --dataset Toy --seeds 2026 --m 64 --k 5 --output results/quickstart
pytest -q
```

Toy is a deterministic fixture with 1,200 rows and eight binary attributes.
The m=64 and k=5 settings above are for a quick check. The standard defaults are
ε=1, δ=10^-5, m=2048, k=15, K=4, itemset sizes 1–4, and an equal budget split between
the two rounds. To run the full default configuration:

```bash
privfim configured --config config/default.json
```

Data and output paths in the configuration are resolved relative to the JSON file.
`python main.py ...` is equivalent to `privfim ...`.
The release was verified with Python 3.12.3. See `requirements-lock.txt` for the dependency
snapshot and [VALIDATION.md](docs/VALIDATION.md) for the checks performed and their scope.

## Data preparation

```bash
privfim prepare --datasets Bank DefaultCredit CensusIncomeKDD MiniBooNE PokerHand LetterRecognition
privfim run --dataset DefaultCredit --seeds 2026 2027 2028 2029 2030 --output results/credit
```

Raw downloads are stored in `data/raw/`. Integer CSV files and encoding metadata are stored
in `data/real/`.

The CSV header must be `0,1,...,M-1`. All remaining rows contain integer category codes.
Codes must be consistent within each column, and missing values should use predefined
categories. Identical numbers in different columns represent different items.
Public domains and preprocessing rules are experiment inputs.

## Methods

| CLI name | Reporting and estimation |
|---|---|
| MAP-M (PriVFim in figures) | Jointly ranks target items and local joint projections, reports at most k keys per owner, and uses positive-set Alpha with MAP |
| IE-Full | Reports the full domains of relevant attributes and subtracts the estimated complement union from noisy N |
| IE-Other | Merges untargeted values into an OTHER bucket per attribute, then uses complement-union estimation |
| FO | Uses aligned OUE reports and debiased intersection estimation |
| First-round | Uses one round of noisy item counts and frequency products to guess candidate support |
| Second-round | Uses one round to report Alpha for all public local singletons, pairs, and triples, then applies MAP |
| Items-only | Reports only public-domain singleton Alpha in one round, as an additional ablation |

```bash
privfim run --dataset Toy --methods MAP-S MAP-L MAP-M --seeds 2026 --output results/map_compare
```

## Eleven experiment families

Each experiment has a separate entry-point file and uses the same protocol and evaluation workflow.

| No. | Varied parameter | Default datasets |
|---|---|---|
| 1 | ε = 0.25, 0.5, 1, 2, 4 | CensusIncomeKDD, MiniBooNE |
| 2 | k = 5, 10, 15, 20, 25 | CensusIncomeKDD, MiniBooNE |
| 3 | K = 2, 4, 8 | CensusIncomeKDD, MiniBooNE |
| 4 | Sample ratio: 0.2, 0.4, 0.6, 0.8, 1 | Bank, DefaultCredit |
| 5 | Attribute-merging ratio: same values as experiment 4 | Bank, DefaultCredit |
| 6 | Domain-reduction ratio: same values as experiment 4 | Bank, DefaultCredit |
| 7 | First-round report limit / k = 0.5, 0.75, 1, 1.5, 2 | DefaultCredit, MiniBooNE |
| 8 | Second-round report limit / k: same values as experiment 7 | Bank, MiniBooNE |
| 9 | Candidate count / k = 1, 1.5, 2, 2.5, 5 | CensusIncomeKDD, PokerHand |
| 10 | First-round budget fraction: 0.1 through 0.9 | PokerHand, MiniBooNE |
| 11 | First-round / Second-round / MAP-M | Bank, DefaultCredit |

```bash
# Run experiment 4 independently.
python new-Experiments/experiment04_sample_size.py --datasets Bank DefaultCredit --seeds 2026 --output results/exp4

# Run all experiments with the five default seeds: 2026–2030.
privfim suite --experiments 1 2 3 4 5 6 7 8 9 10 11 --output results/suite

# Check all eleven entry points offline, without launching large cryptographic runs.
privfim suite --datasets Toy --seeds 2026 --m 32 --output results/all11_smoke

# Override the values of a single parameter axis.
privfim suite --experiments 2 --datasets Bank --values 5 15 25 --seeds 2026 --output results/k_custom
```

## Outputs and evaluation

Each experiment produces `runs.csv`, `aggregated.csv`, and `figures/*.pdf`.
Each task also saves `config.json`, `result.json`, `status.json`, `estimates.csv/json`,
and `global_truth.json`.

- F1 measures set overlap between the predicted top-k and the true top-k over the complete legal itemset space.
- NCR uses zero-based true ranks, weights k−rank, and normalization k(k+1)/2.
- Equal true support counts are resolved by canonical itemset lexicographic order, consistently across methods.
- Support-count MSE and frequency MSE are saved separately. Frequency MSE equals support-count MSE/N².
- Runtime includes the first round, shared-rank construction, second-round reporting, and server estimation. It excludes data loading, offline ground-truth computation, and plotting.
- Communication is calculated from a compact message format. It is not a network capture and excludes transport headers.

`protocol/summary.json` retains the core implementation's candidate-restricted evaluation
for diagnostics. **Use the outer `result.json` and `runs.csv` metrics with
`evaluation_scope=global_exact` for formal comparisons.** Global ground truth is computed
independently after the protocol and is never fed back into candidate generation or estimation.

```bash
privfim plot --input results/credit/aggregated.csv --output results/credit/redraw
```

## Repository layout

```text
privfim/                 Clients, server, DP-FM/MAP, complement unions, budgets, metrics
workflows/               Data preparation, execution, global audit, CSV/PDF, crypto entry point
new-Experiments/         One entry-point file for each of experiments 1–11
experiments/             Reused experiment modules and native cryptographic adapters
scripts/setup_crypto.py  Fetch pinned upstream sources and build TFHE
config/                 Default JSON configuration
data/reference/         Historical Bank integer table and checksums
tests/                  Algorithm, ground-truth, workflow, and crypto correctness tests
docs/                   Data, protocol, cryptography, and validation documentation
```
