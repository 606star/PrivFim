from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "CensusIncomeKDD"
DOWNLOAD_URL = (
    "https://archive.ics.uci.edu/static/public/117/"
    "census%2Bincome%2Bkdd.zip"
)
DOI = "https://doi.org/10.24432/C5N30T"
EXPECTED_ROWS = 299_285
RAW_COLUMNS = 42
IGNORED_SOURCE_COLUMNS = frozenset({24})  # UCI explicitly instructs users to ignore instance weights.
TARGET_SOURCE_COLUMN = 41
CONTINUOUS_SOURCE_COLUMNS = frozenset({0, 5, 16, 17, 18, 30, 39})
ZERO_AWARE_SOURCE_COLUMNS = frozenset({5, 16, 17, 18})
RAW_FILES = ("census-income.data", "census-income.test")

RAW_COLUMN_NAMES = (
    "age",
    "class of worker",
    "detailed industry recode",
    "detailed occupation recode",
    "education",
    "wage per hour",
    "enroll in edu inst last wk",
    "marital stat",
    "major industry code",
    "major occupation code",
    "race",
    "hispanic origin",
    "sex",
    "member of a labor union",
    "reason for unemployment",
    "full or part time employment stat",
    "capital gains",
    "capital losses",
    "dividends from stocks",
    "tax filer stat",
    "region of previous residence",
    "state of previous residence",
    "detailed household and family stat",
    "detailed household summary in household",
    "instance weight",
    "migration code-change in msa",
    "migration code-change in reg",
    "migration code-move within reg",
    "live in this house 1 year ago",
    "migration prev res in sunbelt",
    "num persons worked for employer",
    "family members under 18",
    "country of birth father",
    "country of birth mother",
    "country of birth self",
    "citizenship",
    "own business or self employed",
    "fill inc questionnaire for veteran's admin",
    "veterans benefits",
    "weeks worked in year",
    "year",
    "income class",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive_path.with_suffix(".download")
    request = urllib.request.Request(
        DOWNLOAD_URL, headers={"User-Agent": "PrivFim dataset preparation"}
    )
    with urllib.request.urlopen(request) as response, temporary.open("wb") as target:
        shutil.copyfileobj(response, target)
    temporary.replace(archive_path)


def _extract_raw_files(archive_path: Path, output_dir: Path) -> tuple[Path, ...]:
    # The UCI ZIP contains census.tar.gz; extract only the two explicitly named data members.
    with zipfile.ZipFile(archive_path) as outer:
        with outer.open("census.tar.gz") as source:
            tar_path = output_dir / "census.tar.gz"
            with tar_path.open("wb") as target:
                shutil.copyfileobj(source, target)

    paths = []
    with tarfile.open(tar_path, mode="r:gz") as archive:
        for filename in RAW_FILES:
            member = archive.getmember(filename)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"无法从 UCI 归档读取 {filename}")
            destination = output_dir / filename
            with source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            paths.append(destination)
    return tuple(paths)


def _normalise_row(row: Sequence[str]) -> list[str]:
    if len(row) != RAW_COLUMNS:
        raise ValueError(f"Census-Income 原始行应有 {RAW_COLUMNS} 列，实际为 {len(row)}")
    values = [value.strip() for value in row]
    values[-1] = values[-1].removesuffix(".")
    return values


def _iter_rows(paths: Sequence[Path]) -> Iterator[list[str]]:
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.reader(handle):
                yield _normalise_row(row)


def _quantile_edges(
    values: Sequence[float], bins: int, zero_aware: bool
) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if zero_aware:
        array = array[array > 0]
    if array.size == 0:
        return []
    quantiles = np.linspace(0.0, 1.0, bins + 1)[1:-1]
    return [
        float(value)
        for value in np.unique(np.quantile(array, quantiles, method="linear"))
    ]


def convert_raw_files(
    raw_paths: Sequence[Path],
    output_path: Path,
    *,
    quantile_bins: int = 10,
    target_path: Path | None = None,
) -> dict:
    if quantile_bins < 2:
        raise ValueError("quantile_bins 至少为 2")

    included = tuple(
        index
        for index in range(RAW_COLUMNS)
        if index not in IGNORED_SOURCE_COLUMNS and index != TARGET_SOURCE_COLUMN
    )
    continuous_values = {index: [] for index in CONTINUOUS_SOURCE_COLUMNS}
    categories = {
        index: set() for index in included if index not in CONTINUOUS_SOURCE_COLUMNS
    }

    rows = 0
    for row in _iter_rows(raw_paths):
        rows += 1
        for index in CONTINUOUS_SOURCE_COLUMNS:
            continuous_values[index].append(float(row[index]))
        for index in categories:
            categories[index].add(row[index])

    edges = {
        index: _quantile_edges(
            values,
            bins=quantile_bins,
            zero_aware=index in ZERO_AWARE_SOURCE_COLUMNS,
        )
        for index, values in continuous_values.items()
    }
    category_maps = {
        index: {value: code for code, value in enumerate(sorted(values))}
        for index, values in categories.items()
    }
    target_values = sorted({row[TARGET_SOURCE_COLUMN] for row in _iter_rows(raw_paths)})
    target_map = {value: code for code, value in enumerate(target_values)}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_path = target_path or output_path.with_name(
        f"{output_path.stem}_target{output_path.suffix}"
    )
    temporary = output_path.with_suffix(".tmp")
    target_temporary = target_path.with_suffix(".tmp")
    with (
        temporary.open("w", encoding="utf-8", newline="") as handle,
        target_temporary.open("w", encoding="utf-8", newline="") as target_handle,
    ):
        writer = csv.writer(handle, lineterminator="\n")
        target_writer = csv.writer(target_handle, lineterminator="\n")
        writer.writerow(range(len(included)))
        target_writer.writerow((0,))
        for row in _iter_rows(raw_paths):
            encoded = []
            for index in included:
                if index in CONTINUOUS_SOURCE_COLUMNS:
                    value = float(row[index])
                    offset = int(index in ZERO_AWARE_SOURCE_COLUMNS)
                    if offset and value == 0:
                        code = 0
                    else:
                        code = offset + bisect.bisect_right(edges[index], value)
                else:
                    code = category_maps[index][row[index]]
                encoded.append(code)
            writer.writerow(encoded)
            target_writer.writerow((target_map[row[TARGET_SOURCE_COLUMN]],))
    temporary.replace(output_path)
    target_temporary.replace(target_path)

    attributes = []
    for output_index, source_index in enumerate(included):
        if source_index in CONTINUOUS_SOURCE_COLUMNS:
            domain = len(edges[source_index]) + 1 + int(
                source_index in ZERO_AWARE_SOURCE_COLUMNS
            )
            attributes.append(
                {
                    "attribute": output_index,
                    "source_index": source_index,
                    "name": RAW_COLUMN_NAMES[source_index],
                    "encoding": "zero_plus_positive_quantiles"
                    if source_index in ZERO_AWARE_SOURCE_COLUMNS
                    else "quantiles",
                    "domain": domain,
                    "bin_edges": edges[source_index],
                }
            )
        else:
            ordered = sorted(categories[source_index])
            attributes.append(
                {
                    "attribute": output_index,
                    "source_index": source_index,
                    "name": RAW_COLUMN_NAMES[source_index],
                    "encoding": "categorical",
                    "domain": len(ordered),
                    "categories": ordered,
                }
            )

    return {
        "rows": rows,
        "attributes": len(included),
        "predictor_attributes": len(included),
        "includes_income_label": False,
        "target_source_column": TARGET_SOURCE_COLUMN,
        "target_path": str(target_path.resolve()),
        "target_categories": target_values,
        "ignored_source_columns": [24],
        "quantile_bins": quantile_bins,
        "attribute_metadata": attributes,
    }


def prepare_census_income_kdd(
    archive_path: Path,
    output_path: Path,
    *,
    quantile_bins: int = 10,
    force: bool = False,
) -> dict:
    metadata_path = output_path.with_suffix(".json")
    if not archive_path.exists():
        print(f"下载 {DATASET_NAME}: {DOWNLOAD_URL}")
        _download(archive_path)
    archive_sha256 = _sha256(archive_path)

    if output_path.exists() and metadata_path.exists() and not force:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        source = existing.get("source", {})
        if (
            source.get("archive_sha256") == archive_sha256
            and existing.get("processing", {}).get("quantile_bins") == quantile_bins
            and existing.get("includes_income_label") is False
            and Path(existing.get("target_csv", "")).is_file()
        ):
            return existing

    with tempfile.TemporaryDirectory(prefix="privfim-census-") as temporary:
        raw_paths = _extract_raw_files(archive_path, Path(temporary))
        converted = convert_raw_files(
            raw_paths, output_path, quantile_bins=quantile_bins
        )

    if converted["rows"] != EXPECTED_ROWS:
        raise ValueError(
            f"Census-Income 行数异常: {converted['rows']}，期望 {EXPECTED_ROWS}"
        )

    metadata = {
        "name": DATASET_NAME,
        "description": "UCI Census-Income (KDD), train/test 合并后的 40 特征离散版本",
        "source": {
            "repository": "UCI Machine Learning Repository",
            "url": "https://archive.ics.uci.edu/dataset/117/census-income+kdd",
            "download_url": DOWNLOAD_URL,
            "doi": DOI,
            "license": "CC BY 4.0",
            "archive": str(archive_path.resolve()),
            "archive_sha256": archive_sha256,
        },
        "processing": {
            "raw_rows": converted["rows"],
            "raw_columns": RAW_COLUMNS,
            "ignored_instance_weight": True,
            "merged_train_and_test": True,
            "quantile_bins": quantile_bins,
            "missing_values_as_category": True,
            "income_label_separated": True,
        },
        "csv": str(output_path.resolve()),
        "rows": converted["rows"],
        "attributes": converted["attributes"],
        "predictor_attributes": converted["predictor_attributes"],
        "includes_income_label": converted["includes_income_label"],
        "target_csv": converted["target_path"],
        "target_categories": converted["target_categories"],
        "sha256": _sha256(output_path),
        "target_sha256": _sha256(Path(converted["target_path"])),
        "attribute_metadata": converted["attribute_metadata"],
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def _update_real_manifest(metadata: dict) -> None:
    manifest_path = PROJECT_ROOT / "data" / "real" / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {
            "provenance": "official_uci_download_plus_deterministic_integer_encoding",
            "datasets": [],
        }
    )
    entry = {
        "name": metadata["name"],
        "source": metadata["source"]["url"],
        "download_url": metadata["source"]["download_url"],
        "csv": metadata["csv"],
        "target_csv": metadata["target_csv"],
        "rows": metadata["rows"],
        "attributes": metadata["attributes"],
        "predictor_attributes": metadata["predictor_attributes"],
        "includes_income_label": metadata["includes_income_label"],
        "sha256": metadata["sha256"],
        "target_sha256": metadata["target_sha256"],
        "processing": metadata["processing"],
    }
    datasets = [
        existing
        for existing in manifest.get("datasets", [])
        if existing.get("name") != metadata["name"]
    ]
    datasets.append(entry)
    manifest["datasets"] = datasets
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="准备大规模 UCI 真实数据集")
    parser.add_argument("--force", action="store_true", help="强制重新转换")
    parser.add_argument("--quantile-bins", type=int, default=10)
    args = parser.parse_args()

    metadata = prepare_census_income_kdd(
        PROJECT_ROOT / "data" / "raw" / "CensusIncomeKDD.zip",
        PROJECT_ROOT / "data" / "real" / f"{DATASET_NAME}.csv",
        quantile_bins=args.quantile_bins,
        force=args.force,
    )
    _update_real_manifest(metadata)
    print(
        f"{DATASET_NAME}: rows={metadata['rows']}, "
        f"attributes={metadata['attributes']}, csv={metadata['csv']}"
    )


if __name__ == "__main__":
    main()
