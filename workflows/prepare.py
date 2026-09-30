"""官方数据下载、整数编码与来源记录；Toy 可完全离线生成。"""
from __future__ import annotations

import csv
import gzip
import io
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from workflows.common import ROOT, digest, write_json


CATALOG = {
    "CensusIncomeKDD": (117, "census%2Bincome%2Bkdd", 299285, 40),
    "MiniBooNE": (199, "miniboone+particle+identification", 130064, 50),
    "PokerHand": (158, "poker+hand", 1025010, 10),
    "DefaultCredit": (350, "default+of+credit+card+clients", 30000, 23),
    "LetterRecognition": (59, "letter+recognition", 20000, 16),
}
DATASETS = ("Toy", "Bank", *CATALOG)


def download(url, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target
    tmp = target.with_suffix(target.suffix + ".download")
    request = urllib.request.Request(url, headers={"User-Agent": "PrivFim-reproducibility/0.2"})
    with urllib.request.urlopen(request, timeout=180) as source, tmp.open("wb") as dest:
        shutil.copyfileobj(source, dest)
    with zipfile.ZipFile(tmp) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"Invalid archive: {url}")
    tmp.replace(target)
    return target


def write_matrix(path, values):
    with Path(path).open("w", newline="", encoding="utf8") as handle:
        writer = csv.writer(handle)
        writer.writerow(range(values.shape[1]))
        writer.writerows(values.tolist())


def encode_credit(sheet):
    headings = list(map(str, sheet.row_values(1)[1:-1]))
    raw = np.array([sheet.row_values(i)[1:-1] for i in range(2, sheet.nrows)])
    encoded = np.empty(raw.shape, dtype=np.int64)
    mappings = []
    for j in range(raw.shape[1]):
        if j in {0, 4, *range(11, 23)}:
            x = raw[:, j].astype(float)
            nonzero = x[x != 0]
            edges = np.unique(np.quantile(nonzero, np.linspace(0, 1, 11)[1:-1])) if len(nonzero) else np.array([])
            encoded[:, j] = np.where(x == 0, 0, 1 + np.searchsorted(edges, x, side="right"))
            mapping = {"encoding": "zero + up to 10 nonzero quantile bins", "edges": edges.tolist()}
        else:
            strings = [str(x) for x in raw[:, j]]
            categories = sorted(set(strings))
            codes = {value: i for i, value in enumerate(categories)}
            encoded[:, j] = [codes[x] for x in strings]
            mapping = {"encoding": "sorted categorical strings", "values": categories}
        mappings.append({"attribute": j, "name": headings[j], **mapping})
    return encoded, mappings


def prepare(name, data_dir=None):
    base = Path(data_dir or ROOT / "data").resolve()
    out = base / "real" / f"{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out.with_suffix(".json")
    if out.exists():
        if not meta_path.exists():
            raise ValueError(f"CSV exists without provenance: {out}; use another data directory")
        meta = json.loads(meta_path.read_text())
        if digest(out) != meta["csv_sha256"]:
            raise ValueError(f"Modified data: {out}")
        return out
    meta = {"name": name, "preprocessing_is_public": True, "includes_label": False}
    if name == "Toy":
        rng = np.random.default_rng(2026)
        latent = rng.integers(0, 2, (1200, 4))
        values = np.repeat(latent, 2, axis=1)
        values ^= (rng.random(values.shape) < .08).astype(np.int64)
        meta.update(source="deterministic synthetic fixture", generation_seed=2026)
    elif name == "Bank":
        archive = ROOT / "data/reference/Bank.csv.gz"
        reference = json.loads(archive.with_suffix(".json").read_text())
        if digest(archive) != reference["archive_sha256"]:
            raise ValueError("Bank reference checksum mismatch")
        with gzip.open(archive, "rb") as src, out.open("wb") as dest:
            shutil.copyfileobj(src, dest)
        values = np.loadtxt(out, delimiter=",", skiprows=1, dtype=np.int64)
        meta.update(reference)
    else:
        ident, slug, expected_n, expected_m = CATALOG[name]
        url = f"https://archive.ics.uci.edu/static/public/{ident}/{slug}.zip"
        archive = download(url, base / "raw" / f"{name}.zip")
        meta.update(download_url=url, source_page=f"https://archive.ics.uci.edu/dataset/{ident}", archive_sha256=digest(archive))
        if name == "CensusIncomeKDD":
            from experiments.prepare_large_data import prepare_census_income_kdd
            details = prepare_census_income_kdd(archive, out)
            values = np.loadtxt(out, delimiter=",", skiprows=1, dtype=np.int64)
            meta["processing"] = details
        else:
            with zipfile.ZipFile(archive) as z:
                if name == "DefaultCredit":
                    import xlrd
                    member = next(x for x in z.namelist() if x.endswith(".xls"))
                    sheet = xlrd.open_workbook(file_contents=z.read(member)).sheet_by_index(0)
                    values, meta["encoding"] = encode_credit(sheet)
                elif name == "MiniBooNE":
                    from experiments.prepare_miniboone import encode
                    member = next(x for x in z.namelist() if x.endswith("MiniBooNE_PID.txt"))
                    with z.open(member) as handle:
                        counts = [int(x) for x in handle.readline().split()]
                        raw = np.loadtxt(handle)
                    if sum(counts) != len(raw):
                        raise ValueError("MiniBooNE row count mismatch")
                    values, _, maps = encode(raw)
                    meta["encoding"] = "10 full-table quantile bins per feature, repeated edges removed"
                    meta["quantile_edges"] = [
                        np.unique(np.quantile(raw[:, j], np.linspace(0, 1, 11))).tolist()
                        for j in range(raw.shape[1])
                    ]
                    meta["value_labels"] = maps
                elif name == "LetterRecognition":
                    member = next(x for x in z.namelist() if x.endswith("letter-recognition.data"))
                    rows = list(csv.reader(io.StringIO(z.read(member).decode("ascii"))))
                    values = np.array([r[1:] for r in rows], dtype=np.int64)
                    meta["encoding"] = "original integer features, class removed"
                elif name == "PokerHand":
                    parts = []
                    for suffix in ("poker-hand-training-true.data", "poker-hand-testing.data"):
                        member = next(x for x in z.namelist() if x.endswith(suffix))
                        with z.open(member) as handle:
                            parts.append(np.loadtxt(handle, delimiter=",", dtype=np.int64)[:, :-1])
                    values = np.concatenate(parts)
                    # 原始花色、点数已经是整数，保留它们以兼容已有实验的 item 键。
                    if (np.any((values[:, ::2] < 1) | (values[:, ::2] > 4))
                            or np.any((values[:, 1::2] < 1) | (values[:, 1::2] > 13))):
                        raise ValueError("PokerHand suit/rank outside the official domains")
                    meta["encoding"] = "train + test, original integer suits/ranks, class removed"
        if values.shape != (expected_n, expected_m):
            raise ValueError(f"Unexpected shape for {name}: {values.shape}")
    if not out.exists():
        write_matrix(out, values)
    meta.update(rows=len(values), attributes=values.shape[1],
                domain_sizes=[len(np.unique(values[:, j])) for j in range(values.shape[1])],
                csv_sha256=digest(out))
    write_json(meta_path, meta)
    return out
