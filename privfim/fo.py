from __future__ import annotations

import math

import numpy as np


def oue_parameters(epsilon: float) -> tuple[float, float]:
    """经典 OUE 的一元编码参数 (p, q)。"""
    if epsilon <= 0:
        raise ValueError("OUE epsilon 必须为正数")
    return 0.5, 1.0 / (math.exp(epsilon) + 1.0)


def oue_encode(values: np.ndarray, domain: tuple[int, ...], epsilon: float, seed: int) -> np.ndarray:
    """对分类取值执行 OUE，返回 shape=(n, |domain|) 的二进制报告。"""
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError("OUE 输入必须是一维分类值")
    if not domain:
        raise ValueError("OUE 值域不能为空")
    p, q = oue_parameters(epsilon)
    domain_array = np.asarray(domain, dtype=values.dtype)
    truth = values[:, None] == domain_array[None, :]
    probabilities = np.where(truth, p, q)
    rng = np.random.default_rng(seed)
    return rng.random(probabilities.shape) < probabilities


def oue_estimate_count(reports: np.ndarray, epsilon: float, category_index: int) -> float:
    """从 OUE 某一类别的报告列作无偏频数估计。"""
    reports = np.asarray(reports, dtype=np.float64)
    if reports.ndim != 2:
        raise ValueError("OUE 报告必须是二维数组")
    p, q = oue_parameters(epsilon)
    return float((reports[:, category_index].sum() - len(reports) * q) / (p - q))


def oue_column_sums(
    values: np.ndarray,
    domain: tuple[int, ...],
    epsilon: float,
    seed: int,
    max_chunk_cells: int = 1_000_000,
) -> np.ndarray:
    """分批生成相同 OUE 随机位并累计列和，避免保留 N×值域矩阵。

    随机数按原实现的逐行顺序生成；固定输入和 seed 时，列和与完整
    oue_encode 的结果完全一致。通信量仍按实际模拟的报告位数统计。
    """
    values = np.asarray(values)
    if values.ndim != 1 or not domain or max_chunk_cells < 1:
        raise ValueError("OUE 需要一维输入、非空值域和正的分块上限")
    p, q = oue_parameters(epsilon)
    domain_array = np.asarray(domain, dtype=values.dtype)
    rows_per_chunk = max(1, max_chunk_cells // len(domain))
    sums = np.zeros(len(domain), dtype=np.int64)
    rng = np.random.default_rng(seed)
    for start in range(0, len(values), rows_per_chunk):
        chunk = values[start:start + rows_per_chunk]
        probabilities = np.where(chunk[:, None] == domain_array[None, :], p, q)
        sums += (rng.random(probabilities.shape) < probabilities).sum(axis=0)
    return sums


def oue_positive_membership(membership: np.ndarray, epsilon: float, seed: int) -> np.ndarray:
    """二值 OUE 的正类别位；该位是交集估计所需的充分统计量。"""
    membership = np.asarray(membership, dtype=bool)
    p, q = oue_parameters(epsilon)
    rng = np.random.default_rng(seed)
    return rng.random(len(membership)) < np.where(membership, p, q)


def oue_intersection_estimate(
    positive_reports: list[np.ndarray], epsilons: list[float], n_rows: float
) -> float:
    """由对齐用户的独立 OUE 二值报告无偏估计多块交集。

    OUE 的向量长度仍由实际对齐报告确定；``n_rows`` 仅作为服务端发布的
    noisy N，用于结果截断，不再要求它是整数或等于消息长度。
    """
    if not positive_reports:
        return 0.0
    if len(positive_reports) != len(epsilons):
        raise ValueError("OUE 报告数与预算数不一致")
    reports = [np.asarray(report, dtype=np.float64) for report in positive_reports]
    report_length = reports[0].shape
    if any(report.shape != report_length for report in reports):
        raise ValueError("OUE 二值报告长度必须在各块之间一致")
    centered = []
    for report, epsilon in zip(reports, epsilons, strict=True):
        p, q = oue_parameters(epsilon)
        centered.append((report - q) / (p - q))
    return float(np.clip(np.prod(np.stack(centered, axis=0), axis=0).sum(), 0, n_rows))
