"""Plot measured aggregates only, without automatically filling missing points."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = {"MAP-M": "PriVFim", "IE-Full": "IE-Full", "IE-Other": "IE-Other", "FO": "FO"}
AXES = {"epsilon": r"$\epsilon$", "k": "$k$", "clients": "$K$", "sample_ratio": "$N/N_{full}$",
        "feature_ratio": "$M/M_{full}$", "domain_ratio": "Domain proportion",
        "first_stage_pool": r"$|\mathcal{P}_i|/k$", "second_stage_pool": r"$|\mathcal{Q}_i|/k$",
        "candidate_pool": r"$|\mathcal{C}|/k$", "budget_split": r"$\epsilon_1/\epsilon$"}


def plot(source, output):
    with Path(source).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(r["result_kind"] != "measured" for r in rows):
        raise ValueError("Expected nonempty measured aggregates")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "axes.labelsize": 10, "legend.fontsize": 10})
    for dataset in dict.fromkeys(r["dataset"] for r in rows):
        selected = [r for r in rows if r["dataset"] == dataset]
        methods = list(dict.fromkeys(r["method"] for r in selected))
        axis = selected[0]["axis"]
        fig, axes = plt.subplots(1, 4, figsize=(12, 2.8))
        for ax, metric, title in zip(axes, ("f1", "ncr", "total_seconds", "communication_bytes"),
                                    ("F1", "NCR", "Time (s)", "Communication (MiB)")):
            for i, method in enumerate(methods):
                records = sorted((r for r in selected if r["method"] == method), key=lambda r: float(r["value"]))
                x = np.array([float(r["value"]) for r in records])
                scale = 2 ** 20 if metric == "communication_bytes" else 1
                means = np.array([float(r[metric + "_mean"]) / scale for r in records])
                errors = np.array([float(r[metric + "_std"]) / scale for r in records])
                if axis in {"default", "mechanism"}:
                    ax.bar(i, means[0], yerr=errors[0], label=LABELS.get(method, method), alpha=.8)
                else:
                    ax.errorbar(x, means, yerr=errors, label=LABELS.get(method, method),
                                marker=("o", "s", "^", "D", "p", "h")[i % 6], mfc="white", capsize=2)
            ax.set_ylabel(title)
            if axis in {"default", "mechanism"}:
                ax.set_xticks(range(len(methods)), [LABELS.get(m, m) for m in methods], rotation=25, ha="right")
            else:
                ax.set_xlabel(AXES.get(axis, axis))
                ticks = sorted({float(r["value"]) for r in selected})
                ax.set_xticks(ticks)
            ax.grid(alpha=.2, axis="y")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=min(len(methods), 6), frameon=False)
        fig.tight_layout(rect=(0, 0, 1, .88))
        fig.savefig(output / f"{dataset}_{axis}.pdf")
        plt.close(fig)
