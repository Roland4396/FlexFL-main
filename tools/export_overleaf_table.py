#!/usr/bin/env python3
"""Export an Overleaf-ready LaTeX table from batch log files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


DEFAULT_CLIENT_RATIO = "4:3:2:1"
METHODS = [
    ("fedavg", "FedAvg"),
    ("fedprox", "FedProx"),
    ("heterofl", "HeteroFL"),
    ("scalefl", "ScaleFL"),
    ("decoupled", "Decoupled"),
    ("flexfl", "FlexFL"),
]
BACKBONES = [
    ("resnet", "ResNet-110"),
    ("vgg", "VGG-16"),
]
DATASETS = [
    ("cifar10", "CIFAR-10"),
    ("cifar100", "CIFAR-100"),
    ("tinyimagenet", "Tiny-ImageNet"),
]
CONFIGS = [
    ("iid", "IID"),
    ("noniid_beta1", r"$\alpha=1$"),
    ("noniid_beta100", r"$\alpha=100$"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export current batch results as a LaTeX table.")
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs_batch_all"),
        help="Root directory that contains timestamped batch runs.",
    )
    parser.add_argument(
        "--batch",
        type=str,
        default=None,
        help="Specific batch directory name. Default: latest timestamped directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper_tables/current_batch_results_table.tex"),
        help="Output .tex file.",
    )
    parser.add_argument(
        "--client-ratio",
        type=str,
        default=DEFAULT_CLIENT_RATIO,
        help="Fallback client ratio for Decoupled when it cannot be parsed from the log.",
    )
    return parser.parse_args()


def latest_batch(outputs_root: Path) -> Path:
    candidates = sorted([p for p in outputs_root.iterdir() if p.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"No batch directories found under {outputs_root}")
    return candidates[-1]


def ratio_to_weights(ratio: str) -> list[float]:
    values = [float(item) for item in ratio.split(":")]
    total = sum(values)
    if total <= 0:
        raise ValueError(f"Invalid client ratio: {ratio}")
    return [item / total for item in values]


def parse_client_ratio(text: str, fallback: str) -> str:
    matches = re.findall(r"--client_hetero_ration\s+([^\s]+)", text)
    return matches[-1] if matches else fallback


def parse_final_accuracy(method: str, text: str, fallback_ratio: str) -> float | None:
    accs = [float(item) for item in re.findall(r"Testing accuracy:\s*([0-9.]+)", text)]
    if not accs:
        return None
    if method != "decoupled":
        return accs[-1]

    weights = ratio_to_weights(parse_client_ratio(text, fallback_ratio))
    if len(accs) < len(weights):
        return None
    final_level_accs = accs[-len(weights):]
    return sum(acc * weight for acc, weight in zip(final_level_accs, weights))


def parse_logs(log_dir: Path, fallback_ratio: str) -> dict[tuple[str, str, str, str], str]:
    results: dict[tuple[str, str, str, str], str] = {}
    pattern = re.compile(r"([^_]+)_([^_]+)_([^_]+)_(.+)\.log$")

    for log_path in sorted(log_dir.glob("*.log")):
        match = pattern.match(log_path.name)
        if not match:
            continue
        method, backbone, dataset, config = match.groups()
        text = log_path.read_text(errors="ignore")
        done = "returncode: 0" in text or "save finished" in text
        final_acc = parse_final_accuracy(method, text, fallback_ratio)
        if done and final_acc is not None:
            results[(method, backbone, dataset, config)] = f"{final_acc:.2f}"
    return results


def render_table(batch_name: str, results: dict[tuple[str, str, str, str], str]) -> str:
    lines: list[str] = []
    lines.append(f"% Auto-generated from outputs_batch_all/{batch_name}")
    lines.append(r"\documentclass[11pt]{article}")
    lines.append(r"\usepackage[margin=1in]{geometry}")
    lines.append(r"\usepackage{graphicx}")
    lines.append(r"\pagestyle{empty}")
    lines.append(r"")
    lines.append(r"\begin{document}")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{4.5pt}")
    lines.append(
        r"\caption{Current Top-1 accuracy (\%) snapshot on ResNet-110 and VGG-16. "
        r"Entries are filled only when the corresponding run has completed. "
        r"\texttt{IID} denotes homogeneous partition, while $\alpha=1$ and $\alpha=100$ "
        r"denote Dirichlet non-IID splits. For Decoupled, we report the client-ratio-weighted "
        r"average over resource-level models. \texttt{--} indicates unfinished runs.}"
    )
    lines.append(r"\label{tab:current_batch_results}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lllcccccc}")
    lines.append(r"\hline")
    lines.append(
        r"Model & Dataset & Split & FedAvg & FedProx & HeteroFL & ScaleFL & Decoupled & FlexFL \\"
    )
    lines.append(r"\hline")

    for backbone_key, backbone_name in BACKBONES:
        for dataset_key, dataset_name in DATASETS:
            for config_key, config_name in CONFIGS:
                row = [backbone_name, dataset_name, config_name]
                for method_key, _ in METHODS:
                    row.append(results.get((method_key, backbone_key, dataset_key, config_key), r"\texttt{--}"))
                lines.append(" & ".join(row) + r" \\")
            lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")
    lines.append(r"\end{document}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    batch_dir = args.outputs_root / args.batch if args.batch else latest_batch(args.outputs_root)
    log_dir = batch_dir / "logs"
    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    results = parse_logs(log_dir, args.client_ratio)
    content = render_table(batch_dir.name, results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
