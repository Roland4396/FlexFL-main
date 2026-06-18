#!/usr/bin/env python3
"""Export a standalone LaTeX table that combines prior AlignFL results with completed baselines."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


DEFAULT_CLIENT_RATIO = "4:3:2:1"
ROWS = [
    ("resnet", "cifar10", "IID"),
    ("resnet", "cifar10", "1"),
    ("resnet", "cifar10", "100"),
    ("resnet", "cifar100", "IID"),
    ("resnet", "cifar100", "1"),
    ("resnet", "cifar100", "100"),
    ("resnet", "tinyimagenet", "IID"),
    ("resnet", "tinyimagenet", "1"),
    ("resnet", "tinyimagenet", "100"),
    ("vgg", "cifar10", "IID"),
    ("vgg", "cifar10", "1"),
    ("vgg", "cifar10", "100"),
    ("vgg", "cifar100", "IID"),
    ("vgg", "cifar100", "1"),
    ("vgg", "cifar100", "100"),
    ("vgg", "tinyimagenet", "IID"),
    ("vgg", "tinyimagenet", "1"),
    ("vgg", "tinyimagenet", "100"),
]

MODEL_LABELS = {
    "resnet": "ResNet-110",
    "vgg": "VGG-16",
}

DATASET_LABELS = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "tinyimagenet": "Tiny-ImageNet",
}

CONFIG_TO_ALPHA = {
    "iid": "IID",
    "noniid_beta1": "1",
    "noniid_beta100": "100",
}

METHOD_LABELS = {
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "heterofl": "HeteroFL",
    "scalefl": "ScaleFL",
    "decoupled": "Decoupled",
    "flexfl": "FlexFL",
}

METHOD_ORDER = ["fedavg", "fedprox", "heterofl", "scalefl", "decoupled", "flexfl"]

# Numbers transcribed from the user's existing AlignFL/TDD result figure.
ALIGNFL_RESULTS = {
    ("resnet", "cifar10", "1"): (74.7697, 75.6977),
    ("resnet", "cifar10", "100"): (85.3168, 86.1422),
    ("resnet", "cifar100", "1"): (46.6637, 48.8634),
    ("resnet", "cifar100", "100"): (47.8172, 50.4517),
    ("resnet", "tinyimagenet", "1"): (31.8826, 33.0451),
    ("resnet", "tinyimagenet", "100"): (32.1977, 33.9825),
    ("vgg", "cifar10", "1"): (80.0788, 83.9255),
    ("vgg", "cifar10", "100"): (87.7547, 88.6560),
    ("vgg", "cifar100", "1"): (52.1499, 57.7052),
    ("vgg", "cifar100", "100"): (54.3757, 60.4234),
    ("vgg", "tinyimagenet", "1"): (30.2509, 31.2359),
    ("vgg", "tinyimagenet", "100"): (30.9179, 31.2196),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export AlignFL comparison table as standalone TeX.")
    parser.add_argument(
        "--logs",
        type=Path,
        default=Path("outputs_batch_all") / "20260421_174531" / "logs",
        help="Path to the batch log directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper_tables") / "alignfl_baseline_gap_table.tex",
        help="Output standalone TeX file.",
    )
    parser.add_argument(
        "--client-ratio",
        type=str,
        default=DEFAULT_CLIENT_RATIO,
        help="Fallback client ratio for Decoupled when it cannot be parsed from the log.",
    )
    return parser.parse_args()


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


def parse_completed_results(log_dir: Path, fallback_ratio: str) -> dict[tuple[str, str, str, str], float]:
    results: dict[tuple[str, str, str, str], float] = {}
    pattern = re.compile(r"([^_]+)_([^_]+)_([^_]+)_(.+)\.log$")

    for log_path in sorted(log_dir.glob("*.log")):
        match = pattern.match(log_path.name)
        if not match:
            continue
        method, backbone, dataset, config = match.groups()
        alpha = CONFIG_TO_ALPHA.get(config)
        if alpha is None:
            continue

        text = log_path.read_text(errors="ignore")
        done = "returncode: 0" in text or "save finished" in text
        if not done:
            continue

        final_acc = parse_final_accuracy(method, text, fallback_ratio)
        if final_acc is None:
            continue

        results[(method, backbone, dataset, alpha)] = final_acc
    return results


def select_available_methods(results: dict[tuple[str, str, str, str], float]) -> list[str]:
    complete = []
    for method in METHOD_ORDER:
        if any(key[0] == method for key in results):
            complete.append(method)
    return complete


def format_num(value: float) -> str:
    return f"{value:.4f}"


def render_table(completed_methods: list[str], results: dict[tuple[str, str, str, str], float]) -> str:
    dynamic_cols = "c" * (2 + len(completed_methods) + 1)
    lines: list[str] = []
    lines.append(r"\documentclass[11pt]{article}")
    lines.append(r"\usepackage[margin=0.8in]{geometry}")
    lines.append(r"\usepackage{graphicx}")
    lines.append(r"\pagestyle{empty}")
    lines.append("")
    lines.append(r"\begin{document}")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{4.5pt}")
    lines.append(
        r"\caption{Comparison between prior AlignFL results and baselines completed in the current batch. "
        r"Methods with at least one finished run are shown. "
        r"Cells without a finished result are marked as \texttt{--}. "
        r"For Decoupled, we report the client-ratio-weighted average over resource-level models. "
        r"Gap is computed as the row-wise maximum minus minimum over the available displayed numbers.}"
    )
    lines.append(r"\label{tab:alignfl_baseline_gap}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lll" + dynamic_cols + r"}")
    lines.append(r"\hline")

    header = ["Model", "Dataset", r"$\alpha$", "AlignFL", "AlignFL-TDD"]
    header.extend(METHOD_LABELS[m] for m in completed_methods)
    header.append("Gap")
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\hline")

    for idx, (backbone, dataset, alpha) in enumerate(ROWS):
        prior_pair = ALIGNFL_RESULTS.get((backbone, dataset, alpha))
        displayed_values = []
        row = [
            MODEL_LABELS[backbone],
            DATASET_LABELS[dataset],
            alpha,
        ]
        if prior_pair is None:
            row.extend([r"\texttt{--}", r"\texttt{--}"])
        else:
            alignfl, tdd = prior_pair
            displayed_values.extend([alignfl, tdd])
            row.extend([format_num(alignfl), format_num(tdd)])

        for method in completed_methods:
            value = results.get((method, backbone, dataset, alpha))
            if value is None:
                row.append(r"\texttt{--}")
            else:
                displayed_values.append(value)
                row.append(format_num(value))

        if len(displayed_values) >= 2:
            row.append(format_num(max(displayed_values) - min(displayed_values)))
        else:
            row.append(r"\texttt{--}")
        lines.append(" & ".join(row) + r" \\")
        if idx in {2, 5, 8, 11, 14, 17}:
            lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")
    lines.append(r"\end{document}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    results = parse_completed_results(args.logs, args.client_ratio)
    completed_methods = select_available_methods(results)
    content = render_table(completed_methods, results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(f"Wrote {args.output}")
    print("Included baselines:", ", ".join(completed_methods) if completed_methods else "(none)")


if __name__ == "__main__":
    main()
