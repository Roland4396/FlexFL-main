#!/usr/bin/env python3
"""Extract final accuracies from batch logs with Decoupled handled correctly."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


DEFAULT_CLIENT_RATIO = "4:3:2:1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract final accuracies from batch log files.")
    parser.add_argument("logs", type=Path, help="Directory containing *.log files.")
    parser.add_argument(
        "--client-ratio",
        default=DEFAULT_CLIENT_RATIO,
        help="Fallback Decoupled client ratio when it cannot be parsed from a log.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional TSV output path.")
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


def parse_final_accuracy(method: str, text: str, fallback_ratio: str) -> tuple[float | None, str]:
    accs = [float(item) for item in re.findall(r"Testing accuracy:\s*([0-9.]+)", text)]
    if not accs:
        return None, "missing"
    if method != "decoupled":
        return accs[-1], "final_testing_accuracy"

    ratio = parse_client_ratio(text, fallback_ratio)
    weights = ratio_to_weights(ratio)
    if len(accs) < len(weights):
        return None, "missing_decoupled_levels"
    final_level_accs = accs[-len(weights):]
    return sum(acc * weight for acc, weight in zip(final_level_accs, weights)), f"weighted_levels_{ratio}"


def parse_log_name(path: Path) -> tuple[str, str, str, str] | None:
    match = re.match(r"([^_]+)_([^_]+)_([^_]+)_(.+)\.log$", path.name)
    if not match:
        return None
    return match.groups()


def main() -> None:
    args = parse_args()
    rows = ["name\tmethod\tbackbone\tdataset\tconfig\tdone\tfinal_acc\tmetric\tlog"]
    for log_path in sorted(args.logs.glob("*.log")):
        parsed = parse_log_name(log_path)
        if parsed is None:
            continue
        method, backbone, dataset, config = parsed
        text = log_path.read_text(errors="ignore")
        done = "returncode: 0" in text or "save finished" in text
        final_acc, metric = parse_final_accuracy(method, text, args.client_ratio)
        acc_text = "" if final_acc is None else f"{final_acc:.4f}"
        rows.append(
            "\t".join([
                log_path.stem,
                method,
                backbone,
                dataset,
                config,
                str(done),
                acc_text,
                metric,
                str(log_path),
            ])
        )

    content = "\n".join(rows) + "\n"
    if args.output is None:
        print(content, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
