#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一批量训练脚本：
- 方法: FedAvg / FedProx / HeteroFL / ScaleFL / Decoupled / FlexFL / FedRolex
- Backbone: vgg / resnet / vit
- 数据集: cifar10 / cifar100 / TinyImagenet
- VGG/ResNet 会自动使用对齐后的四档 profile，无需手动传缩放率
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_METHODS = ["FedAvg", "FedProx", "HeteroFL", "ScaleFL", "Decoupled", "FlexFL", "FedRolex"]
DEFAULT_BACKBONES = ["vgg", "resnet", "vit"]
DEFAULT_DATASETS = ["cifar10", "cifar100", "TinyImagenet"]
CONFIGS = {
    "iid": {"iid": 1, "data_beta": None},
    "noniid_beta1": {"iid": 0, "data_beta": 1},
    "noniid_beta100": {"iid": 0, "data_beta": 100},
}
DATASET_INFO = {
    "cifar10": {"num_channels": 3, "num_classes": 10},
    "cifar100": {"num_channels": 3, "num_classes": 100},
    "TinyImagenet": {"num_channels": 3, "num_classes": 200},
}


def ratio_to_weights(ratio):
    values = [float(item) for item in ratio.split(":")]
    total = sum(values)
    if total <= 0:
        raise ValueError(f"Invalid client ratio: {ratio}")
    return [item / total for item in values]


def parse_client_ratio(text, fallback):
    matches = re.findall(r"--client_hetero_ration\s+([^\s]+)", text)
    return matches[-1] if matches else fallback


def parse_final_accuracy(method, log_path, fallback_ratio):
    text = Path(log_path).read_text(errors="ignore")
    accs = [float(item) for item in re.findall(r"Testing accuracy:\s*([0-9.]+)", text)]
    if not accs:
        return None
    if method.lower() != "decoupled":
        return accs[-1]

    weights = ratio_to_weights(parse_client_ratio(text, fallback_ratio))
    if len(accs) < len(weights):
        return None
    final_level_accs = accs[-len(weights):]
    return sum(acc * weight for acc, weight in zip(final_level_accs, weights))
FLEXFL_APOZ = {
    ("vgg", "cifar10"): 9,
    ("resnet", "cifar10"): 8,
    ("vit", "cifar10"): 0,
    ("vgg", "cifar100"): 12,
    ("resnet", "cifar100"): 11,
    ("vit", "cifar100"): 0,
    ("vgg", "TinyImagenet"): 13,
    ("resnet", "TinyImagenet"): 14,
    ("vit", "TinyImagenet"): 0,
}


@dataclass
class Experiment:
    method: str
    backbone: str
    dataset: str
    config_name: str
    generate_data: int


def parse_args():
    parser = argparse.ArgumentParser(description="Run all VGG/ResNet experiments from one script.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=DEFAULT_METHODS)
    parser.add_argument("--backbones", nargs="+", default=DEFAULT_BACKBONES, choices=DEFAULT_BACKBONES)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, choices=DEFAULT_DATASETS)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()), choices=list(CONFIGS.keys()))
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="outputs_batch_all")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--num-users", type=int, default=100)
    parser.add_argument("--frac", type=float, default=0.1)
    parser.add_argument("--local-ep", type=int, default=5)
    parser.add_argument("--local-bs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr-decay", type=float, default=0.998)
    parser.add_argument("--momentum", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam", "adamw", "adaBelief"])
    parser.add_argument("--vit-lr", type=float, default=5e-4)
    parser.add_argument("--vit-weight-decay", type=float, default=0.05)
    parser.add_argument("--vit-image-size", type=int, default=224)
    parser.add_argument("--vit-cifar-local-bs", type=int, default=16)
    parser.add_argument("--vit-tiny-local-bs", type=int, default=50)
    parser.add_argument("--prox-alpha", type=float, default=0.01)
    parser.add_argument("--client-ratio", type=str, default="4:3:2:1")
    parser.add_argument("--scalefl-widths", nargs="+", type=float, default=[0.5, 0.63, 0.794, 1.0])
    parser.add_argument("--scalefl-gamma", type=float, default=0.1)
    parser.add_argument("--flexfl-pretrain", type=int, default=0)
    parser.add_argument("--flexfl-gamma", type=float, default=10.0)
    parser.add_argument("--only", type=int, default=1)
    parser.add_argument("--kd-temperature", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu-threads-per-job", type=int, default=4)
    parser.add_argument("--prepare-data", action="store_true", default=True)
    parser.add_argument("--no-prepare-data", dest="prepare_data", action="store_false")
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip experiments that already have success=True under --output-dir")
    parser.add_argument("--existing-summary-globs", nargs="*", default=None,
                        help="optional summary file globs used by --skip-existing; defaults to --output-dir only")
    parser.add_argument("--no-save-checkpoints", dest="save_checkpoints", action="store_false",
                        help="disable final checkpoint saving for launched jobs")
    parser.set_defaults(save_checkpoints=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def format_beta_suffix(beta):
    beta_float = float(beta)
    if beta_float.is_integer():
        return str(int(beta_float))
    return str(beta_float)


def partition_file(dataset, num_users, config_name):
    cfg = CONFIGS[config_name]
    suffix = "iid" if cfg["iid"] else f"noniid_beta{format_beta_suffix(cfg['data_beta'])}"
    return REPO_ROOT / "data" / f"{dataset}_{num_users}_{suffix}.json"


def prepare_partitions(args):
    if not args.prepare_data:
        missing = [str(partition_file(dataset, args.num_users, config_name))
                   for dataset in args.datasets
                   for config_name in args.configs
                   if not partition_file(dataset, args.num_users, config_name).exists()]
        if missing:
            raise FileNotFoundError("Missing partition files:\n" + "\n".join(missing))
        return

    sys.path.insert(0, str(REPO_ROOT))
    from utils.get_dataset import get_dataset

    for dataset in args.datasets:
        ds_info = DATASET_INFO[dataset]
        for config_name in args.configs:
            target = partition_file(dataset, args.num_users, config_name)
            if target.exists():
                continue
            cfg = CONFIGS[config_name]
            print(f"[prepare] {dataset} {config_name}")
            prepare_args = SimpleNamespace(
                dataset=dataset,
                num_users=args.num_users,
                iid=cfg["iid"],
                data_beta=cfg["data_beta"] if cfg["data_beta"] is not None else 100,
                generate_data=1,
                seed=args.seed,
                num_channels=ds_info["num_channels"],
                num_classes=ds_info["num_classes"],
            )
            get_dataset(prepare_args)


def build_experiments(args):
    experiments = []
    for method in args.methods:
        for backbone in args.backbones:
            for dataset in args.datasets:
                for config_name in args.configs:
                    experiments.append(
                        Experiment(
                            method=method,
                            backbone=backbone,
                            dataset=dataset,
                            config_name=config_name,
                            generate_data=0,
                        )
                    )
    return experiments


def actual_model_name(method, backbone):
    if backbone == "resnet":
        return "resnet_smart"
    return backbone


def experiment_name(exp):
    return f"{exp.method.lower()}_{exp.backbone}_{exp.dataset.lower()}_{exp.config_name}"


def load_existing_successes(args):
    if not args.skip_existing:
        return set()

    patterns = args.existing_summary_globs
    if patterns is None:
        patterns = [
            f"{args.output_dir}/**/summary.tsv",
            f"{args.output_dir}/**/summary.json",
        ]

    successes = set()
    for pattern in patterns:
        for path in REPO_ROOT.glob(pattern):
            if path.suffix == ".tsv":
                with open(path, newline="", encoding="utf-8") as handle:
                    for row in csv.DictReader(handle, delimiter="\t"):
                        if row.get("success") == "True":
                            successes.add(row.get("name", ""))
            elif path.suffix == ".json":
                try:
                    rows = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                for row in rows:
                    if row.get("success") is True:
                        successes.add(row.get("name", ""))
    successes.discard("")
    return successes


def build_command(exp, args, checkpoint_dir=None):
    cfg = CONFIGS[exp.config_name]
    ds_info = DATASET_INFO[exp.dataset]
    is_vit = exp.backbone == "vit"
    local_bs = args.local_bs
    lr = args.lr
    lr_decay = args.lr_decay
    weight_decay = args.weight_decay
    optimizer = args.optimizer
    if is_vit:
        local_bs = args.vit_tiny_local_bs if exp.dataset == "TinyImagenet" else args.vit_cifar_local_bs
        lr = args.vit_lr
        lr_decay = 1.0
        weight_decay = args.vit_weight_decay
        optimizer = "adamw"

    cmd = [
        sys.executable,
        "main_fed.py",
        "--gpu", str(args.gpu),
        "--algorithm", exp.method,
        "--model", actual_model_name(exp.method, exp.backbone),
        "--dataset", exp.dataset,
        "--num_channels", str(ds_info["num_channels"]),
        "--num_classes", str(ds_info["num_classes"]),
        "--num_users", str(args.num_users),
        "--frac", str(args.frac),
        "--iid", str(cfg["iid"]),
        "--generate_data", str(exp.generate_data),
        "--epochs", str(args.epochs),
        "--local_ep", str(args.local_ep),
        "--local_bs", str(local_bs),
        "--bs", str(local_bs if is_vit else 128),
        "--lr", str(lr),
        "--lr_decay", str(lr_decay),
        "--momentum", str(args.momentum),
        "--weight_decay", str(weight_decay),
        "--optimizer", optimizer,
        "--T", str(args.kd_temperature),
        "--seed", str(args.seed),
    ]
    if checkpoint_dir is not None:
        cmd.extend(["--save_checkpoint_dir", str(checkpoint_dir)])
    if is_vit:
        cmd.extend(["--image_size", str(args.vit_image_size)])
    if cfg["data_beta"] is not None:
        cmd.extend(["--data_beta", str(cfg["data_beta"])])

    if exp.method == "FedProx":
        cmd.extend(["--prox_alpha", str(args.prox_alpha)])
    elif exp.method in {"HeteroFL", "Decoupled", "FedRolex"}:
        cmd.extend([
            "--client_hetero_ration", args.client_ratio,
            "--client_chosen_mode", "available",
        ])
    elif exp.method == "ScaleFL":
        cmd.extend([
            "--client_hetero_ration", args.client_ratio,
            "--client_chosen_mode", "available",
            "--gamma", str(args.scalefl_gamma),
        ])
    elif exp.method == "FlexFL":
        cmd.extend([
            "--client_hetero_ration", args.client_ratio,
            "--client_chosen_mode", "available",
            "--pretrain", str(args.flexfl_pretrain),
            "--gamma", str(args.flexfl_gamma),
            "--only", str(args.only),
        ])
        if args.flexfl_pretrain == 0:
            cmd.extend(["--apoz", str(FLEXFL_APOZ[(exp.backbone, exp.dataset)])])
    return cmd


def should_skip_experiment(exp, args, existing_successes=None):
    _ = exp, args
    if existing_successes is not None and experiment_name(exp) in existing_successes:
        return "existing successful result"
    return None


def build_env(args):
    env = os.environ.copy()
    thread_count = str(max(1, int(args.cpu_threads_per_job)))
    env["OMP_NUM_THREADS"] = thread_count
    env["MKL_NUM_THREADS"] = thread_count
    env["OPENBLAS_NUM_THREADS"] = thread_count
    env["NUMEXPR_NUM_THREADS"] = thread_count
    return env


def effective_training_hparams(exp, args):
    is_vit = exp.backbone == "vit"
    local_bs = args.local_bs
    test_bs = 128
    lr = args.lr
    lr_decay = args.lr_decay
    weight_decay = args.weight_decay
    optimizer = args.optimizer
    if is_vit:
        local_bs = args.vit_tiny_local_bs if exp.dataset == "TinyImagenet" else args.vit_cifar_local_bs
        test_bs = local_bs
        lr = args.vit_lr
        lr_decay = 1.0
        weight_decay = args.vit_weight_decay
        optimizer = "adamw"
    client_ratio = args.client_ratio if exp.method in {"HeteroFL", "ScaleFL", "Decoupled", "FlexFL", "FedRolex"} else "N/A"
    if exp.method == "ScaleFL":
        kd = f"gamma={args.scalefl_gamma}, T={args.kd_temperature}, active_round>{args.epochs * 0.25:g}"
    elif exp.method == "FlexFL":
        kd = f"method_internal, gamma={args.flexfl_gamma}, T={args.kd_temperature}"
    else:
        kd = "disabled"
    return {
        "epochs": args.epochs,
        "num_users": args.num_users,
        "frac": args.frac,
        "local_ep": args.local_ep,
        "local_bs": local_bs,
        "bs": test_bs,
        "optimizer": optimizer,
        "lr": lr,
        "lr_decay": lr_decay,
        "weight_decay": weight_decay,
        "seed": args.seed,
        "client_ratio": client_ratio,
        "kd": kd,
    }


def write_header(handle, exp, cmd, args):
    handle.write(f"experiment: {experiment_name(exp)}\n")
    handle.write(f"method: {exp.method}\n")
    handle.write(f"backbone: {exp.backbone}\n")
    handle.write(f"dataset: {exp.dataset}\n")
    handle.write(f"config: {exp.config_name}\n")
    for key, value in effective_training_hparams(exp, args).items():
        handle.write(f"{key}: {value}\n")
    handle.write(f"command: {' '.join(cmd)}\n")
    handle.write("=" * 80 + "\n")
    handle.flush()


def dump_summary(path, results):
    serializable = []
    for result in results:
        serializable.append({
            "name": result["name"],
            "method": result["exp"].method,
            "backbone": result["exp"].backbone,
            "dataset": result["exp"].dataset,
            "config": result["exp"].config_name,
            "success": result["success"],
            "returncode": result["returncode"],
            "elapsed_minutes": round(result["elapsed"] / 60.0, 2),
            "final_acc": result["final_acc"],
            "metric": result["metric"],
            "log": result["log"],
            "checkpoint_dir": result.get("checkpoint_dir", ""),
            "skip_reason": result.get("skip_reason", ""),
        })
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(serializable, handle, ensure_ascii=False, indent=2)


def dump_summary_tsv(path, results):
    lines = ["name\tmethod\tbackbone\tdataset\tconfig\tsuccess\treturncode\telapsed_minutes\tfinal_acc\tmetric\tlog\tcheckpoint_dir\tskip_reason"]
    for result in results:
        final_acc = "" if result["final_acc"] is None else f"{result['final_acc']:.4f}"
        lines.append(
            "\t".join([
                result["name"],
                result["exp"].method,
                result["exp"].backbone,
                result["exp"].dataset,
                result["exp"].config_name,
                str(result["success"]),
                str(result["returncode"]),
                f"{result['elapsed'] / 60.0:.2f}",
                final_acc,
                result["metric"],
                result["log"],
                result.get("checkpoint_dir", ""),
                result.get("skip_reason", ""),
            ])
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def run_experiments(experiments, args, run_dir, existing_successes=None):
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    env = build_env(args)
    pending = list(experiments)
    running = []
    results = []
    total = len(experiments)
    launched = 0
    completed = 0

    while pending or running:
        while pending and len(running) < args.max_parallel:
            exp = pending.pop(0)
            name = experiment_name(exp)
            log_path = logs_dir / f"{name}.log"
            skip_reason = should_skip_experiment(exp, args, existing_successes)
            if skip_reason is not None:
                completed += 1
                print(f"[skip {completed}/{total}] {name} {skip_reason}")
                results.append({
                    "exp": exp,
                    "name": name,
                    "success": False,
                    "returncode": "SKIP",
                    "elapsed": 0.0,
                    "final_acc": None,
                    "metric": "skipped",
                    "log": str(log_path),
                    "checkpoint_dir": "",
                    "skip_reason": skip_reason,
                })
                with open(log_path, "w", encoding="utf-8") as handle:
                    handle.write(f"experiment: {name}\n")
                    handle.write(f"skipped: {skip_reason}\n")
                continue

            checkpoint_dir = (run_dir / "checkpoints" / name) if args.save_checkpoints else None
            cmd = build_command(exp, args, checkpoint_dir)
            handle = open(log_path, "w", encoding="utf-8")
            write_header(handle, exp, cmd, args)
            process = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            launched += 1
            print(f"[launch {launched}/{total}] {name}")
            running.append({
                "exp": exp,
                "name": name,
                "process": process,
                "handle": handle,
                "start": time.time(),
                "log": str(log_path),
                "checkpoint_dir": "" if checkpoint_dir is None else str(checkpoint_dir),
            })

        time.sleep(5 if running else 0)
        still_running = []
        for item in running:
            returncode = item["process"].poll()
            if returncode is None:
                still_running.append(item)
                continue

            item["handle"].write("\n" + "=" * 80 + "\n")
            item["handle"].write(f"returncode: {returncode}\n")
            item["handle"].flush()
            item["handle"].close()
            elapsed = time.time() - item["start"]
            completed += 1
            success = (returncode == 0)
            final_acc = parse_final_accuracy(item["exp"].method, item["log"], args.client_ratio) if success else None
            metric = "client_ratio_weighted_level_acc" if item["exp"].method == "Decoupled" else "final_testing_accuracy"
            print(f"[done {completed}/{total}] {item['name']} {'OK' if success else 'FAIL'} {elapsed / 60.0:.1f} min")
            results.append({
                "exp": item["exp"],
                "name": item["name"],
                "success": success,
                "returncode": returncode,
                "elapsed": elapsed,
                "final_acc": final_acc,
                "metric": metric,
                "log": item["log"],
                "checkpoint_dir": item["checkpoint_dir"],
            })
        running = still_running

    return results


def main():
    args = parse_args()
    os.chdir(REPO_ROOT)

    if args.prepare_data:
        prepare_partitions(args)

    experiments = build_experiments(args)
    existing_successes = load_existing_successes(args)
    if existing_successes:
        print(f"existing successful experiments loaded: {len(existing_successes)}")

    if args.dry_run:
        for exp in experiments:
            skip_reason = should_skip_experiment(exp, args, existing_successes)
            if skip_reason is not None:
                print(f"# {experiment_name(exp)} SKIP: {skip_reason}")
                continue
            checkpoint_dir = Path("<run_dir>") / "checkpoints" / experiment_name(exp) if args.save_checkpoints else None
            print(" ".join(build_command(exp, args, checkpoint_dir)))
        return

    run_dir = REPO_ROOT / args.output_dir / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"total experiments: {len(experiments)}")
    print(f"output dir: {run_dir}")
    results = run_experiments(experiments, args, run_dir, existing_successes)
    results.sort(key=lambda item: item["name"])

    summary_json = run_dir / "summary.json"
    summary_tsv = run_dir / "summary.tsv"
    dump_summary(summary_json, results)
    dump_summary_tsv(summary_tsv, results)

    success_count = sum(1 for result in results if result["success"])
    print(f"success: {success_count}/{len(results)}")
    print(f"summary: {summary_tsv}")


if __name__ == "__main__":
    main()
