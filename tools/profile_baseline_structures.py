#!/usr/bin/env python
"""Profile active params and FLOPs for aligned baseline structures.

The script builds the exact model profiles used by the current training code
and runs one dummy forward pass per method/backbone/dataset/level. FLOPs are
reported as MAC-style Conv/Linear/Attention operations for batch size 1.
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import nn

from arch_profiles import (
    build_flexfl_scale_list,
    build_smallest_width_model,
    build_vit_mlp_only_scale_list,
    build_width_only_scale_list,
    get_scalefl_profile,
)
from getAPOZ import getNet
from models.resnet_smart_scaleFL import ResNet110_cifar_scaleFL
from models.vgg_scaleFL import vgg_16_scaleFL
from models.vit_flexfl import (
    VIT_DEPTH,
    vit_rate_uses_hidden_width,
    vit_small_scalefl,
)


METHODS = ("FedAvg", "FedProx", "HeteroFL", "ScaleFL", "Decoupled", "FlexFL")
BACKBONES = ("resnet_smart", "vgg", "vit")
DATASETS = ("cifar10", "cifar100", "TinyImagenet")
DATASET_INFO = {
    "cifar10": {"num_classes": 10, "num_channels": 3},
    "cifar100": {"num_classes": 100, "num_channels": 3},
    "TinyImagenet": {"num_classes": 200, "num_channels": 3},
}
MODEL_LABEL = {
    "resnet_smart": "ResNet110",
    "vgg": "VGG16",
    "vit": "ViT-Small",
}
ORIGINAL_VGG_CFG_D = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    parser.add_argument("--backbones", nargs="+", default=list(BACKBONES), choices=BACKBONES)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--output", default="paper_tables/baseline_structure_flops.csv")
    parser.add_argument("--markdown-output", default="paper_tables/baseline_structure_flops.md")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--vit-image-size", type=int, default=224)
    return parser.parse_args()


def reset_vgg_cfgs():
    # models.vgg.make_layers mutates cfg['D']; reset before every VGG build.
    import models.vgg as vgg_mod
    import models.vgg_scaleFL as vgg_scalefl_mod

    vgg_mod.cfg["D"] = list(ORIGINAL_VGG_CFG_D)
    vgg_scalefl_mod.cfg["D"] = list(ORIGINAL_VGG_CFG_D)


def make_profile_args(method, backbone, dataset, device, vit_image_size):
    info = DATASET_INFO[dataset]
    return SimpleNamespace(
        algorithm=method,
        model=backbone,
        dataset=dataset,
        num_classes=info["num_classes"],
        num_channels=info["num_channels"],
        image_size=vit_image_size if backbone == "vit" else (64 if dataset == "TinyImagenet" else 32),
        device=device,
        client_hetero_ration="4:3:2:1",
        client_chosen_mode="available",
        width_ration=[0.5, 0.63, 0.794, 1.0],
        depth_saved=[4, 6, 8],
        pretrain=0,
        gamma=10.0,
        apoz=0,
        only=1,
        log=0,
        e=0,
    )


def rate_to_text(rate):
    if rate is None:
        return ""
    if isinstance(rate, torch.Tensor):
        values = rate.detach().cpu().tolist()
    elif isinstance(rate, (list, tuple)):
        values = list(rate)
    else:
        return str(rate)
    return "[" + ",".join(f"{float(value):.6g}" for value in values) + "]"


def rate_exit(rate, default="full"):
    if isinstance(rate, torch.Tensor):
        values = rate.detach().cpu().tolist()
    elif isinstance(rate, (list, tuple)):
        values = list(rate)
    else:
        return default
    if len(values) >= 5 and abs(float(values[4]) - 1.0) > 1e-9:
        exit_loc = int(round(float(values[4])))
        return "full" if exit_loc >= VIT_DEPTH else str(exit_loc)
    return default


def rate_width_mode(rate, backbone, method):
    if backbone != "vit":
        return "conv_width"
    return "hidden_qkv" if vit_rate_uses_hidden_width(rate) else "mlp_only"


def build_standard_model(args, rate):
    reset_vgg_cfgs()
    return getNet(args, rate).to(args.device).eval()


def build_scale_model(args, scale, exits):
    reset_vgg_cfgs()
    if args.model == "vgg":
        return vgg_16_scaleFL(
            num_classes=args.num_classes,
            track_running_stats=False,
            num_channels=args.num_channels,
            scale=scale,
            exit0=exits[0],
            exit1=exits[1],
            exit2=exits[2],
        ).to(args.device).eval()
    if args.model == "resnet_smart":
        return ResNet110_cifar_scaleFL(
            num_channels=args.num_channels,
            num_classes=args.num_classes,
            track_running_stats=False,
            scale=scale,
            exit0=exits[0],
            exit1=exits[1],
            exit2=exits[2],
        ).to(args.device).eval()
    if args.model == "vit":
        return vit_small_scalefl(
            num_classes=args.num_classes,
            num_channels=args.num_channels,
            image_size=args.image_size,
            scale=scale,
            exits=exits,
        ).to(args.device).eval()
    raise ValueError(f"Unsupported ScaleFL backbone: {args.model}")


def scalefl_selected_forward(model, backbone, level, x):
    """Forward only the selected ScaleFL level path.

    Training uses auxiliary earlier exits for distillation, but the model budget
    table should count the deployable level-k path: prefix backbone + one head.
    """
    if backbone == "vgg":
        if level == 1:
            features = model.features[:model.exitpos0](x)
            return model.classifier[0](features)
        if level == 2:
            features = model.features[:model.exitpos1](x)
            return model.classifier[1](features)
        if level == 3:
            features = model.features[:model.exitpos2](x)
            return model.classifier[2](features)
        features = model.features(x)
        return model.classifier[3](features)

    if backbone == "resnet_smart":
        x = model.conv1(x)
        if level == 1:
            limit = model.exit0
        elif level == 2:
            limit = model.exit1
        elif level == 3:
            limit = model.exit2
        else:
            limit = len(model.blocks)
        for idx in range(limit):
            x = model.blocks[idx](x)
        return model.classifiers[level - 1](x)

    if backbone == "vit":
        tokens = model._process_input(x)
        if level < 4:
            exit_loc = model.exit_locations[level - 1]
            for block in model.blocks[:exit_loc]:
                tokens = block(tokens)
            return model.classifiers[level - 1](tokens)
        for block in model.blocks:
            tokens = block(tokens)
        return model._forward_final(tokens)

    raise ValueError(f"Unsupported ScaleFL backbone: {backbone}")


def build_level_specs(args, method):
    if method in {"FedAvg", "FedProx"}:
        reset_vgg_cfgs()
        model = build_smallest_width_model(args).to(args.device).eval()
        return [
            {
                "level": level,
                "model": model,
                "rate": None,
                "scale": None,
                "exit": rate_exit(None),
                "forward_kwargs": {},
                "note": "single_smallest_model",
            }
            for level in range(1, 5)
        ]

    if method == "HeteroFL":
        rates = build_width_only_scale_list(args)
        return [
            {
                "level": idx,
                "model": build_standard_model(args, rate),
                "rate": rate,
                "scale": None,
                "exit": rate_exit(rate),
                "forward_kwargs": {},
                "note": "width_only",
            }
            for idx, rate in enumerate(rates, start=1)
        ]

    if method == "Decoupled":
        rates = build_vit_mlp_only_scale_list(args) if args.model == "vit" else build_width_only_scale_list(args)
        return [
            {
                "level": idx,
                "model": build_standard_model(args, rate),
                "rate": rate,
                "scale": None,
                "exit": rate_exit(rate),
                "forward_kwargs": {},
                "note": "independent_level_model",
            }
            for idx, rate in enumerate(rates, start=1)
        ]

    if method == "FlexFL":
        _scale_rate, rates = build_flexfl_scale_list(args, None)
        return [
            {
                "level": idx,
                "model": build_standard_model(args, rate),
                "rate": rate,
                "scale": None,
                "exit": rate_exit(rate),
                "forward_kwargs": {},
                "note": "apoz_profile",
            }
            for idx, rate in enumerate(rates, start=1)
        ]

    if method == "ScaleFL":
        profile = get_scalefl_profile(args)
        exits = tuple(profile["exits"])
        specs = []
        for idx, scale in enumerate(profile["scales"], start=1):
            specs.append(
                {
                    "level": idx,
                    "model": build_scale_model(args, float(scale), exits),
                    "rate": None,
                    "scale": float(scale),
                    "exit": str(exits[idx - 1]) if idx < 4 else "full",
                    "forward_kwargs": {"ee": idx},
                    "forward_fn": scalefl_selected_forward,
                    "note": f"width_depth_exits={exits}; selected_head_budget",
                }
            )
        return specs

    raise ValueError(f"Unsupported method: {method}")


def _add_active_params(module, seen_param_ids, counter, recurse=False):
    for param in module.parameters(recurse=recurse):
        param_id = id(param)
        if param_id not in seen_param_ids:
            seen_param_ids.add(param_id)
            counter[0] += param.numel()


def profile_forward(model, dummy, forward_kwargs=None, forward_fn=None, backbone=None, level=None):
    forward_kwargs = forward_kwargs or {}
    flops = [0]
    active_params = [0]
    seen_param_ids = set()

    _add_active_params(model, seen_param_ids, active_params, recurse=False)

    mha_child_ids = set()
    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention):
            for child in module.modules():
                if child is not module:
                    mha_child_ids.add(id(child))

    hooks = []

    def conv_hook(module, inputs, output):
        _add_active_params(module, seen_param_ids, active_params, recurse=False)
        if not torch.is_tensor(output):
            return
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
        flops[0] += int(output.numel()) * int(kernel_ops)

    def linear_hook(module, inputs, output):
        _add_active_params(module, seen_param_ids, active_params, recurse=False)
        if not torch.is_tensor(output):
            return
        output_positions = int(output.numel()) // int(module.out_features)
        flops[0] += output_positions * int(module.in_features) * int(module.out_features)

    def mha_hook(module, inputs, output):
        _add_active_params(module, seen_param_ids, active_params, recurse=True)
        query = inputs[0]
        if not torch.is_tensor(query):
            return
        if getattr(module, "batch_first", False):
            batch, q_len, embed_dim = query.shape
        else:
            q_len, batch, embed_dim = query.shape
        head_dim = embed_dim // module.num_heads
        qkv = 3 * batch * q_len * embed_dim * embed_dim
        attn = 2 * batch * module.num_heads * q_len * q_len * head_dim
        proj = batch * q_len * embed_dim * embed_dim
        flops[0] += int(qkv + attn + proj)

    def active_only_hook(module, inputs, output):
        _add_active_params(module, seen_param_ids, active_params, recurse=False)

    for module in model.modules():
        if module is model or id(module) in mha_child_ids:
            continue
        if isinstance(module, nn.MultiheadAttention):
            hooks.append(module.register_forward_hook(mha_hook))
        elif isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
        elif len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(active_only_hook))

    with torch.inference_mode():
        if forward_fn is None:
            _ = model(dummy, **forward_kwargs)
        else:
            _ = forward_fn(model, backbone, level, dummy)

    for hook in hooks:
        hook.remove()

    return active_params[0], flops[0]


def build_full_reference(args):
    if args.model == "vit":
        rate = torch.tensor([1.0, 1.0, 1.0, 1.0, float(VIT_DEPTH)], dtype=torch.float32)
    elif args.model == "vgg":
        rate = [1.0] * 15
    else:
        rate = [1.0] * 5
    model = build_standard_model(args, rate)
    dummy = torch.zeros(1, args.num_channels, args.image_size, args.image_size, device=args.device)
    active_params, flops = profile_forward(model, dummy)
    total_params = sum(param.numel() for param in model.parameters())
    return active_params, total_params, flops


def write_markdown(rows, path):
    columns = [
        "method",
        "backbone",
        "dataset",
        "level",
        "active_params_m",
        "total_params_m",
        "flops_m",
        "active_param_ratio",
        "flop_ratio",
        "width_mode",
        "exit",
    ]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    cli_args = parse_args()
    torch.set_num_threads(max(1, int(cli_args.threads)))
    device = torch.device("cuda:0" if cli_args.device == "cuda" and torch.cuda.is_available() else "cpu")

    rows = []
    for backbone in cli_args.backbones:
        for dataset in cli_args.datasets:
            ref_args = make_profile_args("FedAvg", backbone, dataset, device, cli_args.vit_image_size)
            ref_active_params, ref_total_params, ref_flops = build_full_reference(ref_args)
            for method in cli_args.methods:
                args = make_profile_args(method, backbone, dataset, device, cli_args.vit_image_size)
                specs = build_level_specs(args, method)
                for spec in specs:
                    dummy = torch.zeros(1, args.num_channels, args.image_size, args.image_size, device=device)
                    active_params, flops = profile_forward(
                        spec["model"],
                        dummy,
                        spec["forward_kwargs"],
                        spec.get("forward_fn"),
                        backbone,
                        spec["level"],
                    )
                    total_params = sum(param.numel() for param in spec["model"].parameters())
                    row = {
                        "method": method,
                        "backbone": MODEL_LABEL[backbone],
                        "dataset": dataset,
                        "level": spec["level"],
                        "active_params_m": f"{active_params / 1e6:.3f}",
                        "total_params_m": f"{total_params / 1e6:.3f}",
                        "flops_m": f"{flops / 1e6:.3f}",
                        "active_param_ratio": f"{active_params / ref_active_params:.4f}",
                        "total_param_ratio": f"{total_params / ref_total_params:.4f}",
                        "flop_ratio": f"{flops / ref_flops:.4f}",
                        "input_size": args.image_size,
                        "width_mode": rate_width_mode(spec["rate"], backbone, method),
                        "exit": spec["exit"],
                        "scale": "" if spec["scale"] is None else f"{spec['scale']:.6g}",
                        "rate": rate_to_text(spec["rate"]),
                        "note": spec["note"],
                    }
                    rows.append(row)
                    print(
                        f"{method:9s} {MODEL_LABEL[backbone]:10s} {dataset:13s} "
                        f"L{spec['level']} params={row['active_params_m']}M "
                        f"flops={row['flops_m']}M"
                    )

    output = Path(cli_args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    markdown_output = Path(cli_args.markdown_output)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(rows, markdown_output)
    print(f"\nWrote {len(rows)} rows to {output}")
    print(f"Wrote markdown table to {markdown_output}")


if __name__ == "__main__":
    main()
