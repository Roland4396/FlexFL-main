#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
计算VGG16、ResNet110、MobileNetV2的参数量分布
用于确定最佳的早退点位置，使得参数量比例接近 1/8, 1/4, 1/2, 1
"""

import torch
import torch.nn as nn
import numpy as np


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters())


def analyze_vgg16():
    """分析VGG16的参数量分布"""
    print("="*80)
    print("VGG16 参数量分布分析")
    print("="*80)

    # VGG16的配置（D配置）
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]

    # 计算每层的参数量
    in_channels = 3
    conv_layers = []
    layer_params = []
    layer_idx = 0

    for v in cfg:
        if v == 'M':
            continue
        else:
            # Conv2d参数量 = in_channels * out_channels * kernel_size^2
            params = in_channels * v * 3 * 3
            conv_layers.append((layer_idx, v, params))
            layer_params.append(params)
            in_channels = v
            layer_idx += 1

    # 累积参数量
    total_conv_params = sum(layer_params)
    cumsum_params = np.cumsum(layer_params)
    cumsum_ratio = cumsum_params / total_conv_params

    print("\n层数\t通道数\t参数量\t累积比例")
    print("-"*50)
    for i, (idx, channels, params) in enumerate(conv_layers):
        print(f"{i+1}\t{channels}\t{params:,}\t{cumsum_ratio[i]:.2%}")

    # 找最接近目标比例的层
    target_ratios = [0.125, 0.25, 0.5, 1.0]  # 1/8, 1/4, 1/2, 1
    exit_points = []

    print("\n最佳退出点（目标：1/8, 1/4, 1/2, 1）：")
    for target in target_ratios[:-1]:  # 最后一个是完整模型
        diff = np.abs(cumsum_ratio - target)
        best_layer = np.argmin(diff) + 1
        actual_ratio = cumsum_ratio[best_layer - 1]
        exit_points.append((best_layer, actual_ratio))
        print(f"  目标{target:.2%} -> 第{best_layer}层退出，实际{actual_ratio:.2%}")

    return exit_points


def analyze_resnet110():
    """分析ResNet110的参数量分布"""
    print("\n" + "="*80)
    print("ResNet110 (SmartFL) 参数量分布分析")
    print("="*80)

    # ResNet110配置：3 stages，每个18 blocks
    # 通道数：16, 32, 64
    blocks_per_stage = [18, 18, 18]
    channels = [16, 32, 64]

    # 计算参数量
    # 初始conv层
    initial_conv_params = 3 * 16 * 3 * 3  # 3通道输入，16输出

    # 每个block有2个3x3卷积
    stage_params = []
    for i, (blocks, ch) in enumerate(zip(blocks_per_stage, channels)):
        if i == 0:
            # 第一个stage
            params_per_block = ch * ch * 3 * 3 * 2  # 2个3x3卷积
        else:
            # 后续stage，第一个block有下采样
            first_block_params = channels[i-1] * ch * 3 * 3 + ch * ch * 3 * 3
            first_block_params += channels[i-1] * ch * 1 * 1  # shortcut 1x1卷积
            other_blocks_params = (blocks - 1) * ch * ch * 3 * 3 * 2
            params_per_block = (first_block_params + other_blocks_params) / blocks

        for b in range(blocks):
            stage_params.append(params_per_block)

    # 累积参数量
    total_params = initial_conv_params + sum(stage_params)
    block_cumsum = np.cumsum([initial_conv_params] + stage_params)
    cumsum_ratio = block_cumsum / total_params

    print("\nBlock\tStage\t累积比例")
    print("-"*30)
    print(f"0\tConv\t{cumsum_ratio[0]:.2%}")

    block_idx = 1
    for stage_idx, blocks in enumerate(blocks_per_stage):
        for b in range(blocks):
            if b % 6 == 0:  # 每6个block打印一次
                print(f"{block_idx}\tS{stage_idx+1}-B{b+1}\t{cumsum_ratio[block_idx]:.2%}")
            block_idx += 1

    # 找最接近目标比例的block
    target_ratios = [0.125, 0.25, 0.5, 1.0]
    exit_points = []

    print("\n最佳退出点（目标：1/8, 1/4, 1/2, 1）：")
    for target in target_ratios[:-1]:
        diff = np.abs(cumsum_ratio - target)
        best_block = np.argmin(diff)
        actual_ratio = cumsum_ratio[best_block]
        exit_points.append((best_block, actual_ratio))

        # 转换为stage和block位置
        if best_block == 0:
            print(f"  目标{target:.2%} -> 初始Conv后，实际{actual_ratio:.2%}")
        else:
            cum_blocks = 0
            for s_idx, blocks in enumerate(blocks_per_stage):
                if best_block <= cum_blocks + blocks:
                    block_in_stage = best_block - cum_blocks
                    print(f"  目标{target:.2%} -> Stage{s_idx+1}-Block{block_in_stage}后，实际{actual_ratio:.2%}")
                    break
                cum_blocks += blocks

    return exit_points


def analyze_mobilenet():
    """分析MobileNetV2的参数量分布"""
    print("\n" + "="*80)
    print("MobileNetV2 参数量分布分析")
    print("="*80)

    # MobileNetV2的inverted residual配置
    # t: expansion factor, c: output channels, n: repeat, s: stride
    cfgs = [
        # t, c, n, s
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]

    # 计算参数量
    block_params = []
    in_channels = 32  # 初始卷积输出

    # 初始卷积
    initial_params = 3 * 32 * 3 * 3
    block_params.append(initial_params)

    for t, c, n, s in cfgs:
        for i in range(n):
            stride = s if i == 0 else 1

            # Inverted residual block参数量估算
            # 1x1 expand + 3x3 depthwise + 1x1 project
            expand_channels = in_channels * t

            if t != 1:
                # Expansion 1x1
                params = in_channels * expand_channels * 1 * 1
                # Depthwise 3x3
                params += expand_channels * 3 * 3
                # Projection 1x1
                params += expand_channels * c * 1 * 1
            else:
                # 没有expansion，直接depthwise + projection
                params = in_channels * 3 * 3
                params += in_channels * c * 1 * 1

            block_params.append(params)
            in_channels = c

    # 累积参数量
    total_params = sum(block_params)
    cumsum_params = np.cumsum(block_params)
    cumsum_ratio = cumsum_params / total_params

    print("\nBlock\t累积比例")
    print("-"*30)
    for i in range(0, len(cumsum_ratio), 3):  # 每3个block打印一次
        print(f"{i}\t{cumsum_ratio[i]:.2%}")

    # 找最接近目标比例的block
    target_ratios = [0.125, 0.25, 0.5, 1.0]
    exit_points = []

    print("\n最佳退出点（目标：1/8, 1/4, 1/2, 1）：")
    for target in target_ratios[:-1]:
        diff = np.abs(cumsum_ratio - target)
        best_block = np.argmin(diff)
        actual_ratio = cumsum_ratio[best_block]
        exit_points.append((best_block, actual_ratio))
        print(f"  目标{target:.2%} -> Block {best_block}后退出，实际{actual_ratio:.2%}")

    return exit_points


def calculate_scale_values(exit_points, model_name):
    """根据退出点计算推荐的scale值"""
    print(f"\n{model_name} 推荐配置：")
    print("-"*50)
    print("等级\t退出点\t深度比\tScale值\t目标参数比")

    # 假设最后一层是完整模型
    if model_name == "VGG16":
        total_layers = 13
    elif model_name == "ResNet110":
        total_layers = 54  # 3*18 blocks
    else:  # MobileNet
        total_layers = 17  # 根据配置

    scales = []
    for i, (exit_layer, actual_ratio) in enumerate(exit_points):
        target_ratio = [0.125, 0.25, 0.5][i]
        depth_ratio = exit_layer / total_layers

        # 计算scale值使得 depth_ratio * scale^2 ≈ target_ratio
        scale = np.sqrt(target_ratio / actual_ratio)
        scales.append(scale)

        print(f"{i+1}\t{exit_layer}\t{depth_ratio:.2f}\t{scale:.3f}\t{target_ratio:.2%}")

    # 最后一个等级（完整模型）
    print(f"4\t{total_layers}\t1.00\t1.000\t100.00%")
    scales.append(1.0)

    return scales


def main():
    print("计算最佳退出点和Scale值配置")
    print("目标参数量比例：1/8 (12.5%), 1/4 (25%), 1/2 (50%), 1 (100%)")

    # VGG16分析
    vgg_exits = analyze_vgg16()
    vgg_scales = calculate_scale_values(vgg_exits, "VGG16")

    # ResNet110分析
    resnet_exits = analyze_resnet110()
    resnet_scales = calculate_scale_values(resnet_exits, "ResNet110")

    # MobileNetV2分析
    mobile_exits = analyze_mobilenet()
    mobile_scales = calculate_scale_values(mobile_exits, "MobileNetV2")

    # 总结
    print("\n" + "="*80)
    print("最终推荐配置")
    print("="*80)

    print("\nVGG16:")
    print(f"  退出点：exit0={vgg_exits[0][0]}, exit1={vgg_exits[1][0]}, exit2={vgg_exits[2][0]}")
    print(f"  Scale值：{[f'{s:.3f}' for s in vgg_scales]}")

    print("\nResNet110:")
    resnet_blocks = [e[0] for e in resnet_exits]
    print(f"  退出点：Block {resnet_blocks}")
    print(f"  Scale值：{[f'{s:.3f}' for s in resnet_scales]}")

    print("\nMobileNetV2:")
    mobile_blocks = [e[0] for e in mobile_exits]
    print(f"  退出点：Block {mobile_blocks}")
    print(f"  Scale值：{[f'{s:.3f}' for s in mobile_scales]}")


if __name__ == "__main__":
    main()