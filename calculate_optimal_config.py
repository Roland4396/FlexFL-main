#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
计算ScaleFL最优配置
目标：4个等级模型，参数量比例 1/8, 1/4, 1/2, 1
要求：scale和深度比例相近
"""

import numpy as np

def calculate_optimal_config():
    """
    使用立方根方法计算最优配置
    参数量 = 深度 × scale²
    如果 深度 ≈ scale，则参数量 ≈ scale³
    """

    target_ratios = [1/8, 1/4, 1/2, 1.0]

    # 方法1：scale = 深度比例 = ³√(参数量比例)
    print("="*60)
    print("方法1：scale = 深度比例 = ³√(参数量比例)")
    print("="*60)

    scales_method1 = []
    depth_ratios_method1 = []

    for ratio in target_ratios:
        scale = ratio ** (1/3)
        scales_method1.append(scale)
        depth_ratios_method1.append(scale)

    print("\n等级\t目标参数比\tScale\t深度比例\t验证(深度×scale²)")
    for i, (target, scale, depth) in enumerate(zip(target_ratios, scales_method1, depth_ratios_method1)):
        actual = depth * scale**2
        print(f"{i+1}\t{target:.3f}\t\t{scale:.3f}\t{depth:.3f}\t\t{actual:.3f}")

    # 方法2：微调以更接近实际层数
    print("\n" + "="*60)
    print("方法2：根据实际模型结构微调")
    print("="*60)

    # VGG16有13个卷积层
    vgg_layers = 13
    vgg_exits = []
    vgg_scales = []

    print("\nVGG16配置（13个卷积层）：")
    print("等级\t退出层\t深度比例\tScale\t预期参数比")

    for i, ratio in enumerate(target_ratios):
        if i < 3:
            # 使用立方根方法
            ideal_depth = ratio ** (1/3)
            exit_layer = round(ideal_depth * vgg_layers)
            actual_depth = exit_layer / vgg_layers

            # 调整scale使得 actual_depth × scale² = target_ratio
            scale = np.sqrt(ratio / actual_depth)

            vgg_exits.append(exit_layer)
            vgg_scales.append(scale)

            print(f"{i+1}\t{exit_layer}\t{actual_depth:.3f}\t\t{scale:.3f}\t{ratio:.3f}")
        else:
            print(f"{i+1}\t{vgg_layers}\t1.000\t\t1.000\t1.000")
            vgg_exits.append(vgg_layers)
            vgg_scales.append(1.0)

    # ResNet110有54个block
    resnet_blocks = 54
    resnet_exits = []
    resnet_scales = []

    print("\nResNet110配置（54个blocks）：")
    print("等级\tBlock\t深度比例\tScale\t预期参数比")

    for i, ratio in enumerate(target_ratios):
        if i < 3:
            ideal_depth = ratio ** (1/3)
            exit_block = round(ideal_depth * resnet_blocks)
            actual_depth = exit_block / resnet_blocks

            scale = np.sqrt(ratio / actual_depth)

            resnet_exits.append(exit_block)
            resnet_scales.append(scale)

            print(f"{i+1}\t{exit_block}\t{actual_depth:.3f}\t\t{scale:.3f}\t{ratio:.3f}")
        else:
            print(f"{i+1}\t{resnet_blocks}\t1.000\t\t1.000\t1.000")
            resnet_exits.append(resnet_blocks)
            resnet_scales.append(1.0)

    # MobileNetV2有17个block
    mobile_blocks = 17
    mobile_exits = []
    mobile_scales = []

    print("\nMobileNetV2配置（17个blocks）：")
    print("等级\tBlock\t深度比例\tScale\t预期参数比")

    for i, ratio in enumerate(target_ratios):
        if i < 3:
            ideal_depth = ratio ** (1/3)
            exit_block = round(ideal_depth * mobile_blocks)
            actual_depth = exit_block / mobile_blocks

            scale = np.sqrt(ratio / actual_depth)

            mobile_exits.append(exit_block)
            mobile_scales.append(scale)

            print(f"{i+1}\t{exit_block}\t{actual_depth:.3f}\t\t{scale:.3f}\t{ratio:.3f}")
        else:
            print(f"{i+1}\t{mobile_blocks}\t1.000\t\t1.000\t1.000")
            mobile_exits.append(mobile_blocks)
            mobile_scales.append(1.0)

    # 输出最终配置
    print("\n" + "="*60)
    print("最终推荐配置")
    print("="*60)

    print("\n统一Scale配置（所有模型通用）：")
    print(f"WIDTH_RATIOS = {[f'{s:.3f}' for s in scales_method1]}")

    print("\n各模型早退点配置：")
    print(f"VGG16:      exit0={vgg_exits[0]}, exit1={vgg_exits[1]}, exit2={vgg_exits[2]}")
    print(f"            scales={[f'{s:.3f}' for s in vgg_scales]}")

    print(f"\nResNet110:  exit0={resnet_exits[0]}, exit1={resnet_exits[1]}, exit2={resnet_exits[2]}")
    print(f"            scales={[f'{s:.3f}' for s in resnet_scales]}")

    print(f"\nMobileNet:  exit0={mobile_exits[0]}, exit1={mobile_exits[1]}, exit2={mobile_exits[2]}")
    print(f"            scales={[f'{s:.3f}' for s in mobile_scales]}")

    print("\n注意：")
    print("1. ee=1对应exit0（最早退出）")
    print("2. ee=2对应exit1")
    print("3. ee=3对应exit2")
    print("4. ee=4对应完整模型")

if __name__ == "__main__":
    calculate_optimal_config()