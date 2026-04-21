#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ScaleFL 批量训练脚本 - 最终版本
支持4个模型等级，参数量比例 1/8, 1/4, 1/2, 1
使用优化的scale和ee配置
"""

import os
import subprocess
import time
from datetime import datetime

# ==================== 配置区域 ====================

# GPU配置
GPU = 0

# 模型列表 (使用resnet_smart而不是resnet)
MODELS = ["vgg", "resnet_smart", "mobilenet"]

# 数据集配置
DATASETS = ["cifar10", "cifar100", "TinyImagenet"]

# 数据分布配置
DATA_CONFIGS = [
    {"name": "noniid_beta1", "iid": 0, "data_beta": 1},       # 高度Non-IID
    {"name": "noniid_beta100", "iid": 0, "data_beta": 100}    # 接近IID
]

# 数据集信息
DATASET_INFO = {
    "cifar10": {"num_channels": 3, "num_classes": 10},
    "cifar100": {"num_channels": 3, "num_classes": 100},
    "TinyImagenet": {"num_channels": 3, "num_classes": 200}
}

# ScaleFL 4等级配置 (基于计算结果)
# 统一的scale配置，对应参数量比例：1/8, 1/4, 1/2, 1
WIDTH_RATIOS = [0.5, 0.63, 0.794, 1.0]

# 客户端分布（4个等级均匀分布）
CLIENT_RATIO = "1:1:1:1"

# ScaleFL特定参数
GAMMA = 0.1  # ScaleFL的gamma参数（知识蒸馏权重）
EPOCHS = 400  # 训练轮数
LOCAL_EP = 5  # 本地训练轮数
LOCAL_BS = 50  # 本地批大小
LR = 0.01  # 学习率
LR_DECAY = 0.998  # 学习率衰减
MOMENTUM = 0.5  # SGD动量
WEIGHT_DECAY = 1e-4  # 权重衰减

# 输出目录
OUTPUT_BASE = "outputs_scalefl_4levels"
os.makedirs(OUTPUT_BASE, exist_ok=True)

# ==================== 函数定义 ====================

def get_exp_name(model, dataset, data_config):
    """生成实验名称"""
    return f"scalefl_{model}_{dataset}_{data_config['name']}"

def run_experiment(model, dataset, data_config, exp_id, total_exp):
    """运行单个ScaleFL实验"""

    exp_name = get_exp_name(model, dataset, data_config)
    log_file = os.path.join(OUTPUT_BASE, f"{exp_name}.log")

    print("\n" + "="*80)
    print(f"[{exp_id}/{total_exp}] 开始实验: {exp_name}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"模型: {model}")
    print(f"数据集: {dataset}")
    print(f"数据分布: {data_config['name']}")
    print(f"Scale配置: {WIDTH_RATIOS} (参数比例: 1/8, 1/4, 1/2, 1)")
    print(f"客户端分布: {CLIENT_RATIO}")
    print("="*80)

    # 获取数据集配置
    ds_info = DATASET_INFO[dataset]

    # 构建命令
    cmd = [
        "python", "main_fed.py",
        "--gpu", str(GPU),
        "--algorithm", "ScaleFL",
        "--model", model,
        "--dataset", dataset,
        "--num_channels", str(ds_info["num_channels"]),
        "--num_classes", str(ds_info["num_classes"]),
        "--iid", str(data_config["iid"]),
        "--width_ration"] + [str(w) for w in WIDTH_RATIOS] + [
        "--client_hetero_ration", CLIENT_RATIO,
        "--gamma", str(GAMMA),
        "--epochs", str(EPOCHS),
        "--local_ep", str(LOCAL_EP),
        "--local_bs", str(LOCAL_BS),
        "--lr", str(LR),
        "--lr_decay", str(LR_DECAY),
        "--momentum", str(MOMENTUM),
        "--weight_decay", str(WEIGHT_DECAY)
    ]

    # 添加data_beta参数（如果需要）
    if data_config["data_beta"] is not None:
        cmd.extend(["--data_beta", str(data_config["data_beta"])])

    print(f"命令: {' '.join(cmd)}")
    print(f"日志文件: {log_file}")
    print("-"*80)

    # 运行实验
    start_time = time.time()
    try:
        with open(log_file, "w", encoding='utf-8') as f:
            # 写入实验信息
            f.write("="*80 + "\n")
            f.write(f"实验名称: {exp_name}\n")
            f.write(f"算法: ScaleFL (4 levels)\n")
            f.write(f"模型: {model}\n")
            f.write(f"数据集: {dataset}\n")
            f.write(f"数据分布: {data_config['name']}\n")
            f.write(f"Scale配置: {WIDTH_RATIOS}\n")
            f.write(f"参数量比例: 1/8, 1/4, 1/2, 1\n")
            f.write(f"客户端分布: {CLIENT_RATIO}\n")
            f.write(f"开始时间: {datetime.now()}\n")
            f.write(f"命令: {' '.join(cmd)}\n")
            f.write("="*80 + "\n\n")
            f.flush()

            # 运行进程
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            # 实时输出和保存日志
            for line in process.stdout:
                print(line, end='')
                f.write(line)
                f.flush()

            process.wait()
            return_code = process.returncode

        elapsed = time.time() - start_time

        # 检查结果
        if return_code == 0:
            print(f"\n✓ 实验完成: {exp_name}")
            print(f"  耗时: {elapsed/60:.1f} 分钟")
            success = True
        else:
            print(f"\n✗ 实验失败: {exp_name} (返回码: {return_code})")
            print(f"  查看日志: {log_file}")
            success = False

        return {
            "exp_name": exp_name,
            "success": success,
            "elapsed": elapsed
        }

    except Exception as e:
        print(f"\n✗ 实验异常: {exp_name}")
        print(f"  错误: {str(e)}")
        print(f"  查看日志: {log_file}")
        return {
            "exp_name": exp_name,
            "success": False,
            "elapsed": 0
        }

def main():
    """主函数"""
    print("\n" + "="*80)
    print("ScaleFL 批量训练 - 4等级版本")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    print("\n配置信息:")
    print(f"  Scale配置: {WIDTH_RATIOS}")
    print(f"  参数量比例: 1/8, 1/4, 1/2, 1")
    print(f"  客户端分布: {CLIENT_RATIO}")
    print(f"  训练轮数: {EPOCHS}")
    print(f"  本地轮数: {LOCAL_EP}")
    print(f"  学习率: {LR} (衰减: {LR_DECAY})")
    print(f"  Gamma: {GAMMA}")

    # 计算总实验数
    total_exp = len(MODELS) * len(DATASETS) * len(DATA_CONFIGS)
    print(f"\n总实验数: {total_exp}")
    print(f"  模型: {MODELS}")
    print(f"  数据集: {DATASETS}")
    print(f"  数据分布: {[dc['name'] for dc in DATA_CONFIGS]}")
    print("="*80)

    # 询问用户是否继续
    response = input("\n是否开始批量训练？(y/n): ")
    if response.lower() != 'y':
        print("训练已取消")
        return

    # 初始化统计
    exp_id = 0
    success_count = 0
    fail_count = 0
    results = []

    overall_start = time.time()

    # 批量运行实验
    for model in MODELS:
        for dataset in DATASETS:
            for data_config in DATA_CONFIGS:
                exp_id += 1

                # 运行实验
                result = run_experiment(model, dataset, data_config, exp_id, total_exp)
                results.append(result)

                # 更新统计
                if result["success"]:
                    success_count += 1
                else:
                    fail_count += 1

                # 短暂休息（避免过热）
                if exp_id < total_exp:
                    print("\n等待5秒后继续下一个实验...")
                    time.sleep(5)

    # 计算总耗时
    overall_elapsed = time.time() - overall_start

    # 打印总结
    print("\n" + "="*80)
    print("批量训练完成")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {overall_elapsed/3600:.1f} 小时")
    print("="*80)

    print(f"\n实验统计:")
    print(f"  总数: {total_exp}")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")

    print("\n实验结果详情:")
    print("-"*80)
    print(f"{'状态':<6} {'实验名称':<50} {'耗时':<10}")
    print("-"*80)
    for result in results:
        status = "✓" if result["success"] else "✗"
        elapsed_str = f"{result['elapsed']/60:.1f}min" if result['success'] else "失败"
        print(f"{status:<6} {result['exp_name']:<50} {elapsed_str:<10}")
    print("="*80)

    # 生成总结文件
    summary_file = os.path.join(OUTPUT_BASE, "summary.txt")
    with open(summary_file, "w", encoding='utf-8') as f:
        f.write("ScaleFL 批量训练总结 (4等级版本)\n")
        f.write("="*80 + "\n")
        f.write(f"生成时间: {datetime.now()}\n")
        f.write(f"总耗时: {overall_elapsed/3600:.1f} 小时\n\n")

        f.write("配置信息:\n")
        f.write(f"  Scale配置: {WIDTH_RATIOS}\n")
        f.write(f"  参数量比例: 1/8, 1/4, 1/2, 1\n")
        f.write(f"  客户端分布: {CLIENT_RATIO}\n")
        f.write(f"  训练轮数: {EPOCHS}\n")
        f.write(f"  Gamma: {GAMMA}\n\n")

        f.write(f"实验统计:\n")
        f.write(f"  总数: {total_exp}\n")
        f.write(f"  成功: {success_count}\n")
        f.write(f"  失败: {fail_count}\n\n")

        f.write("实验结果:\n")
        f.write("-"*80 + "\n")
        for result in results:
            status = "成功" if result["success"] else "失败"
            elapsed_str = f"{result['elapsed']/60:.1f}分钟" if result['success'] else "N/A"
            f.write(f"{status}: {result['exp_name']} (耗时: {elapsed_str})\n")

    print(f"\n总结已保存到: {summary_file}")
    print("\n所有任务完成！")

if __name__ == "__main__":
    main()