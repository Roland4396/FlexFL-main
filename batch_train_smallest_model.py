#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量训练最小模型 (1/8参数量)
使用标准FedAvg，但模型大小只有原来的1/8
"""

import os
import subprocess
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ==================== 配置区域 ====================

# GPU配置
GPU = 0

# 并行配置
MAX_PARALLEL_JOBS = 3  # 同时运行3个实验

# 模型列表
MODELS = ["vgg", "resnet", "mobilenet"]

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

# 训练参数（标准FedAvg参数）
EPOCHS = 400
LOCAL_EP = 5
LOCAL_BS = 50
LR = 0.01
LR_DECAY = 0.998
MOMENTUM = 0.5
WEIGHT_DECAY = 1e-4
FRAC = 0.1  # 每轮选择10%的客户端

# 输出目录
OUTPUT_BASE = "outputs_smallest_model"
os.makedirs(OUTPUT_BASE, exist_ok=True)

# ==================== 函数定义 ====================

def run_single_experiment(exp_info):
    """运行单个实验（用于并行）"""
    import traceback

    model, dataset, data_config, exp_id, total_exp = exp_info

    # 构建实验名称
    exp_name = f"smallest_{model}_{dataset}_{data_config['name']}"
    log_file = os.path.join(OUTPUT_BASE, f"{exp_name}.log")

    print(f"\n[{exp_id}/{total_exp}] 启动实验: {exp_name}")
    print(f"  PID: {os.getpid()}")
    print(f"  时间: {datetime.now().strftime('%H:%M:%S')}")
    print(f"  模型参数: 1/8 of full model")

    # 获取数据集配置
    ds_info = DATASET_INFO[dataset]

    # 构建命令 - 使用Decoupled算法但只有一个模型等级
    cmd = [
        "python", "-c",
        f"""
import sys
sys.path.insert(0, '/root/autodl-tmp/FlexFL-main')

from utils.options import args_parser
from utils.get_dataset import get_dataset
from Algorithm.Training_Decoupled_SingleLevel import Decoupled_SingleLevel
from utils.set_seed import set_random_seed

# 解析参数
args = args_parser()
args.gpu = {GPU}
args.algorithm = 'Decoupled_SingleLevel'
args.model = '{model}'
args.dataset = '{dataset}'
args.num_channels = {ds_info["num_channels"]}
args.num_classes = {ds_info["num_classes"]}
args.iid = {data_config["iid"]}
args.data_beta = {data_config.get("data_beta", "None")}
args.epochs = {EPOCHS}
args.local_ep = {LOCAL_EP}
args.local_bs = {LOCAL_BS}
args.lr = {LR}
args.lr_decay = {LR_DECAY}
args.num_users = 100
args.momentum = {MOMENTUM}
args.weight_decay = {WEIGHT_DECAY}
args.frac = {FRAC}
args.num_users = 100
args.device = 'cuda:{GPU}' if args.gpu >= 0 else 'cpu'
args.log = 0  # 不使用wandb

# 设置随机种子
set_random_seed(2023)

# 获取数据集
dataset_train, dataset_test, dict_users = get_dataset(args)

# 训练
print('Starting training with smallest model (1/8 parameters)...')
Decoupled_SingleLevel(args, dataset_train, dataset_test, dict_users)
        """
    ]

    # 运行实验
    start_time = time.time()
    try:
        with open(log_file, "w", encoding='utf-8') as f:
            f.write(f"实验: {exp_name}\n")
            f.write(f"模型大小: 1/8 of full model\n")
            f.write(f"PID: {os.getpid()}\n")
            f.write(f"开始时间: {datetime.now()}\n")
            f.write("="*80 + "\n\n")

            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True
            )

            elapsed = time.time() - start_time
            f.write(f"\n\n{'='*80}\n")
            f.write(f"结束时间: {datetime.now()}\n")
            f.write(f"耗时: {elapsed/60:.1f} 分钟\n")

            success = (result.returncode == 0)

        if success:
            print(f"✓ [{exp_id}/{total_exp}] 完成: {exp_name} ({elapsed/60:.1f}分钟)")
        else:
            print(f"✗ [{exp_id}/{total_exp}] 失败: {exp_name}")

        return {
            "exp_name": exp_name,
            "success": success,
            "elapsed": elapsed
        }

    except Exception as e:
        print(f"✗ [{exp_id}/{total_exp}] 异常: {exp_name}")
        print(f"  错误: {str(e)}")

        # 写入日志
        try:
            with open(log_file, "a", encoding='utf-8') as f:
                f.write(f"\n\n{'='*80}\n")
                f.write(f"异常信息:\n")
                f.write(f"错误: {str(e)}\n\n")
                f.write("完整堆栈:\n")
                f.write(traceback.format_exc())
        except:
            pass

        return {
            "exp_name": exp_name,
            "success": False,
            "elapsed": 0
        }

def main():
    """主函数"""
    print("\n" + "="*80)
    print("最小模型批量训练 (1/8参数量)")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"最大并行数: {MAX_PARALLEL_JOBS}")
    print("="*80)

    print("\n配置信息:")
    print(f"  模型大小: 1/8 of full model (scale ≈ 0.354)")
    print(f"  训练算法: 标准FedAvg")
    print(f"  训练轮数: {EPOCHS}")
    print(f"  客户端选择: {FRAC*100:.0f}%每轮")
    print(f"  并行数: {MAX_PARALLEL_JOBS}")

    # 准备所有实验
    experiments = []
    exp_id = 0
    for model in MODELS:
        for dataset in DATASETS:
            for data_config in DATA_CONFIGS:
                exp_id += 1
                experiments.append((model, dataset, data_config, exp_id,
                                  len(MODELS) * len(DATASETS) * len(DATA_CONFIGS)))

    total_exp = len(experiments)
    print(f"\n总实验数: {total_exp}")
    print(f"  模型: {MODELS}")
    print(f"  数据集: {DATASETS}")
    print(f"  数据分布: {[dc['name'] for dc in DATA_CONFIGS]}")
    print("="*80)

    # 询问用户是否继续
    response = input("\n是否开始批量训练最小模型？(y/n): ")
    if response.lower() != 'y':
        print("训练已取消")
        return

    # 并行执行
    overall_start = time.time()
    results = []

    print(f"\n开始并行训练，每次运行{MAX_PARALLEL_JOBS}个实验...")
    print("="*80)

    with ProcessPoolExecutor(max_workers=MAX_PARALLEL_JOBS) as executor:
        # 提交所有任务
        future_to_exp = {executor.submit(run_single_experiment, exp): exp
                        for exp in experiments}

        # 收集结果
        completed = 0
        for future in as_completed(future_to_exp):
            completed += 1
            result = future.result()
            results.append(result)

            # 显示进度
            print(f"\n进度: {completed}/{total_exp} 完成 " +
                  f"({completed/total_exp*100:.1f}%)")

            # 显示当前运行状态
            running = min(MAX_PARALLEL_JOBS, total_exp - completed)
            if running > 0:
                print(f"当前并行运行: {running} 个实验")

    overall_elapsed = time.time() - overall_start

    # 统计
    success_count = sum(1 for r in results if r["success"])
    fail_count = total_exp - success_count
    total_time = sum(r["elapsed"] for r in results) / 3600

    # 总结
    print("\n" + "="*80)
    print("最小模型批量训练完成")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"实际耗时: {overall_elapsed/3600:.1f} 小时")
    print(f"累计训练时间: {total_time:.1f} 小时")
    print(f"加速比: {total_time/max(overall_elapsed/3600, 0.01):.1f}x")
    print("="*80)

    print(f"\n实验统计:")
    print(f"  总数: {total_exp}")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")

    print("\n实验结果:")
    print("-"*80)

    # 按实验名排序
    results.sort(key=lambda x: x["exp_name"])
    for result in results:
        status = "✓" if result["success"] else "✗"
        elapsed_str = f"{result['elapsed']/60:.1f}min" if result['success'] else "失败"
        print(f"{status} {result['exp_name']:50s} {elapsed_str}")
    print("="*80)

    # 生成总结文件
    summary_file = os.path.join(OUTPUT_BASE, "summary.txt")
    with open(summary_file, "w", encoding='utf-8') as f:
        f.write("最小模型批量训练总结 (1/8参数量)\n")
        f.write(f"生成时间: {datetime.now()}\n")
        f.write(f"实际耗时: {overall_elapsed/3600:.1f} 小时\n")
        f.write(f"累计训练时间: {total_time:.1f} 小时\n")
        f.write(f"加速比: {total_time/max(overall_elapsed/3600, 0.01):.1f}x\n")
        f.write(f"并行数: {MAX_PARALLEL_JOBS}\n")
        f.write("="*80 + "\n\n")
        f.write("配置信息:\n")
        f.write(f"  模型大小: 1/8 of full model\n")
        f.write(f"  Scale: 0.354\n")
        f.write(f"  训练轮数: {EPOCHS}\n")
        f.write(f"  算法: 标准FedAvg\n\n")
        f.write(f"实验统计:\n")
        f.write(f"  总数: {total_exp}\n")
        f.write(f"  成功: {success_count}\n")
        f.write(f"  失败: {fail_count}\n\n")
        f.write("实验结果:\n")
        f.write("-"*80 + "\n")
        for result in results:
            status = "成功" if result["success"] else "失败"
            elapsed_str = f"{result['elapsed']/60:.1f}min" if result['success'] else "N/A"
            f.write(f"{status}: {result['exp_name']:50s} {elapsed_str}\n")

    print(f"\n总结已保存: {summary_file}")
    print("\n所有任务完成！")

if __name__ == "__main__":
    # 设置启动方法为spawn（Windows兼容）
    multiprocessing.set_start_method('spawn', force=True)
    main()