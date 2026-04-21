#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FlexFL 并行批量训练脚本（4090优化）
支持多进程并行训练，充分利用GPU
"""

import os
import subprocess
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# 配置
GPU = 0  # 4090
MODELS = ["resnet_smart"]  # 只训练 SmartFL ResNet110
DATASETS = ["TinyImagenet"]
DATA_CONFIGS = [
    {"name": "noniid_alpha1", "iid": 0, "data_beta": 1},      # 高度Non-IID
    {"name": "noniid_alpha100", "iid": 0, "data_beta": 100}   # 接近IID
]

# 数据集配置
DATASET_INFO = {
    "TinyImagenet": {"num_channels": 3, "num_classes": 200},
}

# 训练参数
CLIENT_RATIO = "1:1:1:1"
PRETRAIN = 200
GAMMA = 10
ONLY = 1
EPOCHS = 400

# 并行配置（4090可以同时跑2-3个实验）
MAX_PARALLEL_JOBS = 2  # 同时运行的实验数，根据显存调整

# 输出目录
OUTPUT_BASE = "outputs"
os.makedirs(OUTPUT_BASE, exist_ok=True)

def run_single_experiment(exp_info):
    """运行单个实验（用于并行）"""
    import traceback

    model, dataset, data_config, exp_id, total_exp = exp_info

    # 构建实验名称
    exp_name = f"{model}_{dataset}_{data_config['name']}"
    log_file = os.path.join(OUTPUT_BASE, f"{exp_name}.log")

    print(f"\n[{exp_id}/{total_exp}] 启动实验: {exp_name}")
    print(f"  PID: {os.getpid()}")
    print(f"  时间: {datetime.now().strftime('%H:%M:%S')}")

    # 获取数据集配置
    ds_info = DATASET_INFO[dataset]

    # 构建命令
    cmd = [
        "python", "main_fed.py",
        "--gpu", str(GPU),
        "--algorithm", "FlexFL",
        "--model", model,
        "--dataset", dataset,
        "--num_channels", str(ds_info["num_channels"]),
        "--num_classes", str(ds_info["num_classes"]),
        "--iid", str(data_config["iid"]),
        "--client_hetero_ration", CLIENT_RATIO,
        "--client_chosen_mode", "available",
        "--pretrain", str(PRETRAIN),
        "--gamma", str(GAMMA),
        "--only", str(ONLY),
        "--epochs", str(EPOCHS)
    ]

    if data_config["data_beta"] is not None:
        cmd.extend(["--data_beta", str(data_config["data_beta"])])

    # 运行实验
    start_time = time.time()
    try:
        with open(log_file, "w", encoding='utf-8') as f:
            f.write(f"实验: {exp_name}\n")
            f.write(f"PID: {os.getpid()}\n")
            f.write(f"命令: {' '.join(cmd)}\n")
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
        print(f"  错误类型: {type(e).__name__}")
        print(f"  错误信息: {str(e)}")
        print(f"\n完整错误堆栈:")
        print("-"*80)
        traceback.print_exc()
        print("-"*80)

        # 写入日志
        try:
            with open(log_file, "a", encoding='utf-8') as f:
                f.write(f"\n\n{'='*80}\n")
                f.write(f"异常信息:\n")
                f.write(f"错误类型: {type(e).__name__}\n")
                f.write(f"错误信息: {str(e)}\n\n")
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
    print("FlexFL 并行批量训练开始（4090优化）")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"最大并行数: {MAX_PARALLEL_JOBS}")
    print("="*80)

    # 准备所有实验
    experiments = []
    exp_id = 0
    for model in MODELS:
        for dataset in DATASETS:
            for data_config in DATA_CONFIGS:
                exp_id += 1
                experiments.append((model, dataset, data_config, exp_id, len(MODELS) * len(DATASETS) * len(DATA_CONFIGS)))

    total_exp = len(experiments)
    print(f"\n总实验数: {total_exp}")
    print(f"模型: {MODELS}")
    print(f"数据集: {DATASETS}")
    print(f"数据分布: {[dc['name'] for dc in DATA_CONFIGS]}")
    print("="*80 + "\n")

    # 并行执行
    overall_start = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=MAX_PARALLEL_JOBS) as executor:
        # 提交所有任务
        future_to_exp = {executor.submit(run_single_experiment, exp): exp for exp in experiments}

        # 收集结果
        completed = 0
        for future in as_completed(future_to_exp):
            completed += 1
            result = future.result()
            results.append(result)
            print(f"\n进度: {completed}/{total_exp} 完成")

    overall_elapsed = time.time() - overall_start

    # 统计
    success_count = sum(1 for r in results if r["success"])
    fail_count = total_exp - success_count
    total_time = sum(r["elapsed"] for r in results) / 3600

    # 总结
    print("\n" + "="*80)
    print("并行批量训练完成")
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
        print(f"{status} {result['exp_name']:40s} {elapsed_str}")
    print("="*80)

    # 生成总结文件
    summary_file = os.path.join(OUTPUT_BASE, "summary.txt")
    with open(summary_file, "w", encoding='utf-8') as f:
        f.write("FlexFL 并行批量训练总结\n")
        f.write(f"生成时间: {datetime.now()}\n")
        f.write(f"实际耗时: {overall_elapsed/3600:.1f} 小时\n")
        f.write(f"累计训练时间: {total_time:.1f} 小时\n")
        f.write(f"加速比: {total_time/max(overall_elapsed/3600, 0.01):.1f}x\n")
        f.write(f"并行数: {MAX_PARALLEL_JOBS}\n")
        f.write("="*80 + "\n\n")
        f.write(f"实验统计:\n")
        f.write(f"  总数: {total_exp}\n")
        f.write(f"  成功: {success_count}\n")
        f.write(f"  失败: {fail_count}\n\n")
        f.write("实验结果:\n")
        f.write("-"*80 + "\n")
        for result in results:
            status = "成功" if result["success"] else "失败"
            elapsed_str = f"{result['elapsed']/60:.1f}min" if result['success'] else "N/A"
            f.write(f"{status}: {result['exp_name']:40s} {elapsed_str}\n")

    print(f"\n总结已保存: {summary_file}")
    print("\n所有任务完成！")

if __name__ == "__main__":
    # 设置启动方法为spawn（Windows兼容）
    multiprocessing.set_start_method('spawn', force=True)
    main()
