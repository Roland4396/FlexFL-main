#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FlexFL 批量训练脚本
3个模型 × 3个数据集 × 2个数据分布 = 18个实验
"""

import os
import subprocess
import time
from datetime import datetime

# 配置
GPU = 0
MODELS = ["vgg", "mobilenet", "resnet"]
DATASETS = ["cifar10", "cifar100", "mnist"]  # 3个数据集
DATA_CONFIGS = [
    {"name": "iid", "iid": 1, "data_beta": None},
    {"name": "noniid_beta100", "iid": 0, "data_beta": 100}  # 接近IID的non-IID
]

# 数据集配置
DATASET_INFO = {
    "cifar10": {"num_channels": 3, "num_classes": 10},
    "cifar100": {"num_channels": 3, "num_classes": 100},
    "mnist": {"num_channels": 1, "num_classes": 10}
}

# 训练参数
CLIENT_RATIO = "1:1:1:1"
PRETRAIN = 200
GAMMA = 10
ONLY = 1
EPOCHS = 1000

# 输出目录
OUTPUT_BASE = "outputs"
os.makedirs(OUTPUT_BASE, exist_ok=True)

def run_experiment(model, dataset, data_config, exp_id, total_exp):
    """运行单个实验"""

    # 构建实验名称
    exp_name = f"{model}_{dataset}_{data_config['name']}"
    log_file = os.path.join(OUTPUT_BASE, f"{exp_name}.log")

    print("\n" + "="*80)
    print(f"[{exp_id}/{total_exp}] 开始实验: {exp_name}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

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

    # 添加data_beta参数（如果需要）
    if data_config["data_beta"] is not None:
        cmd.extend(["--data_beta", str(data_config["data_beta"])])

    print(f"命令: {' '.join(cmd)}")
    print(f"日志: {log_file}")
    print("-"*80)

    # 运行实验
    start_time = time.time()
    try:
        with open(log_file, "w", encoding='utf-8') as f:
            f.write(f"实验: {exp_name}\n")
            f.write(f"命令: {' '.join(cmd)}\n")
            f.write(f"开始时间: {datetime.now()}\n")
            f.write("="*80 + "\n\n")
            f.flush()

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            # 实时输出
            for line in process.stdout:
                print(line, end='')
                f.write(line)
                f.flush()

            process.wait()
            return_code = process.returncode

        elapsed = time.time() - start_time

        if return_code == 0:
            print(f"\n✓ 实验完成: {exp_name}")
            print(f"  耗时: {elapsed/60:.1f} 分钟")
            return True
        else:
            print(f"\n✗ 实验失败: {exp_name} (返回码: {return_code})")
            print(f"  查看日志: {log_file}")
            return False

    except Exception as e:
        print(f"\n✗ 实验异常: {exp_name}")
        print(f"  错误: {str(e)}")
        print(f"  查看日志: {log_file}")
        return False

def main():
    """主函数"""
    print("\n" + "="*80)
    print("FlexFL 批量训练开始")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # 计算总实验数
    total_exp = len(MODELS) * len(DATASETS) * len(DATA_CONFIGS)
    print(f"\n总实验数: {total_exp}")
    print(f"模型: {MODELS}")
    print(f"数据集: {DATASETS}")
    print(f"数据分布: {[dc['name'] for dc in DATA_CONFIGS]}")
    print("="*80)

    # 统计
    exp_id = 0
    success_count = 0
    fail_count = 0
    results = []

    overall_start = time.time()

    # 批量运行
    for model in MODELS:
        for dataset in DATASETS:
            for data_config in DATA_CONFIGS:
                exp_id += 1

                success = run_experiment(model, dataset, data_config, exp_id, total_exp)

                results.append({
                    "exp_name": f"{model}_{dataset}_{data_config['name']}",
                    "success": success
                })

                if success:
                    success_count += 1
                else:
                    fail_count += 1

                # 短暂休息
                if exp_id < total_exp:
                    print("\n等待5秒后继续下一个实验...")
                    time.sleep(5)

    # 总结
    overall_elapsed = time.time() - overall_start

    print("\n" + "="*80)
    print("批量训练完成")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {overall_elapsed/3600:.1f} 小时")
    print("="*80)
    print(f"\n实验统计:")
    print(f"  总数: {total_exp}")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    print("\n实验结果:")
    print("-"*80)
    for result in results:
        status = "✓" if result["success"] else "✗"
        print(f"{status} {result['exp_name']}")
    print("="*80)

    # 生成总结文件
    summary_file = os.path.join(OUTPUT_BASE, "summary.txt")
    with open(summary_file, "w", encoding='utf-8') as f:
        f.write("FlexFL 批量训练总结\n")
        f.write(f"生成时间: {datetime.now()}\n")
        f.write(f"总耗时: {overall_elapsed/3600:.1f} 小时\n")
        f.write("="*80 + "\n\n")
        f.write(f"实验统计:\n")
        f.write(f"  总数: {total_exp}\n")
        f.write(f"  成功: {success_count}\n")
        f.write(f"  失败: {fail_count}\n\n")
        f.write("实验结果:\n")
        f.write("-"*80 + "\n")
        for result in results:
            status = "成功" if result["success"] else "失败"
            f.write(f"{status}: {result['exp_name']}\n")

    print(f"\n总结已保存: {summary_file}")
    print("\n所有任务完成！")

if __name__ == "__main__":
    main()
