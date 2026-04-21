# HeteroFL 4等级模型实现

## 概述

成功将HeteroFL扩展为支持4个模型等级，实现参数量比例为 **1/8, 1/4, 1/2, 1** 的异构联邦学习。

## HeteroFL vs ScaleFL 对比

| 特性 | HeteroFL | ScaleFL |
|-----|----------|---------|
| **模型缩放方式** | Uniform Scaling（所有层相同比例） | Width Scaling + Early Exit |
| **参数控制** | 仅通过scale参数 | scale参数 + ee参数 |
| **知识蒸馏** | 无 | 有（gamma参数） |
| **实现复杂度** | 简单 | 复杂 |

## 主要修改

### 1. 新增文件

#### `Algorithm/Training_HeteroFL_4levels.py`
- 实现了4个等级的HeteroFL
- 使用uniform scaling策略
- Scale值：[0.354, 0.5, 0.707, 1.0]（对应参数量1/8, 1/4, 1/2, 1）

#### `batch_train_heterofl_parallel.py`
- 并行批量训练脚本
- 支持3个并行实验
- 18个实验（3模型 × 3数据集 × 2分布）

#### `test_heterofl_4levels.py`
- 测试脚本，验证4等级功能

### 2. 修改的文件

#### `Algorithm/__init__.py`
- 导入新的HeteroFL_4levels版本
- 保持向后兼容性

## 技术细节

### Scale计算方法

HeteroFL使用uniform scaling，参数量与scale的平方成正比：
```
参数量 ≈ scale²
因此：scale = √(目标参数比例)
```

| 等级 | 目标参数比 | Scale值 | 实际参数比 |
|-----|-----------|---------|-----------|
| 1 | 1/8 (12.5%) | 0.354 | ~12.5% |
| 2 | 1/4 (25%) | 0.5 | 25% |
| 3 | 1/2 (50%) | 0.707 | ~50% |
| 4 | 1 (100%) | 1.0 | 100% |

## 使用方法

### 1. 测试功能
```bash
python test_heterofl_4levels.py
```

### 2. 批量训练（3并行）
```bash
python batch_train_heterofl_parallel.py
```

### 3. 单个实验
```bash
python main_fed.py \
  --gpu 0 \
  --algorithm HeteroFL \
  --model vgg \
  --dataset cifar10 \
  --num_channels 3 \
  --num_classes 10 \
  --iid 0 \
  --data_beta 1 \
  --client_hetero_ration 1:1:1:1 \
  --epochs 400
```

## 配置参数

### 训练配置
- **训练轮数**: 400
- **本地训练轮数**: 5
- **本地批大小**: 50
- **学习率**: 0.01
- **学习率衰减**: 0.998
- **客户端比例**: 1:1:1:1（4个等级均匀分布）

### 实验配置
- **模型**: vgg, resnet, mobilenet
- **数据集**: cifar10, cifar100, TinyImagenet
- **数据分布**: Non-IID beta=1, Non-IID beta=100
- **总实验数**: 18个

## 输出文件

训练结果保存在：
```
outputs_heterofl_4levels/
├── heterofl_vgg_cifar10_noniid_beta1.log
├── heterofl_vgg_cifar10_noniid_beta100.log
├── ...
└── summary.txt
```

## 与ScaleFL的区别

1. **HeteroFL**:
   - 简单的uniform scaling
   - 所有层使用相同的缩放比例
   - 无早退机制
   - 无知识蒸馏

2. **ScaleFL**:
   - 复杂的width scaling + early exit
   - 可以有不同层的缩放比例
   - 4个早退点（ee=1,2,3,4）
   - 有知识蒸馏（gamma参数）

## 执行命令汇总

```bash
# HeteroFL测试
python test_heterofl_4levels.py

# HeteroFL批量训练
python batch_train_heterofl_parallel.py

# ScaleFL批量训练
python batch_train_scalefl_parallel.py

# 查看结果
ls outputs_heterofl_4levels/
ls outputs_scalefl_parallel/
```

## 注意事项

1. HeteroFL使用标准的`resnet`模型，不是`resnet_smart`
2. HeteroFL不需要gamma参数（无知识蒸馏）
3. HeteroFL的实现相对简单，训练速度可能比ScaleFL快
4. 两种算法的性能对比需要通过实验结果来判断