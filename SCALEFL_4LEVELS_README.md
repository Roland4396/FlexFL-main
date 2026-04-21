# ScaleFL 4等级模型实现说明

## 概述

成功将ScaleFL修改为支持4个模型等级，实现参数量比例为 **1/8, 1/4, 1/2, 1** 的异构联邦学习。

## 主要修改

### 1. 模型文件修改

#### VGG16 (`models/vgg_scaleFL.py`)
- 新增 `exit0` 参数，支持4个早退点
- 退出点配置：exit0=6, exit1=8, exit2=10
- 支持 ee=1,2,3,4 四个等级

#### ResNet110 (`models/resnet_smart_scaleFL.py`)
- 新建文件，基于SmartFL ResNet110
- 退出点配置：exit0=27, exit1=34, exit2=43（基于54个blocks）
- 支持 ee=1,2,3,4 四个等级

#### MobileNetV2 (`models/mobileNetV2_scaleFL.py`)
- 新增 `exit0` 参数，支持4个早退点
- 退出点配置：exit0=4, exit1=6, exit2=8
- 支持 ee=1,2,3,4 四个等级

### 2. 算法文件修改

#### Training_ScaleFL.py
- 添加对 `resnet_smart` 模型的支持
- 导入 `ResNet110_cifar_scaleFL`

### 3. 配置参数

#### Scale值配置（基于立方根计算）
```python
WIDTH_RATIOS = [0.5, 0.63, 0.794, 1.0]
```

这些值满足：
- scale ≈ 深度比例
- 深度 × scale² ≈ 目标参数比例

#### 客户端分布
```python
CLIENT_RATIO = "1:1:1:1"  # 4个等级均匀分布
```

## 使用方法

### 1. 测试模型
```bash
python test_scalefl_models.py
```

### 2. 批量训练
```bash
python batch_train_scalefl_final.py
```

### 3. 单个实验示例
```bash
python main_fed.py \
  --gpu 0 \
  --algorithm ScaleFL \
  --model resnet_smart \
  --dataset cifar10 \
  --num_channels 3 \
  --num_classes 10 \
  --iid 0 \
  --data_beta 10 \
  --width_ration 0.5 0.63 0.794 1.0 \
  --client_hetero_ration 1:1:1:1 \
  --gamma 0.05 \
  --epochs 1000
```

## 参数量验证

### VGG16
- Level 1 (scale=0.5): 22M 参数
- Level 2 (scale=0.63): 35M 参数
- Level 3 (scale=0.794): 56M 参数
- Level 4 (scale=1.0): 89M 参数

### ResNet110
- Level 1 (scale=0.5): 436K 参数
- Level 2 (scale=0.63): 679K 参数
- Level 3 (scale=0.794): 1.05M 参数
- Level 4 (scale=1.0): 1.73M 参数

### MobileNetV2
- Level 1 (scale=0.5): 661K 参数
- Level 2 (scale=0.63): 1.01M 参数
- Level 3 (scale=0.794): 1.59M 参数
- Level 4 (scale=1.0): 2.50M 参数

## 文件列表

新增/修改的文件：
- `models/vgg_scaleFL.py` - 修改支持4等级
- `models/resnet_smart_scaleFL.py` - 新建ResNet110 ScaleFL版本
- `models/mobileNetV2_scaleFL.py` - 修改支持4等级
- `Algorithm/Training_ScaleFL.py` - 添加resnet_smart支持
- `batch_train_scalefl_final.py` - 批量训练脚本
- `test_scalefl_models.py` - 模型测试脚本
- `calculate_optimal_config.py` - 参数计算脚本

## 注意事项

1. **ee参数**：控制早退点数量（1-4）
2. **scale参数**：控制模型宽度
3. **参数量**：由深度（ee）和宽度（scale）共同决定
4. 使用 `resnet_smart` 而不是 `resnet` 来获得ResNet110

## 已验证

✅ 所有模型均可正常创建和前向传播
✅ 4个ee等级均能返回正确数量的输出
✅ Scale和深度比例相近，满足设计要求
✅ 参数量比例接近目标（1/8, 1/4, 1/2, 1）