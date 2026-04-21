# FlexFL 训练命令（简化版）

本文档包含更新后的训练命令，支持以下特性：
- **4个模型等级**：1/8, 1/4, 1/2, 1（原来是3个等级）
- **SmartFL ResNet模型**：已集成到框架中
- **SmartFL Dirichlet数据切分**：通过alpha参数控制Non-IID程度

## 参数说明

### 关键参数

- `--iid`: 数据分布方式
  - `1`: IID（独立同分布）
  - `0`: Non-IID（使用Dirichlet分布，通过 `--data_beta` 控制）

- `--data_beta`: Dirichlet分布的alpha参数（**整数，范围 1-100+**）
  - `1`: 高度非独立同分布（highly non-IID）
  - `10`: 中等非独立同分布（moderate non-IID）
  - `50`: 轻度非独立同分布
  - `100`: 接近独立同分布（near IID）

- `--client_hetero_ration`: 客户端异构比例
  - **支持4个等级**，例如：`1:1:1:1`（均匀分布）或 `4:3:2:1`（递减分布）
  - 对应模型大小：1/8, 1/4, 1/2, 1

- `--model`: 模型选择
  - `vgg`: VGG模型
  - `resnet`: **SmartFL ResNet**（新集成，推荐用于对比实验）

## FlexFL训练命令

### 1. IID数据分布

#### 使用VGG模型（4个等级，均匀分布）
```bash
python main_fed.py --gpu 0 --algorithm FlexFL --model vgg --dataset cifar10 \
  --num_channels 3 --num_classes 10 --iid 1 \
  --client_hetero_ration 1:1:1:1 --client_chosen_mode available \
  --pretrain 200 --gamma 10 --only 1
```

#### 使用SmartFL ResNet模型（4个等级）
```bash
python main_fed.py --gpu 0 --algorithm FlexFL --model resnet --dataset cifar10 \
  --num_channels 3 --num_classes 10 --iid 1 \
  --client_hetero_ration 1:1:1:1 --client_chosen_mode available \
  --pretrain 200 --gamma 10 --only 1
```

### 2. Non-IID数据分布（使用Dirichlet方法）

#### 高度非IID（alpha=1）
```bash
python main_fed.py --gpu 0 --algorithm FlexFL --model vgg --dataset cifar10 \
  --num_channels 3 --num_classes 10 --iid 0 --data_beta 1 \
  --client_hetero_ration 1:1:1:1 --client_chosen_mode available \
  --pretrain 200 --gamma 10 --only 1
```

#### 中等非IID（alpha=10，推荐）
```bash
python main_fed.py --gpu 0 --algorithm FlexFL --model vgg --dataset cifar10 \
  --num_channels 3 --num_classes 10 --iid 0 --data_beta 10 \
  --client_hetero_ration 1:1:1:1 --client_chosen_mode available \
  --pretrain 200 --gamma 10 --only 1
```

#### 轻度非IID（alpha=50）
```bash
python main_fed.py --gpu 0 --algorithm FlexFL --model vgg --dataset cifar10 \
  --num_channels 3 --num_classes 10 --iid 0 --data_beta 50 \
  --client_hetero_ration 1:1:1:1 --client_chosen_mode available \
  --pretrain 200 --gamma 10 --only 1
```

#### 使用ResNet + 中等非IID
```bash
python main_fed.py --gpu 0 --algorithm FlexFL --model resnet --dataset cifar10 \
  --num_channels 3 --num_classes 10 --iid 0 --data_beta 10 \
  --client_hetero_ration 1:1:1:1 --client_chosen_mode available \
  --pretrain 200 --gamma 10 --only 1
```

### 3. 不同客户端异构分布

#### 递减分布（4:3:2:1）
```bash
python main_fed.py --gpu 0 --algorithm FlexFL --model vgg --dataset cifar10 \
  --num_channels 3 --num_classes 10 --iid 0 --data_beta 10 \
  --client_hetero_ration 4:3:2:1 --client_chosen_mode available \
  --pretrain 200 --gamma 10 --only 1
```

### 4. 使用预计算的APoZ分数（跳过预训练）

```bash
python main_fed.py --gpu 0 --algorithm FlexFL --model vgg --dataset cifar10 \
  --num_channels 3 --num_classes 10 --iid 0 --data_beta 10 \
  --client_hetero_ration 1:1:1:1 --client_chosen_mode available \
  --pretrain 0 --apoz 9 --gamma 10 --only 1
```

## 其他算法训练命令

### HeteroFL（4个等级）

#### IID
```bash
python main_fed.py --gpu 0 --algorithm HeteroFL --model vgg --dataset cifar10 \
  --num_channels 3 --num_classes 10 --iid 1 \
  --client_hetero_ration 1:1:1:1 --client_chosen_mode available
```

#### Non-IID（alpha=10）
```bash
python main_fed.py --gpu 0 --algorithm HeteroFL --model vgg --dataset cifar10 \
  --num_channels 3 --num_classes 10 --iid 0 --data_beta 10 \
  --client_hetero_ration 1:1:1:1 --client_chosen_mode available
```

### Decoupled（4个等级）

#### IID
```bash
python main_fed.py --gpu 0 --algorithm Decoupled --model vgg --dataset cifar10 \
  --num_channels 3 --num_classes 10 --iid 1 \
  --client_hetero_ration 1:1:1:1 --client_chosen_mode available
```

#### Non-IID（alpha=10）
```bash
python main_fed.py --gpu 0 --algorithm Decoupled --model vgg --dataset cifar10 \
  --num_channels 3 --num_classes 10 --iid 0 --data_beta 10 \
  --client_hetero_ration 1:1:1:1 --client_chosen_mode available
```

### ScaleFL

#### IID
```bash
python main_fed.py --gpu 0 --algorithm ScaleFL --model vgg --dataset cifar10 \
  --num_channels 3 --num_classes 10 --iid 1 \
  --width_ration 0.75 0.82 1.0 --client_hetero_ration 4:3:3 --gamma 0.05
```

#### Non-IID（alpha=10）
```bash
python main_fed.py --gpu 0 --algorithm ScaleFL --model vgg --dataset cifar10 \
  --num_channels 3 --num_classes 10 --iid 0 --data_beta 10 \
  --width_ration 0.75 0.82 1.0 --client_hetero_ration 4:3:3 --gamma 0.05
```

## 数据集选项

支持的数据集：
- `cifar10`（默认，10类）
- `cifar100`（100类，需设置 `--num_classes 100`）
- `mnist`
- `fashion-mnist`
- `widar`
- `TinyImagenet`（200类）

### CIFAR-100示例
```bash
python main_fed.py --gpu 0 --algorithm FlexFL --model resnet --dataset cifar100 \
  --num_channels 3 --num_classes 100 --iid 0 --data_beta 10 \
  --client_hetero_ration 1:1:1:1 --client_chosen_mode available \
  --pretrain 200 --gamma 10 --only 1
```

## Alpha参数选择指南

根据实验需求选择合适的alpha值：

| alpha值 | Non-IID程度 | 使用场景 |
|---------|------------|----------|
| 1       | 极高        | 极端异构数据测试 |
| 10      | 高          | 一般Non-IID实验（推荐） |
| 50      | 中等        | 轻度Non-IID测试 |
| 100     | 低          | 接近IID的场景 |

## 关键改进总结

1. **简化配置**：移除了 `--noniid_case` 参数，直接使用 `--iid` 和 `--data_beta` 控制
2. **统一方法**：Non-IID模式统一使用SmartFL的Dirichlet分布
3. **模型等级**：从3个（1/4, 1/2, 1）扩展到4个（1/8, 1/4, 1/2, 1）
4. **ResNet集成**：SmartFL的ResNet实现，更标准的BasicBlock结构
5. **客户端分布**：支持4级客户端异构配置

## 注意事项

1. `--data_beta` 参数**必须使用整数**（1-100+），不能使用浮点数
2. `--client_hetero_ration` 现在应该配置为4个等级（例如`1:1:1:1`）
3. 使用ResNet模型时，建议设置较大的预训练轮数（`--pretrain 200`）
4. Dirichlet方法比传统标签分片方法更适合模拟真实的Non-IID场景
5. 不再需要指定 `--noniid_case` 参数
