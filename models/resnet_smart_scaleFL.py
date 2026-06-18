"""
ResNet110 for ScaleFL - 支持4个等级
基于SmartFL ResNet110，修改为支持ScaleFL的早退机制
- 总共54个blocks (3 stages × 18 blocks)
- 4个早退点：Block 27, 34, 43, 54
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic residual block"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, track_running_stats=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=track_running_stats)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=track_running_stats)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes, track_running_stats=track_running_stats)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet110_ScaleFL(nn.Module):
    """
    ResNet110 for ScaleFL with 4 exit points
    - exit0 / exit1 / exit2 由外部 profile 传入
    - full:  Block 54 (100%深度，全部参数)
    """
    def __init__(self, num_channels=3, num_classes=10, track_running_stats=True, scale=1.0,
                 exit0=27, exit1=34, exit2=43):
        super(ResNet110_ScaleFL, self).__init__()

        self.exit0 = exit0
        self.exit1 = exit1
        self.exit2 = exit2
        self.total_blocks = 54

        # 根据scale调整通道数
        base_channels = [16, 32, 64]
        channels = [int(c * scale) for c in base_channels]

        self.inchannel = channels[0]

        # 初始卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, self.inchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.inchannel, track_running_stats=track_running_stats),
            nn.ReLU()
        )

        # 构建所有blocks作为一个列表，便于按照退出点切分
        self.blocks = nn.ModuleList()

        # Stage 1: 18 blocks, 16 channels
        for i in range(18):
            self.blocks.append(BasicBlock(self.inchannel, channels[0], 1, track_running_stats))
            self.inchannel = channels[0]

        # Stage 2: 18 blocks, 32 channels (第一个block有stride=2)
        for i in range(18):
            stride = 2 if i == 0 else 1
            self.blocks.append(BasicBlock(self.inchannel, channels[1], stride, track_running_stats))
            self.inchannel = channels[1]

        # Stage 3: 18 blocks, 64 channels (第一个block有stride=2)
        for i in range(18):
            stride = 2 if i == 0 else 1
            self.blocks.append(BasicBlock(self.inchannel, channels[2], stride, track_running_stats))
            self.inchannel = channels[2]

        exit_channels = [
            self._exit_channels(channels, exit0),
            self._exit_channels(channels, exit1),
            self._exit_channels(channels, exit2),
            channels[2],
        ]

        # 为每个退出点创建分类器
        self.classifiers = nn.ModuleList()

        # 每个分类头的输入通道必须与 exit 实际落到的 stage 对齐。
        # 否则只要切到 stage3，Linear 就会出现维度不匹配。
        self.classifiers.append(nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(exit_channels[0], num_classes)
        ))

        self.classifiers.append(nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(exit_channels[1], num_classes)
        ))

        self.classifiers.append(nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(exit_channels[2], num_classes)
        ))

        # 完整模型分类器
        self.classifiers.append(nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(exit_channels[3], num_classes)
        ))

    def _exit_channels(self, channels, exit_block):
        if not 1 <= exit_block <= self.total_blocks:
            raise ValueError(f"exit_block must be in [1, {self.total_blocks}], got {exit_block}")
        if exit_block <= 18:
            return channels[0]
        if exit_block <= 36:
            return channels[1]
        return channels[2]

    def forward(self, x, ee=1):
        """
        Forward with early exit
        ee=1: exit at configured exit0
        ee=2: exit at configured exit1
        ee=3: exit at configured exit2
        ee=4: full model (all 54 blocks)
        """
        x = self.conv1(x)

        if ee == 1:
            # 只运行到exit0
            for i in range(self.exit0):
                x = self.blocks[i](x)
            output = self.classifiers[0](x)
            return [{'output': output}]

        elif ee == 2:
            # 运行到exit0和exit1
            results = []

            # 到exit0
            for i in range(self.exit0):
                x = self.blocks[i](x)
            results.append({'output': self.classifiers[0](x)})

            # 到exit1
            for i in range(self.exit0, self.exit1):
                x = self.blocks[i](x)
            results.append({'output': self.classifiers[1](x)})

            return results

        elif ee == 3:
            # 运行到exit0、exit1和exit2
            results = []

            # 到exit0
            for i in range(self.exit0):
                x = self.blocks[i](x)
            results.append({'output': self.classifiers[0](x)})

            # 到exit1
            for i in range(self.exit0, self.exit1):
                x = self.blocks[i](x)
            results.append({'output': self.classifiers[1](x)})

            # 到exit2
            for i in range(self.exit1, self.exit2):
                x = self.blocks[i](x)
            results.append({'output': self.classifiers[2](x)})

            return results

        elif ee == 4:
            # 完整模型
            results = []

            # 到exit0
            for i in range(self.exit0):
                x = self.blocks[i](x)
            results.append({'output': self.classifiers[0](x)})

            # 到exit1
            for i in range(self.exit0, self.exit1):
                x = self.blocks[i](x)
            results.append({'output': self.classifiers[1](x)})

            # 到exit2
            for i in range(self.exit1, self.exit2):
                x = self.blocks[i](x)
            results.append({'output': self.classifiers[2](x)})

            # 到最后
            for i in range(self.exit2, len(self.blocks)):
                x = self.blocks[i](x)
            results.append({'output': self.classifiers[3](x)})

            return results


def ResNet110_cifar_scaleFL(num_channels=3, num_classes=10, track_running_stats=True,
                             scale=1.0, exit0=27, exit1=34, exit2=43):
    """ResNet110 for ScaleFL"""
    return ResNet110_ScaleFL(num_channels, num_classes, track_running_stats, scale, exit0, exit1, exit2)


# 兼容性别名
ResNet18_cifar_scaleFL = ResNet110_cifar_scaleFL


if __name__ == '__main__':
    print("Testing ResNet110 ScaleFL with 4 levels:")
    print("=" * 60)

    # 测试4个scale值
    scales = [0.5, 0.63, 0.794, 1.0]

    for i, scale in enumerate(scales, 1):
        net = ResNet110_cifar_scaleFL(num_classes=10, scale=scale)
        params = sum(p.numel() for p in net.parameters())
        print(f"Level {i} (scale={scale:.3f}): {params:,} parameters")

    # 测试forward
    print("\nTesting forward pass with ee=4:")
    net = ResNet110_cifar_scaleFL(num_classes=10, scale=1.0)
    data = torch.randn(2, 3, 32, 32)
    results = net(data, ee=4)
    print(f"Number of outputs: {len(results)}")
    for i, result in enumerate(results, 1):
        print(f"  Output {i} shape: {result['output'].shape}")

    print("\n[SUCCESS] ResNet110 ScaleFL with 4 levels!")
