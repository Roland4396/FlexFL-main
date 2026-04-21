"""
SmartFL ResNet110 - 保持原始结构
- 层数：ResNet110 [18, 18, 18] blocks（与SmartFL完全一致）
- 通道数：16, 32, 64（与SmartFL完全一致）
- FlexFL适配：将3个stage展开成5层结构，接收5个rate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """SmartFL BasicBlock - 适配FlexFL的ReLU层检测"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, track_running_stats=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=track_running_stats)
        self.relu1 = nn.ReLU(inplace=True)  # 改为nn.ReLU层
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=track_running_stats)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes, track_running_stats=track_running_stats)
            )
        self.relu2 = nn.ReLU(inplace=True)  # 改为nn.ReLU层

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out)
        return out


class ResNet110(nn.Module):
    """
    SmartFL ResNet110 - 展开成FlexFL的5层结构

    SmartFL原始：3 stages [18, 18, 18]，通道 16→32→64
    FlexFL展开：5 layers，通道对应5个rate
      - 初始conv: 16 * rate[0]
      - layer1: 16 * rate[1]，9 blocks (拆分stage1前半)
      - layer2: 16 * rate[2]，9 blocks (拆分stage1后半)
      - layer3: 32 * rate[3]，18 blocks (stage2)
      - layer4: 64 * rate[4]，18 blocks (stage3)
    """
    def __init__(self, num_channels=3, num_classes=10, track_running_stats=True, rate=None, dataset='cifar'):
        super(ResNet110, self).__init__()

        if rate is None:
            rate = [1.0, 1.0, 1.0, 1.0, 1.0]

        self.dataset = dataset
        self.inchannel = int(16 * rate[0])

        # FlexFL的features结构：初始conv + 4个stage
        self.features = nn.Sequential(
            # 初始conv: 16 * rate[0]
            nn.Sequential(
                nn.Conv2d(num_channels, self.inchannel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.inchannel, track_running_stats=track_running_stats),
                nn.ReLU()
            ),

            # Layer1: 16 * rate[1]，9 blocks (stage1前半)
            self._make_layer(BasicBlock, int(16 * rate[1]), 9, stride=1, track_running_stats=track_running_stats),

            # Layer2: 16 * rate[2]，9 blocks (stage1后半)
            self._make_layer(BasicBlock, int(16 * rate[2]), 9, stride=1, track_running_stats=track_running_stats),

            # Layer3: 32 * rate[3]，18 blocks (stage2)
            self._make_layer(BasicBlock, int(32 * rate[3]), 18, stride=2, track_running_stats=track_running_stats),

            # Layer4: 64 * rate[4]，18 blocks (stage3)
            self._make_layer(BasicBlock, int(64 * rate[4]), 18, stride=2, track_running_stats=track_running_stats)
        )

        # Classifier
        self.classifier = nn.Linear(int(64 * rate[4]) * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, track_running_stats):
        """Create a layer with multiple blocks"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, planes, stride, track_running_stats))
            self.inchannel = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass - FlexFL格式"""
        out = self.features(x)
        result = {'representation': out}

        # Global average pooling and classification
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        result['output'] = out

        return result


def ResNet18_cifar(num_channels=3, num_classes=10, track_running_stats=True, rate=None):
    """SmartFL ResNet110 for FlexFL"""
    if rate is None:
        rate = [1.0] * 5
    return ResNet110(num_channels, num_classes, track_running_stats, rate, 'cifar')


if __name__ == '__main__':
    print("Testing SmartFL ResNet110 (5-layer expansion):")
    print("=" * 60)

    # Test with FlexFL's 5-rate format
    net_full = ResNet18_cifar(num_classes=10, track_running_stats=True, rate=[1.0] * 5)
    full_params = sum([param.nelement() for param in net_full.parameters()])
    print(f"Full model: {full_params:,} parameters")
    print(f"Features layers: {len(net_full.features)}")

    net_half = ResNet18_cifar(num_classes=10, track_running_stats=True, rate=[0.5] * 5)
    half_params = sum([param.nelement() for param in net_half.parameters()])
    print(f"1/2 model: {half_params:,} parameters ({half_params/full_params:.2%})")

    # Test forward
    data = torch.randn(2, 3, 32, 32)
    result = net_full(data)
    print(f"\nOutput shape: {result['output'].shape}")
    print(f"Representation shape: {result['representation'].shape}")
    print("\n[SUCCESS] SmartFL ResNet110 with 5-rate FlexFL structure!")
