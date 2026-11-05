import torch
import torch.nn as nn

__all__ = ['spiking_resnet19']


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout=0.0, neuron: callable = None, **kwargs):
        super(BasicBlock, self).__init__()
        if neuron is None:
            neuron = nn.ReLU
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = neuron(**kwargs)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = neuron(**kwargs)
        self.dropout = nn.Dropout(dropout)

        self.stride = stride
        self.downsample = downsample
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual

        out = self.relu2(out)

        return out


class ResNet19(nn.Module):
    def __init__(self, block, num_blocks, c_in=3, num_classes=10, dropout=0.0, neuron: callable = None, **kwargs):
        super(ResNet19, self).__init__()
        if neuron is None:
            neuron = nn.ReLU
        inplanes = 128
        self.inplanes = inplanes
        self.conv1 = nn.Conv2d(c_in, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = neuron(**kwargs)

        self.layer1 = self._make_layer(block, inplanes, num_blocks[0], dropout, neuron, **kwargs)
        self.layer2 = self._make_layer(block, inplanes * 2, num_blocks[1], dropout, neuron, stride=2, **kwargs)
        self.layer3 = self._make_layer(block, inplanes * 4, num_blocks[2], dropout, neuron, stride=2, **kwargs)
        self.dropout = nn.Dropout(dropout)
        self.flat = nn.Flatten()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(inplanes * 4 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, dropout, neuron, stride=1, **kwargs):
        layers = []
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers.append(block(self.inplanes, planes, stride, downsample, dropout, neuron=neuron, **kwargs))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, neuron=neuron, **kwargs))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = self.flat(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out


def spiking_resnet19(neuron: callable = None, num_classes=10, neuron_dropout=0.0, **kwargs):
    return ResNet19(BasicBlock, [3, 3, 2], num_classes=num_classes, dropout=neuron_dropout, neuron=neuron, **kwargs)