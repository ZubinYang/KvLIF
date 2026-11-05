import torch
import torch.nn as nn
from spikingjelly.activation_based import layer

__all__ = ['vggsnn']

class Layer(nn.Module):
    def __init__(self, in_plane, out_plane, kernel_size, stride, padding, neuron, **kwargs):
        super(Layer, self).__init__()
        self.fwd = nn.Sequential(
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding),
            nn.BatchNorm2d(out_plane)
        )
        self.act = neuron(**kwargs)

    def forward(self, x):
        x = self.fwd(x)
        x = self.act(x)
        return x


class VGGSNN(nn.Module):
    def __init__(self, neuron, num_classes=10, neuron_dropout=0.0, c_in=3, **kwargs):
        super(VGGSNN, self).__init__()
        pool = nn.Sequential(nn.AvgPool2d(2))
        self.features = nn.Sequential(
            Layer(c_in, 64, 3, 1, 1, neuron, **kwargs),
            Layer(64, 128, 3, 1, 1, neuron, **kwargs),
            pool,
            Layer(128, 256, 3, 1, 1, neuron, **kwargs),
            Layer(256, 256, 3, 1, 1, neuron, **kwargs),
            pool,
            Layer(256, 512, 3, 1, 1, neuron, **kwargs),
            Layer(512, 512, 3, 1, 1, neuron, **kwargs),
            pool,
            Layer(512, 512, 3, 1, 1, neuron, **kwargs),
            Layer(512, 512, 3, 1, 1, neuron, **kwargs),
            pool,
        )
        W = int(32 / 2 / 2 / 2 / 2)
        self.classifier = nn.Linear(512 * W * W, num_classes)
        self.drop = layer.Dropout(neuron_dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x = self.features(input)
        x = self.drop(torch.flatten(x, start_dim=-3, end_dim=-1))
        x = self.classifier(x)
        return x


def vggsnn(neuron: callable = None, num_classes=10, neuron_dropout=0.0, **kwargs):
    return VGGSNN(neuron=neuron, num_classes=num_classes, neuron_dropout=neuron_dropout, **kwargs)