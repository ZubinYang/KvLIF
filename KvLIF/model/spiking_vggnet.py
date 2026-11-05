import torch.nn as nn
from spikingjelly.activation_based import layer

__all__ = ['spiking_vggnet11', 'spiking_vggnet13',
           'spiking_vggnet16', 'spiking_vggnet19']

cfg = {
    'VGG11': [
        [64, 'M'],
        [128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512, 'M']
    ],
    'VGG13': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512, 'M']
    ],
    'VGG16': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 'M'],
        [512, 512, 512, 'M'],
        [512, 512, 512, 'M']
    ],
    'VGG19': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 256, 'M'],
        [512, 512, 512, 512, 'M'],
        [512, 512, 512, 512, 'M']
    ]
}


class SpikingVGGBN(nn.Module):
    def __init__(self, dataset, vgg_name, neuron: callable = None, dropout=0.0, num_classes=10, c_in=3,**kwargs):
        super(SpikingVGGBN, self).__init__()
        self.whether_bias = True
        self.init_channels = c_in
        self.layer1 = self._make_layers(cfg[vgg_name][0], dropout, neuron, **kwargs)
        self.layer2 = self._make_layers(cfg[vgg_name][1], dropout, neuron, **kwargs)
        self.layer3 = self._make_layers(cfg[vgg_name][2], dropout, neuron, **kwargs)
        self.layer4 = self._make_layers(cfg[vgg_name][3], dropout, neuron, **kwargs)
        self.layer5 = self._make_layers(cfg[vgg_name][4], dropout, neuron, **kwargs)

        if dataset == 'cifar10_dvs':
            self.pool = nn.AdaptiveAvgPool2d((7, 7))
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512 * 7 * 7, num_classes),
            )
        elif dataset == 'tiny_imagenet':
            self.pool = nn.Identity()
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512 * 2 * 2, num_classes),
            )
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, num_classes),
            )

    def _make_layers(self, cfg, dropout, neuron, **kwargs):
        layers = []
        for v in cfg:
            if v == 'M':
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(self.init_channels, v, kernel_size=3, padding=1, bias=self.whether_bias))
                layers.append(nn.BatchNorm2d(v))
                layers.append(neuron(**kwargs))
                layers.append(layer.Dropout(dropout))
                self.init_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.pool(out)
        out = self.classifier(out)
        return out


def spiking_vggnet11(dataset, neuron: callable = None, num_classes=10, neuron_dropout=0.0, **kwargs):
    return SpikingVGGBN(dataset, 'VGG11', neuron=neuron, dropout=neuron_dropout, num_classes=num_classes, **kwargs)


def spiking_vggnet13(dataset, neuron: callable = None, num_classes=10, neuron_dropout=0.0, **kwargs):
    return SpikingVGGBN(dataset, 'VGG13', neuron=neuron, dropout=neuron_dropout, num_classes=num_classes, **kwargs)


def spiking_vggnet16(dataset, neuron: callable = None, num_classes=10, neuron_dropout=0.0, **kwargs):
    return SpikingVGGBN(dataset, 'VGG16', neuron=neuron, dropout=neuron_dropout, num_classes=num_classes, **kwargs)


def spiking_vggnet19(dataset, neuron: callable = None, num_classes=10, neuron_dropout=0.0, **kwargs):
    return SpikingVGGBN(dataset, 'VGG19', neuron=neuron, dropout=neuron_dropout, num_classes=num_classes, **kwargs)