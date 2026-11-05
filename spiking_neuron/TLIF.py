from typing import Callable
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron
from surrogate import Rectangle


class TLIFNeuron(neuron.BaseNode):
    """The KvLIF neuron described in the paper, retaining TLIFNeuron as the class name."""
    def __init__(self,
                 v_threshold=1.,
                 v_reset=None,
                 surrogate_function: Callable = None,
                 detach_reset=False,
                 step_mode='s',
                 ):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode)
        self.register_memory('a', 0.)

    def neuronal_charge(self, x: torch.Tensor):
        self.v = 0.5 * self.v + x

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        self.v = self.v - spike_d * self.v_threshold

    def forward(self, x: torch.Tensor):
        if isinstance(self.a, float):
            self.a = torch.zeros_like(x)
        x = x * (1 - 0.05 * torch.sigmoid(self.a))
        self.neuronal_charge(x)
        self.a = 0.8 * self.a + 0.1 * self.v  # beta: 0.3 for others, 0.1 for cifar100 and cifar10dvs
        spike = self.neuronal_fire()
        self.a = self.a + spike
        self.neuronal_reset(spike)
        self.v = self.v - spike * torch.sigmoid(self.a)
        return spike


class TLIFNeuron_S(neuron.BaseNode):
    def __init__(self,
                 v_threshold=1.,
                 v_reset=None,
                 surrogate_function: Callable = None,
                 detach_reset=False,
                 step_mode='s',
                 ):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode)
        self.register_memory('a', 0.)

    def neuronal_charge(self, x: torch.Tensor):
        self.v = 0.5 * self.v + x

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        self.v = self.v - spike_d * self.v_threshold

    def forward(self, x: torch.Tensor):
        if isinstance(self.a, float):
            self.a = torch.zeros_like(x)
        # x = x * (1 - 0.05 * torch.sigmoid(self.a))
        self.neuronal_charge(x)
        self.a = 0.8 * self.a + 0.1 * self.v  # beta: 0.3 for others, 0.1 for cifar100 and cifar10dvs
        spike = self.neuronal_fire()
        self.a = self.a + spike
        self.neuronal_reset(spike)
        self.v = self.v - spike * torch.sigmoid(self.a)
        return spike


class TLIFNeuron_T(neuron.BaseNode):
    def __init__(self,
                 v_threshold=1.,
                 v_reset=None,
                 surrogate_function: Callable = None,
                 detach_reset=False,
                 step_mode='s',
                 ):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode)
        self.register_memory('a', 0.)

    def neuronal_charge(self, x: torch.Tensor):
        self.v = 0.5 * self.v + x

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        self.v = self.v - spike_d * self.v_threshold

    def forward(self, x: torch.Tensor):
        if isinstance(self.a, float):
            self.a = torch.zeros_like(x)
        x = x * (1 - 0.05 * torch.sigmoid(self.a))
        self.neuronal_charge(x)
        self.a = 0.8 * self.a + 0.1 * self.v  # beta: 0.3 for others, 0.1 for cifar100 and cifar10dvs
        spike = self.neuronal_fire()
        self.a = self.a + spike
        self.neuronal_reset(spike)
        # self.v = self.v - spike * torch.sigmoid(self.a)
        return spike
