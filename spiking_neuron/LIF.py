from typing import Callable
import torch
from spikingjelly.activation_based import neuron


class LIFNeuron(neuron.BaseNode):
    def __init__(self,
                 tau=2.,
                 v_threshold=1.,
                 v_reset=None,
                 surrogate_function: Callable = None,
                 detach_reset=False,
                 step_mode='s',
                 ):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode)
        self.tau = tau

    def neuronal_charge(self, x: torch.Tensor):
        self.v = (1 - 1 / self.tau) * self.v + x

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        self.v = self.v - spike_d * self.v_threshold

    def forward(self, x: torch.Tensor):
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

