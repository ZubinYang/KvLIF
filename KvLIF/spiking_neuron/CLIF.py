from typing import Callable
import torch
from spikingjelly.activation_based.neuron import LIFNode as LIFNode_sj


class CLIFNeuron(LIFNode_sj):
    def __init__(self,
                 tau: float = 2.,
                 decay_input: bool = False,
                 v_threshold: float = 1.,
                 v_reset: float = None,
                 surrogate_function: Callable = None,
                 detach_reset: bool = False,
                 **kwargs):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, **kwargs)
        self.register_memory('m', 0.)  # Complementary memory

    def forward(self, x: torch.Tensor):
        self.neuronal_charge(x)  # LIF charging
        self.m = self.m * torch.sigmoid(self.v / self.tau)  # Forming
        spike = self.neuronal_fire()  # LIF fire
        self.m += spike  # Strengthen
        self.neuronal_reset(spike)  # LIF reset
        self.v = self.v - spike * torch.sigmoid(self.m)  # Reset
        return spike

    def neuronal_charge(self, x: torch.Tensor):
        self._charging_v(x)

    def neuronal_reset(self, spike: torch.Tensor):
        self._reset(spike)

    def _charging_v(self, x: torch.Tensor):
        if self.decay_input:
            x = x / self.tau

        if self.v_reset is None or self.v_reset == 0:
            if type(self.v) is float:
                self.v = x
            else:
                self.v = self.v * (1 - 1. / self.tau) + x
        else:
            if type(self.v) is float:
                self.v = self.v_reset * (1 - 1. / self.tau) + self.v_reset / self.tau + x
            else:
                self.v = self.v * (1 - 1. / self.tau) + self.v_reset / self.tau + x

    def _reset(self, spike):
        if self.v_reset is None:
            # soft reset
            self.v = self.v - spike * self.v_threshold
        else:
            # hard reset
            self.v = (1. - spike) * self.v + spike * self.v_reset