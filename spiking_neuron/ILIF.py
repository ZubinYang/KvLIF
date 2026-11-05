from typing import Callable
import torch
from spikingjelly.activation_based.neuron import LIFNode as LIFNode_sj


class ILIFNeuron(LIFNode_sj):
    def __init__(self,
                 tau: float = 2.0,
                 decay_input: bool = False,
                 v_threshold: float = 1.,
                 v_reset: float = None,
                 surrogate_function: Callable = None,
                 detach_reset: bool = False,
                 **kwargs):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, **kwargs)
        self.register_memory('inhibitory_memory', 0.)  # Inhibitory memory
        self.register_memory('prev_input', 0.)
        self.register_memory('prev_spike', 0.)
        self.register_memory('inhibition', 0.)

    def forward(self, x: torch.Tensor):
        if isinstance(self.prev_input, float):
            self.prev_input = torch.zeros_like(x)
            self.prev_spike = torch.zeros_like(x)
            self.inhibition = torch.zeros_like(x)
        current_input = x
        self.inhibition = 0.05 * (self.inhibition + self.prev_input * self.prev_spike)  # for dvs_gesture
        # self.inhibition = 0.03 * (self.inhibition + self.prev_input * self.prev_spike)
        self.neuronal_charge(x-torch.clamp(self.inhibition, min=0))  # LIF charging
        self.prev_input = current_input
        spike = self.neuronal_fire()  # LIF fire
        self.prev_spike = spike
        self.neuronal_reset(spike)  # LIF reset
        self.inhibitory_memory = 1*(self.inhibitory_memory + spike * self.v)
        self.v = self.v - spike * torch.sigmoid(self.inhibitory_memory)  # Reset
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