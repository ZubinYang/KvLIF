from typing import Callable
import torch
import torch.nn as nn
from spikingjelly.activation_based.neuron import LIFNode as LIFNode_sj


class XLIFNeuron(LIFNode_sj):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = None, surrogate_function: Callable = None,
                 detach_reset: bool = False, **kwargs):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, **kwargs)

        # ADDED Parameters
        self.a = nn.Parameter(torch.tensor(.0), requires_grad=True)
        self.c = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        # ADDED Adaptive Reset V
        self.register_memory('m', 0)

    def forward(self, x: torch.Tensor):

        if type(self.m) is not torch.Tensor:
            self.m = torch.zeros_like(x)
        # Leaky and Integrate
        self.v = self.v * 0.5 + x
        # Spike
        spike = self.surrogate_function(self.v - self.v_threshold - self.a.tanh() * x.tanh())
        # ADD
        # reset voltage leaky
        self.m = self.m.relu() * torch.sigmoid(self.c * x) + -(-self.m).relu() * (1 - torch.sigmoid(self.c * x))
        # reset voltage integrate
        self.m += spike * torch.sigmoid(x)
        self.m -= (1 - spike) * torch.sigmoid(x)
        # Adaptive reset
        self.v = self.v - spike * (self.v_threshold + torch.sigmoid(self.m) + self.a.tanh() * x.tanh())

        return spike