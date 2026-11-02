import torch
import torch.nn as nn
from typing import List, Type

from .layers import SelfCompressingLinear
from .quantize import quantize


class SelfCompressingMLP(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_shape: List,
            init_num_bits: int,
            init_exponent: int,
            activation_layer: Type[nn.Module] = nn.ReLU
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activation_layer = activation_layer()
        current_input_size = input_size
        for hidden_size in hidden_shape[:-1]:
            self.layers.append(SelfCompressingLinear(current_input_size, hidden_size, True, init_num_bits, init_exponent))
            self.layers.append(activation_layer())
            current_input_size = hidden_size
        self.layers.append(SelfCompressingLinear(current_input_size, hidden_shape[-1], False, init_num_bits, init_exponent))

    @property
    def linear_layers(self):
        """Return only the linear layers skiping the activation functions"""
        return [layer for layer in self.layers if isinstance(layer, SelfCompressingLinear)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def keep_and_cut_indices(self):
        indices = []
        for layer in self.linear_layers:
            indices.append(layer.get_keep_and_cut_indices())
        return indices

    def prune_MLP(self, optimizer, keep_and_cut_indices: List):
        linear_layers = self.linear_layers
        for l, (keep_indices, cut_indices) in enumerate(keep_and_cut_indices[:-1]):
            if len(cut_indices) == 0:
                continue
            current_layer, next_layer = linear_layers[l], linear_layers[l+1]

            # --- compute the bias (from dead neurones i.e. bits = 0) than we need to add to the next layer ---
            removed_bias = self.activation_layer(current_layer.bias.data[cut_indices])
            quantize_next_weight = quantize(next_layer.weight.data, next_layer.bits.data, next_layer.exponent.data)
            next_layer.bias.data += quantize_next_weight[:, cut_indices] @ removed_bias

            # --- prune output current layer and intput next layer ---
            current_layer.prune_output(keep_indices, optimizer)
            next_layer.prune_input(keep_indices, optimizer)

    def get_tot_num_bits(self):
        tot_bits = 0
        for layer in self.linear_layers:
            tot_bits += layer.get_num_bits()
        return tot_bits

    def get_num_param(self):
            return sum(layer.weight.numel() for layer in self.linear_layers)

    @torch.no_grad()
    def count_tot_num_bits(self):
        tot_bits = 0
        for layer in self.linear_layers:
            tot_bits += layer.count_num_bits()
        return tot_bits

    @torch.no_grad()
    def count_num_neurones_by_layer(self):
        return [layer.count_num_neurones() for layer in self.linear_layers]