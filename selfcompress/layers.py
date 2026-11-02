import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .quantize import quantize


class SelfCompressingLinear(nn.Module):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            prune_out: bool,
            init_num_bits: int = 32,
            init_exponent: int = -32,
    ):
        super().__init__()

        self.prune_out   = prune_out
        self.input_size  = input_size
        self.output_size = output_size

        std = 1.0 / math.sqrt(input_size)

        self.weight = nn.Parameter(torch.empty(output_size, input_size).uniform_(-std, std))
        self.bias   = nn.Parameter(torch.zeros(output_size))

        self.bits     = nn.Parameter(torch.full((output_size, 1), float(init_num_bits)))
        self.exponent = nn.Parameter(torch.full((output_size, 1), float(init_exponent)))

    def forward(self, x):
        quantized_weight = quantize(self.weight, self.bits, self.exponent)
        return F.linear(x, quantized_weight, self.bias)

    @torch.no_grad()
    def get_keep_and_cut_indices(self):
        """ We remove a weight if its number of bits is 0. If the bias is not zero, we add this bias to the next layer """
        if not self.prune_out:    # To not prune the final output layer (for classification for exemple -> fixe number of prediction)
            return torch.arange(self.output_size, device=self.weight.device)
        bits_at_zero = (F.relu(self.bits).squeeze(-1) == 0)
        cut_mask     = bits_at_zero
        keep_mask    = ~bits_at_zero
        return (keep_mask.nonzero(as_tuple=True)[0], cut_mask.nonzero(as_tuple=True)[0])

    def prune_output(self, keep_indices: torch.Tensor, optimizer):
        self.output_size = len(keep_indices)
        for param in self.parameters():
            param.data = torch.index_select(input=param.data, dim=0, index=keep_indices)
            param.grad = None
            optimizer.cut_off_parameter(id(param), keep_indices, dim=0)

    def prune_input(self, keep_indices: torch.Tensor, optimizer):
        # prune only the input of the weight
        self.input_size  = len(keep_indices)
        self.weight.data = torch.index_select(input=self.weight.data, dim=1, index=keep_indices)
        self.weight.grad = None
        optimizer.cut_off_parameter(id(self.weight), keep_indices, dim=1)

    def get_num_bits(self):
        return (F.relu(self.bits) * self.weight.shape[1]).sum()  # self.weight.shape[1] -> num of weights by nurone

    @torch.no_grad()
    def count_num_bits(self):
        return self.get_num_bits().item()

    @torch.no_grad()
    def count_num_neurones(self):
        return self.weight.shape[0]