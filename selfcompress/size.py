import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import SelfCompressingLinear


FLOAT_BITS     = 32   # a bias, a normalization parameter or an exponent is stored as a float32
BIT_WIDTH_BITS = 8    # the bit-width of a neuron is stored as one byte


@torch.no_grad()
def stored_bit_width(bits: torch.Tensor) -> torch.Tensor:
    """
    Whole number of bits needed to store one weight of each neuron.

    The learned bit-width b is continuous, but a quantized weight can only take the integer levels
    round(-2^(b-1)) ... round(2^(b-1) - 1): storing one of them takes ceil(log2(number of levels)) bits.
    e.g. b = 2.71 -> levels -3 ... 2 -> 3 bits, b = 1.5 -> levels -1, 0 -> 1 bit, b = 0 -> level 0 -> 0 bit
    """
    b = F.relu(bits)
    lowest  = torch.round(-2.0 ** (b - 1.0))
    highest = torch.round( 2.0 ** (b - 1.0) - 1.0)
    num_levels = highest - lowest + 1.0
    return torch.ceil(torch.log2(num_levels) - 1e-6).clamp(min=0.0)


@torch.no_grad()
def model_size_bits(model: nn.Module) -> dict:
    """
    Bits needed to store the model as it is now, everything included:
        weight_bits   : the weights of every neuron with b > 0, on a whole number of bits (see stored_bit_width)
        overhead_bits : what is stored next to the weights
                            - one bit-width (a byte) and one exponent (a float32) per neuron with b > 0
                            - every other parameter as a float32 (biases, normalization layers)
        total_bits    : weight_bits + overhead_bits
    """
    weight_bits   = 0.0
    overhead_bits = 0.0
    counted = set()     # ids of the parameters counted in the loop: all the others are stored as float32

    for module in model.modules():
        if isinstance(module, SelfCompressingLinear):
            bit_width = stored_bit_width(module.bits).squeeze(-1)
            weight_bits   += (bit_width * module.weight.shape[1]).sum().item()
            overhead_bits += (bit_width > 0).sum().item() * (BIT_WIDTH_BITS + FLOAT_BITS)
            counted.update(id(p) for p in (module.weight, module.bits, module.exponent))

    overhead_bits += FLOAT_BITS * sum(p.numel() for p in model.parameters() if id(p) not in counted)

    return {"weight_bits": weight_bits, "overhead_bits": overhead_bits, "total_bits": weight_bits + overhead_bits}


@torch.no_grad()
def float32_size_bits(model: nn.Module) -> int:
    """
    Size of the same model without quantization: 32 bits per weight, bias and normalization parameter.
    Call it before any pruning to get the reference the compressed model is compared to.
    """
    quantization = set()
    for module in model.modules():
        if isinstance(module, SelfCompressingLinear):
            quantization.update(id(p) for p in (module.bits, module.exponent))
    return FLOAT_BITS * sum(p.numel() for p in model.parameters() if id(p) not in quantization)
