import torch
import torch.nn.functional as F


def ste_round(x: torch.Tensor) -> torch.Tensor:
    """Straight-Through Estimator (STE) with round to the closest int."""
    return (torch.round(x) - x).detach() + x

def quantize(weight: torch.Tensor, bits: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
    """
    Take in input:
        weight : a tensor of weighs
        bits : a tensor of bits
        exponent : a tensor of exponents
    Return:
        quantized weight
    """
    b = F.relu(bits)
    scale = 2.0 ** exponent
    lower_bound = (-2.0 ** (b - 1.0))
    upper_bound = ( 2.0 ** (b - 1.0)) - 1

    scaled_weight  = weight / scale
    clamped_weight = torch.clamp(
        scaled_weight, 
        min=lower_bound, 
        max=upper_bound
    )
    return scale * ste_round(clamped_weight)