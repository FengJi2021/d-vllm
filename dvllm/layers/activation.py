import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # 忽略多余参数
        # Expected usage: activation(up, gate) -> return silu(gate) * up
        # Some call sites previously passed a single concatenated tensor; we
        # support both forms for robustness.
        if len(args) >= 1:
            up = x
            gate = args[0]
        else:
            # single tensor case: split into two halves
            up, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * up
