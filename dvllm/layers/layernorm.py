import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x.to(orig_dtype) * self.weight

    def add_rms_forward(
        self, x: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 调整 residual 形状
        if residual.shape != x.shape:
            if residual.dim() == 2 and x.dim() == 3:
                # residual: (batch, hidden), x: (batch, seq, hidden)
                residual = residual.unsqueeze(1).expand(-1, x.size(1), -1)
            else:
                # 其他情况尝试 reshape
                residual = residual.view(x.shape)

        orig_dtype = x.dtype
        x = x.float() + residual.float()
        residual_out = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x.to(orig_dtype) * self.weight, residual_out

    @torch.compile
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
