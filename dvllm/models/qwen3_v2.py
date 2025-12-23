from torch import nn
from torch import Tensor as T


class Qwen3Model(nn.Module):
    def __init__(self, *args, **kwargs):
        self._emb_tokens = ...
        self._final_norm = ...
        self._transformer_blocks = ...

    def forward(self, input_ids: T) -> T:
        x = self._emb_tokens(input_ids)
        if x.dim() == 2:
            # prevent batch size 1
            x = x.unsqueeze(0)

        for transformer_block in self._transformer_blocks:
            x = transformer_block(x)

        x = self._final_norm(x)
        return x
