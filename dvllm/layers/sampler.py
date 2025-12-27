import torch
from torch import nn
import logging

logger = logging.getLogger(__name__)


class Sampler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, logits: torch.Tensor, temperatures: torch.Tensor, seq_chunk: int = 32
    ):
        """
        logits: [B, L, V]，代表每个序列的每个位置的预测分布
        temperatures: [B] 或 [B, 1]

        返回: [B, L] 张量，每个位置采样一个 token
        """
        B, L, V = logits.shape

        # 处理 temperatures 形状
        if temperatures is None:
            temperatures = torch.ones(B, device=logits.device)

        if temperatures.dim() == 1:
            temperatures = temperatures.unsqueeze(-1)  # [B] -> [B, 1]

        # 确保 temperatures 能够与 logits 正确广播
        # logits: [B, L, V], temperatures: [B, 1] -> 需要广播到 [B, L, V]
        # 但如果 B != logits.shape[0]，则 flatten
        if temperatures.shape[0] != logits.shape[0]:
            # temperatures 的 batch 与 logits 的 batch 不匹配
            # 这种情况下可能所有序列被 flatten 成一个更大的 batch
            # 需要根据 temperatures 的长度重新处理
            logger.warning(
                f"temperature B ({temperatures.shape[0]}) != logits B ({logits.shape[0]}), reshaping"
            )
            # 假设 logits 已经 flatten 成 [1, total_len, V]，temperatures 是每个序列一个
            # 我们需要展开 logits 或者复制 temperatures
            # 为简单起见，假设所有序列用同一个 temperature（第一个）
            temperatures = temperatures[:1]

        logits_float = logits.float()  # [B, L, V]

        # 应用 temperature scaling: logits / T
        # 确保广播兼容性：[B, L, V] / [B, 1] -> [B, L, V]
        scaled_logits = logits_float / temperatures.reshape(
            -1, 1, 1
        )  # [B, 1, 1] 显式广播

        # 计算概率
        probs = torch.softmax(scaled_logits, dim=-1)  # [B, L, V]

        # Gumbel-max sampling：采样所有位置
        gumbel_noise = torch.empty_like(probs).exponential_().clamp_min_(1e-10)
        sampled = (probs / gumbel_noise).argmax(dim=-1)  # [B, L]

        logger.debug(
            f"Sampler: logits shape={logits.shape}, sampled shape={sampled.shape}"
        )
        logger.debug(f"Sampler: sampled tokens (first seq): {sampled[0]}")

        return sampled  # [B, L]
