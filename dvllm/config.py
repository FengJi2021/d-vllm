import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    device_type: str = "auto"  # "cuda" | "mps" | "cpu" | "auto"

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(
            self.max_model_len, self.hf_config.max_position_embeddings
        )

        # 针对 MPS (Apple Metal) 的内存优化
        if self.device_type == "mps":
            # MPS 内存受限，减小批处理大小和序列长度
            self.max_num_batched_tokens = min(self.max_num_batched_tokens, 2048)  # 减半
            self.max_num_seqs = min(self.max_num_seqs, 16)  # 大幅减小
            self.max_model_len = min(self.max_model_len, 2048)  # 限制上下文长度

        assert self.max_num_batched_tokens >= self.max_model_len
