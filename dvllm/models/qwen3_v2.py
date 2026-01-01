import logging
import math
import torch
from torch import nn
from torch import Tensor as T
import torch.nn.functional as F
from transformers import Qwen3Config
from typing import Optional

from dvllm.layers.activation import SiluAndMul
from dvllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from dvllm.layers.layernorm import RMSNorm
from dvllm.layers.rotary_embedding import get_rope

DEFAULT_EPS = 1e-6

logger = logging.getLogger(__name__)


class TransfomerBlock(nn.Module):
    def __init__(
        self, hidden_size, num_heads, num_kv_heads, head_dim, ff_hidden_size, scaling
    ):
        super().__init__()
        # set variables
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scaling = scaling

        # Multi-Group-Attention
        # q=2048, k=1024, v=1024
        # 2 query share one k, v
        qkv_output_size = 2048 + 1024 + 1024
        self.qkv_proj = nn.Linear(hidden_size, qkv_output_size, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # attention norm
        # TODO eps from config?
        self.attn_norm = RMSNorm(hidden_size, eps=DEFAULT_EPS)
        self.ffn_norm = RMSNorm(hidden_size, eps=DEFAULT_EPS)

        # q, k norm
        self.q_norm = RMSNorm(head_dim, eps=DEFAULT_EPS)
        self.k_norm = RMSNorm(head_dim, eps=DEFAULT_EPS)

        # FFN
        # gate/up -> activation -> down
        self.gate_proj = nn.Linear(hidden_size, ff_hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, ff_hidden_size, bias=False)
        self.down_proj = nn.Linear(ff_hidden_size, hidden_size, bias=False)
        self.ffn_activation = SiluAndMul()

        # debug flag
        self.debug = False

    def forward(
        self,
        positions: Optional[T],
        hidden_states: T,
        rope_fn=None,
        device_type=None,
        sdpa_scale=None,
        chunked_threshold=4096,
        chunk_size=1024,
        dump_path=None,
    ):
        B, L, C = hidden_states.shape

        # Attention
        x = hidden_states  # embeddings B, L, H
        # norm
        x_attn = self.attn_norm(x)  # B, L, H

        # qkv: by chunk, sdpa as default backend operato
        qkv: T = self.qkv_proj(x_attn)  # B, L, 4096
        q, k, v = qkv.split(
            [2048, 1024, 1024], dim=2
        )  # B, L, 2048 | B, L, 1024 | B, L, 1024

        # GQA
        assert (
            q.shape[-1] == self.num_heads * self.head_dim
        ), f"q shape {q.shape[-1]} not match {self.num_heads} * {self.head_dim}"
        assert (
            k.shape[-1] == self.num_kv_heads * self.head_dim
        ), f"k shape {k.shape[-1]} not match {self.num_kv_heads} * {self.head_dim}"
        assert (
            v.shape[-1] == self.num_kv_heads * self.head_dim
        ), f"v shape {v.shape[-1]} not match {self.num_kv_heads} * {self.head_dim}"

        q = q.view(B, L, self.num_heads, self.head_dim)
        k = k.view(B, L, self.num_kv_heads, self.head_dim)
        v = v.view(B, L, self.num_kv_heads, self.head_dim)

        # q, k norm
        q: T = self.q_norm(q)  # B, L, n_heads, head_dim
        k: T = self.k_norm(k)  # B, L, n_heads, head_dim

        # multi head attention
        q = q.transpose(1, 2)  # B, n_heads, L, head_dim
        k = k.transpose(1, 2)  # B, n_kv_heads, L, head_dim
        v = v.transpose(1, 2)  # B, n_kv_heads, L, head_dim

        # rotary embedding for q, k
        logger.debug(f"q shape: {q.shape}, k shape: {k.shape}")
        q, k = rope_fn(positions, q, k)

        # sdpa backbone
        def attention_range_checker(mean, std, min, max) -> bool:
            def is_norm(value, lower, upper) -> bool:
                return lower <= value <= upper

            mean_c = is_norm(mean, 0, 0.5)
            std_c = is_norm(std, 0.5, 2)
            min_c = is_norm(min, -10, 10)
            max_c = is_norm(max, -10, 10)

            # log
            logger.debug(
                f"[DEBUG] Attention Range Check: Mean {'Norm' if mean_c else 'abnormal'}: {mean} | "
                f"Std {'Norm' if std_c else 'abnormal'}: {std} | "
                f"Min {'Norm' if min_c else 'abnormal'}: {min} | "
                f"Max {'Norm' if max_c else 'abnormal'}: {max}"
            )

            return all([mean_c, std_c, min_c, max_c])

        def sdpa(q: T, k: T, v: T) -> T:
            scores = torch.einsum("bhld,bhmd->bhlm", q, k) * (
                sdpa_scale if sdpa_scale else 1.0 / math.sqrt(self.head_dim)
            )
            with torch.no_grad():
                q_stats = (
                    q.mean().item(),
                    q.std().item(),
                    q.min().item(),
                    q.max().item(),
                )
                k_stats = (
                    k.mean().item(),
                    k.std().item(),
                    k.min().item(),
                    k.max().item(),
                )
                v_stats = (
                    v.mean().item(),
                    v.std().item(),
                    v.min().item(),
                    v.max().item(),
                )
                sc_stats = (
                    scores.mean().item(),
                    scores.std().item(),
                    scores.min().item(),
                    scores.max().item(),
                )

                for name, stats in zip(
                    ["q", "k", "v", "sc"], [q_stats, k_stats, v_stats, sc_stats]
                ):
                    is_good = attention_range_checker(*stats)
                    if not is_good:
                        logger.warning(f"[WARNING] {name} attention range is abnormal")

                logger.debug(
                    f"[DEBUG] q mean,std,min,max={q_stats} | k={k_stats} | v={v_stats} | scores mean,std,min,max={sc_stats}"
                )
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
                scale=sdpa_scale,
            )

            return out

        # chunked attention
        factor = self.num_heads // self.num_kv_heads
        device = q.device.type
        attn_out: Optional[T] = None
        if device == "mps" and L > chunked_threshold:
            out_chunks = []
            for start in range(0, L, chunk_size):
                end = min(L, start + chunk_size)
                cur_chunk_size = end - start
                # chunk q, k, v
                q_chunk = q[:, :, start:end, :]
                k_chunk = k[:, :, start:end, :]
                v_chunk = v[:, :, start:end, :]
                # broadcast head in GQA
                if factor > 1:
                    k_chunk = (
                        k_chunk[:, :, None, :, :]
                        .expand(
                            B, self.num_kv_heads, factor, cur_chunk_size, self.head_dim
                        )
                        .reshape(
                            B, self.num_kv_heads * factor, cur_chunk_size, self.head_dim
                        )
                    )
                    v_chunk = (
                        v_chunk[:, :, None, :, :]
                        .expand(
                            B, self.num_kv_heads, factor, cur_chunk_size, self.head_dim
                        )
                        .reshape(
                            B, self.num_kv_heads * factor, cur_chunk_size, self.head_dim
                        )
                    )

                out_chunks.append(sdpa(q_chunk, k_chunk, v_chunk))
            out_t = torch.cat(out_chunks, dim=2)  # cat on L dimension
            out_t = out_t.transpose(1, 2).contiguous()  # B, L, n_heads, head_dim
            attn_out = out_t.view(B, L, self.num_heads * self.head_dim)
            attn_out = self.o_proj(attn_out)
        else:
            if factor > 1:
                k = (
                    k[:, :, None, :, :]
                    .expand(B, self.num_kv_heads, factor, L, self.head_dim)
                    .reshape(B, self.num_kv_heads * factor, L, self.head_dim)
                )
                v = (
                    v[:, :, None, :, :]
                    .expand(B, self.num_kv_heads, factor, L, self.head_dim)
                    .reshape(B, self.num_kv_heads * factor, L, self.head_dim)
                )
            out_t = sdpa(q, k, v)
            out_t = out_t.transpose(1, 2).contiguous()  # B, L, n_heads, head_dim
            attn_out = out_t.view(B, L, self.num_heads * self.head_dim)
            attn_out = self.o_proj(attn_out)

        # residual layer
        x = x + attn_out

        # ffn: norm, gate, up, down and residual
        x_ffn = self.ffn_norm(x)
        gate = self.gate_proj(x_ffn)
        up = self.up_proj(x_ffn)

        activated_ffn = self.ffn_activation(up, gate)
        down = self.down_proj(activated_ffn)

        x = x + down

        # return
        return x


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config

        # get variable from config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_layers = config.num_hidden_layers
        self.head_dim = config.head_dim

        # embedding
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, self.hidden_size)

        # position embedding
        self.rope_fn = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
        )

        # transformer layers
        ff_hidden = config.intermediate_size
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = TransfomerBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                ff_hidden_size=ff_hidden,
                scaling=(1.0 / math.sqrt(self.head_dim)),
            )
            # set layer id
            layer.layer_id = i

            self.layers.append(layer)

        # final norm
        self.final_norm = RMSNorm(hidden_size=self.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: T, positions: T = None, dump_path: str = None) -> T:
        x: T = self.embed_tokens(input_ids)
        if x.dim() == 2:
            # prevent batch size 1
            x = x.unsqueeze(0)

        # inheritate device type from input ids
        device_type = x.device.type

        for transformer_block in self.layers:
            x = transformer_block(
                positions,
                x,
                rope_fn=self.rope_fn,
                device_type=device_type,
                sdpa_scale=(1.0 / math.sqrt(self.head_dim)),
                dump_path=dump_path,
            )

        x = self.final_norm(x)
        return x


class Qwen3ForCausalLM_V2(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.model = Qwen3Model(config)

        # require by loader
        self.layers = self.model.layers

        self.norm = self.model.final_norm
        # use vocab_size * hidden_size to do chunk calculation easier
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

        if getattr(config, "tie_word_embeddings", True):
            try:
                self.lm_head.weight = self.model.embed_tokens.weight
            except AttributeError:
                import logging

                logger.info("embed_tokens.weight not found, skipping tie_weights")

    def forward(self, input_ids: T, positions: T = None, dump_path: str = None) -> T:
        return self.model(input_ids, positions, dump_path=dump_path)

    def compute_logits(
        self,
        hidden_states: T,
        batch_chunk: int = 1,
        seq_chunk: int = 32,
        vocab_chunk: int = 512,
    ) -> T:
        """
        MPS: calculate once on GPU
        CUDA: calcuate by chunk
        """
        B, L, H = hidden_states.shape
        device = hidden_states.device

        if device.type == "mps" or device.type == "cpu":
            return F.linear(hidden_states, self.lm_head.weight)

        logits_chunks = []
        weight_cpu = self.lm_head.weight.cpu()

        for b_start in range(0, B, batch_chunk):
            b_end = min(B, b_start + batch_chunk)
            b_hidden = hidden_states[b_start:b_end, :, :]

            b_logits = []

            for seq_start in range(0, L, seq_chunk):
                seq_end = min(0, L, seq_chunk)
                seq_hidden = b_hidden[:, seq_start:seq_end, :].cpu()

                seq_logits_chunks = []

                for vocab_start in range(0, weight_cpu.size(0), vocab_chunk):
                    vocab_end = min(weight_cpu.size(0), vocab_start + vocab_chunk)
                    w_chunk = weight_cpu[vocab_start:vocab_end, :]  # vocab * hidden
                    logits_chunk = F.linear(seq_hidden, w_chunk)  # B_C, L_C, V_C
                    seq_logits_chunks.append(logits_chunk)

                # cat sequence chunk
                b_logits.append(torch.cat(seq_logits_chunks), dim=2)  # B_C, L_C, V

            logits_chunks.append(torch.cat(b_logits, dim=1))  # B_C, L, V

        return torch.cat(logits_chunk, dim=0)  # B, L, V
