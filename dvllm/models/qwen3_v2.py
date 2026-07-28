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
        qkv_output_size = (self.num_heads + self.num_kv_heads + self.num_kv_heads) * self.head_dim
        
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
        self.debug = True

    def _attention_range_checker(self, mean, std, min_val, max_val) -> bool:
        def is_norm(value, lower, upper) -> bool:
            return lower <= value <= upper

        mean_c = is_norm(mean, 0, 0.5)
        std_c = is_norm(std, 0.5, 2)
        min_c = is_norm(min_val, -10, 10)
        max_c = is_norm(max_val, -10, 10)

        # log
        # logger.debug(
        #     f"[DEBUG] Attention Range Check: Mean {'Norm' if mean_c else 'abnormal'}: {mean} | "
        #     f"Std {'Norm' if std_c else 'abnormal'}: {std} | "
        #     f"Min {'Norm' if min_c else 'abnormal'}: {min_val} | "
        #     f"Max {'Norm' if max_c else 'abnormal'}: {max_val}"
        # )

        return all([mean_c, std_c, min_c, max_c])

    def _debug_sdpa(self, q, k, v, scale):
        # Calculate scores for debug stats (expensive!)
        scores = torch.einsum("bhld,bhmd->bhlm", q, k) * scale
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
                is_good = self._attention_range_checker(*stats)
                if not is_good:
                    logger.warning(f"[WARNING] {name} attention range is abnormal")

            # logger.debug(
            #     f"[DEBUG] q stats={q_stats} | k={k_stats} | v={v_stats} | scores={sc_stats}"
            # )

    def _sdpa(self, q: T, k: T, v: T, scale: float, mask: Optional[T] = None) -> T:
        if self.debug:
            self._debug_sdpa(q, k, v, scale)

        # If mask is provided, we must set is_causal=False
        use_causal = True if mask is None else False

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=use_causal,
            scale=scale,
        )
        return out

    def _chunked_attention(self, q: T, k: T, v: T, chunk_size: int, sdpa_scale: float):
        B, n_heads, L, head_dim = q.shape
        num_kv_heads = k.shape[1]
        factor = n_heads // num_kv_heads

        out_chunks = []
        for start in range(0, L, chunk_size):
            end = min(L, start + chunk_size)
            cur_chunk_size = end - start

            # chunk q (current window)
            q_chunk = q[:, :, start:end, :]

            # k and v must include history for causal attention
            k_chunk = k[:, :, 0:end, :]
            v_chunk = v[:, :, 0:end, :]

            # broadcast head in GQA
            if factor > 1:
                k_chunk = (
                    k_chunk[:, :, None, :, :]
                    .expand(B, num_kv_heads, factor, end, head_dim)
                    .reshape(B, num_kv_heads * factor, end, head_dim)
                )
                v_chunk = (
                    v_chunk[:, :, None, :, :]
                    .expand(B, num_kv_heads, factor, end, head_dim)
                    .reshape(B, num_kv_heads * factor, end, head_dim)
                )

            # Create Custom Mask
            attn_mask = torch.zeros(
                (cur_chunk_size, end), device=q.device, dtype=q.dtype
            )
            causal_block = torch.ones(
                (cur_chunk_size, cur_chunk_size), device=q.device, dtype=torch.bool
            ).triu(1)
            attn_mask[:, start:end].masked_fill_(causal_block, float("-inf"))

            out_chunks.append(
                self._sdpa(q_chunk, k_chunk, v_chunk, scale=sdpa_scale, mask=attn_mask)
            )

        out_t = torch.cat(out_chunks, dim=2)  # cat on L dimension
        out_t = out_t.transpose(1, 2).contiguous()  # B, L, n_heads, head_dim
        attn_out = out_t.view(B, L, n_heads * head_dim)
        return self.o_proj(attn_out)

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

        # --- DEBUG CHECK 1: Input Hidden States ---
        if self.debug:
            if torch.isnan(x).any():
                logger.error("!!! NaN detected in Input Hidden States !!!")
            if torch.isinf(x).any():
                logger.error("!!! Inf detected in Input Hidden States !!!")

        # norm
        x_attn = self.attn_norm(x)  # B, L, H

        # qkv: by chunk, sdpa as default backend operato
        qkv: T = self.qkv_proj(x_attn)  # B, L, 4096
        
        # MQA
        q, k, v = qkv.split(
            [
                self.num_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
            ],
            dim=2,
        )  # B, L, 2048 | B, L, 1024 | B, L, 1024

        # --- DEBUG CHECK 2: QKV Projection Output ---
        if self.debug:
            for name, tensor in [("q", q), ("k", k), ("v", v)]:
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    logger.error(f"!!! NaN/Inf detected in {name} (projection) !!!")

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
        q, k = rope_fn(positions, q, k)

        # Determine scale
        real_sdpa_scale = sdpa_scale if sdpa_scale else 1.0 / math.sqrt(self.head_dim)

        # chunked attention
        device = q.device.type
        attn_out: Optional[T] = None
        if device == "mps" and L > chunked_threshold:
            attn_out = self._chunked_attention(q, k, v, chunk_size, real_sdpa_scale)
        else:
            factor = self.num_heads // self.num_kv_heads
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
            out_t = self._sdpa(q, k, v, scale=real_sdpa_scale)
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
                logger.info("embed_tokens.weight not found, skipping tie_weights")

    def forward(self, input_ids: T, positions: T = None, dump_path: str = None) -> T:
        return self.model(input_ids, positions, dump_path=dump_path)

    def _chunked_compute_logits(
        self, hidden_states: T, batch_chunk, seq_chunk, vocab_chunk
    ) -> T:
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(1)

        B, L, H = hidden_states.shape
        logits_chunks = []

        for b_start in range(0, B, batch_chunk):
            b_end = min(B, b_start + batch_chunk)
            b_hidden = hidden_states[b_start:b_end, :, :]

            b_logits = []

            for seq_start in range(0, L, seq_chunk):
                seq_end = min(L, seq_start + seq_chunk)
                seq_hidden = b_hidden[:, seq_start:seq_end, :]

                seq_logits_chunks = []

                for vocab_start in range(0, self.lm_head.weight.size(0), vocab_chunk):
                    vocab_end = min(
                        self.lm_head.weight.size(0), vocab_start + vocab_chunk
                    )
                    w_chunk = self.lm_head.weight[
                        vocab_start:vocab_end, :
                    ]  # vocab * hidden
                    logits_chunk = F.linear(seq_hidden, w_chunk)  # B_C, L_C, V_C
                    seq_logits_chunks.append(logits_chunk)

                # cat sequence chunk
                b_logits.append(torch.cat(seq_logits_chunks, dim=2))  # B_C, L_C, V

            logits_chunks.append(torch.cat(b_logits, dim=1))  # B_C, L, V

        return torch.cat(logits_chunks, dim=0)  # B, L, V

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
        Only last logits
        """
        B, L, H = hidden_states.shape

        if B == 1 and L > 1:
            hidden_states = hidden_states[:, -1, :]
            L = 1

        device = hidden_states.device

        if device.type == "mps" or device.type == "cpu":
            return F.linear(hidden_states, self.lm_head.weight)

        # other device (cuda), use chunked compute
        return self._chunked_compute_logits(
            hidden_states, batch_chunk, seq_chunk, vocab_chunk
        )
