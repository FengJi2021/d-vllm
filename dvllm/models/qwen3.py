# dvllm/models/qwen3.py
import math
import torch
from torch import nn
import torch.nn.functional as F
from transformers import Qwen3Config

from dvllm.layers.activation import SiluAndMul
from dvllm.layers.attention import Attention
from dvllm.layers.layernorm import RMSNorm
from dvllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from dvllm.layers.rotary_embedding import get_rope
from dvllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, ff_hidden_size, scaling, use_rope=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scaling = scaling
        self.use_rope = use_rope

        # qkv combined projection
        # 权重文件中 q 被扩大了 2 倍：[q=2048, k=1024, v=1024] = 4096
        # 但我们在 forward 中只用 q 的前 1024 维，所以合并后仍然是 2048
        # （即 q[:1024] + k + v = 1024 + 1024 + 1024 = 3072）
        # 不过为了完整加载权重，我们让 qkv_proj 接受完整的 4096 维，然后在 forward 中处理
        qkv_output_size = 4096  # q(2048) + k(1024) + v(1024)
        self.qkv_proj = nn.Linear(hidden_size, qkv_output_size, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # norms
        self.attn_norm = RMSNorm(hidden_size, eps=1e-6)
        self.ffn_norm = RMSNorm(hidden_size, eps=1e-6)
        # q_norm and k_norm are applied to Q and K after projection
        # They normalize per head_dim (not per hidden_size like attn_norm)
        self.q_norm = RMSNorm(head_dim, eps=1e-6)
        self.k_norm = RMSNorm(head_dim, eps=1e-6)

        # gated-FFN
        # gated-FFN: gate_proj/up_proj -> activation -> down_proj
        # gate_proj/up_proj map hidden_size -> ff_hidden_size, activation returns a tensor
        # of shape [B, L, ff_hidden_size], so down_proj must accept ff_hidden_size
        self.gate_proj = nn.Linear(hidden_size, ff_hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, ff_hidden_size, bias=False)
        self.down_proj = nn.Linear(ff_hidden_size, hidden_size, bias=False)
        self.activation = SiluAndMul()
        # debug flag to enable per-layer tensor stats printing
        self.debug = False

    def forward(self, positions, hidden_states, rope_fn=None, device_type=None, sdpa_scale=None,
                chunked_threshold=4096, chunk_size=1024, dump_path=None):
        # 确保 hidden_states 是三维 (B, L, C)
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
        B, L, C = hidden_states.shape

        # Attention
        x = hidden_states
        x_attn = self.attn_norm(x)

        qkv = self.qkv_proj(x_attn)  # [B, L, 4096]
        # 权重文件中是 [q=2048, k=1024, v=1024] = 4096
        # 我们的模型只需要 [q=1024, k=512, v=512] = 2048
        # 所以 split 为 [2048, 1024, 1024]，然后都砍掉一半
        q, k, v = qkv.split([2048, 1024, 1024], dim=-1)  # q:[B,L,2048], k:[B,L,1024], v:[B,L,1024]
        # TODO add assert for debug
        q = q[:, :, :self.num_heads * self.head_dim]  # 保留前 1024 维
        k = k[:, :, :self.num_kv_heads * self.head_dim]  # 保留前 512 维
        v = v[:, :, :self.num_kv_heads * self.head_dim]  # 保留前 512 维
        
        q = q.view(B, L, self.num_heads, self.head_dim)
        k = k.view(B, L, self.num_kv_heads, self.head_dim)
        v = v.view(B, L, self.num_kv_heads, self.head_dim)
        
        # Apply q_norm and k_norm (after projection, on head_dim)
        # q, k have shape [B, L, num_heads, head_dim] at this point
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Transpose to [B, num_heads, L, head_dim] for rope and attention computation
        # This matches HF Qwen3 which does transpose AFTER q_norm/k_norm
        q = q.transpose(1, 2)  # [B, L, num_heads, head_dim] -> [B, num_heads, L, head_dim]
        k = k.transpose(1, 2)  # [B, L, num_kv_heads, head_dim] -> [B, num_kv_heads, L, head_dim]
        v = v.transpose(1, 2)  # [B, L, num_kv_heads, head_dim] -> [B, num_kv_heads, L, head_dim]

        # rotary
        if self.use_rope and (rope_fn is not None):
            try:
                q, k = rope_fn(positions, q, k)
            except Exception:
                pass

        factor = self.num_heads // self.num_kv_heads
        assert self.num_heads % self.num_kv_heads == 0

        device = q.device.type
        
        # 临时存储 attention 内部张量用于 dump
        attn_internals = {}

        def sdpa(q_t, k_t, v_t):
            # compute raw attention scores for debugging/consistency
            scores = torch.einsum("bhld,bhmd->bhlm", q_t, k_t) * (sdpa_scale if sdpa_scale else 1.0 / math.sqrt(self.head_dim))
            # debug: print q/k/v and score stats when enabled
            if self.debug:
                try:
                    with torch.no_grad():
                        q_stats = (q_t.mean().item(), q_t.std().item(), q_t.min().item(), q_t.max().item())
                        k_stats = (k_t.mean().item(), k_t.std().item(), k_t.min().item(), k_t.max().item())
                        v_stats = (v_t.mean().item(), v_t.std().item(), v_t.min().item(), v_t.max().item())
                        sc_stats = (scores.mean().item(), scores.std().item(), scores.min().item(), scores.max().item())
                        print(f"[DEBUG] q mean,std,min,max={q_stats} | k={k_stats} | v={v_stats} | scores mean,std,min,max={sc_stats}")
                except Exception:
                    pass
            
            # 如果需要 dump，保存这一步的张量
            if dump_path is not None:
                try:
                    attn_internals['q_t'] = q_t.detach().cpu()
                    attn_internals['k_t'] = k_t.detach().cpu()
                    attn_internals['v_t'] = v_t.detach().cpu()
                    attn_internals['scores'] = scores.detach().cpu()
                except Exception:
                    pass

            try:
                out = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=None, dropout_p=0.0,
                                                     is_causal=True, scale=sdpa_scale)
                if dump_path is not None:
                    try:
                        weights = torch.softmax(scores, dim=-1)
                        attn_internals['softmax_weights'] = weights.detach().cpu()
                        attn_internals['attn_out'] = out.detach().cpu()
                    except Exception:
                        pass
                return out
            except Exception:
                Lq = q_t.size(2)
                Lk = k_t.size(2)
                if Lk == Lq:
                    mask = torch.tril(torch.ones(Lq, Lk, device=scores.device, dtype=torch.bool))
                    scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
                weights = torch.softmax(scores, dim=-1)
                out = torch.einsum("bhlm,bhmd->bhld", weights, v_t)
                if dump_path is not None:
                    try:
                        attn_internals['softmax_weights'] = weights.detach().cpu()
                        attn_internals['attn_out'] = out.detach().cpu()
                    except Exception:
                        pass
                return out

        # q, k, v are now [B, num_heads, L, head_dim] after transpose
        # This is the correct format for attention computation
        if device == "mps" and L > chunked_threshold:
            out_chunks = []
            for s in range(0, L, chunk_size):
                e = min(L, s + chunk_size)
                q_chunk = q[:, :, s:e, :]  # [B, num_heads, chunk_size, head_dim]
                k_slice = k[:, :, s:e, :]  # [B, num_kv_heads, chunk_size, head_dim]
                v_slice = v[:, :, s:e, :]  # [B, num_kv_heads, chunk_size, head_dim]
                if factor > 1:
                    k_chunk = k_slice.repeat_interleave(factor, dim=1)  # Repeat on heads dimension
                    v_chunk = v_slice.repeat_interleave(factor, dim=1)
                else:
                    k_chunk, v_chunk = k_slice, v_slice
                out_chunk = sdpa(q_chunk, k_chunk, v_chunk)  # [B, num_heads, chunk_size, head_dim]
                out_chunks.append(out_chunk)
            out_t = torch.cat(out_chunks, dim=2)  # Concatenate on L dimension
            # Reshape from [B, num_heads, L, head_dim] to [B, L, num_heads * head_dim]
            out_t = out_t.transpose(1, 2).contiguous()  # [B, L, num_heads, head_dim]
            attn_out = out_t.view(B, L, self.num_heads * self.head_dim)
            attn_out = self.o_proj(attn_out)
        else:
            if factor > 1:
                k = k.repeat_interleave(factor, dim=1)  # Repeat on heads dimension
                v = v.repeat_interleave(factor, dim=1)
            # q, k, v are already [B, num_heads, L, head_dim]
            out_t = sdpa(q, k, v)  # [B, num_heads, L, head_dim]
            # Reshape from [B, num_heads, L, head_dim] to [B, L, num_heads * head_dim]
            out_t = out_t.transpose(1, 2).contiguous()  # [B, L, num_heads, head_dim]
            attn_out = out_t.view(B, L, self.num_heads * self.head_dim)
            attn_out = self.o_proj(attn_out)

        hidden_states = hidden_states + attn_out

        # FFN
        x_ffn = self.ffn_norm(hidden_states)
        gate = self.gate_proj(x_ffn)
        up = self.up_proj(x_ffn)
        # debug: print gate/up stats
        if self.debug:
            try:
                with torch.no_grad():
                    g_stats = (gate.mean().item(), gate.std().item(), gate.min().item(), gate.max().item())
                    u_stats = (up.mean().item(), up.std().item(), up.min().item(), up.max().item())
                    print(f"[DEBUG-FFN] gate mean,std,min,max={g_stats} | up={u_stats}")
            except Exception:
                pass

        acted = self.activation(up, gate)
        # debug: post-activation and down_proj stats
        if self.debug:
            try:
                with torch.no_grad():
                    a_stats = (acted.mean().item(), acted.std().item(), acted.min().item(), acted.max().item())
                    print(f"[DEBUG-FFN] acted mean,std,min,max={a_stats}")
            except Exception:
                pass

        down = self.down_proj(acted)
        if self.debug:
            try:
                with torch.no_grad():
                    d_stats = (down.mean().item(), down.std().item(), down.min().item(), down.max().item())
                    print(f"[DEBUG-FFN] down mean,std,min,max={d_stats}")
            except Exception:
                pass
        hidden_states = hidden_states + down
        
        # 如果需要 dump，保存 FFN 张量
        if dump_path is not None:
            try:
                attn_internals['gate'] = gate.detach().cpu()
                attn_internals['up'] = up.detach().cpu()
                attn_internals['acted'] = acted.detach().cpu()
                attn_internals['down'] = down.detach().cpu()
                import pickle
                with open(f"{dump_path}/layer_{self.layer_id}.pkl", 'wb') as f:
                    pickle.dump(attn_internals, f)
            except Exception as e:
                print(f"Warning: failed to dump layer {self.layer_id}: {e}")

        return hidden_states


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.num_layers = config.num_hidden_layers
        # allow overriding head_dim from config (HF provides head_dim=128)
        self.head_dim = getattr(config, 'head_dim', self.hidden_size // self.num_heads)

        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, self.hidden_size)
        try:
            self.rope_fn = get_rope(config.hidden_size)  # 注意这里改成 dim
        except Exception:
            self.rope_fn = None

        ff_hidden = getattr(config, "intermediate_size", self.hidden_size * 4)
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = TransformerBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                ff_hidden_size=ff_hidden,
                scaling=(1.0 / math.sqrt(self.head_dim)),
                use_rope=(self.rope_fn is not None)
            )
            layer.layer_id = i
            self.layers.append(layer)
        self.final_norm = RMSNorm(self.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6))

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor = None, dump_path: str = None) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        device_type = x.device.type
        for layer in self.layers:
            x = layer(positions, x, rope_fn=self.rope_fn, device_type=device_type,
                      sdpa_scale=(1.0 / math.sqrt(self.head_dim)), dump_path=dump_path)
        return self.final_norm(x)


class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.model = Qwen3Model(config)
        self.layers = self.model.layers  # loader 需要
        self.norm = self.model.final_norm
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if getattr(config, "tie_word_embeddings", True):
            try:
                self.lm_head.weight = self.model.embed_tokens.weight
            except AttributeError:
                import logging
                logging.info("embed_tokens.weight not found, skipping tie_weights")

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor = None, dump_path: str = None) -> torch.Tensor:
        return self.model(input_ids, positions, dump_path=dump_path)
    

    def compute_logits(self, hidden_states: torch.Tensor,
                    batch_chunk: int = 1,
                    seq_chunk: int = 32,
                    vocab_chunk: int = 512) -> torch.Tensor:
        """
        三重分块计算 logits，避免内存爆炸
        hidden_states: [B, L, H]
        
        对于 MPS，直接在 MPS 上计算通常更快，避免 CPU-GPU 频繁数据搬运
        对于 CUDA，分块可以缓解大显存压力
        """
        B, L, H = hidden_states.shape
        device = hidden_states.device
        
        # 对于 MPS 或内存充足的情况，直接计算
        if device.type == "mps" or device.type == "cpu":
            # 简单直接的方式：一次性计算
            return torch.nn.functional.linear(hidden_states, self.lm_head.weight)
        
        # CUDA 的分块计算方式
        logits_list = []
        weight_cpu = self.lm_head.weight.cpu()  # lm_head 权重放 CPU

        for batch_start in range(0, B, batch_chunk):
            batch_end = min(B, batch_start + batch_chunk)
            batch_hidden = hidden_states[batch_start:batch_end, :, :]

            batch_logits = []

            for seq_start in range(0, L, seq_chunk):
                seq_end = min(L, seq_start + seq_chunk)
                seq_hidden = batch_hidden[:, seq_start:seq_end, :].cpu()

                seq_logits_chunks = []

                for vocab_start in range(0, weight_cpu.size(0), vocab_chunk):
                    vocab_end = min(weight_cpu.size(0), vocab_start + vocab_chunk)
                    w_chunk = weight_cpu[vocab_start:vocab_end, :]
                    # 计算线性
                    logits_chunk = F.linear(seq_hidden, w_chunk)
                    seq_logits_chunks.append(logits_chunk)

                # 拼接 vocab chunk
                seq_logits = torch.cat(seq_logits_chunks, dim=-1)
                # 立即搬回 GPU
                batch_logits.append(seq_logits.to(hidden_states.device))

            # 拼接 seq chunk
            batch_logits = torch.cat(batch_logits, dim=1)
            logits_list.append(batch_logits)

        # 拼接 batch chunk
        return torch.cat(logits_list, dim=0)


                

