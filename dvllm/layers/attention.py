import torch
from torch import nn
import torch.nn.functional as F

# lazy imports for optional accel libs
try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    HAS_TRITON = False

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

    HAS_FLASH_ATTN = True
except Exception:
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None
    HAS_FLASH_ATTN = False

from dvllm.utils.context import get_context

# Only define Triton kernel if triton is present
if HAS_TRITON:

    @triton.jit
    def store_kvcache_kernel(
        key_ptr,
        key_stride,
        value_ptr,
        value_stride,
        k_cache_ptr,
        v_cache_ptr,
        slot_mapping_ptr,
        D: tl.constexpr,
    ):
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)
        if slot == -1:
            return
        key_offsets = idx * key_stride + tl.arange(0, D)
        value_offsets = idx * value_stride + tl.arange(0, D)
        key = tl.load(key_ptr + key_offsets)
        value = tl.load(value_ptr + value_offsets)
        cache_offsets = slot * D + tl.arange(0, D)
        tl.store(k_cache_ptr + cache_offsets, key)
        tl.store(v_cache_ptr + cache_offsets, value)

    def store_kvcache_triton(key, value, k_cache, v_cache, slot_mapping):
        N, num_heads, head_dim = key.shape
        D = num_heads * head_dim
        store_kvcache_kernel[(N,)](
            key,
            key.stride(0),
            value,
            value.stride(0),
            k_cache,
            v_cache,
            slot_mapping,
            D,
        )

else:

    def store_kvcache_triton(*args, **kwargs):
        raise RuntimeError("triton not available")


class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def _fallback_prefill_attention(self, q, k, v, context):
        if q.dim() == 4:
            try:
                return F.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, is_causal=True
                )
            except Exception:
                B, Lq, H, D = q.shape
                _, Lk, _, _ = k.shape
                q2 = q.reshape(B * H, Lq, D)
                k2 = k.reshape(B * H, Lk, D)
                v2 = v.reshape(B * H, Lk, D)
                scores = torch.bmm(q2, k2.transpose(1, 2)) * (1.0 / (D**0.5))
                mask = torch.tril(
                    torch.ones(Lq, Lk, device=scores.device, dtype=torch.bool)
                )
                scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
                probs = F.softmax(scores, dim=-1)
                out = torch.bmm(probs, v2)
                return out.reshape(B, Lq, H, D)
        else:
            N, H, D = q.shape
            q2 = q.reshape(N * H, 1, D)
            k2 = k.reshape(N * H, -1, D)
            v2 = v.reshape(N * H, -1, D)
            scores = torch.bmm(q2, k2.transpose(1, 2)) * (1.0 / (D**0.5))
            probs = F.softmax(scores, dim=-1)
            out = torch.bmm(probs, v2)
            return out.reshape(N, H, D)

    def _fallback_decode_attention(self, q, k_cache, v_cache, context):
        N, H, D = q.shape
        q2 = q.reshape(H * N, 1, D)
        k2 = k_cache.permute(1, 0, 2).reshape(H * N, k_cache.size(0), D)
        v2 = v_cache.permute(1, 0, 2).reshape(H * N, v_cache.size(0), D)
        scores = torch.bmm(q2, k2.transpose(1, 2)) * (1.0 / (D**0.5))
        probs = F.softmax(scores, dim=-1)
        out = torch.bmm(probs, v2)
        return out.reshape(N, H, D)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        if k_cache.numel() and v_cache.numel():
            if HAS_TRITON and q.device.type == "cuda":
                store_kvcache_triton(k, v, k_cache, v_cache, context.slot_mapping)
            else:
                slots = context.slot_mapping.cpu().tolist()
                for i, slot in enumerate(slots):
                    if slot == -1:
                        continue
                    key_row = k[i].reshape(-1)
                    val_row = v[i].reshape(-1)
                    D = k.shape[1] * k.shape[2]
                    k_cache.view(-1, D)[slot] = key_row.to(k_cache.device)
                    v_cache.view(-1, D)[slot] = val_row.to(v_cache.device)

        use_flash = (q.device.type == "cuda") and HAS_FLASH_ATTN

        if context.is_prefill:
            if use_flash:
                return flash_attn_varlen_func(
                    q,
                    k,
                    v,
                    max_seqlen_q=context.max_seqlen_q,
                    cu_seqlens_q=context.cu_seqlens_q,
                    max_seqlen_k=context.max_seqlen_k,
                    cu_seqlens_k=context.cu_seqlens_k,
                    softmax_scale=self.scale,
                    causal=True,
                    block_table=context.block_tables,
                )
            else:
                return self._fallback_prefill_attention(q, k, v, context)
        else:
            if use_flash:
                return flash_attn_with_kvcache(
                    q.unsqueeze(1),
                    k_cache,
                    v_cache,
                    cache_seqlens=context.context_lens,
                    block_table=context.block_tables,
                    softmax_scale=self.scale,
                    causal=True,
                )
            else:
                return self._fallback_decode_attention(q, k_cache, v_cache, context)
