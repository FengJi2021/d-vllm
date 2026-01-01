# Debugging Plan for QKV Explosion

This document outlines the steps to debug the issue where the model output becomes nonsensical, likely due to exploding values in the QKV computations.

## 1. Enable Debug Mode

The codebase already supports a debug flag for specific layers via the `DV_DEBUG_LAYERS` environment variable.

**Usage:**
Set the environment variable before running your script:
```bash
export DV_DEBUG_LAYERS="0-3"  # Debug layers 0, 1, 2, and 3
# OR
export DV_DEBUG_LAYERS="0,10,20" # Debug layers 0, 10, and 20
```

## 2. Instrument `dvllm/models/qwen3_v2.py`

The `TransfomerBlock` class has a `self.debug` attribute, but it is currently unused in the `forward` method. We need to add checks to inspect the values of `q`, `k`, and `v` at various stages.

**Action:**
Modify `dvllm/models/qwen3_v2.py` in the `TransfomerBlock.forward` method.

**Code to Insert:**

Locate the `forward` method in `dvllm/models/qwen3_v2.py` and insert the following checks.

```python
    def forward(
        self,
        positions: Optional[T],
        hidden_states: T,
        # ... other args
    ):
        B, L, C = hidden_states.shape

        # --- DEBUG CHECK 1: Input Hidden States ---
        if self.debug:
            print(f"\n[Layer DEBUG] Input Hidden States: shape={hidden_states.shape}")
            if torch.isnan(hidden_states).any():
                print("!!! NaN detected in Input Hidden States !!!")
            if torch.isinf(hidden_states).any():
                print("!!! Inf detected in Input Hidden States !!!")
            print(f"  Max: {hidden_states.max().item():.4f}, Min: {hidden_states.min().item():.4f}, Mean: {hidden_states.mean().item():.4f}")

        # Attention
        x = hidden_states
        x_attn = self.attn_norm(x)

        # qkv projection
        qkv: T = self.qkv_proj(x_attn)
        q, k, v = qkv.split([2048, 1024, 1024], dim=2)

        # --- DEBUG CHECK 2: QKV Projection Output ---
        if self.debug:
            print(f"[Layer DEBUG] QKV Raw Output")
            for name, tensor in [('q', q), ('k', k), ('v', v)]:
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    print(f"!!! NaN/Inf detected in {name} (projection) !!!")
                print(f"  {name} - Max: {tensor.max().item():.4f}, Min: {tensor.min().item():.4f}")

        # ... reshape logic ...
        q = q.view(B, L, self.num_heads, self.head_dim)
        k = k.view(B, L, self.num_kv_heads, self.head_dim)
        v = v.view(B, L, self.num_kv_heads, self.head_dim)

        # q, k norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # --- DEBUG CHECK 3: QK Norm Output ---
        if self.debug:
            print(f"[Layer DEBUG] QK Norm Output")
            for name, tensor in [('q_norm', q), ('k_norm', k)]:
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    print(f"!!! NaN/Inf detected in {name} !!!")
                print(f"  {name} - Max: {tensor.max().item():.4f}, Min: {tensor.min().item():.4f}")

        # ... RoPE application (if applicable) ...
        # Note: RoPE is usually applied inside the attention mechanism or passed as a function.
        # If applied here, check after RoPE.

        # ... Attention call ...
        # If you are using a custom attention function, pass the debug flag or inspect inputs there.
```

## 3. Instrument `dvllm/layers/attention.py`

If the explosion happens during the dot product (attention scores), we need to check inside the attention mechanism.

**Action:**
Modify `dvllm/layers/attention.py`.

**Code to Insert:**

In `_fallback_prefill_attention` (or whichever attention implementation is active):

```python
    def _fallback_prefill_attention(self, q, k, v, context):
        # ... existing code ...
        
        # Calculate scores
        # scores = torch.bmm(q2, k2.transpose(1, 2)) * (1.0 / (D**0.5))
        
        # --- DEBUG CHECK 4: Attention Scores ---
        # Note: You might need to pass 'debug' flag to this method or use a global/env var check
        if torch.isnan(scores).any() or torch.isinf(scores).any():
             print("!!! NaN/Inf detected in Attention Scores (pre-softmax) !!!")
             print(f"  Scores Max: {scores.max().item()}, Min: {scores.min().item()}")

        # ... mask application ...
        
        # probs = F.softmax(scores, dim=-1)
        
        # --- DEBUG CHECK 5: Attention Probabilities ---
        if torch.isnan(probs).any():
             print("!!! NaN detected in Attention Probabilities !!!")

        # out = torch.bmm(probs, v2)
        
        # --- DEBUG CHECK 6: Attention Output ---
        if torch.isnan(out).any() or torch.isinf(out).any():
             print("!!! NaN/Inf detected in Attention Output !!!")
```

## 4. Analyze the Logs

1.  Run the model with `DV_DEBUG_LAYERS="0-31"` (or a subset).
2.  Look for the first occurrence of `!!! NaN/Inf detected ... !!!`.
3.  **If it happens at Input Hidden States:** The issue is in the previous layer or embedding.
4.  **If it happens at QKV Raw Output:** The `qkv_proj` weights might be corrupted, or the input was already unstable (but not Inf/NaN).
5.  **If it happens at QK Norm Output:** The RMSNorm might be unstable (check epsilon).
6.  **If it happens at Attention Scores:** The dot product is too large. This often happens if Q/K are not properly normalized or scaled. Check `self.scaling` or the `1.0 / (D**0.5)` factor.
7.  **If it happens at Attention Output:** The value `v` might be exploded, or the weighted sum is overflowing.

## 5. Common Fixes

*   **Precision:** Ensure you are using `bfloat16` or `float32`. `float16` can easily overflow.
*   **Scaling:** Verify the attention scaling factor (`1 / sqrt(head_dim)`).
*   **Norm Epsilon:** Ensure RMSNorm epsilon is sufficient (e.g., `1e-6` or `1e-5`).
*   **Softmax:** Ensure the mask is not setting everything to `-inf` (which leads to NaNs in softmax).

## 6. Senior Engineer's Hypothesis & Starting Plan

Based on the code analysis, here are the most likely culprits for the issues you are facing.

### A. The "Nonsense Output" Cause: Incorrect Chunking Logic
**Location:** `dvllm/models/qwen3_v2.py` (Lines ~160-180)

The current chunked attention implementation for MPS devices has a critical semantic bug.
```python
# Current Code
q_chunk = q[:, :, start:end, :]
k_chunk = k[:, :, start:end, :]  # <--- PROBLEM
v_chunk = v[:, :, start:end, :]  # <--- PROBLEM
```
**The Issue:** By slicing `k` and `v` to `[start:end]`, you are restricting the attention to **only the current chunk**. The model effectively "forgets" everything before the current chunk. Token 2000 cannot attend to Token 0. This destroys the causal context, leading to garbage output.

**The Fix:** `k_chunk` and `v_chunk` must include the **entire history** up to `end` (or at least a large sliding window), not just the current chunk.
```python
# Proposed Fix Concept
k_chunk = k[:, :, 0:end, :] 
v_chunk = v[:, :, 0:end, :]
# Note: You will need to handle the causal mask carefully if using F.sdpa with different q/k lengths.
```

### B. The "Explosion" Cause: Float16 Overflow
**Location:** `dvllm/models/qwen3_v2.py` (Attention Calculation)

You mentioned "middle result exploded".
*   **Observation:** Qwen models use `RMSNorm` on Q and K before attention. This usually keeps values small.
*   **Hypothesis:** If you are running on MPS (Mac), PyTorch often defaults to `float16` (Half Precision) for performance.
*   **The Math:** `float16` max value is **65,504**.
    *   If `q` and `k` have values around ~2-3 (after norm), and `head_dim` is 128.
    *   Dot product sum: `128 * 2 * 2 = 512`.
    *   Scaled by `1/sqrt(128)` (~0.088) -> `45`.
    *   `exp(45)` is fine.
    *   **However**, if there is a spike or the scaling is omitted/wrong, the dot product can easily exceed `88` (since `exp(88) > 65504` in some contexts, or if the dot product itself exceeds 65k).
    *   **More likely:** If `sdpa_scale` is missing or `None`, and the fallback logic is slightly off, or if `q_norm` parameters were not loaded (zeros), it could cause issues.

### C. Immediate Action Plan

1.  **Disable Chunking (Test):**
    Set `chunked_threshold` to a very high value (e.g., 100000) in `dvllm/models/qwen3_v2.py` or via config to bypass the faulty chunking logic. Run the model.
    *   *If output makes sense:* The problem was the chunking logic (Hypothesis A).
    *   *If output is still nonsense:* The problem is elsewhere.

2.  **Force Float32 (Test):**
    Temporarily force the model to run in `float32` (or `bfloat16` if supported).
    *   *If explosion stops:* The problem is FP16 overflow (Hypothesis B).

3.  **Run the Debugger:**
    Use the `DV_DEBUG_LAYERS` instrumentation added in Step 2 to confirm if `q` or `k` are exploding *before* attention, or if the result explodes *during* attention.
