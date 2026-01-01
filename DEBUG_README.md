# Debugging Plan for Qwen3 Model

## Current Status
*   **Issue:** Model output is nonsensical, and debug logs indicate values are "exploding" (NaN/Inf or extremely large) in the middle layers.
*   **Recent Fixes:**
    *   Fixed a logic error in chunked attention where history was being ignored.
    *   Implemented custom masking for chunked causal attention.
    *   Refactored `dvllm/models/qwen3_v2.py` to separate attention logic into `_sdpa` and `_chunked_attention` methods.

## Debugging Steps

### 1. Enable Debug Mode
Set the environment variable to debug specific layers. Start with the first few layers and move deeper if they are clean.
```bash
export DV_DEBUG_LAYERS="0-5"
```

### 2. Analyze Logs for Explosion
Run your generation script and grep for "WARNING" or "error".
```bash
python test/example.py 2>&1 | grep -E "WARNING|error|NaN|Inf"
```

**What to look for:**
*   `[WARNING] q attention range is abnormal`: Indicates Query values are too large/small.
*   `[WARNING] sc attention range is abnormal`: **CRITICAL**. This means the dot product scores (before softmax) are exploding.
    *   If `sc_stats` max is > 88 (for float16) or extremely high, `exp(score)` will overflow to `inf`.
    *   `inf` in softmax leads to `NaN` output.

### 3. Hypothesis: FP16 Overflow
If you see `sc` (scores) exploding, it is likely due to `float16` range limitations on MPS/GPU.

**Action:**
Force the model to run in `float32` to confirm.
Modify `dvllm/engine/run_model.py`:
```python
# In __init__
torch.set_default_dtype(torch.float32) 
# Ensure model loading also uses float32
```
If the model works in `float32`, the issue is definitely FP16 overflow.

### 4. Hypothesis: Scaling Factor
Check the `sdpa_scale` value being used.
In `dvllm/models/qwen3_v2.py`, we use:
```python
real_sdpa_scale = sdpa_scale if sdpa_scale else 1.0 / math.sqrt(self.head_dim)
```
If `head_dim` is small (e.g., 128), scale is ~0.088.
If the input `q` and `k` are large (e.g., magnitude 10+), `10 * 10 * 128 * 0.088` = `1126`. `exp(1126)` is infinity.

**Action:**
If `q` or `k` stats show large means/max values (e.g., > 5.0 after norm), check the `RMSNorm` implementation or weights.

### 5. Code Structure (Refactored)
The `TransfomerBlock` in `dvllm/models/qwen3_v2.py` is now structured as:
*   `forward`: Main entry, handles Norms, Projections, RoPE.
*   `_chunked_attention`: Handles the loop for long sequences on MPS.
*   `_sdpa`: Wrapper around `F.scaled_dot_product_attention` with debug logging.
*   `_attention_range_checker`: Helper to validate tensor statistics.

Use `_sdpa` to inspect the exact values entering the attention mechanism.