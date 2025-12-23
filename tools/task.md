# Task：如何一层一层检查 Qwen3 的数值

目标：基于当前的 Qwen3 实现和 tools/get_params.py，给出一套**按层排查数值问题**的实战流程。

分两部分：

1. **理解 Qwen3 里已经提供了哪些“按层观测”的入口**（debug / dump_path）。
2. **理解 get_params.py 能看到什么（权重形状），以及什么时候该直接在 Qwen3 里打 log 看激活值。**

> 你可以把这份文档当成：
> - “怎么一层层 debug 这个模型的数值”的操作手册。

---

## 1. Qwen3 目前是如何一层一层前向的？

关键文件：

- 模型结构： dvllm/models/qwen3.py
- 我们关心的类：
  - `TransformerBlock`
  - `Qwen3Model`
  - `Qwen3ForCausalLM`

### 1.1 总体调用链

大致流程：

1. 外部一般拿的是 `Qwen3ForCausalLM`：

   ```python
   model = Qwen3ForCausalLM(config)
   logits = model(input_ids, positions)
   ```

2. `Qwen3ForCausalLM.forward` 只是把调用转发给内部的 `Qwen3Model`：

   ```python
   def forward(self, input_ids, positions=None, dump_path=None):
       return self.model(input_ids, positions, dump_path=dump_path)
   ```

3. `Qwen3Model.forward` 做几件事：

   ```python
   x = self.embed_tokens(input_ids)      # [B, L, H]
   for layer in self.layers:             # self.layers 是一组 TransformerBlock
       x = layer(positions, x, rope_fn=self.rope_fn, device_type=device_type,
                 sdpa_scale=..., dump_path=dump_path)
   x = self.final_norm(x)
   ```

4. 每个 `TransformerBlock` 负责：

   - 做一次 **Attention + 残差**
   - 再做一次 **FFN + 残差**
   - 内部就包含我们关注的：Q/K/V、scores、FFN 激活、以及 dump / debug 逻辑。

### 1.2 TransformerBlock 内部重要参数

构造时：

```python
self.qkv_proj = nn.Linear(hidden_size, 4096, bias=False)
self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
...
self.q_norm = RMSNorm(head_dim, eps=1e-6)
self.k_norm = RMSNorm(head_dim, eps=1e-6)
...
self.gate_proj = nn.Linear(hidden_size, ff_hidden_size, bias=False)
self.up_proj   = nn.Linear(hidden_size, ff_hidden_size, bias=False)
self.down_proj = nn.Linear(ff_hidden_size, hidden_size, bias=False)

self.debug = False  # 控制是否在控制台打印统计信息
self.layer_id = i   # 在 Qwen3Model 中赋值，用于 dump 文件命名
```

forward 的签名：

```python
def forward(self, positions, hidden_states, rope_fn=None, device_type=None,
            sdpa_scale=None, chunked_threshold=4096, chunk_size=1024,
            dump_path=None):
    ...
```

这里有两个和“逐层观察数值”直接相关的入口：

- `self.debug`：
  - 如果设为 `True`，会在控制台打印 Q/K/V、scores、FFN 等的 mean/std/min/max。
- `dump_path`：
  - 如果不为 `None`，每一层会把中间张量存成一个 `layer_{layer_id}.pkl`，方便事后离线分析。

---

## 2. 一层一层输出：Attention 部分的数值

### 2.1 Attention 内部的形状流转（回顾）

在 `TransformerBlock.forward` 中（已在 EASY_README 里解释过，这里只快速串一下）：

1. 归一化并线性映射：

   ```python
   x_attn = self.attn_norm(hidden_states)          # [B, L, H]
   qkv = self.qkv_proj(x_attn)                    # [B, L, 4096]
   q, k, v = qkv.split([2048, 1024, 1024], dim=-1)
   ```

2. 只保留前半部分维度（对应较小的 num_heads/num_kv_heads）：

   ```python
   q = q[:, :, :self.num_heads * self.head_dim]    # 例如 1024
   k = k[:, :, :self.num_kv_heads * self.head_dim] # 例如 512
   v = v[:, :, :self.num_kv_heads * self.head_dim] # 例如 512
   ```

3. reshape 成多头：

   ```python
   q = q.view(B, L, self.num_heads, self.head_dim)
   k = k.view(B, L, self.num_kv_heads, self.head_dim)
   v = v.view(B, L, self.num_kv_heads, self.head_dim)
   ```

4. 对每个 head 的向量做 RMSNorm（`head_dim` 粒度）：

   ```python
   q = self.q_norm(q)
   k = self.k_norm(k)
   ```

5. 转成注意力习惯的布局：

   ```python
   q = q.transpose(1, 2)  # [B, num_heads,    L, head_dim]
   k = k.transpose(1, 2)  # [B, num_kv_heads, L, head_dim]
   v = v.transpose(1, 2)
   ```

6. 可选：RoPE

7. GQA 扩展（如果 `num_heads > num_kv_heads`）：

   ```python
   factor = self.num_heads // self.num_kv_heads
   if factor > 1:
       k = k.repeat_interleave(factor, dim=1)
       v = v.repeat_interleave(factor, dim=1)
   ```

8. 调用内部的 `sdpa(q, k, v)` 计算注意力：

   ```python
   out_t = sdpa(q, k, v)               # [B, num_heads, L, head_dim]
   out_t = out_t.transpose(1, 2)       # [B, L, num_heads, head_dim]
   attn_out = out_t.view(B, L, -1)     # [B, L, H]
   attn_out = self.o_proj(attn_out)    # [B, L, H]
   hidden_states = hidden_states + attn_out
   ```

### 2.2 sdpa 内部怎么调试？

在 `TransformerBlock` 里定义了一个内部函数 `sdpa(q_t, k_t, v_t)`，里面有所有我们关心的 debug / dump 逻辑：

1. 先自己算一遍原始 scores：

   ```python
   scores = torch.einsum("bhld,bhmd->bhlm", q_t, k_t) * (
       sdpa_scale if sdpa_scale else 1.0 / math.sqrt(self.head_dim)
   )
   ```

2. 如果开启 `self.debug`，打印统计信息：

   ```python
   if self.debug:
       q_stats = (q_t.mean(), q_t.std(), q_t.min(), q_t.max())
       k_stats = (k_t.mean(), k_t.std(), k_t.min(), k_t.max())
       v_stats = (v_t.mean(), v_t.std(), v_t.min(), v_t.max())
       sc_stats = (scores.mean(), scores.std(), scores.min(), scores.max())
       print(f"[DEBUG] q mean,std,min,max={q_stats} | ... | scores={sc_stats}")
   ```

3. 如果 `dump_path` 不为 None，会把 q/k/v/scores 存到 `attn_internals` 里：

   ```python
   if dump_path is not None:
       attn_internals['q_t'] = q_t.detach().cpu()
       attn_internals['k_t'] = k_t.detach().cpu()
       attn_internals['v_t'] = v_t.detach().cpu()
       attn_internals['scores'] = scores.detach().cpu()
   ```

4. 调用 `F.scaled_dot_product_attention` 得到 `out`，并可选地保存 softmax 权重和输出：

   ```python
   out = F.scaled_dot_product_attention(...)
   if dump_path is not None:
       weights = torch.softmax(scores, dim=-1)
       attn_internals['softmax_weights'] = weights.detach().cpu()
       attn_internals['attn_out'] = out.detach().cpu()
   ```

5. 在整层 forward 的末尾，如果 `dump_path` 不为 None，会把 `attn_internals` 写成一个 pkl 文件：

   ```python
   if dump_path is not None:
       attn_internals['gate'] = gate.detach().cpu()
       attn_internals['up'] = up.detach().cpu()
       attn_internals['acted'] = acted.detach().cpu()
       attn_internals['down'] = down.detach().cpu()
       with open(f"{dump_path}/layer_{self.layer_id}.pkl", 'wb') as f:
           pickle.dump(attn_internals, f)
   ```

也就是说：

> **对于每一层 TransformerBlock，你都可以得到一个包含该层 Q/K/V、scores、softmax 权重、FFN 激活的字典，离线慢慢看。**

---

## 3. FFN（MLP）部分的数值怎么一层层看？

Attention 之后是 FFN：

```python
x_ffn = self.ffn_norm(hidden_states)

gate = self.gate_proj(x_ffn)  # [B, L, ff_hidden]
up   = self.up_proj(x_ffn)    # [B, L, ff_hidden]

acted = self.activation(up, gate)  # SiluAndMul

down = self.down_proj(acted)      # [B, L, H]

hidden_states = hidden_states + down
```

如果 `self.debug == True`，会额外打印：

- gate 的 mean/std/min/max
- up 的 mean/std/min/max
- acted（激活后）的 mean/std/min/max
- down（映射回 hidden_size） 的 mean/std/min/max

如果 `dump_path` 不为 None，这几个张量也会一起写入 `layer_{id}.pkl` 中：

- `gate`
- `up`
- `acted`
- `down`

> 这样一层层往下，你可以分别观察：
> - Attention 输出前后的数值分布是否合理
> - FFN 激活前后是否有数值爆炸 / 退化

---

## 4. 实战：怎么在代码里一层层看？

### 4.1 开启 debug，直接看控制台统计

1. 构造模型：

   ```python
   from dvllm.models.qwen3 import Qwen3ForCausalLM
   from transformers import Qwen3Config

   config = Qwen3Config(...)  # 按你的模型配置
   model = Qwen3ForCausalLM(config)
   ```

2. 打开所有层的 debug 开关：

   ```python
   for layer in model.model.layers:
       layer.debug = True
   ```

3. 准备一个小 batch 输入：

   ```python
   input_ids = ...  # [B, L]
   positions = ...  # [B, L]，通常是递增的 position_ids

   outputs = model(input_ids, positions)
   ```

4. 运行时，你会在 stdout 看到类似：

   ```text
   [DEBUG] q mean,std,min,max=(..., ..., ..., ...) | k=(...) | v=(...) | scores mean,std,min,max=(...)
   [DEBUG-FFN] gate mean,std,min,max=(...) | up=(...)
   [DEBUG-FFN] acted mean,std,min,max=(...)
   [DEBUG-FFN] down mean,std,min,max=(...)
   ```

5. 对照 EASY_README 中的“正常 / 异常范围”，就可以一层层判断：

   - 哪一层的 Q/K/V 或 scores 开始异常
   - 是 Attention 部分炸了，还是 FFN 部分炸了

### 4.2 使用 dump_path，离线逐层分析

如果你不想刷控制台，而是想把每层的张量都存下来慢慢看，可以用 `dump_path`：

1. 创建一个目录：

   ```bash
   mkdir -p /tmp/qwen_debug
   ```

2. 调用时传入 `dump_path`：

   ```python
   outputs = model(input_ids, positions, dump_path="/tmp/qwen_debug")
   ```

3. 运行后，你会在 `/tmp/qwen_debug` 里看到：

   ```text
   layer_0.pkl
   layer_1.pkl
   ...
   layer_{num_layers-1}.pkl
   ```

4. 读取某一层的 pkl：

   ```python
   import pickle

   with open("/tmp/qwen_debug/layer_0.pkl", "rb") as f:
       data = pickle.load(f)

   print(data.keys())
   # 可能包含：q_t, k_t, v_t, scores, softmax_weights, attn_out,
   #          gate, up, acted, down

   q = data["q_t"]       # tensor, 形状大致 [B, num_heads, L, head_dim]
   scores = data["scores"]

   print(q.shape, q.mean(), q.std())
   ```

5. 你可以写一个小脚本：

   - 遍历所有 `layer_i.pkl`
   - 打印每一层 Q/K/V/scores/FFN 的 mean/std/min/max
   - 用简单逻辑标记“超出正常范围”的层

---

## 5. get_params.py：看权重形状用的，而不是看运行时数值

脚本位置： tools/get_params.py

核心逻辑：

```python
from safetensors.torch import safe_open

with safe_open(model_path, framework="pt") as f:
    for key in f.keys():
        tensor = f.get_tensor(key)
        print(f"{key}: {tuple(tensor.shape)}")
```

使用方式：

```bash
./tools/get_params.py --model ~/huggingface/Qwen3-0.6B/model.safetensors
```

你会看到：

```text
model.layers.0.self_attn.q_proj.weight: (2048, 1024)
model.layers.0.self_attn.k_proj.weight: (1024, 1024)
model.layers.0.self_attn.v_proj.weight: (1024, 1024)
...
```

这个脚本告诉你的是：

- **每个权重张量（参数）的名字和形状**：
  - 比如 q_proj/k_proj/v_proj 的输出/输入维度
- 方便你确认：
  - Safetensors 里的权重形状和你在 Qwen3 代码中设置的 `hidden_size` / `num_heads` / `head_dim` 是否一致
  - 比如 q 是 2048，k/v 是 1024，这和 `qkv_proj` 的 4096 一一对应

它**不会**给你：

- 运行时的 Q/K/V/scores/FFN 的数值
- 任何 batch 维度 / 序列维度的信息

> 换句话说：
> - `get_params.py` 用来**检查“静态”权重和你的代码设计是否匹配**。
> - Qwen3 里的 debug / dump_path 用来**检查“动态”前向过程中的激活是否健康**。

---

## 6. 什么时候用 get_params，什么时候直接在 Qwen3 里 log？

可以简单分两类场景：

### 6.1 场景 A：模型刚接进来 / loader 有问题

症状：

- 加载权重时报 shape mismatch
- 你怀疑 **权重文件和代码的 hidden_size/num_heads/head_dim 没对上**

操作建议：

1. 先用 `get_params.py` 看一眼关键层权重的形状：
   - `q_proj.weight` / `k_proj.weight` / `v_proj.weight`
   - `gate_proj` / `up_proj` / `down_proj`
   - embedding / lm_head
2. 对照 qwen3.py 里的线性层定义，确认维度是否对齐。

### 6.2 场景 B：前向可以跑，但 loss/输出异常（怀疑数值问题）

症状：

- loss NaN / 无穷大
- logits 全部接近某个常数 / attention 看起来“锁死”在一个 token

操作建议：

1. 在 Qwen3 里开 `layer.debug = True`，先跑一个小 batch：
   - 看控制台上的 Q/K/V/scores/FFN 的 mean/std/min/max
2. 如果想更系统地分析：
   - 用 `dump_path` 把每一层的中间张量存下来
   - 写脚本离线遍历所有 layer 的 pkl，画图 or 打统计

> 总结成一张简单表：
>
> - **要看参数 shape 是否匹配** → 用 `tools/get_params.py`
> - **要看具体前向激活是否健康** → 在 qwen3 的 `TransformerBlock` 里用 `debug` / `dump_path`

---

## 7. 为啥还要这么认真 debug？torch/MPS 不应该自己搞定吗？

你现在的直觉是对的：

- 维度都被 constrain 住了；
- 计算都是走 PyTorch + MPS/CUDA 官方算子；
- 乍一看“只要不写 bug，结果就应该是对的”。

但在真实工程里，很多“看起来没问题”的实现，最后问题都出在：

1. 语义层面的小差异（形状对了，但“意义”错了）。
2. 精度 / 缩放 / norm 顺序导致的数值不稳定。
3. 边界条件（超长序列、分块、极端输入）导致的后端细节问题。

这些都是 **PyTorch 和 MPS 不会替你兜底** 的部分，只能靠你自己用 debug 钩子去观测。

### 7.1 维度对了 ≠ 算法语义对了

典型例子：

- Q/K/V 的拆分、裁剪、reshape / transpose 顺序。
  - 现在是：`[2048,1024,1024]` split → 裁剪 → `view` 成 `[B,L,num_heads,head_dim]`；
  - 如果以后调整 head 数、head_dim 或换别的模型结构，**很容易出现“形状还能对上，但某些 head 对应错了”的情况**；
  - MPS/torch 不会告诉你“这个 head 里的语义错位了”，只会老老实实算。

- RoPE / mask / 因果结构：
  - position 张错了、RoPE 维度划分错了、mask 搞错看到了未来 token，这些都不会导致 crash；
  - 但会让 scores / softmax_weights 呈现出“很不正常”的分布（比如永远盯着某几个 token）。

这类问题，只看代码很难一眼看出，**但看 Q/K/V/scores 的 mean/std/min/max 或可视化，很容易发现“不像正常注意力”的行为**。

### 7.2 精度 & 缩放 & Norm 顺序

你提到“是不是精度导致的”其实非常关键：

- 混合精度（fp16/bf16）下：
  - qkv 投影 + FFN 的大矩阵乘法，如果没有合适的缩放 / norm，很容易把中间 activations 推到很大的数值范围；
  - 后端只负责算结果，不会自动 clip 到“合理范围”。

- Norm 放置的位置和维度：
  - 现在：`attn_norm` 是在 hidden_size 上，`q_norm`/`k_norm` 是在 head_dim 上，且在 transpose 之前做；
  - 这一点如果以后改动（例如换成别的顺序、换 eps），**完全可能形状仍然对，数值统计却完全不一样**。

- scores 的缩放因子：
  - 现在 sdpa 里手动算了一遍 `scores = q·k^T * scale`，又把 `sdpa_scale` 传给 `scaled_dot_product_attention`；
  - 如果未来哪次 refactor 把 scale 叠加两次 / 忘记传，`scores` 的 std 可能从 ~1 变成 ~10，softmax 直接变成“只看一个 token”。

这些不会在代码层面直接报错，但：

- Q/K/V/scores 的 mean/std/min/max 会明显“出圈”；
- logits / loss 表现会很奇怪（NaN / 极端偏置 / 不收敛）。

### 7.3 后端实现差异 & 边界条件

同一份 PyTorch 代码，在 CPU / CUDA / MPS 上：

- 浮点舍入方式、kernel 实现细节不一样；
- 在极端输入（超长序列、chunk 边界、batch=1 或超大 batch）下，可能只在某一个后端暴露问题。

你当前的实现里有这些比较“敏感”的区域：

- MPS 上的 chunked attention：
  - 对 L>chunked_threshold 时做分块；
  - 每块单独算 sdpa，再 `cat` 回去；
  - 一旦在 slice 范围、repeat_interleave、cat 维度上有 off-by-one / 维度选错，**形状仍然成立，但语义错位**。

- 回退路径（scaled_dot_product_attention 抛异常时自己算 mask+softmax+einsum）：
  - 极端长度 / 特殊 mask 情况下可能走到这个分支；
  - 一旦 mask 广播/形状有细微 bug，也不会立刻 crash，而是“悄悄影响注意力权重”。

这些问题，**光靠“我相信 torch/MPS”是不足够的**，必须有手段看到：

- 每一层 / 每一种输入场景下，实际的 Q/K/V/scores/FFN 的统计；
- 哪一层开始出现 NaN / Inf 或极端值。

### 7.4 debug 钩子是给“未来的你”留的保险

总结一下：

- 现在这版代码在你当前配置和输入下“看起来没问题”，不代表：
  - 换一个 head 数 / hidden_size / 精度配置仍然没问题；
  - 换一套权重 / 换一个 loader 仍然没问题；
  - 在极端长度/边界条件下仍然没问题。

而一旦以后你或别人改了：

- RoPE / mask 逻辑；
- 分块逻辑（chunk_size / threshold）；
- norm / scale / dtype；

> 这些 debug / dump_path 钩子，就是用最少的代码，帮“未来的你”快速定位：
> - 哪一层出问题；
> - 是 Attention 还是 FFN；
> - 是 Q/K/V/scores 还是激活后的 FFN 在爆。

理解成：

- 平时不开，模型照常跑；
- 一旦出现“输出怪怪的”，你可以在几分钟内把“黑盒”变成“每层都有统计和中间激活的白盒”。