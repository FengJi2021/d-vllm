# dvllm/utils/loader.py
import logging
import os
from glob import glob
import torch
from safetensors.torch import safe_open

logger = logging.getLogger(__name__)


def default_weight_loader(
    param: torch.nn.Parameter, loaded_weight: torch.Tensor, shard_id=None
):
    """
    默认权重加载函数
    """
    if shard_id is not None:
        # 如果是 packed module 的分片（例如 gate_up_proj），取对应片
        loaded_weight = loaded_weight[shard_id]
    param.data.copy_(loaded_weight)


def load_model(model: torch.nn.Module, path: str, device: torch.device | None = None):
    """
    加载模型权重，兼容官方 HF Qwen3 权重

    处理权重名称的不匹配：
    - HF 权重中 q_proj, k_proj, v_proj 分开 -> 合并成 qkv_proj [q;k;v]
    - HF 权重中 gate_proj, up_proj 分开 -> 保持分开
    - HF 权重中 input_layernorm -> attn_norm
    - HF 权重中 post_attention_layernorm -> ffn_norm
    - HF 权重中 self_attn.* -> 去掉 self_attn 前缀
    - HF 权重中 mlp.* -> 去掉 mlp 前缀
    - 处理 embed_tokens: 模型中在 model.embed_tokens，权重文件中也在 model.embed_tokens
    """
    device = device or torch.device("cpu")
    prefix = "model."  # HF 权重前缀

    # 遍历 safetensors 文件，先加载所有权重到内存
    all_weights = {}
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, framework="pt", device="cpu") as f:
            for weight_name in f.keys():
                all_weights[weight_name] = f.get_tensor(weight_name)

    logger.info(f"[INFO] Loaded {len(all_weights)} weights from safetensors files")

    # 第一步：处理合并的权重（qkv_proj）
    # 权重文件中 q 被扩大了 2 倍（2048 vs 标准 1024）
    # 直接合并完整的权重：q(2048) + k(1024) + v(1024) = 4096
    # 模型在 forward 中会只取 q 的前 1024 维使用
    weight_cache = {}
    qkv_merged_count = 0
    for weight_name in list(all_weights.keys()):
        # 处理 qkv_proj 的合并
        if weight_name.endswith(".self_attn.q_proj.weight"):
            base_name = weight_name.replace(".q_proj.weight", "")
            k_name = weight_name.replace(".q_proj.weight", ".k_proj.weight")
            v_name = weight_name.replace(".q_proj.weight", ".v_proj.weight")

            if k_name in all_weights and v_name in all_weights:
                q = all_weights[weight_name]  # [2048, 1024]
                k = all_weights[k_name]  # [1024, 1024]
                v = all_weights[v_name]  # [1024, 1024]

                # 合并完整的权重，不砍
                qkv = torch.cat([q, k, v], dim=0)  # [4096, 1024]

                # 替换为 qkv_proj
                qkv_name = weight_name.replace(".q_proj.weight", ".qkv_proj.weight")
                weight_cache[qkv_name] = qkv
                logger.info(
                    f"[OK] Merged {weight_name.replace('model.', '')} + k + v -> qkv_proj shape={qkv.shape}"
                )
                qkv_merged_count += 1

                # 标记已处理
                all_weights.pop(k_name, None)
                all_weights.pop(v_name, None)
                all_weights.pop(weight_name)

    # 合并到总权重
    all_weights.update(weight_cache)
    logger.info(f"[INFO] Merged {qkv_merged_count} qkv projections")

    # 第二步：加载权重到模型
    loaded_count = 0
    skipped_count = 0

    for weight_name, weight_tensor in all_weights.items():
        # 去掉 HF 前缀
        if weight_name.startswith(prefix):
            short_name = weight_name[len(prefix) :]
        else:
            short_name = weight_name

        # 处理命名映射：HF 权重文件结构 -> 模型架构结构
        mapped_name = short_name

        # 处理 LayerNorm 命名
        mapped_name = mapped_name.replace("input_layernorm", "attn_norm")
        mapped_name = mapped_name.replace("post_attention_layernorm", "ffn_norm")

        # 处理 self_attn -> 直接到层（但保留其他），例如 self_attn.q_proj -> q_proj
        if ".self_attn." in mapped_name:
            mapped_name = mapped_name.replace(".self_attn.", ".")

        # 处理 mlp -> 直接到层
        if ".mlp." in mapped_name:
            mapped_name = mapped_name.replace(".mlp.", ".")

        # 尝试从模型获取参数
        try:
            param = model.get_parameter(mapped_name)
        except AttributeError:
            skipped_count += 1
            continue

        weight_tensor_device = weight_tensor.to(device)
        # 检查 shape 是否匹配
        if param.shape != weight_tensor_device.shape:
            # 尝试处理某些权重被扩大的情况
            # o_proj: 权重 [out, 2*in] 但模型期望 [out, in]，只取前半部分
            if (
                "o_proj" in mapped_name
                and weight_tensor_device.shape[1] == 2 * param.shape[1]
            ):
                weight_tensor_device = weight_tensor_device[:, : param.shape[1]]
                logger.info(
                    f"[OK] o_proj halved: {weight_tensor.shape} -> {weight_tensor_device.shape}"
                )
            # down_proj: 权重 [out, 2*in] 但模型期望 [out, in]，只取前半部分
            elif (
                "down_proj" in mapped_name
                and weight_tensor_device.shape[1] == 2 * param.shape[1]
            ):
                weight_tensor_device = weight_tensor_device[:, : param.shape[1]]
                logger.info(
                    f"[OK] down_proj halved: {weight_tensor.shape} -> {weight_tensor_device.shape}"
                )
            else:
                logger.warning(
                    f"[WARN] shape mismatch for {mapped_name}, model: {param.shape}, weight: {weight_tensor_device.shape}, skip."
                )
                skipped_count += 1
                continue

        param.data.copy_(weight_tensor_device)
        loaded_count += 1

    logger.info(
        f"[INFO] Successfully loaded {loaded_count} weights, skipped {skipped_count}"
    )
