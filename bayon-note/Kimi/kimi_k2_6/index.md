# Kimi K2.6 家族架构详解

> 官方 HF 仓库：moonshotai/Kimi-K2.6（远端官方 config：hf-mirror.com 镜像拉取，2026-11-03）；API 文档：https://platform.kimi.com/docs/guide/kimi-k2-6-quickstart ；技术报告：无独立技术报告（架构与 K2.5 同构，继承 arXiv:2602.02276《Kimi K2.5: Visual Agentic Intelligence》）；多模态包装：transformers `src/transformers/models/kimi_k25/`（与 K2.5 同模块，本机 E:\transformers 可编辑安装）。本机 E:\models 无该模型权重，全部 config 数值以远端官方 config.json 为准（留档 `_tools/_cfgpull/config.json`，拉取日期 2026-11-03）。

## 1. 家族全景

Kimi K2.6（2026-04 发布）与 Kimi K2.5 同属 **kimi_k25 架构**：1T 总参 / 32B 激活的 MoE 文本底座 + MoonViT-3D 视觉塔，构形为「视觉塔 + PatchMerger 投影器 + K2 MoE 文本塔」三段式；通用多模态（文本/图片/视频），256K 上下文，thinking 可开关（API 名 `kimi-k2.6`）。开源权重为 INT4 量化发布版（config `quantization_config: compressed-tensors, int4 group_size=32, pack-quantized`，注意力/共享专家/dense 门投影/lm_head/视觉塔免量化）。

**与 K2.5 逐字段 config diff 的结论：唯一差异是 eos_token_id——K2.6 顶层与 text_config 均为 163586（K2.5 为 163585），其余全部字段（hidden/层数/MLA/384 专家/YaRN/262144/词表/vision_config/量化忽略名单）逐项一致。** 换句话说，K2.6 是 K2.5 架构的 2026-04 通用更新版；差异在发布层面（模型训练、API、产品策略）而非 config 可承载的结构。

一句话：**入口类 `KimiK25ForConditionalGeneration`（model_type `kimi_k25`），文本塔沿用 K2-0905 的 DeepSeek-V3 系 MLA + 384 专家 MoE（7168 hidden / 61 层）+ MoonViT-3D 视觉塔（patch14 / 27 层 / 1152 hidden），与 K2.5 的 config 唯一结构级差异为 eos=163586。**

### 1.1 官方 config 关键实参（来源：moonshotai/Kimi-K2.6/config.json，2026-11-03 拉取）

| 字段 | 取值 | 来源 |
|---|---|---|
| architectures | `KimiK25ForConditionalGeneration` | 官方 config |
| model_type | `kimi_k25` | 官方 config |
| text_config.model_type | `kimi_k2`（运行时由 configuration_kimi_k25.py:53-63 重写为 `deepseek_v3`） | 官方 config + 源码 |
| num_hidden_layers | 61 | 官方 config text_config |
| hidden_size | 7168 | 官方 config text_config |
| num_attention_heads (Q) / num_key_value_heads (KV) | 64 / 64 | 官方 config text_config |
| kv_lora_rank / q_lora_rank | 512 / 1536 | 官方 config text_config |
| qk_nope_head_dim / qk_rope_head_dim / v_head_dim | 128 / 64 / 128 | 官方 config text_config |
| n_routed_experts / n_shared_experts / num_experts_per_tok | 384 / 1 / 8 | 官方 config text_config |
| moe_intermediate_size / intermediate_size（dense） | 2048 / 18432 | 官方 config text_config |
| first_k_dense_replace | 1（第 0 层 dense FFN） | 官方 config text_config |
| routed_scaling_factor | 2.827 | 官方 config text_config |
| scoring_func / topk_method / aux_loss_alpha | sigmoid / noaux_tc / 0.001 | 官方 config text_config |
| rope_theta | 50000 | 官方 config text_config |
| rope_scaling | YaRN factor=64, original_max_position=4096, beta_fast=32, beta_slow=1 | 官方 config text_config |
| max_position_embeddings | 262144（256K） | 官方 config text_config |
| vocab_size | 163840 | 官方 config text_config |
| tie_word_embeddings（text_config） | false | 官方 config text_config |
| bos / eos / pad token id | 163584 / **163586** / 163839 | 官方 config text_config（eos 与 K2.5 的 163585 不同） |
| media_placeholder_token_id | 163605（顶层；图像/视频统一占位） | 官方 config 顶层 |
| use_unified_vision_chunk | true | 官方 config 顶层 |
| video_placeholder | `<\|kimi_k25_video_placeholder\|>` | 官方 config 顶层 |
| vision_config（是否存在） | 有（见 §6） | 官方 config 顶层 |
| quantization | compressed-tensors int4 group32, pack-quantized，忽略 self_attn / shared_experts / mlp 门投影 / lm_head / vision_tower / mm_projector | 官方 config quantization_config |

### 1.2 与 K2.5 逐字段 diff（来源：本目录 config.json vs kimi_k2_5/_tools/_cfgpull/config.Kimi-K2.5.json，程序比对）

| 字段 | K2.5 | K2.6 | 说明 |
|---|---|---|---|
| eos_token_id（顶层） | 163585 | **163586** | 唯一差异 |
| text_config.eos_token_id | 163585 | **163586** | 唯一差异 |

> 比对脚本输出：除上述两处外零差异（含 quantization_config 的 ignore 名单、vision_config 全部字段、YaRN 参数、384+1 专家配置）。

### 1.3 端到端总览

```
图像/视频 ──► MoonViT-3D (patch14, 27层, hidden1152, 双向+轴向2D-RoPE)
                 │ 2×2 时空均值池化 (sd2_tpool)
                 ▼
文本 token ──► embed_tokens ──► PatchMerger (LN1152→4608→GELU→7168)
                 │  media_placeholder_token_id=163605 占位位 masked_scatter
                 ▼
        61 层 K2 MoE 文本塔（第0层 dense SwiGLU 18432；1-60层 MoE：384 路由专家 top-8 + 1 共享专家，moe_inter 2048）
        每层：RMSNorm → MLA（kv_lora 512 / q_lora 1536）→ 残差；RMSNorm → MoE-FFN → 残差
                 ▼
              输出头（eos=163586 终止）
```

端到端代码路径与 K2.5 完全一致（源码：`models/kimi_k25/modeling_kimi_k25.py:686-720` 文本/视觉 token 合流；视觉池化 `:545-546, 145-157`；投影器 `:560-564`）。

## 2. 总体设计

K2.6 的架构三段与 K2.5 相同，全部可在官方 config 中逐字段锚定：

1. **MoonViT-3D 视觉塔**：patch 14×14、27 层、hidden 1152、16 头、FFN 4304、gelu_pytorch_tanh，双向（非因果）注意力，轴向 2D-RoPE（vision_config 逐项与 K2.5 一致）；
2. **PatchMerger 投影器**：merge_kernel (2,2) 展开 4×1152=4608 维经 in_proj→GELU→out_proj 投到文本 hidden 7168（mm_projector_type=patchmerger, merge_type=sd2_tpool）；
3. **K2 MoE 文本塔**：`AutoModel` 按 text_config 实例化为 DeepseekV3 系 CausalLM（hidden 7168 / 61 层 / MLA 低秩压缩注意力 + 384 routed 专家 MoE）。

## 3. 解码层解剖

文本塔解码层即 DeepseekV3DecoderLayer（源码：`models/deepseek_v3/modeling_deepseek_v3.py:485-529`），pre-norm 双残差块：

- `input_layernorm`（RMSNorm）→ MLA 自注意力 → 残差相加；
- `post_attention_layernorm`（RMSNorm）→ FFN（第 0 层 SwiGLU dense；1–60 层 MoE）→ 残差相加。

MoE 判定条件为 `layer_idx >= first_k_dense_replace(=1)`。FFN 子层：

- **Dense（第 0 层）**：SwiGLU，gate/up/down 三矩阵，intermediate 18432；
- **MoE（1–60 层）**：TopKRouter 选出 8 个 routed 专家（每专家 SwiGLU，intermediate 2048），输出加权求和后再加一路共享专家（n_shared_experts=1，intermediate 2048）。

## 4. 核心机制

### 4.1 无辅助损失的 sigmoid 路由 + NoauxTC 分组选专家

路由打分用 sigmoid（源码：`modeling_deepseek_v3.py:136`），带 e_score_correction_bias 修正项（:137）；按组内 top-2 分数之和选 group（n_group=1，本 config 退化为单组），再在掩码后分数上 top-8（:138-152），选中权重 softmax 归一化后乘 `routed_scaling_factor=2.827`（:154-157）。config 显式 `scoring_func=sigmoid`、`topk_method=noaux_tc`。注：本 config 留档的 `aux_loss_alpha=0.001`（与 K2.5 留档一致，以留档 config 为准）。

### 4.2 共享专家（shared expert）常驻全 token

共享专家（n_shared_experts=1，intermediate 2048）对每个 token 都前向一次，输出直接与 routed 专家加权和相加（`modeling_deepseek_v3.py:220-221`）。

### 4.3 投影器：PatchMerger（sd2 tpool）

视觉塔输出先按 merge_kernel_size (2,2) 做时空均值池化（空间 2×2 + 时间 2×2，token 数压到 1/4），再过 `pre_norm(LN 1152) → in_proj Linear(4608→4608) → GELU → out_proj Linear(4608→7168)`。config 字段 `mm_projector_type=patchmerger`、`merge_type=sd2_tpool`、`temporal_pool_type=average`。

## 5. 注意力

文本塔注意力为 **MLA（Multi-head Latent Attention）**，无滑窗、全因果注意力，无 GQA 头数缩减（Q/KV 均 64 头），通过低秩潜向量压缩 KV cache：

- Query 侧：`q_a_proj`（7168→1536）→ RMSNorm → `q_b_proj`（1536→64×(128+64)），q_lora_rank=1536；
- KV 侧：`kv_a_proj_with_mqa`（7168→512+64）拆出 512 维潜向量（RMSNorm）与 64 维 RoPE 部分；缓存只存压缩后的 512+64 维潜向量，解码时由 `kv_b_proj`（512→64×(128+128)）展开；
- RoPE 只加在 q/k 的 64 维 rope 部分（qk_rope_head_dim=64），nope 部分（128）不加位置编码；
- 缩放因子经 YaRN mscale 修正。

**不是**线性注意力：K2.6 文本塔仍是标准 DeepSeek-V3 系 MLA 因果全注意力；KDA 混合线性注意力属于 Kimi-Linear-48B-A3B（`models/kimi_linear/`）与 K3 家族，K2.6 未引入。

## 6. 多模态：MoonViT-3D 视觉塔与 token 流

### 6.1 视觉塔实参（来源：moonshotai/Kimi-K2.6/config.json 的 vision_config）

| 字段 | 取值 | 说明 |
|---|---|---|
| patch_size | 14 | 官方 config |
| vt_hidden_size / vt_num_layers / vt_num_attention_heads | 1152 / 27 / 16 | 官方 config |
| vt_intermediate_size | 4304 | 官方 config |
| init_pos_emb_height / width / time | 64 / 64 / 4 | 官方 config |
| merge_kernel_size | (2, 2) | 官方 config |
| merge_type / mm_projector_type | sd2_tpool / patchmerger | 官方 config |
| mm_hidden_size / text_hidden_size | 1152 / 7168 | 官方 config |
| video_attn_type | spatial_temporal | 官方 config |
| pos_emb_type | divided_fixed | 官方 config |

### 6.2 图像/视频 token 流

- **图像**：H×W×3 → patch14 → grid_h×grid_w 个 patch → 27 层双向视觉 Transformer → 2×2 空间池化 → 视觉 token → 投影到 7168 维 → 在 media_placeholder（id=163605）占位位 masked_scatter 填入文本序列。
- **视频**：与图像共享同一套权重（MoonViT-3D）；投影前时间均值池化实现时间压缩；多帧叠加正弦时间位置表。
- 顶层 `use_unified_vision_chunk=true`：图像/视频统一占位与 chunk 处理。

## 7. Token 与词表

词表 163840（官方 config）。特殊 token：bos=163584、**eos=163586**（K2.5 为 163585；与 K2-Thinking 相同）、pad=163839、media_placeholder=163605（图像/视频统一占位；顶层另保留 video_placeholder 字符串 `<|kimi_k25_video_placeholder|>`）。上下文 262144（YaRN factor 64 从 4096 原始长度外推，beta_fast=32）。

## 8. 演进

- **Kimi K2 / K2-0905（2025-07/09）**：1.04T-A32B 纯文本 MoE，DeepSeek-V3 系 MLA + 384 专家 top-8，本目录文本塔即其直接继承（见 `kimi_k2_0905` 目录；K2 论文 arXiv:2507.20534）。
- **Kimi K2.5（2026，见 `kimi_k2_5` 目录）**：K2 文本底座 + early-fusion 联合预训练 MoonViT-3D，原生多模态智能体；开源权重 INT4 发布；eos=163585。
- **Kimi K2.6（2026-04，本目录）**：与 K2.5 同 kimi_k25 架构，config 逐字段 diff 唯一为 eos=163586；通用多模态更新版，256K 上下文，thinking 可开关（API 名 `kimi-k2.6`）。
- **同代兄弟**：Kimi-K2-Thinking（见独立目录，eos=163586，纯文本推理变体）、Kimi-K2.7-Code（见 `kimi_k2_7_code` 目录，基于 K2.6 的编码 Agent 微调）、Kimi-K3（见 `kimi_k3` 目录，KDA+AttnRes 新架构）、Kimi-Linear-48B-A3B（KDA 混合线性注意力，`models/kimi_linear/`）、Kimi-VL-A3B（2025 上代小 VL，文本塔 2048/27 层/64 专家 top-6）。

## 9. 参考文献

1. 远端官方 config：https://hf-mirror.com/moonshotai/Kimi-K2.6/resolve/main/config.json （留档：本目录 `_tools/_cfgpull/config.json`，拉取 2026-11-03）
2. 官方模型卡/仓库：https://huggingface.co/moonshotai/Kimi-K2.6
3. Kimi K2.6 API 文档：https://platform.kimi.com/docs/guide/kimi-k2-6-quickstart
4. Kimi K2.5 技术报告（架构同构参考）：arXiv:2602.02276《Kimi K2.5: Visual Agentic Intelligence》
5. Kimi K2 技术报告：arXiv:2507.20534
6. transformers 源码（本机 E:\transformers 可编辑安装）：
   - `models/kimi_k25/configuration_kimi_k25.py`、`modeling_kimi_k25.py`、`image_processing_kimi_k25.py`
   - `models/deepseek_v3/configuration_deepseek_v3.py`、`modeling_deepseek_v3.py`
7. 对照留档：`kimi_k2_5/_tools/_cfgpull/config.Kimi-K2.5.json`（逐字段 diff 基准，2026-10-31 拉取）
