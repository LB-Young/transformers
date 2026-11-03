# Kimi K2.7-Code 家族架构详解

> 官方 HF 仓库：moonshotai/Kimi-K2.7-Code（远端官方 config：hf-mirror.com 镜像拉取，2026-11-03）；发布：2026-06-12（2026-06-15 出 HighSpeed 变体）；技术报告：无独立技术报告（架构基于 K2.6，同 kimi_k25 系）；多模态包装：transformers `src/transformers/models/kimi_k25/`（与 K2.5/K2.6 同模块，本机 E:\transformers 可编辑安装）。本机 E:\models 无该模型权重，全部 config 数值以远端官方 config.json 为准（留档 `_tools/_cfgpull/config.json`，拉取日期 2026-11-03）。config 无法承载的字段（preserve_thinking、协议、评测分数、权重格式细节）以官方模型卡/公告为来源并在文中逐项标注。

## 1. 家族全景

Kimi K2.7-Code 是 **基于 K2.6 的编码 Agent 微调版**：1T 总参 / 32B 激活，61 层（第 0 层 dense），384 路由专家（每 token 激活 8 + 1 共享专家），DeepSeek-V3 系 MLA，256K 上下文，MoonViT 视觉塔（约 400M）。**config 层面与 K2.6 逐字段比对几乎零差异**（仅 quantization_config.ignore 里一条正则由 `vision_tower` 变为 `vision_tower.*`，属忽略名单写法收紧，不影响权重结构）——其差异在发布层面：编码专用后训练（preserve_thinking 强制思维链）、Modified MIT 协议、FP8 + 原生 INT4（QAT）发布。

一句话：**入口类 `KimiK25ForConditionalGeneration`（model_type `kimi_k25`），文本塔与 K2.6 完全同构（7168 hidden / 61 层 / MLA 低秩 / 384+1 专家 top-8），Kimi Code Bench v2 官方评测 62.0 vs K2.6 的 50.9（来源：官方模型卡）。**

### 1.1 官方 config 关键实参（来源：moonshotai/Kimi-K2.7-Code/config.json，2026-11-03 拉取）

| 字段 | 取值 | 来源 |
|---|---|---|
| architectures | `KimiK25ForConditionalGeneration` | 官方 config |
| model_type | `kimi_k25` | 官方 config |
| text_config.model_type | `kimi_k2`（运行时重写为 `deepseek_v3`） | 官方 config + 源码 |
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
| bos / eos / pad token id | 163584 / 163586 / 163839 | 官方 config text_config |
| media_placeholder_token_id | 163605（顶层；图像/视频统一占位） | 官方 config 顶层 |
| use_unified_vision_chunk | true | 官方 config 顶层 |
| vision_config（是否存在） | 有（见 §6） | 官方 config 顶层 |
| quantization | compressed-tensors int4 group32, pack-quantized；ignore 名单：self_attn / shared_experts / mlp 门投影 / lm_head / vision_tower.*/mm_projector | 官方 config quantization_config |

### 1.2 与 K2.6 逐字段 diff（来源：本目录 config.json vs kimi_k2_6/_tools/_cfgpull/config.json，程序比对）

| 字段 | K2.6 | K2.7-Code | 说明 |
|---|---|---|---|
| text_config.quantization_config.ignore | 含 `re:vision_tower.*` | 含 `re:.*vision_tower.*` | 仅视觉塔忽略正则写法收紧（加 `.*` 前缀），不影响权重结构 |

> 比对脚本输出：除上述一条忽略正则外零差异。**架构层面 K2.7-Code = K2.6**；编码 Agent 能力来自后训练微调（config 不可承载）。

### 1.3 端到端总览

```
图像/视频 ──► MoonViT-3D (patch14, 27层, hidden1152, 双向+轴向2D-RoPE)
                 │ 2×2 时空均值池化 (sd2_tpool)
                 ▼
文本 token ──► embed_tokens ──► PatchMerger (LN1152→4608→GELU→7168)
                 │  media_placeholder_token_id=163605 占位位 masked_scatter
                 ▼
        61 层 K2 MoE 文本塔（第0层 dense SwiGLU 18432；1-60层 MoE：384 路由专家 top-8 + 1 共享专家）
                 ▼
        输出头（eos=163586）——后训练强制 preserve_thinking：输出必带思维链
```

端到端代码路径与 K2.5/K2.6 一致（源码：`models/kimi_k25/modeling_kimi_k25.py:686-720`）。

## 2. 总体设计

三段式架构与 K2.6 相同，全部可在官方 config 中逐字段锚定：MoonViT-3D 视觉塔（patch14 / 27 层 / hidden 1152 / 16 头 / FFN 4304 / gelu_pytorch_tanh / 轴向 2D-RoPE）→ PatchMerger 投影器（merge (2,2) → 4608→7168）→ K2 MoE 文本塔（7168 / 61 层 / MLA + 384 专家）。

K2.7-Code 与 K2.6 的设计差异不在 config，而在**发布层**：

- **强制 thinking（preserve_thinking）**：编码 Agent 使用中保留完整思维链历史，多轮对话需回传 assistant 消息全文（含 reasoning_content）——来源：官方模型卡（config 无对应字段）；
- **协议 Modified MIT**：相较 K2.5/K2.6 的开源协议更宽松——来源：官方模型卡；
- **权重格式：FP8 + 原生 INT4（QAT，量化感知训练）**——来源：官方模型卡（config 内 quantization_config 为 compressed-tensors int4 group32 pack-quantized，与 K2.6 同构，原生 INT4/QAT 的发布口径以模型卡为准）；
- **评测：Kimi Code Bench v2 得分 62.0，对比 K2.6 的 50.9**——来源：官方模型卡；
- **HighSpeed 变体（2026-06-15）**：官方公告发布的高速推理变体——来源：官方公告（config 不可区分，同一仓库主线）。

## 3. 解码层解剖

文本塔解码层即 DeepseekV3DecoderLayer（源码：`models/deepseek_v3/modeling_deepseek_v3.py:485-529`），pre-norm 双残差块：`input_layernorm`（RMSNorm）→ MLA 自注意力 → 残差；`post_attention_layernorm`（RMSNorm）→ FFN（第 0 层 SwiGLU dense，intermediate 18432；1–60 层 MoE，moe_intermediate 2048 + 共享专家 intermediate 2048）→ 残差。

## 4. 核心机制

### 4.1 无辅助损失的 sigmoid 路由 + NoauxTC

路由打分 sigmoid，带 e_score_correction_bias 修正项；组内 top-2 分数之和选 group（n_group=1，退化为单组），掩码后 top-8，归一化后乘 routed_scaling_factor=2.827。config 显式 `scoring_func=sigmoid`、`topk_method=noaux_tc`。

### 4.2 共享专家常驻全 token

n_shared_experts=1，intermediate 2048，对每个 token 前向一次，输出与 routed 专家加权和相加。

### 4.3 投影器：PatchMerger（sd2 tpool）

merge_kernel_size (2,2) 时空均值池化 → LN(1152) → in_proj(4608→4608) → GELU → out_proj(4608→7168)。config 字段 `mm_projector_type=patchmerger`、`merge_type=sd2_tpool`。

## 5. 注意力

文本塔注意力为 **MLA**（无滑窗、全因果、Q/KV 均 64 头）：q_a_proj(7168→1536) → RMSNorm → q_b_proj(1536→64×(128+64))；kv_a_proj_with_mqa(7168→512+64) 拆出 512 维潜向量 + 64 维 RoPE 段，缓存只存潜向量，解码时 kv_b_proj(512→64×(128+128)) 展开；RoPE 只加在 64 维 rope 部分；YaRN mscale 修正。**不是**线性注意力（KDA 属 Kimi-Linear/K3 家族，本模型未引入）。

## 6. 多模态：MoonViT 视觉塔

### 6.1 视觉塔实参（来源：moonshotai/Kimi-K2.7-Code/config.json 的 vision_config）

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

> 视觉塔参数量约 400M 的口径与 MoonViT-3D（1152/27 层）规模相符；精确参数量未在 config 单列（待确认：模型卡若未给出精确值则不补数）。

### 6.2 图像/视频 token 流

图像：patch14 → 27 层双向视觉 Transformer → 2×2 空间池化 → 视觉 token → 投影 7168 维 → media_placeholder（163605）占位位 masked_scatter。视频：与图像共享权重，时间均值池化压缩 token。

## 7. Token 与词表

词表 163840（官方 config）。特殊 token：bos=163584、eos=163586、pad=163839、media_placeholder=163605。上下文 262144（YaRN factor 64 从 4096 外推，beta_fast=32）。

## 8. 演进

- **Kimi K2.6（2026-04，见 `kimi_k2_6` 目录）**：K2.5 架构的通用更新版，eos=163586，config 与 K2.5 仅 eos 差异。
- **Kimi K2.7-Code（2026-06-12，本目录）**：基于 K2.6 的编码 Agent 微调；config 与 K2.6 几乎零差异，差异在 preserve_thinking 强制思维链、Modified MIT 协议、FP8 + 原生 INT4（QAT）发布；Kimi Code Bench v2 62.0（vs K2.6 50.9）。
- **同代兄弟**：Kimi-K2-Thinking（见独立目录）、Kimi-K3（见 `kimi_k3` 目录，KDA+AttnRes 新架构旗舰）、K2.8 Preview（2026-09-11 接管 kimi-for-coding；HF 无官方权重仓库，搜索 0 命中——不臆造配置）、Kimi-Linear-48B-A3B（KDA 混合线性注意力，`models/kimi_linear/`）、Kimi-VL-A3B（2025 上代小 VL，文本塔 2048/27 层/64 专家 top-6）。

## 9. 参考文献

1. 远端官方 config：https://hf-mirror.com/moonshotai/Kimi-K2.7-Code/resolve/main/config.json （留档：本目录 `_tools/_cfgpull/config.json`，拉取 2026-11-03）
2. 官方模型卡/仓库：https://huggingface.co/moonshotai/Kimi-K2.7-Code （preserve_thinking、Modified MIT、FP8 + 原生 INT4 QAT、Kimi Code Bench v2 62.0 vs 50.9、HighSpeed 变体出处）
3. 架构同构参考：`kimi_k2_6/index.md`（K2.6）、`kimi_k2_5/index.md`（K2.5 技术报告 arXiv:2602.02276）
4. Kimi K2 技术报告：arXiv:2507.20534
5. transformers 源码（本机 E:\transformers 可编辑安装）：
   - `models/kimi_k25/configuration_kimi_k25.py`、`modeling_kimi_k25.py`
   - `models/deepseek_v3/configuration_deepseek_v3.py`、`modeling_deepseek_v3.py`
