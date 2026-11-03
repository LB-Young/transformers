# DeepSeek-V4.1-Flash 家族架构详解 — 5120·40层 · CED 非对称 · CSA2+FP4 · Engram · DSpark

> 说明：本目录对应官方仓库 **deepseek-ai/DeepSeek-V4.1-Flash**（model_type `deepseek_v41`，入口类 `DeepseekV41ForCausalLM`，config 标注 transformers 5.6.0）。2026-09-10 发布上线，2026-10-17 已迭代至 **V4.1-Flash-1017**（网页端/APP/API 同步，API 名仍为 `deepseek-flash`）。本地 `E:\transformers\src\transformers\models\` 暂**无** `deepseek_v41` 模块（2026-11-03 核查，仅有 v2/v3/v32/v4/vl/vl_hybrid/ocr2），无法本地断点；以下数字一律以官方 config（hf-mirror 拉取，存 `_tools\_cfgpull\`）、arXiv:2609.19969、官方新闻页与 NVIDIA NIM 部署文档为准。

- 徽标：入口类 DeepseekV41ForCausalLM · model_type deepseek_v41 · **552B 主干 MoE**（论文/新闻口径）
- 徽标：**CED 因果编解码非对称架构**：prefill 每 token 仅激活 **8B**，decode 激活 **16B**
- 徽标：注意力 **MQA**（64 头 · kv_head=1 · head_dim=512）+ **CSA2 跨层 KV 复用**（源层 2/8/14/20）+ **FP4 KV** + sliding_window=128
- 徽标：MoE **384 路由专家选 6 + 1 共享**（V4 的 256→384）· sqrtsoftplus / noaux_tc
- 徽标：1M 上下文 · 全局 KV 缓存 **890 字节/token**（V4-Flash 的 1/4，HBM）；常驻 KV 约 **1/8**（SSD/主机内存，SWA Bounded Replay）
- 徽标：**Engram 条件记忆**（层 1/14 · 16M 词表 · ~196B 参数）+ **DSpark 投机解码** + **MTP ×3** · fp8 主权重 + **fp4 专家**
- 徽标：**原生多模态**（vision_config：32 层视觉塔 · patch 14 · downsample 3 · 单图 ≤1024 token）

## 1 家族全景：V4.1-Flash config

| 字段 | 取值 | 来源 |
|---|---|---|
| architectures / model_type | DeepseekV41ForCausalLM / deepseek_v41 | config 顶层 |
| 主干规模 / 激活 | 552B 总参（MoE）；**CED：prefill 8B / decode 16B** 每 token | 论文摘要 + 官方新闻 |
| hidden_size / num_hidden_layers | 5120 / 40 | config |
| num_attention_heads / num_key_value_heads / head_dim | 64 / **1（MQA）** / 512 | config |
| q_lora_rank / o_lora_rank / qk_rope_head_dim | 1280 / 1024 / 64 | config（Q 低秩高于 V4 的 1024） |
| n_routed_experts / num_experts_per_tok / n_shared | 384 / **6** / 1 | config（NVIDIA 文档同口径） |
| moe_intermediate_size | 2304 | config |
| scoring_func / topk_method / routed_scaling_factor / norm_topk_prob | sqrtsoftplus / noaux_tc / 1.5 / true | config |
| o_groups | 8 | config |
| sliding_window | 128 | config |
| max_position_embeddings | 1048576（1M） | config |
| rope_scaling | yarn factor=16 · original_max=65536 · beta_fast=32 / beta_slow=1 · rope_theta=10000 · compress_rope_theta=160000 | config |
| vocab_size / tie_word_embeddings | 129280 / false | config |
| compress_ratios | 40 层列表：前 2 层 0，第 3–20 层 2，第 21–37 层 1，末 3 层 0（CSA2 跨层压缩比） | config |
| kv_source_layer_ids | [2, 8, 14, 20]（CSA2 跨层 KV 复用源层） | config |
| index_source_layer_ids / index_n_heads / index_head_dim / index_topk | [2,8,14,20,24,28,32,36] / 32 / 128 / **512**（DSA 索引器源层） | config |
| candidate_source_layer_id / candidate_topk_blocks / candidate_block_size | 20 / 2048 / 8（Engram 检索候选） | config |
| engram_layer_ids / engram_num_embeddings | [1, 14] / [384006168, 384016682] | config |
| engram_vocab_size / engram_n_heads / engram_head_dim / engram_compressed_vocab_size | 16000000 / 8 / 256 / 99092 | config（≈196B 条件记忆参数，NVIDIA 口径） |
| hc_mult / hc_sinkhorn_iters / hc_eps | 4 / 20 / 1e-6（mHC 超连接，延续 V4） | config |
| num_nextn_predict_layers | **3（MTP ×3）** | config（V4 为 ×1） |
| dspark_block_size / dspark_target_layer_ids / dspark_markov_rank | 5 / [37, 38, 39] / 256 | config（DSpark 投机解码） |
| dspark_n_routed_experts / dspark_num_experts_per_tok | 128 / 3 | config |
| quantization | fp8（dynamic）· weight_block [32,32] · scale_fmt ue8m0 · **expert_dtype=fp4** | config |
| image_token_id / bos / eos / pad | 129264 / 0 / 1 / 2 | config |
| vision_config | deepseek_v41_vision：32 层 · hidden 1024 · 16 头 · inter 2816 · patch 14 · downsample_ratio 3 · max_image_tokens 1024 · min_pixels 295936 | config |

## 2 端到端数据流总览（CED）

> 文本/图像 token 输入 → **CED 编解码主干**（40×DecoderLayer：输入阶段仅激活 8B/ token，输出阶段激活 16B/token）→ 最终 RMSNorm → LM Head；MTP ×3 多 token 预测头 + DSpark 投机解码器挂在末 3 层（37/38/39）。

- 官方定位：面向**输入重负载的 Agent 场景**（长上下文 prefill 成本高），非对称激活把成本压向 decode 侧更重的生成阶段。
- 与 V4-Flash 的 KV 演进（官方图，字节/token）：V1 389,120 → V3.2 48,068（8.1×）→ V4-Flash 3,514（13.7×，2026-04）→ **V4.1-Flash 890（3.9×，2026-09）**。

## 3 核心机制

### 3.1 ① CED 因果编解码（Causal Encoder-Decoder）非对称架构

- 论文口径：**decode 每 token 激活 16B 参数，prefill 仅激活 8B**，主干总参 552B（MoE）。
- 目的：prefill（输入处理）是长上下文 Agent 的主要算力与成本来源，非对称设计显著改善 input-heavy 工作负载的单位成本；官方新闻称其性能、成本、速度、总耗时全面领先 V4-Pro。

### 3.2 ② CSA2 压缩稀疏注意力 + FP4 KV 缓存（KV 压缩极限）

- **CSA2（Compressed Sparse Attention 2）**：在 V4 的 HCA/CSA 基础上引入**跨层 KV 缓存复用**——`kv_source_layer_ids=[2,8,14,20]` 四个源层的 KV 被下游层复用，`compress_ratios` 逐层指定压缩比（前 2 层不压缩，第 3–20 层 2，第 21–37 层 1，末 3 层 0）。
- **FP4 主 KV 缓存**：KV 主体以 FP4 存储（`expert_dtype=fp4`，主权重仍 fp8）。
- 效果：全局 KV 缓存（常驻 HBM）压到 **890 字节/token ≈ V4-Flash 的 1/4**。
- **SWA Bounded Replay**：滑窗注意力（sliding_window=128）的有界回放优化，把**常驻 KV 缓存**（常驻 SSD/主机内存）压到 **V4-Flash 的约 1/8**。

### 3.3 ③ mHC 超连接（延续 V4，Single-Pass）

- hc_mult=4 通道扩展 + Sinkhorn 20 迭代门控，继续替代标准残差；NVIDIA 文档标注为 Single-Pass mHC。

### 3.4 ④ Engram 条件记忆（conditional memory）

- 新增**条件记忆**模块，挂在 `engram_layer_ids=[1,14]` 两层：16M 词表（`engram_vocab_size=16000000`）× 8 头 × head_dim 256，压缩词表 99092，ngram 上限 4；候选检索自第 20 层（`candidate_source_layer_id=20`，top-2048 blocks、block size 8）。
- NVIDIA 口径：该记忆模块约 **196B 参数**——解释为何 552B 主干在激活侧仍只有 8B/16B（大部分参数为条件记忆与 MoE 专家，按需激活）。

### 3.5 ⑤ DSpark 投机解码 + MTP ×3

- **DSpark**：挂在末 3 层（`dspark_target_layer_ids=[37,38,39]`）的投机解码器，独立小 MoE（128 路由专家选 3，`dspark_markov_rank=256`，block_size=5，noise token 128799），加速自回归。
- **MTP**：多 token 预测头从 V4 的 ×1 扩到 **×3**（`num_nextn_predict_layers=3`）。

### 3.6 ⑥ 原生多模态视觉

- 官方新闻：V4.1-Flash 为**新架构家族最小尺寸成员，原生视觉理解**（V4-Flash 与 V4-Flash-Vision-Exp 均已退役，`deepseek-v4-flash` / `deepseek-v4-flash-vision-exp` API 名暂时路由到 V4.1-Flash）。
- vision_config：32 层视觉塔（hidden 1024 / 16 头 / inter 2816 / patch 14 / downsample 3），单图 ≤1024 token（max_image_tokens），min_pixels 295936；`image_token_id=129264`。

## 4 注意力机制一览

| 项 | V4-Flash-Base | V4.1-Flash |
|---|---|---|
| 形式 | MQA（kv_head=1，head_dim=512） | MQA（kv_head=1，head_dim=512） |
| Q/O 低秩 | q_lora=o_lora=1024 | q_lora=1280 / o_lora=1024 |
| 稀疏 | DSA index_topk=512 + sliding_window=128 | DSA index_topk=512 + 滑窗 128 + **CSA2 跨层 KV 复用** |
| KV 缓存 | 3,514 字节/token（官方图，2026-04） | **890 字节/token**（2026-09，3.9×） |
| KV 精度 | — | **FP4 主 KV** |
| 上下文 | 1,048,576（YaRN 16） | 1,048,576（YaRN 16） |
| RoPE | qk_rope_head_dim=64，compress_rope_theta=160000 | 同 V4 |

## 5 特殊 Token 与词表

| token | ID | 来源 |
|---|---|---|
| bos / eos / pad | 0 / 1 / 2 | config |
| image_token | 129264 | config（原生多模态） |
| dspark_noise | 128799 | config |
| engram_pad | 2 | config |
| vocab_size | 129280 | config（同 V3/V4） |

## 6 演进视角：V4-Flash → V4.1-Flash

| 维度 | V4-Flash-Base（2026-04） | V4.1-Flash（2026-09） |
|---|---|---|
| hidden / 层数 | 4096 / 43 | **5120 / 40** |
| 架构 | 标准因果 LM（MQA+DSA+mHC+Hash MoE） | **CED 非对称**：prefill 8B / decode 16B |
| MoE | 256 选 6 + 1 共享，前 3 层 Hash MoE | **384 选 6 + 1 共享**（config 无 num_hash_layers 字段，Hash MoE 未延续） |
| 注意力 | MQA + DSA(top512) + 滑窗 128 | MQA + DSA(top512) + 滑窗 128 + **CSA2 跨层复用** |
| KV 缓存 | 3,514 字节/token | **890 字节/token（FP4 + CSA2）** |
| 残差 | mHC（hc_mult=4） | mHC（Single-Pass） |
| 投机 | MTP ×1 | **MTP ×3 + DSpark** |
| 条件记忆 | 无 | **Engram（层 1/14，~196B 参数）** |
| 上下文 | 1M | 1M |
| 量化 | fp8 e4m3 | fp8（dynamic）+ **专家 fp4** |
| 多模态 | 无（纯文本 Flash-Base） | **原生多模态**（vision_config + image token） |

> 产品线现状（官方新闻/更新日志）：V4-Flash 与 V4-Flash-Vision-Exp 已退役；V4-Pro 正在退出（2026-09-14 起 `deepseek-v4-pro` 请求按 V4.1-Flash 计费路由），**V4.1-Pro 待发布**；V4.1-Flash 于 2026-10-17 升级为 **1017** 并开放 Search / Fetch 公共 API。完整尺寸 V4 主仓（deepseek-ai/DeepSeek-V4）仍未见公开（此前 404，**待确认**）。

## 7 来源

1. 官方 config（hf-mirror 2026-11-03 拉取，存 `_tools\_cfgpull\`）`deepseek-ai/DeepSeek-V4.1-Flash.config.json`（deepseek_v41 / 5120 / 40 层 / 384 专家 / CSA2+FP4 / Engram / DSpark / MTP×3 / 原生多模态）+ tokenizer_config.json（generation_config 仓库 404）
2. 官方论文：arXiv:2609.19969（2026-09-17，DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression）——552B 主干、CED 8B/16B、CSA2+FP4 → 890B/token、SWA Bounded Replay → 常驻 KV 1/8、45T token 多模态预训练
3. 官方新闻：https://www.deepseek.com/en/news/deepseek-v4-1-flash/ （2026-09-10，非对称架构、KV 演进图、API 退役/路由策略）
4. NVIDIA NIM：https://docs.api.nvidia.com/nim/reference/nvidia-deepseek-v4_1-flash （CSA2 / SWA Bounded Replay / FP4 KV / Single-Pass mHC / 196B Engram / DSpark / 384 路由选 6）
5. API 更新日志：https://api-docs.deepseek.com/zh-cn/updates/ （2026-10-17，V4.1-Flash-1017 升级 + Search/Fetch API）
6. 官方模型仓库：https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash

> 标注约定：全部数字来自官方 config / arXiv 摘要 / 官方新闻 / NVIDIA 部署文档；本地 transformers 尚无 `deepseek_v41` 模块，无法本地断点（**待 transformers 版本更新**）。访问与核验日期：2026-11-03。

---

*页脚：DeepSeek-V4.1-Flash 家族架构详解 · 数据来源：官方 config（hf-mirror）+ arXiv:2609.19969 + 官方新闻 + NVIDIA NIM · 仅供技术学习与论文研究*
