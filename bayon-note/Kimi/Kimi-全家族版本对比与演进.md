# Kimi（Moonshot）全家族版本对比与演进 — Moonlight → K2-Base → K2-0905 → K2-Thinking → K2.5 / K2.5-VL

> 整理日期：2026-11-02
> 数据来源：各版本目录 `index.md`（其事实全部源自远端官方 config（hf-mirror 拉取留档 `_tools\_cfgpull\`）、transformers 本地源码（`models\deepseek_v3\` / `models\kimi_k25\`）、官方论文与官方仓库，详见文末来源）
> 覆盖版本：moonshot_8b（官方 404，待确认）、moonshot_32b_a32b（官方 404，待确认）、kimi_k2_0905（K2-Base/Instruct-0905）、kimi_k2_thinking、kimi_k2_5（K2.5，含 K2.5-VL 视觉塔）、kimi_k2_vl（= K2.5 多模态，无独立仓库）
> 用途：配合 `E:\transformers` 源码研读与断点 debug 的家族横向速查

---

## 结论速览

Kimi 家族的演进主线是一条「**DeepSeek-V3 系 MLA-MoE 范式下的规模跃迁 + 模态扩展**」路线：

1. **Moonlight-16B-A3B（2025-02/03）**：Moonshot 首个真实开源稀疏模型——DeepSeek-V3 式 MLA + 64 路由专家/2 共享/top-6，Muon 优化器大规模训练配套发布；规模 16B-A3B。
2. **Kimi-K2-Base（2025-07）**：**1T-A32B 级跃迁**——hidden 7168/61 层、MLA 新增 q 侧低秩（q_lora 1536）、路由专家扩到 **384 + 1 共享 top-8**（稀疏度 48），上下文 131072；官方论文 arXiv:2507.20534。
3. **Kimi-K2-0905（2025-09）**：与 Base 同文本塔，工程升级——上下文经 YaRN 扩到 **262144**、权重改 **fp8 e4m3** 官方发布。
4. **Kimi-K2-Thinking（2025-09）**：**架构零改动**的推理变体——config 与 K2-0905 逐字段一致（仅 eos=163586、YaRN beta_fast=1.0），差异在长思维链 SFT + RL；INT4 发布，纯文本。
5. **Kimi-K2.5（2026）**：**原生多模态智能体**——K2 文本塔 + **MoonViT-3D 视觉塔**（patch14/27 层/1152，双向、轴向 2D-RoPE，初始化自 SigLIP-SO-400M）+ **PatchMerger 投影器**（时空 2×2 池化 → 4608→7168）；early-fusion 联合预训练约 15T token；INT4 发布，图像/视频权重共享、视频 4× 时间压缩；K2.5-VL 即其视觉塔专述（官方无独立 K2-VL 仓库）。
6. **Moonshot-8B / Moonshot-32B-A32B**：经 hf-mirror / ModelScope 多通道核验 **404（不存在或已下架）**，无 config 留痕，全部规格标「待确认」；任务书描述若属实，最接近的真实对象是 Moonlight-16B-A3B（16B-A3B）与 K2 系（1T-A32B）——按源文档口径如实记录，不臆造数字。

一句话：**K2 系是"DeepSeek-V3 架构模板的 1T 级复用"（MLA 加 q 低秩、专家 64→384），Thinking 是"同架构换训练"（推理 RL），K2.5 是"文本塔加视觉塔换模态"（MoonViT-3D + PatchMerger），早期"Moonshot-8B/32B-A32B"不可考**。

---

## 一、全家族核心规格对比表

| 项目 | Moonlight-16B-A3B（真实锚点） | Kimi-K2-Base | Kimi-K2-0905 | Kimi-K2-Thinking | Kimi-K2.5（含 K2.5-VL） | Moonshot-8B / 32B-A32B |
|---|---|---|---|---|---|---|
| 时间 | 2025-02/03 | 2025-07 | 2025-09 | 2025-09 | 2026 | —（404） |
| 入口类 / model_type | DeepseekV3ForCausalLM / deepseek_v3 | DeepseekV3ForCausalLM / kimi_k2 | 同左 | 同左 | **KimiK25ForConditionalGeneration / kimi_k25**（text 重写为 deepseek_v3） | 待确认 |
| 形态 | MoE（16B-A3B） | **MoE（1.04T-A32B）** | 同 Base | 同 Base | 同 K2 + 视觉塔 | 8B dense? / A32B MoE?（待确认） |
| hidden / 层数 | 2048 / 27 | 7168 / 61（第 0 层 dense） | 7168 / 61 | 7168 / 61 | 7168 / 61 | 待确认 |
| 注意力 | **MLA**（kv_lora 512，**无 q_lora**，nope128/rope64/v128） | MLA（kv 512 + **q 1536**） | 同 Base | 同 Base | 同 K2（文本塔） | 待确认（是否 MLA 未知） |
| Q/KV 头 | 16 / 16 | 64 / 64 | 64 / 64 | 64 / 64 | 64 / 64 | 待确认 |
| 路由专家 / 共享 / top-k | 64 / 2 / 6 | **384 / 1 / 8** | 同 Base | 同 Base | 同 K2 | 待确认 |
| moe_intermediate | 1408 | 2048 | 2048 | 2048 | 2048 | 待确认 |
| dense intermediate | 11264 | 18432 | 18432 | 18432 | 18432 | 待确认 |
| 路由策略 | sigmoid + noaux_tc | sigmoid + noaux_tc（scaling 2.827） | 同 Base | 同 Base（aux_loss 0.0） | 同 K2 | 待确认 |
| 上下文 | 8192 | 131072 | **262144**（YaRN ×64） | 262144 | 262144 | 128K？（待确认） |
| rope_theta | 50000 | 50000 | 50000 | 50000（beta_fast 1.0） | 50000（beta_fast 32） | 待确认 |
| 词表 | 163840 | 163840 | 163840 | 163840 | 163840 | 待确认 |
| 权重格式 | bf16 | bf16 | **fp8 e4m3** | **INT4**（compressed-tensors g32） | **INT4**（注意力/共享专家/lm_head/视觉塔免量化） | 待确认 |
| eos_token_id | — | 163585 | 163585 | **163586** | 163585 | 待确认 |
| 视觉塔 | 无 | 无 | 无 | 无 | **MoonViT-3D**：patch14 / 27 层 / hidden 1152 / 16 头 / FFN 4304 / gelu_pytorch_tanh / 双向 / 轴向 2D-RoPE / 初始化自 SigLIP-SO-400M | 无 |
| 投影器 | 无 | 无 | 无 | 无 | **PatchMerger**：merge(2,2) 时空均值池化 → LN(1152) → 4608→GELU→7168 | 无 |
| 多模态处理 | — | — | — | — | **NaViT patch packing**（不切 tile）+ max_patches 16384 + 视频 4× 时间压缩 | — |
| 训练特点 | Muon 优化器首次大规模实践 | 15.5T tokens、MuonClip | 同 Base | **长思维链 SFT + RL**（架构外） | 视觉塔先对齐 Moonlight（1T）→ 投影桥接 → 联合预训练约 15T | — |
| 主要论文 | 官方博客 | arXiv:2507.20534 | 同左 | 同左 | arXiv:2602.02276 | — |

> 说明：① Moonshot-8B / Moonshot-32B-A32B 两个目录记录的是核验过程——hf-mirror 组织列表 19 个仓库、全文搜索、ModelScope 直连均无此二模型，Wayback 不可达；按源文档约定**不臆造任何数字**。② K2.5-VL 无独立 HF 仓库（官方多模态即 Kimi-K2.5，NVIDIA 文档称 "Kimi-K25-VL"），视觉侧规格与 K2.5 完全一致。③ 同代兄弟（源文档提及、未单独立目录）：Kimi-K2.6（同 kimi_k25 架构，eos=163586）、K2.7-Code、Kimi-K3（2026-07，2.8T，KDA 混合线性注意力）、Kimi-Linear-48B-A3B、Kimi-VL-A3B（2025 上代小 VL，文本塔 2048/27 层/64 专家 top-6）。

---

## 二、代际演进详解

### 2.1 早期（2025-02/03）：Moonlight-16B-A3B —— 开源起点与 Muon 实践

- 形态：MoE 16B 总 / 3B 激活；hidden 2048、27 层。
- 注意力：MLA，`kv_lora_rank=512`、无 q 侧低秩、qk_nope 128 / qk_rope 64 / v_head 128。
- MoE：64 路由专家 + 2 共享，top-6，moe_inter 1408。
- 上下文 8192、词表 163840、rope_theta 50000。
- 意义：确认「DeepSeek-V3 系 MLA-MoE」作为 Moonshot 文本塔模板；Muon 优化器首次大规模训练实践配套发布。

### 2.2 Kimi-K2-Base（2025-07）→ K2-0905（2025-09）：1T 级规模跃迁 + 工程升级

| 维度 | Moonlight-16B-A3B | Kimi-K2-Base | Kimi-K2-0905 |
|---|---|---|---|
| 规模 | 16B-A3B | **1.04T-A32B**（总参/激活） | 同 Base |
| hidden / 层 | 2048 / 27 | **7168 / 61**（第 0 层 dense） | 同 Base |
| MLA | kv 512（无 q_lora） | **kv 512 + q_lora 1536** | 同 Base |
| 路由专家 | 64 + 2 共享，top-6 | **384 + 1 共享，top-8**（稀疏度 48） | 同 Base |
| 上下文 | 8192 | 131072 | **262144（YaRN ×64，orig 4096）** |
| 权重 | bf16 | bf16 | **fp8 e4m3**（block 128，dynamic） |
| 优化器 | Muon | MuonClip（15.5T tokens） | 同 Base |

- K2-Base 的关键设计（论文口径）：固定激活参数量下**扩大路由专家总数（384/8=48 稀疏度）可降低 loss**；MLA 增加 q 侧低秩压缩（q_lora 1536）进一步压 KV。
- 0905 是 2025-09 更新的 Instruct 检查点：文本塔与 Base 同架构，工程差异集中在**上下文翻倍到 256K** 与 **fp8 官方权重**。

### 2.3 Kimi-K2-Thinking：同架构推理变体（架构零改动）

- 与 K2-0905 文本塔 **config 逐字段一致**（hidden 7168 / 61 层 / MLA 同维 / 384+1 top-8 / routed_scaling 2.827 / vocab 163840 / max 262144）。
- 唯一差异：`eos_token_id=163586`（0905 为 163585）、YaRN `beta_fast=1.0`（K2.5 为 32）、`aux_loss_alpha=0.0`、无 vision_config、无 nextn 预测层。
- 差异本质在**训练**：长思维链（long CoT）SFT + 强化学习，激发「先思考再回答」；config 无 reasoning 专用开关字段。
- 权重：INT4（compressed-tensors，group32，dynamic，act-order，weight-only）。

### 2.4 Kimi-K2.5 / K2.5-VL（2026）：文本塔 + 视觉塔 = 原生多模态智能体

| 部件 | 规格 | 关键点 |
|---|---|---|
| 文本塔 | = K2（7168/61/MLA/384+1 top-8） | text_config.model_type 运行时重写为 `deepseek_v3`，复用 DeepSeek-V3 实现 |
| 视觉塔 MoonViT-3D | patch14 · 27 层 · hidden 1152 · 16 头 · FFN 4304 · gelu_pytorch_tanh | 双向（非因果）全注意力、轴向 2D-RoPE（head_dim 72 / spatial 36）、可学习 64×64×4 位置网格（超网格双三次插值）、初始化自 SigLIP-SO-400M |
| 投影器 PatchMerger | merge_kernel (2,2) 时空均值池化 → LN(1152) → in_proj(4608→4608) → GELU → out_proj(4608→7168) | 空间 2×2 + 时间 2×2，token 压到 1/4；约 54M 参数 |
| 多模态处理 | **NaViT patch packing**（原生分辨率、不切 tile、不拼缩略图） | max_patches=16384 预算等比缩放；多图 1D 拼接 + cu_seqlens 变长掩码；视频最多 4 帧共享图像权重、时间均值池化 4× 压缩 |
| 特殊 token | vision_start=163602 / image=163605 / video=163840 / vision_end=163604 | masked_scatter 在 embedding 层填入（early fusion） |
| 训练 | ① 视觉塔 caption loss 对齐 Moonlight-16B-A3B（约 1T token）→ ② 短训投影器桥接 → ③ 联合预训练约 15T 视觉-文本（4K 长度、固定低视觉比例）→ ④ YaRN 32K→262144 | 与「文本训完再加视觉」的后期融合路线不同 |
| 发布 | INT4 量化（注意力/共享专家/dense lm_head/视觉塔免量化），BF16 官方未释出 | K2.6 同架构（eos=163586） |

- **K2.5-VL 与 K2.5 是同一模型**：官方无独立 Kimi-K2-VL 仓库；K2 代多模态能力即由 `KimiK25ForConditionalGeneration` 承载。
- 与 InternVL 路线的本质区别：K2.5 走**原生分辨率 + NaViT packing**（不切 tile），InternVL 走**动态 tile + 缩略图**。

---

## 三、关键机制演进一图流

```
文本塔模板：  DeepSeek-V3 系 MLA-MoE（Moonlight 起恒定，K2 全系复用）
MLA：         kv_lora 512 / 无 q_lora（Moonlight）→ kv 512 + q_lora 1536（K2 系）
MoE：         64 路由 + 2 共享 top-6（Moonlight）→ 384 路由 + 1 共享 top-8（K2 系）
上下文：      8192（Moonlight）→ 131072（K2-Base）→ 262144（K2-0905/Thinking/K2.5）
权重格式：    bf16（Moonlight/Base）→ fp8 e4m3（0905）→ INT4（Thinking/K2.5）
模态：        纯文本（Moonlight/K2-Base/0905/Thinking）→ 文本+图像+视频（K2.5）
视觉塔：      — → MoonViT-3D（patch14/27层/1152/双向/轴向2D-RoPE）+ PatchMerger（K2.5）
训练：        Muon 实践（Moonlight）→ 15.5T 预训练（K2）→ 长 CoT RL（Thinking）
             → early-fusion 联合预训练约 15T（K2.5）
```

演进主线解读：
1. **模板恒定**：从 Moonlight 到 K2.5，文本塔始终是 DeepSeek-V3 系 MLA + 细粒度专家 MoE + sigmoid/noaux_tc 路由；代际差异在规模、上下文与模态。
2. **MLA 一次补强**：K2 起加 q_lora 1536（Moonlight 没有），KV cache 低秩 512 不变。
3. **容量路线**：16B-A3B → 1.04T-A32B（专家 64→384）；K2.5 加视觉塔但文本塔不动。
4. **Thinking/K2.5 都是"架构复用 + 训练/模态换新"**：Thinking 换推理训练，K2.5 换多模态塔与投影器。
5. **早期"Moonshot-8B / 32B-A32B"不可考**：官方 404，按核验记录如实呈现，不补数字。

---

## 四、对源码研读 / debug 的启示

1. **按 model_type 分流**：`kimi_k2`（K2-Base/0905/Thinking）config.architectures 直指 `DeepseekV3ForCausalLM`——本地 `models\deepseek_v3\` 即等价实现，打断点走 DeepseekV3 路径；`kimi_k25`（K2.5/K2.6）走 `models\kimi_k25\`（KimiK25ForConditionalGeneration），其 text_config.model_type 运行时被重写为 `deepseek_v3`（configuration_kimi_k25.py:53-63）。
2. **重点调试对象**：
   - MLA 的低秩链路：`q_a_proj(7168→1536) → RMSNorm → q_b_proj(1536→64×192)`；`kv_a_proj_with_mqa(7168→512+64)` 缓存只存 512 维潜向量 + 64 维 rope 段，解码时 `kv_b_proj` 展开（modeling_deepseek_v3.py:349-463）；
   - MoE 路由：sigmoid 打分（:136）→ e_score_correction_bias（:137）→ top-8（:138-152）→ 归一化 × routed_scaling_factor 2.827（:157），第 0 层 dense SwiGLU（inter 18432）、1–60 层 MoE（inter 2048）+ 共享专家（:220-221）；
   - K2.5 视觉链路：MoonViT-3D 的轴向 2D-RoPE（head_dim 72 / spatial 36）、NaViT packing 的 cu_seqlens 变长掩码、PatchMerger 的 merge (2,2) 时空池化（modeling_kimi_k25.py:117-159, 362-390, 545-572, 686-720）。
3. **同架构对照实验**：K2-Base / 0905 / Thinking 共用 `models\deepseek_v3\` 代码，config diff 只有上下文、量化与 eos——适合做「同架构不同训练/格式」对照；K2.5 的视觉塔与文本塔在代码里分属两个模块，可单独断点。
4. **本地可跑性**：K2 系主模型 1T 级本地不可跑；K2-Thinking / K2.5 的 INT4 发布版是唯一可考虑本地加载的形态（仍需较大显存）；Moonlight-16B-A3B（16B-A3B）最适合本地双 16G 卡断点调试。

---

## 参考来源

1. `E:\transformers\bayon-note\Kimi\moonshot_8b\index.md`（Moonshot-8B 404 核验记录 + Moonlight-16B-A3B config 对照）
2. `E:\transformers\bayon-note\Kimi\moonshot_32b_a32b\index.md`（Moonshot-32B-A32B 404 核验记录 + Moonlight 对照）
3. `E:\transformers\bayon-note\Kimi\kimi_k2_0905\index.md`（K2-Base/0905：官方 config 留档 + 本地 deepseek_v3 等价实现 + arXiv:2507.20534）
4. `E:\transformers\bayon-note\Kimi\kimi_k2_thinking\index.md`（K2-Thinking：官方 config 留档 + 与 0905 逐字段比对）
5. `E:\transformers\bayon-note\Kimi\kimi_k2_5\index.md`（K2.5：官方 config 留档 + 本地 kimi_k25 源码 + arXiv:2602.02276）
6. `E:\transformers\bayon-note\Kimi\kimi_k2_vl\index.md`（K2.5-VL：视觉塔与 token 流专述，同上来源）
7. 官方仓库/卡片：https://huggingface.co/moonshotai/Kimi-K2-Instruct-0905 、Kimi-K2-Thinking 、Kimi-K2.5 ；https://hf-mirror.com/api/models?author=moonshotai
8. 论文/博客：https://arxiv.org/abs/2507.20534 ；https://arxiv.org/abs/2602.02276 ；https://platform.moonshot.cn/blog/posts/moonlight

> 标注约定：全部数字来自各版本目录 index.md 所引官方 config / 本地源码 / 官方论文；Moonshot-8B / 32B-A32B 无官方 config，统一标「待确认」并保留核验记录。访问与核验日期：2026-10-31（各版本文档），2026-11-02（本对比文档）。

---

*页脚：Kimi（Moonshot）全家族版本对比与演进 · 数据来源：bayon-note 各版本 index.md（config + transformers 源码 + 官方论文）· 仅供技术学习与论文研究*
