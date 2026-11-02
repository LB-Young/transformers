# InternVL 全家族版本对比与演进 — 1.5 → 2.0 → 2.5 → 3.0 → 3.5

> 整理日期：2026-11-02
> 数据来源：各版本目录 `index.md`（其事实全部源自本地 config.json（`E:\models\model\`）、transformers 本地源码 `models\internvl\modeling_internvl.py`、官方论文与官方博客，详见文末来源）
> 覆盖版本：internvl-1_5（2024-04）、internvl-2（2024-07）、internvl-2_5（2024-12）、internvl-3（2025-04）、internvl-3_5
> 用途：配合 `E:\transformers` 源码研读与断点 debug 的家族横向速查

---

## 结论速览

InternVL 家族的演进主线是一条「**骨架从 1.5 起恒定，能力靠训练范式与底座规模迭代**」的路线：

1. **InternVL 1.5（2024-04）**：砍掉 1.0 的 8B QLLaMA 桥接，确立 **ViT–MLP–LLM 三段式直连**；动态分辨率（448 tile ×1–40）+ 缩略图 + pixel shuffle（256 token/tile）；两阶段训练；主打"逼近 GPT-4V"。
2. **InternVL 2.0（2024-07）**：**渐进扩展策略**把家族铺到 1B–108B，7 种开源语言底座；加入视频/医疗影像输入与 VisionLLMv2 结构化输出（框/掩码/图）；视觉塔固定 300M/6B 两档。
3. **InternVL 2.5（2024-12）**：模型/数据/测试时**三线扩展**——视觉塔 V2.5 增量 NTP 训练、SFT 数据翻倍到 16.3M、引入 CoT 测试时扩展（78B MMMU 首破 70）；语言底座升到 Qwen2.5/InternLM2.5。
4. **InternVL 3.0（2025-04）**：**原生多模态预训练**（文本+多模态 1:3 单阶段联合，约 200B token）+ V2PE 可变视觉位置编码 + LayerScale/mean pooling 视觉塔 + **MPO 混合偏好优化** + Best-of-N/VisualPRM 测试时扩展；格式转为 transformers 原生 `internvl`。
5. **InternVL 3.5**：**Cascade RL**（MPO 离线热身 → GSPO 在线精修）+ **ViR** 视觉分辨率路由 + **DvD** 解耦部署（约 4.05× 加速、推理 +16.0%）；家族首次上 **MoE 底座**（20B-A4B / 30B-A3B / 241B-A28B），最大 241B。

一句话：**1.5 定三段式骨架，2.0 铺开底座谱系，2.5 换底座+测试时扩展，3.0 换训练范式（原生预训练+MPO），3.5 换推理效率与 RL 范式（Cascade RL + MoE）**——架构主线（ViT–MLP–LLM、448 tile、pixel shuffle、masked_scatter 注入）从 1.5 到 3.5 始终未变。

---

## 一、全家族核心规格对比表

| 项目 | InternVL 1.5 | InternVL 2.0 | InternVL 2.5 | InternVL 3.0 | InternVL 3.5 |
|---|---|---|---|---|---|
| 发布时间 | 2024-04 | 2024-07-02 | 2024-12 | 2025-04 | — |
| 入口类 / 格式 | InternVLChatModel（legacy `internvl_chat`） | 同左 | 同左 | **InternVLForConditionalGeneration（原生 `internvl`）** | 同左（部分尺寸 legacy） |
| 架构骨架 | ViT–MLP–LLM（去 QLLaMA 桥） | 同左 | 同左 | 同左 | 同左 |
| 家族规模 | 2B / 26B（Mini 2B/4B 附带） | **1B–108B**（8 尺寸） | **1B–78B**（7 尺寸） | **1B–78B**（7 尺寸） | **1B–241B-A28B**（9 尺寸，含 3 个 MoE） |
| 视觉塔 | 300M（Mini）/ 6B-448px-V1.5（主） | 300M（1B–8B）/ 6B-V1.5（26B+） | **300M/6B-V2.5**（增量 NTP 训练） | 300M-V2.5 / 6B-V2.5（+**LayerScale 0.1 / mean pooling**） | 300M / 6B（同 3.0 规格） |
| 本地 1B 底座 | Mini-2B：InternLM2-1.8B（2048/24/16·8） | Qwen2-0.5B（896/24/14·2） | **Qwen2.5-0.5B**（896/24/14·2，vocab 151674） | Qwen2.5-0.5B（896/24/14·2） | **Qwen3-0.6B**（1024/28/16·8，vocab 151936） |
| 语言底座谱系 | InternLM2（1.8B/20B）、Phi-3 | 7 种：Qwen2 / InternLM2 / Phi-3 / Yi / Llama-3 / InternLM2-102B | Qwen2.5 / InternLM2.5 | Qwen2.5 全系 + InternLM3-8B（base 初始化） | Qwen3 全系 + GPT-OSS-20B（MoE）+ Qwen3 MoE |
| 投影器 | MLP（4096→2048→2048，GELU） | MLP → LLM dim | MLP → LLM dim | MLP（LayerNorm→Linear₁→GELU→Linear₂） | MLP（同 3.0） |
| tile / 分辨率 | 448，1–12 训练 / 40 测试 + 缩略图 | 同左（≤12 / ≤40） | 同左 | 同左（max_patches=12） | 同左 + **ViR**（每 tile 1/4 或 1/16 压缩） |
| token/tile | 256（pixel shuffle 0.5） | 256 | 256 | 256（image_seq_length=256） | 256 |
| 特殊 token | 占位 + internlm2-chat 模板 | 同左（internvl2 模板） | 同左（internvl2_5 模板） | BOI/EOI + image_token_id=151667 | image_token_id=151671 |
| 预训练范式 | 两阶段（对齐预训练 → 全参 SFT） | 两阶段 + 视觉塔连续学习 | 三阶段（MLP warmup → ViT 增量 → 全量 SFT） | **原生多模态预训练**（1:3 单阶段，约 200B token） | 原生预训练（沿用） |
| SFT 数据 | 5.1M | 7.3M | **16.3M** | **21.7M** | — |
| 偏好/RL | 无 | 无 | 无 | **MPO**（DPO+BCO+LM，约 30 万对） | **Cascade RL**（MPO 热身 + **GSPO** 在线精修） |
| 测试时扩展 | 无 | 无 | **CoT**（MMMU>70） | **Best-of-N + VisualPRM-8B** | 同 3.0 + ViR/DvD 部署（约 4.05× 加速） |
| 上下文 | 4K 训练 | 8K | 4K–32K | **32K**（V2PE + 头/序列并行） | 40960（本地 1B max_pos） |
| 主要论文 | arXiv:2404.16821 | 官方博客 | arXiv:2412.05271 | arXiv:2504.10479 | arXiv:2508.18265 |
| 输出能力 | 文本 | 文本 + 框/掩码/图像（VisionLLMv2） | 文本 | 文本 + 思维链 | 文本 + GUI/具身/SVG |

> 说明：① 视觉塔两档贯穿全家族——小尺寸用 InternViT-300M（1024/24 层/16 头/inter 4096，qkv_bias=true、无 QK-Norm），大尺寸用 InternViT-6B（3200/45 层/25 头/inter 12800，RMS+QK-Norm）；3.0 起 300M 规格加 LayerScale(0.1) 与 mean pooling。② 3.5 的 MoE 底座：20B-A4B（GPT-OSS-20B，32 专家 top-4）、30B-A3B（Qwen3-30B-A3B，128 top-8）、241B-A28B（Qwen3-235B-A22B，128 top-8）。③ 格式变化：1.5/2.x 为 legacy `internvl_chat`（顶层 llm_config），3.0/3.5 为原生 `internvl`（顶层嵌套 text_config/vision_config），ModelScope 上 3.5 的 30B-A3B/241B-A28B 仍发 legacy 格式。

---

## 二、代际演进详解

### 2.1 InternVL 1.0 → 1.5：砍桥接，定骨架

| 维度 | InternVL 1.0（arXiv:2312.14238） | InternVL 1.5（arXiv:2404.16821） |
|---|---|---|
| 结构 | InternViT-6B + **8B QLLaMA 桥接** + LLaMA 解码器 | **去除 QLLaMA**，pixel shuffle + MLP 直连 LLM |
| 视觉编码器 | InternViT-6B，224px 固定 | InternViT-6B/300M，**448px 连续学习、动态分辨率** |
| 语言底座 | LLaMA 系 | InternLM2（1.8B/20B） |
| 分辨率 | 固定低分辨率 | 1–12 tile 训练 / 40 tile 测试（4K）+ 缩略图 |
| 训练 | 图文对比对齐 | 阶段一（ViT+MLP 对齐预训练）→ 阶段二（全参 SFT） |

- 1.5 是"骨架定型"代：**ViT–MLP–LLM 三段式 + 动态 tile + pixel shuffle + masked_scatter 注入**从此成为家族不变内核，后续 2.0–3.5 全部沿用。

### 2.2 InternVL 1.5 → 2.0：铺开底座谱系（渐进扩展）

| 维度 | 1.5 | 2.0 |
|---|---|---|
| 家族规模 | 2B / 26B | **1B–108B**（8 尺寸） |
| 语言底座 | InternLM2 / Phi-3 | **7 种**：Qwen2-0.5B / InternLM2-1.8B / Phi-3 / InternLM2.5-7B / InternLM2-20B / Yi-34B / Llama-3-70B / InternLM2-102B |
| 视觉塔 | V1.5 两档 | 300M/6B-V1.5 **连续学习升级**（跨底座复用） |
| 输入模态 | 图 + 文本 | + **视频、医疗影像、多图**（8K 上下文） |
| 输出能力 | 仅文本 | VisionLLMv2 解码器：文本 + **框 + 掩码 + 图像** |
| 数据 | 5.1M | 7.3M |

- 2.0 的工程要点：小尺寸先训好、权重作大模型初始化参考的**渐进扩展策略**；视觉塔与投影器跨底座可复用。

### 2.3 InternVL 2.0 → 2.5：模型/数据/测试时三线升级

| 维度 | 2.0 | 2.5 |
|---|---|---|
| 视觉塔 | V1.5 | **V2.5**（增量 NTP 预训练，全参数不冻结） |
| 语言底座 | Qwen2 / InternLM2 系 | **Qwen2.5 / InternLM2.5** |
| SFT 数据 | 7.3M | **16.3M**（翻倍 + 严格去重） |
| 训练管线 | 两阶段 | **三阶段**：MLP warmup → ViT 增量（可选）→ 全量 SFT |
| 推理 | 直接回答 | **CoT 思维链测试时扩展**（78B MMMU 验证集首破 70） |
| 效率 | — | 78B 仅约 **120B token** 训练（对比 Qwen2-VL 累计 1.4T） |

- 2.5 的相对增量全部不在架构（仍是 internvl_chat legacy），而在**数据规模、视觉塔继续学习、推理期思维链**。

### 2.4 InternVL 2.5 → 3.0：换训练范式（原生预训练 + MPO）

| 维度 | 2.5 | 3.0 |
|---|---|---|
| 预训练 | MLP warmup 对齐 + 多阶段微调（语言/多模态分离） | **原生多模态预训练**：文本 50B + 多模态 150B 单阶段交错、全参数联合（配比 1:3） |
| 视觉编码器 | V2.5 基础 ViT | + **LayerScale(0.1)、mean pooling、绝对位置编码 + V2PE** |
| 语言底座 | Qwen2.5 / InternLM2.5（instruct 版） | Qwen2.5 全系 + InternLM3-8B，**统一用 base 模型初始化** |
| SFT | 16.3M | **21.7M**（+GUI/工具调用/3D/视频/科学图表） |
| 后训练 | 无偏好优化 | **MPO**（DPO 偏好 + BCO 质量 + LM 生成损失，约 30 万偏好对，78B 推理 +4.1） |
| 测试时 | CoT | **Best-of-N + VisualPRM-8B** 分步奖励评分 |
| 上下文 | 4K–32K | **32K**（V2PE + 头/序列并行） |
| 格式 | legacy internvl_chat | **原生 internvl**（text_config/vision_config 嵌套） |

- 3.0 的宣言：语言与视觉能力在同一预训练阶段联合塑形（square averaging 加权、loss 只在文本 token）；视觉塔三个 config 可证新特性（LayerScale 0.1 / mean pooling / V2PE）。

### 2.5 InternVL 3.0 → 3.5：RL 范式 + 效率（Cascade RL / ViR / DvD）+ MoE

| 维度 | 3.0 | 3.5 |
|---|---|---|
| 后训练 | MPO（离线） | **Cascade RL**：MPO 离线热身 → **GSPO 在线精修**（同 query 组内归一化优势，GRPO 同族） |
| 推理效率 | 基线 | **ViR**（每 tile 语义动态 1/4 或 1/16 压缩）+ **DvD**（ViT/LLM 分置 GPU），最高约 **4.05×** |
| 推理能力 | 基线 | 整体推理最高 **+16.0%** |
| 底座 | Qwen2.5（Dense） | **Qwen3 全系 + 3 个 MoE 底座**（GPT-OSS-20B 32×top-4；Qwen3-30B-A3B、235B-A22B 128×top-8） |
| 1B 底座 | Qwen2.5-0.5B（896/24/2KV） | **Qwen3-0.6B**（1024/28/8KV，vocab 151936/max 40960） |
| 能力边界 | 通用图文 / OCR / grounding | + **GUI 交互、具身智能、SVG 理解与生成** |
| 最大模型 | 78B | **241B-A28B**（InternViT-6B + Qwen3-235B-A22B） |

---

## 三、关键机制演进一图流

```
骨架：       ViT–MLP–LLM（1.5 起恒定，1.0 的 QLLaMA 桥被砍）
格式：       legacy internvl_chat（1.5/2.0/2.5）→ 原生 internvl（3.0/3.5）
视觉塔：     300M/6B 两档（1.5–2.5）→ +LayerScale(0.1)/mean pooling/V2PE（3.0+）
语言底座：   InternLM2（1.5）→ 7 种底座（2.0）→ Qwen2.5/InternLM2.5（2.5）
             → Qwen2.5+InternLM3 base（3.0）→ Qwen3 + MoE（3.5）
训练范式：   两阶段（1.5）→ 两阶段+连续学习（2.0）→ 三阶段 MLP warmup（2.5）
             → 原生单阶段预训练 200B（3.0）→ +Cascade RL（3.5）
后训练：     无（1.5–2.5）→ MPO（3.0）→ MPO 热身 + GSPO 在线（3.5）
测试时：     无（1.5–2.0）→ CoT（2.5）→ Best-of-N + VisualPRM（3.0）
             → +ViR 动态压缩 + DvD 解耦部署（3.5）
家族规模：   2B/26B（1.5）→ 1B–108B（2.0）→ 1B–78B（2.5/3.0）→ 1B–241B-A28B（3.5）
SFT 数据：   5.1M（1.5）→ 7.3M（2.0）→ 16.3M（2.5）→ 21.7M（3.0）
上下文：     4K（1.5）→ 8K（2.0）→ 32K（3.0，V2PE）
```

演进主线解读：
1. **骨架不变**：ViT–MLP–LLM、448 tile + 缩略图、pixel shuffle 0.5 → 256 token/tile、masked_scatter 占位注入，从 1.5 到 3.5 始终未动——代际差异几乎全部落在训练范式、底座与推理策略。
2. **视觉塔只在 3.0 有一次结构级更新**（LayerScale/mean pooling），其余是两档规格 + 增量训练。
3. **底座从单一 InternLM2 走向 Qwen3/MoE 谱系**：1.5 一个底座 → 2.0 七种 → 3.5 的 Qwen3 全系 + 三个 MoE。
4. **训练范式三级跳**：两阶段对齐（1.5）→ 三阶段 warmup（2.5）→ 原生单阶段预训练（3.0）→ Cascade RL（3.5）。
5. **推理期能力逐代叠加**：CoT（2.5）→ Best-of-N+VisualPRM（3.0）→ ViR/DvD 效率优化（3.5）。

---

## 四、对源码研读 / debug 的启示

1. **格式分流**：legacy 1.5/2.x 用 `InternVLChatModel`（本地 config 顶层是 llm_config/vision_config，不在 transformers 内置，走官方 modeling_internvl_chat.py）；3.0/3.5 用 `InternVLForConditionalGeneration`（本地 `models\internvl\modeling_internvl.py`，顶层嵌套 text_config/vision_config）。打断点前先看 config 是哪种格式。
2. **重点调试对象**：
   - `get_image_features`（:543-587）→ pixel_shuffle（:670）→ `InternVLMultiModalProjector`（:485）→ 占位校验（:607-610）→ masked_scatter 写回（:646-652）是整条视觉链路；
   - 3.0 起注意 **LayerScale（layer_scale_init_value=0.1）** 与 **mean pooling（use_mean_pooling=true）**、V2PE 的可变位置增量 δ（默认 1）；
   - 3.5 的 MoE 底座路由（Qwen3 MoE / GPT-OSS 32 专家）与 ViR 的按 tile 压缩率切换逻辑。
3. **同代对照**：同一 transformers `models\internvl\` 模块可加载 3.0 与 3.5 的 Dense 尺寸（image_token_id 不同：151667 vs 151671，加载时注意 config）。
4. **本地可跑性**：1B/2B/4B 尺寸（300M 视觉塔 + Qwen2.5/3 小底座）适合本地双 16G 卡断点调试；241B-A28B 仅适合远端。

---

## 参考来源

1. `E:\transformers\bayon-note\InternVL\internvl-1_5\index.md`（1.5：本地 config + arXiv:2404.16821 + arXiv:2312.14238）
2. `E:\transformers\bayon-note\InternVL\internvl-2\index.md`（2.0：本地 config + 官方博客 2024-07-02 + arXiv:2406.08394 VisionLLMv2）
3. `E:\transformers\bayon-note\InternVL\internvl-2_5\index.md`（2.5：本地 config + arXiv:2412.05271）
4. `E:\transformers\bayon-note\InternVL\internvl-3\index.md`（3.0：本地 config + 本地 modeling_internvl.py 行号 + arXiv:2504.10479）
5. `E:\transformers\bayon-note\InternVL\internvl-3_5\index.md`（3.5：本地 config + arXiv:2508.18265）
6. 官方仓库/博客：https://github.com/OpenGVLab/InternVL ；https://internvl.github.io/blog/2024-07-02-InternVL-2.0/ ；https://internvl.github.io/blog/2025-04-11-InternVL-3.0/
7. 论文：https://arxiv.org/abs/2404.16821 ；https://arxiv.org/abs/2412.05271 ；https://arxiv.org/abs/2504.10479 ；https://arxiv.org/abs/2508.18265

> 标注约定：全部数字来自各版本目录 index.md 所引本地 config / 本地源码 / 官方论文；未核实项（如 3.5 发布具体日期）按源文档留空。访问与核验日期：2026-10-26（各版本文档），2026-11-02（本对比文档）。

---

*页脚：InternVL 全家族版本对比与演进 · 数据来源：bayon-note 各版本 index.md（config + transformers 源码 + 官方论文）· 仅供技术学习与论文研究*
