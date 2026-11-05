# Kimi（Moonshot）全家族版本对比与演进 — Moonlight → K2-Base → K2-0905 → K2-Thinking → K2.5 / K2.5-VL → K2.6 → K2.7-Code → K3 → K3.1-Pro

> 整理日期：2026-11-03
> 数据来源：各版本目录 `index.md`（其事实全部源自远端官方 config（hf-mirror 拉取留档 `_tools\_cfgpull\`）、transformers 本地源码（`models\deepseek_v3\` / `models\kimi_k25\` / `models\kimi_linear\`）、官方论文与官方仓库/官网，详见文末来源）
> 覆盖版本：moonshot_8b（官方 404，待确认）、moonshot_32b_a32b（官方 404，待确认）、kimi_k2_0905（K2-Base/Instruct-0905）、kimi_k2_thinking、kimi_k2_5（K2.5，含 K2.5-VL 视觉塔）、kimi_k2_vl（= K2.5 多模态，无独立仓库）、kimi_k2_6（K2.6）、kimi_k2_7_code（K2.7-Code）、kimi_k3（K3，技术报告 PDF 留档）、kimi_k3_1_pro（K3.1-Pro，无官方仓库，多字段待确认）
> 用途：配合 `E:\transformers` 源码研读与断点 debug 的家族横向速查

---

## 结论速览

Kimi 家族的演进主线是一条「**DeepSeek-V3 系 MLA-MoE 范式下的规模跃迁 + 模态扩展，2026-07 在 K3 完成注意力范式换代**」路线：

1. **Moonlight-16B-A3B（2025-02/03）**：Moonshot 首个真实开源稀疏模型——DeepSeek-V3 式 MLA + 64 路由专家/2 共享/top-6，Muon 优化器大规模训练配套发布；规模 16B-A3B。
2. **Kimi-K2-Base（2025-07）**：**1T-A32B 级跃迁**——hidden 7168/61 层、MLA 新增 q 侧低秩（q_lora 1536）、路由专家扩到 **384 + 1 共享 top-8**（稀疏度 48），上下文 131072；官方论文 arXiv:2507.20534。
3. **Kimi-K2-0905（2025-09）**：与 Base 同文本塔，工程升级——上下文经 YaRN 扩到 **262144**、权重改 **fp8 e4m3** 官方发布。
4. **Kimi-K2-Thinking（2025-09）**：**架构零改动**的推理变体——config 与 K2-0905 逐字段一致（仅 eos=163586、YaRN beta_fast=1.0），差异在长思维链 SFT + RL；INT4 发布，纯文本。
5. **Kimi-K2.5（2026-01-27）**：**原生多模态智能体**——K2 文本塔 + **MoonViT-3D 视觉塔**（patch14/27 层/1152，双向、轴向 2D-RoPE，初始化自 SigLIP-SO-400M）+ **PatchMerger 投影器**（时空 2×2 池化 → 4608→7168）；early-fusion 联合预训练约 15T token；INT4 发布，图像/视频权重共享、视频 4× 时间压缩；K2.5-VL 即其视觉塔专述（官方无独立 K2-VL 仓库）。
6. **Kimi-K2.6（2026-04）**：**K2.5 架构的通用更新版**——config 与 K2.5 逐字段 diff 唯一为 eos=163586（K2.5 为 163585），其余全部字段（7168/61/MLA/384+1/YaRN 262144/vision_config/INT4）逐项一致；通用多模态（文本/图片/视频），256K 上下文，thinking 可开关，API 名 `kimi-k2.6`。
7. **Kimi-K2.7-Code（2026-06-12）**：**基于 K2.6 的编码 Agent 微调**——config 与 K2.6 几乎零差异（仅量化忽略名单正则写法收紧），差异在发布层：preserve_thinking 强制思维链、Modified MIT 协议、FP8 + 原生 INT4（QAT）发布；Kimi Code Bench v2 得分 62.0 vs K2.6 的 50.9（来源：官方模型卡）；06-15 出 HighSpeed 变体。
8. **Kimi-K3（2026-07-16 API / 2026-07-27 开源）**：**注意力范式换代的当前旗舰**——2.8T/104B 激活；混合注意力 **69 KDA（Kimi Delta Attention）+ 24 Gated MLA（NoPE）**，3:1 混合 + 末端一层全局；**AttnRes** 跨层注意力残差（block 12）；**Stable LatentMoE**（896 路由 + 2 共享，top-16，latent 3584，稀疏度 56）；**SiTU-GLU**（β₁=4、β₂=25）+ **Quantile Balancing**；**MoonViT-V2**（从零训练，patch14/27 层/1024/12 头，~401M，像素重排 2×2）；**NoPE 直接外推 1M 上下文**；**MXFP4 权重/MXFP8 激活（QAT）**，无 BF16 发布；Per-Head Muon；~2.5× scaling efficiency vs K2；Kimi K3 License（非 MIT，MaaS 年收入 >$20M 需单独协议）；技术报告 arXiv:2607.24653。
9. **Kimi-K3.1-Pro（2026-10-24 官网展示）**：**最新旗舰**——3 万亿参数级、1M 上下文，面向长程 coding/3D 创作/金融研究/复杂 agent 任务；无官方 HF 仓库、无 config、无技术报告，架构字段全部待确认（来源：moonshot.ai 首页）。
10. **Moonshot-8B / Moonshot-32B-A32B（待确认）**：经 hf-mirror / ModelScope 多通道核验 **404（不存在或已下架）**，无 config 留痕，全部规格标「待确认」；最接近的真实对象是 Moonlight-16B-A3B 与 K2 系——按源文档口径如实记录，不臆造数字。

一句话：**K2 系是"DeepSeek-V3 架构模板的 1T 级复用"（MLA 加 q 低秩、专家 64→384），Thinking 是"同架构换训练"（推理 RL），K2.5 是"文本塔加视觉塔换模态"（MoonViT-3D + PatchMerger），K2.6/K2.7-Code 是"同 config 换发布与后训练"，K3 是"注意力范式换代"（KDA 混合线性 + AttnRes + Stable LatentMoE + NoPE 1M + MXFP4 QAT），K3.1-Pro 是"3T 级新一代旗舰（信息待确认）"，早期"Moonshot-8B/32B-A32B"不可考**。

---

## 一、全家族核心规格对比表

| 项目 | Moonlight-16B-A3B（真实锚点） | Kimi-K2-Base | Kimi-K2-0905 | Kimi-K2-Thinking | Kimi-K2.5（含 K2.5-VL） | Kimi-K2.6 | Kimi-K2.7-Code | Kimi-K3 | Kimi-K3.1-Pro | Moonshot-8B / 32B-A32B |
|---|---|---|---|---|---|---|---|---|---|---|
| 时间 | 2025-02/03 | 2025-07 | 2025-09 | 2025-09 | 2026-01-27 | 2026-04 | 2026-06-12 | 2026-07-16 API / 2026-07-27 开源 | 2026-10-24 官网展示 | 待确认（404） |
| 入口类 / model_type | DeepseekV3ForCausalLM / deepseek_v3 | DeepseekV3ForCausalLM / kimi_k2 | 同左 | 同左 | KimiK25ForConditionalGeneration / kimi_k25（text 重写为 deepseek_v3） | 同 K2.5 | 同 K2.6 | **KimiK3ForConditionalGeneration / kimi_k3**（text=kimi_linear → KimiLinearForCausalLM） | 待确认 | 待确认 |
| 形态 | MoE（16B-A3B） | **MoE（1.04T-A32B）** | 同 Base | 同 Base | 同 K2 + 视觉塔 | 同 K2.5 | 同 K2.6（编码 Agent 微调） | **MoE（2.78T/104.2B 激活）** | 3T 级（官网口径） | 8B dense? / A32B MoE?（待确认） |
| hidden / 层数 | 2048 / 27 | 7168 / 61（第 0 层 dense） | 7168 / 61 | 7168 / 61 | 7168 / 61 | 7168 / 61 | 7168 / 61 | **7168 / 93**（1 dense + 69 KDA + 24 Gated MLA） | 待确认 | 待确认 |
| 注意力 | **MLA**（kv_lora 512，**无 q_lora**，nope128/rope64/v128） | MLA（kv 512 + **q 1536**） | 同 Base | 同 Base | 同 K2（文本塔） | 同 K2.5 | 同 K2.6 | **混合：69 KDA（通道级遗忘门 delta-rule）+ 24 Gated MLA（NoPE + 通道级全秩输出门）** | 待确认 | 待确认（是否 MLA 未知） |
| Q/KV 头 | 16 / 16 | 64 / 64 | 64 / 64 | 64 / 64 | 64 / 64 | 64 / 64 | 64 / 64 | **96 / 96**（KDA 头亦 96，head_dim 128） | 待确认 | 待确认 |
| 路由专家 / 共享 / top-k | 64 / 2 / 6 | **384 / 1 / 8** | 同 Base | 同 Base | 同 K2 | 同 K2.5 | 同 K2.6 | **896 / 2 / 16**（latent 3584，稀疏度 56） | 待确认 | 待确认 |
| moe_intermediate | 1408 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 3072（latent 空间） | 待确认 | 待确认 |
| dense intermediate | 11264 | 18432 | 18432 | 18432 | 18432 | 18432 | 18432 | 33792 | 待确认 | 待确认 |
| 路由策略 | sigmoid + noaux_tc | sigmoid + noaux_tc（scaling 2.827） | 同 Base | 同 Base | 同 K2 | 同 K2.5 | 同 K2.6 | sigmoid + noaux_tc（**Quantile Balancing**，routed_scaling 1.0） | 待确认 | 待确认 |
| 激活函数 | SwiGLU | SwiGLU | SwiGLU | SwiGLU | SwiGLU | SwiGLU | SwiGLU | **SiTU-GLU**（β₁=4、β₂=25，输出有界，上限 100） | 待确认 | 待确认 |
| 上下文 | 8192 | 131072 | **262144**（YaRN ×64） | 262144 | 262144 | 262144 | 262144 | **1048576（NoPE，无 RoPE 外推）** | 1M（官网口径） | 128K？（待确认） |
| rope_theta | 50000 | 50000 | 50000 | 50000（beta_fast 1.0） | 50000（beta_fast 32） | 50000（beta_fast 32） | 50000（beta_fast 32） | **无（NoPE：位置由 KDA 循环门控隐式编码）** | 待确认 | 待确认 |
| 词表 | 163840 | 163840 | 163840 | 163840 | 163840 | 163840 | 163840 | 163840 | 待确认 | 待确认 |
| 权重格式 | bf16 | bf16 | **fp8 e4m3** | **INT4**（compressed-tensors g32） | **INT4**（注意力/共享专家/lm_head/视觉塔免量化） | INT4（同 K2.5 留档口径） | **FP8 + 原生 INT4（QAT）**（模型卡；config 内 int4 pack-quantized 同 K2.6） | **MXFP4 权重 / MXFP8 激活（QAT 自 SFT 阶段，无 BF16 发布）** | 待确认 | 待确认 |
| eos_token_id | — | 163585 | 163585 | **163586** | 163585 | **163586** | 163586 | 163586 | 待确认 | 待确认 |
| 视觉塔 | 无 | 无 | 无 | 无 | **MoonViT-3D**：patch14 / 27 层 / hidden 1152 / 16 头 / FFN 4304 / gelu_pytorch_tanh / 双向 / 轴向 2D-RoPE / 初始化自 SigLIP-SO-400M | 同 K2.5 | 同 K2.6（~400M） | **MoonViT-V2**：patch14 / 27 层 / hidden 1024 / 12 头 / FFN 4096 / RMSNorm 无 bias / **从零训练** / ~401M | 待确认 | 无 |
| 投影器 | 无 | 无 | 无 | 无 | **PatchMerger**：merge(2,2) 时空均值池化 → LN(1152) → 4608→GELU→7168 | 同 K2.5 | 同 K2.6 | **PatchMergerV2 + 像素重排 2×2 降采样（token ÷4，输入 ≤3584×3584）** | 待确认 | 无 |
| 多模态处理 | — | — | — | — | **NaViT patch packing**（不切 tile）+ max_patches 16384 + 视频 4× 时间压缩 | 同 K2.5 | 同 K2.6 | 原生多模态（文本/图像/视频共享 backbone，训练起步即联合） | 待确认（官网未提模态） | — |
| 训练特点 | Muon 优化器首次大规模实践 | 15.5T tokens、MuonClip | 同 Base | **长思维链 SFT + RL**（架构外） | 视觉塔先对齐 Moonlight（1T）→ 投影桥接 → 联合预训练约 15T | 通用更新（thinking 可开关） | **编码 Agent 微调，preserve_thinking 强制思维链**（模型卡） | **Per-Head Muon** + K2 权重裁剪 + QB 负载均衡；cosine LR（1% warmup）；8K→64K→1M 课程；~2.5× scaling efficiency vs K2 | 待确认 | — |
| License | 开源 | 开源 | 开源 | 开源 | 开源 | 开源 | **Modified MIT**（模型卡） | **Kimi K3 License**（非 MIT；MaaS 年收入 >$20M 需单独协议） | 待确认 | — |
| 主要论文 | 官方博客 | arXiv:2507.20534 | 同左 | 同左 | arXiv:2602.02276 | 无独立报告（架构同 K2.5） | 无独立报告（模型卡口径） | arXiv:2607.24653 | 无（待检索） | — |

> 说明：① Moonshot-8B / Moonshot-32B-A32B 两个目录记录的是核验过程——hf-mirror 组织列表 19 个仓库、全文搜索、ModelScope 直连均无此二模型，Wayback 不可达；按源文档约定**不臆造任何数字**。② K2.5-VL 无独立 HF 仓库（官方多模态即 Kimi-K2.5，NVIDIA 文档称 "Kimi-K25-VL"），视觉侧规格与 K2.5 完全一致。③ **同代兄弟**：K2.6、K2.7-Code、K3 已单独立目录（见上表三列与 `kimi_k2_6/`、`kimi_k2_7_code/`、`kimi_k3/` 目录）；以下为保留的同代兄弟引用：K2.8 Preview（2026-09-11 接管 kimi-for-coding；**HF 无官方权重仓库，搜索 0 命中**，不臆造配置）、Kimi-Linear-48B-A3B（KDA 混合线性注意力前身小模型，本地 `models/kimi_linear/`）、Kimi-VL-A3B（2025 上代小 VL，文本塔 2048/27 层/64 专家 top-6）。

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

### 2.3 Kimi-K2-Thinking（2025-09）：同架构推理变体（架构零改动）

- 与 K2-0905 文本塔 **config 逐字段一致**（hidden 7168 / 61 层 / MLA 同维 / 384+1 top-8 / routed_scaling 2.827 / vocab 163840 / max 262144）。
- 唯一差异：`eos_token_id=163586`（0905 为 163585）、YaRN `beta_fast=1.0`（K2.5 为 32）、无 vision_config、无 nextn 预测层。
- 差异本质在**训练**：长思维链（long CoT）SFT + 强化学习，激发「先思考再回答」；config 无 reasoning 专用开关字段。
- 权重：INT4（compressed-tensors，group32，dynamic，act-order，weight-only）。

### 2.4 Kimi-K2.5 / K2.5-VL（2026-01-27）：文本塔 + 视觉塔 = 原生多模态智能体

| 部件 | 规格 | 关键点 |
|---|---|---|
| 文本塔 | = K2（7168/61/MLA/384+1 top-8） | text_config.model_type 运行时重写为 `deepseek_v3`，复用 DeepSeek-V3 实现 |
| 视觉塔 MoonViT-3D | patch14 · 27 层 · hidden 1152 · 16 头 · FFN 4304 · gelu_pytorch_tanh | 双向（非因果）全注意力、轴向 2D-RoPE、可学习 64×64×4 位置网格、初始化自 SigLIP-SO-400M |
| 投影器 PatchMerger | merge_kernel (2,2) 时空均值池化 → LN(1152) → in_proj(4608→4608) → GELU → out_proj(4608→7168) | 空间 2×2 + 时间 2×2，token 压到 1/4 |
| 多模态处理 | **NaViT patch packing**（原生分辨率、不切 tile） | max_patches=16384 预算等比缩放；多图 1D 拼接 + cu_seqlens 变长掩码；视频最多 4 帧共享图像权重、时间均值池化 4× 压缩 |
| 特殊 token | vision_start=163602 / image=163605 / video=163840 / vision_end=163604 | masked_scatter 在 embedding 层填入（early fusion） |
| 训练 | ① 视觉塔 caption loss 对齐 Moonlight-16B-A3B（约 1T token）→ ② 短训投影器桥接 → ③ 联合预训练约 15T 视觉-文本 → ④ YaRN 32K→262144 | 与「文本训完再加视觉」的后期融合路线不同 |
| 发布 | INT4 量化（注意力/共享专家/dense lm_head/视觉塔免量化），BF16 官方未释出 | K2.6 同架构（eos=163586） |

- **K2.5-VL 与 K2.5 是同一模型**：官方无独立 Kimi-K2-VL 仓库；K2 代多模态能力即由 `KimiK25ForConditionalGeneration` 承载。
- 与 InternVL 路线的本质区别：K2.5 走**原生分辨率 + NaViT packing**（不切 tile），InternVL 走**动态 tile + 缩略图**。

### 2.5 Kimi-K2.6（2026-04）：同 config 的通用更新版

- config 与 K2.5 **逐字段 diff 唯一为 eos_token_id=163586**（K2.5 为 163585；顶层与 text_config 同步）；7168/61/MLA 低秩（kv 512 + q 1536）/384+1 top-8/YaRN 262144/vision_config（MoonViT-3D）/INT4 量化忽略名单全部逐项一致（程序比对核验，2026-11-03）。
- 产品层：通用多模态（文本/图片/视频），256K 上下文，thinking 可开关；API 名 `kimi-k2.6`；官方文档 https://platform.kimi.com/docs/guide/kimi-k2-6-quickstart 。
- 意义：K2.6 证明 K2.5 架构已进入「config 冻结、按发布节奏迭代」阶段——后续 K2.7-Code 直接在其之上做编码微调。

### 2.6 Kimi-K2.7-Code（2026-06-12）：K2.6 之上的编码 Agent 微调

- config 与 K2.6 几乎零差异（仅 quantization_config.ignore 中 `vision_tower` → `vision_tower.*` 正则收紧，不影响权重结构）；架构 = K2.6。
- 差异全部在发布层（来源：官方模型卡，config 不可承载）：
  - **preserve_thinking 强制思维链**：多轮对话需回传 assistant 完整消息（含 reasoning_content）；
  - **Modified MIT 协议**；
  - **FP8 + 原生 INT4（QAT）发布**；
  - **Kimi Code Bench v2 得分 62.0 vs K2.6 的 50.9**；
  - 2026-06-15 发布 HighSpeed 变体（高速推理）。
- 意义：编码能力的工程化微调样板——config 不动，后训练与发布形态换代。

### 2.7 Kimi-K3（2026-07-16 API / 2026-07-27 开源）：注意力范式换代（当前旗舰）

| 维度 | K2.5 / K2.6 / K2.7-Code | Kimi K3 |
|---|---|---|
| 规模 | 1.04T / 32.6B 激活 | **2.78T / 104.2B 激活** |
| 层数 | 61（第 0 层 dense） | **93**（1 dense + 69 KDA + 24 Gated MLA） |
| 注意力 | 全 MLA（RoPE + YaRN） | **混合：3×KDA + 1×Gated MLA（3:1），NoPE，末端一层全局** |
| 跨层信息流 | 标准残差 | **AttnRes**（伪查询 w 对 embedding + 各 block 输出加权检索，block 12） |
| MoE | 384+1，top-8，稀疏度 48，inter 2048 | **896+2，top-16，latent 3584，稀疏度 56**，Quantile Balancing |
| 激活 | SwiGLU | **SiTU-GLU**（β₁=4、β₂=25，软上限，输出有界 100） |
| 视觉塔 | MoonViT-3D（SigLIP 初始化，1152/27/16） | **MoonViT-V2（从零训练，1024/27/12，~401M，RMSNorm 无 bias，像素重排 2×2）** |
| 上下文 | 262144（YaRN ×64） | **1048576（NoPE 直接外推）** |
| 权重 | INT4 / FP8 | **MXFP4 权重 / MXFP8 激活（QAT，无 BF16 发布）** |
| 优化器 | MuonClip | **Per-Head Muon**（按头 Newton-Schulz 正交化）+ K2 权重裁剪 |

- KDA（Kimi Delta Attention）：delta-rule 递推 + 通道级遗忘门；q/k/v 投影 ShortConv→Swish→L2Norm；分块并行（chunk 间递推、chunk 内 UT 变换）；全秩输出门。长序列混合效率高，KV cache 从 O(n·d) 降为递归状态。
- Gated MLA：保留 K2 系 MLA 低秩压缩（kv 512 / q 1536），但全部加 NoPE（位置由 KDA 承担），并加通道级全秩输出门逐 token 调制全局读出。
- Stable LatentMoE：路由专家在 latent ℓ=3584 内运算，896 专家/16 激活/2 共享；上投影前 RMSNorm + SiTU-GLU 防激活爆炸；Quantile Balancing（分位数 bias + 直方图估计）替代固定步长负载均衡。
- 训练：原生多模态联合预训练（从起步即 interleaved next-token）；上下文 8K→64K→1M 四阶段课程；RL 覆盖长程 coding/通用 agent/推理知识，多 reasoning-effort 级别，多教师 on-policy 蒸馏。
- 发布：thinking 常开（reasoning_effort low/high/max）；API `kimi-k3`（输入 $3 / 输出 $15 / 缓存命中 $0.30 每 1M token）；Kimi K3 License（MaaS 年收入 >$20M 需单独协议）；技术报告 arXiv:2607.24653。
- 本地代码核验（2026-11-03）：`models\kimi_k3\` 模块不存在（顶层走仓库 custom code）；text_config 实例化目标 `models\kimi_linear\`（KimiLinearForCausalLM）本地存在。

### 2.8 Kimi-K3.1-Pro（2026-10-24 官网展示）：3T 级新一代旗舰（信息待确认）

- 官网首页展示（2026-11-03 抓取）：**3 万亿参数级、1M 上下文**，面向长程 coding / 3D 创作 / 金融研究 / 复杂 agent 任务。
- 无官方 HF 仓库（搜索 0 命中）、无 config、无技术报告；是否继承 K3 的 KDA/AttnRes/Stable LatentMoE 架构、激活规模、专家配置、权重格式、协议——全部**待确认**，不臆造数字。
- 按路线推测（仅参照，非事实）：应为 K3 架构路线的规模扩展；待官方 config/报告释出后补全。

---

## 三、关键机制演进一图流

```
文本塔模板：  DeepSeek-V3 系 MLA-MoE（Moonlight 起，K2 全系复用）→ K3 换代为 KDA+Gated MLA 混合
MLA：         kv_lora 512 / 无 q_lora（Moonlight）→ kv 512 + q_lora 1536（K2 系）→ KDA 混合线性 + Gated MLA NoPE（K3）
MoE：         64 路由 + 2 共享 top-6（Moonlight）→ 384 路由 + 1 共享 top-8（K2 系）→ 896 路由 + 2 共享 top-16 / latent 3584（K3）
激活：        SwiGLU（Moonlight~K2.7-Code）→ SiTU-GLU β₁=4/β₂=25 软上限（K3）
负载均衡：    sigmoid + 固定步长 bias（K2 系）→ Quantile Balancing 分位数 + 直方图估计（K3）
跨层信息流：  标准残差累加（Moonlight~K2.7-Code）→ AttnRes 伪查询跨层检索 block 12（K3）
上下文：      8192（Moonlight）→ 131072（K2-Base）→ 262144（K2-0905/Thinking/K2.5/K2.6/K2.7-Code）→ 1048576 NoPE（K3）
权重格式：    bf16（Moonlight/Base）→ fp8 e4m3（0905）→ INT4（Thinking/K2.5/K2.6）→ FP8+原生INT4 QAT（K2.7-Code）→ MXFP4 权重/MXFP8 激活 QAT（K3）
模态：        纯文本（Moonlight/K2-Base/0905/Thinking）→ 文本+图像+视频（K2.5/K2.6/K2.7-Code/K3）
视觉塔：      — → MoonViT-3D（patch14/27层/1152/双向/轴向2D-RoPE/SigLIP初始化）+ PatchMerger（K2.5~K2.7-Code）
             → MoonViT-V2（patch14/27层/1024/从零训练/像素重排2×2）+ PatchMergerV2（K3）
训练：        Muon 实践（Moonlight）→ 15.5T 预训练（K2）→ 长 CoT RL（Thinking）→ early-fusion 联合预训练约 15T（K2.5）
             → 编码 Agent 微调 preserve_thinking（K2.7-Code）→ Per-Head Muon + QB + 原生多模态共训 + 多 effort RL（K3）
```

演进主线解读：
1. **模板恒定**：从 Moonlight 到 K2.7-Code，文本塔始终是 DeepSeek-V3 系 MLA + 细粒度专家 MoE + sigmoid/noaux_tc 路由；代际差异在规模、上下文与模态。
2. **K3 是第一次范式换代**：注意力从全 MLA 换为 KDA 混合线性 + 周期性 Gated MLA（NoPE），专家从 384/8 扩到 896/16（latent 空间），激活/均衡器/跨层机制全部为极端稀疏（56）重做，上下文从 YaRN 外推改为 NoPE 直接外推。
3. **容量路线**：16B-A3B → 1.04T-A32B → 2.78T/104.2B 激活；K3.1-Pro 官网口径再上 3T 级。
4. **Thinking/K2.5/K2.6/K2.7-Code 都是"架构复用 + 训练/模态/发布换新"**：Thinking 换推理训练，K2.5 换多模态塔，K2.6 只换 eos 与发布节奏，K2.7-Code 换编码后训练与协议。
5. **早期"Moonshot-8B / 32B-A32B"不可考**：官方 404，按核验记录如实呈现，不补数字。
6. **K3.1-Pro 信息待确认**：仅官网一句定位；架构字段待官方 config/报告释出后补全。

---

## 四、对源码研读 / debug 的启示

1. **按 model_type 分流**：`kimi_k2`（K2-Base/0905/Thinking）config.architectures 直指 `DeepseekV3ForCausalLM`——本地 `models\deepseek_v3\` 即等价实现；`kimi_k25`（K2.5/K2.6/K2.7-Code）走 `models\kimi_k25\`（KimiK25ForConditionalGeneration），其 text_config.model_type 运行时被重写为 `deepseek_v3`（configuration_kimi_k25.py:53-63）；`kimi_k3`（K3）**本地无 `models\kimi_k3\` 模块**（2026-11-03 Glob 核验），顶层走 HF 仓库 custom code，text_config 实例化目标是本地 `models\kimi_linear\`（KimiLinearForCausalLM）——断点打在 `models\kimi_linear\modeling_kimi_linear.py` 可走 KDA 前向路径。
2. **重点调试对象**：
   - MLA 的低秩链路（K2 系）：`q_a_proj(7168→1536) → RMSNorm → q_b_proj`；`kv_a_proj_with_mqa(7168→512+64)` 缓存只存 512 维潜向量 + 64 维 rope 段，解码时 `kv_b_proj` 展开（modeling_deepseek_v3.py:349-463）；
   - MoE 路由（K2 系）：sigmoid 打分 → e_score_correction_bias → top-8 → 归一化 × routed_scaling_factor 2.827；
   - K3 新链路：KDA 的 delta-rule 状态递推与分块 UT 变换、Gated MLA 的 NoPE + 全秩输出门、AttnRes 的伪查询加权、Stable LatentMoE 的 latent 下投影/上投影与 QB——对照技术报告 §2.1-2.3 与 `models\kimi_linear\` 源码；
   - K2.5~K2.7-Code 视觉链路：MoonViT-3D 的轴向 2D-RoPE、NaViT packing 的 cu_seqlens 变长掩码、PatchMerger 的 merge (2,2) 时空池化；K3 换为 MoonViT-V2 + PatchMergerV2 + 像素重排（config vision_config 字段锚定）。
3. **同架构对照实验**：K2-Base / 0905 / Thinking 共用 `models\deepseek_v3\` 代码，config diff 只有上下文、量化与 eos；K2.5 / K2.6 / K2.7-Code 共用 `models\kimi_k25\` 代码，config diff 只有 eos 与忽略正则——适合做「同架构不同训练/格式」对照。
4. **本地可跑性**：K2 系主模型 1T 级本地不可跑；K2-Thinking / K2.5 / K2.6 的 INT4 发布版是唯一可考虑本地加载的形态（仍需较大显存）；K3 为 2.8T MXFP4 权重，本地双 16G 卡不可跑；Moonlight-16B-A3B（16B-A3B）最适合本地断点调试。

---

## 参考来源

1. `E:\transformers\bayon-note\Kimi\moonshot_8b\index.md`（Moonshot-8B 404 核验记录 + Moonlight-16B-A3B config 对照）
2. `E:\transformers\bayon-note\Kimi\moonshot_32b_a32b\index.md`（Moonshot-32B-A32B 404 核验记录 + Moonlight 对照）
3. `E:\transformers\bayon-note\Kimi\kimi_k2_0905\index.md`（K2-Base/0905：官方 config 留档 + 本地 deepseek_v3 等价实现 + arXiv:2507.20534）
4. `E:\transformers\bayon-note\Kimi\kimi_k2_thinking\index.md`（K2-Thinking：官方 config 留档 + 与 0905 逐字段比对）
5. `E:\transformers\bayon-note\Kimi\kimi_k2_5\index.md`（K2.5：官方 config 留档 + 本地 kimi_k25 源码 + arXiv:2602.02276）
6. `E:\transformers\bayon-note\Kimi\kimi_k2_vl\index.md`（K2.5-VL：视觉塔与 token 流专述，同上来源）
7. `E:\transformers\bayon-note\Kimi\kimi_k2_6\index.md`（K2.6：官方 config 留档 + 与 K2.5 逐字段 diff 程序比对，2026-11-03）
8. `E:\transformers\bayon-note\Kimi\kimi_k2_7_code\index.md`（K2.7-Code：官方 config 留档 + 与 K2.6 逐字段 diff + 模型卡口径字段，2026-11-03）
9. `E:\transformers\bayon-note\Kimi\kimi_k3\index.md`（K3：官方 config 留档 + 技术报告 PDF 留档 _tools/_cfgpull/k3_tech_report.pdf + arXiv:2607.24653，2026-11-03）
10. `E:\transformers\bayon-note\Kimi\kimi_k3_1_pro\index.md`（K3.1-Pro：moonshot.ai 官网抓取 + 无官方仓库核验记录，2026-11-03）
11. 官方仓库/卡片：https://huggingface.co/moonshotai/Kimi-K2-Instruct-0905 、Kimi-K2-Thinking 、Kimi-K2.5 、Kimi-K2.6 、Kimi-K2.7-Code 、Kimi-K3 ；https://hf-mirror.com/api/models?author=moonshotai
12. 论文/博客：https://arxiv.org/abs/2507.20534 ；https://arxiv.org/abs/2602.02276 ；https://arxiv.org/abs/2607.24653 ；https://www.kimi.com/blog/kimi-k3 ；https://platform.kimi.com/docs/guide/kimi-k2-6-quickstart ；https://www.moonshot.ai （K3.1-Pro 展示页）

> 标注约定：全部数字来自各版本目录 index.md 所引官方 config / 本地源码 / 官方论文/模型卡/官网；Moonshot-8B / 32B-A32B 与 K3.1-Pro 无官方 config，统一标「待确认」并保留核验记录。访问与核验日期：2026-10-31（各旧版本文档），2026-11-03（K2.6/K2.7-Code/K3 config 拉取 + K3 技术报告 PDF 留档 + 本对比文档更新）。

---

*页脚：Kimi（Moonshot）全家族版本对比与演进 · 数据来源：bayon-note 各版本 index.md（config + transformers 源码 + 官方论文/模型卡/官网）· 仅供技术学习与论文研究*
