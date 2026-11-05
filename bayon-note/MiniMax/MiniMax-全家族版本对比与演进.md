# MiniMax 全家族版本对比与演进 — Text-01 → VL-01 → M1 → M2 → M2.1 → M2.5 → M2.7 → M3（M3-VL）→ M3.1-Flash

> 整理日期：2026-11-03
> 数据来源：各版本目录 `index.md`（其事实全部源自远端官方 config（hf-mirror 拉取留档）、transformers 本地源码（`models\minimax\` / `models\minimax_m2\` / `models\minimax_m3_vl\`）、官方论文与官方公告/博客，详见文末来源）
> 覆盖版本：minimax_text_01（456B-A46B 基座）、minimax_visual_01（VL-01）、minimax_m1（同构推理版）、minimax_m2（230B-A10B）、minimax_m2_1（M2.1，多语言编程）、minimax_m2_5（M2.5，生产力 SOTA）、minimax_m2_7（M2.7，自我迭代/204K）、minimax_m3_vl（M3，原生多模态）、minimax_m3_1_flash（M3.1-Flash，效率版）
> 用途：配合 `E:\transformers` 源码研读与断点 debug 的家族横向速查

---

## 结论速览

MiniMax 家族的演进主线是一条「**注意力机制两次大转向 + 规模与模态随之调整**」的路线：

1. **Text-01（2025-01）**：首个开源 456B-A46B MoE 基座，招牌是 **Lightning Attention 块递归线性注意力**——70 层线性 + 10 层全 Softmax（7:1），32 专家 top-2、**无共享专家**、门控残差（α≈3.5566），原生 1M 上下文（推理可外推 4M）。
2. **VL-01（2025-01）**：同一文本塔 + **CLIP 视觉塔**（1024/24 层/336px）+ 动态分辨率切块，512B 视觉-语言 token 续训；文本块上下文压到 8192，首开 MiniMax 开源 VLM。
3. **M1（2025-06 论文 / 2025-09 开源）**：**架构逐字段同构**的推理变体——网络一个字段没改，靠大规模 RL（CISPO）练成长链推理，40K/80K 思维预算，100K 生成长度下 FLOPs 约为 DeepSeek-R1 的 25%。
4. **M2（2025-10-27）**：**路线反转**——放弃 Lightning，62 层全改标准 GQA Softmax；专家 32→**256**、top-8；引入 sigmoid 路由偏置、**QK-Norm**、**MTP**（×3）与 FP8 发布；规模降为 230B-A10B，主打智能体/代码。
5. **M2.1（2025-12-22）/ M2.5（2026-02）/ M2.7（2026-03-18）**：**同底座三连迭代**——官方 config 与 M2 逐字段 diff 后网络字段全部一致（62 层 / 256 专家 top-8 / 全 Softmax / MTP×3 / FP8）；M2.1 收窄为「多语言编程专家」，M2.5 主打编程/工具调用/搜索/办公 SOTA，M2.7 唯一真实 config 差异是上下文 196608→**204800**（204K）并提「开启模型的自我迭代」；三版均提供 base / highspeed 双服务变体。
6. **M3（2026-06-01 发布 / 2026-06-12 开源权重）**：**再换注意力为 MSA 块稀疏索引**（index 4 头选 top-16 块 + 主分支精算）；前 3 层 Dense + 后 57 层 MoE（128 专家 top-4 + **带回 1 个共享专家**）；原生多模态（CLIP 32 层/2016px + 视频 + 3D 轴向 RoPE + patch_merge），1M 上下文回归，MTP ×7；1M 下单 token 算力约 M2 的 1/20。
7. **M3.1-Flash（2026-10-19）**：**同一 M3 底座的效率版**——config 侧去 MTP、60 层全部并入 MSA 稀疏路径、rope_theta 5e6→1e7、视觉最大分辨率 2016→672、权重改 MxFP8 + MoE NVFP4 量化发布；官方口径约 3× 推理加速、约 70% agentic 成本下降，保留 1M 上下文与多模态输入，开放权重。

一句话：**Text-01/M1 走"线性注意力省长上下文成本"，M2 全退 Softmax 换规模效率，M3 用块稀疏索引（MSA）在 Softmax 精度与稀疏成本间取折中**；而 M2.1/M2.5/M2.7 证明「同架构换训练」可连续迭代，M3.1-Flash 则在 M3 底座上做推理侧瘦身——是家族最鲜明的演进线索。

---

## 一、全家族核心规格对比表

| 项目 | MiniMax-Text-01 | MiniMax-VL-01 | MiniMax-M1 | MiniMax-M2 | M2.1 / M2.5 | MiniMax-M2.7 | MiniMax-M3（M3-VL） | MiniMax-M3.1-Flash |
|---|---|---|---|---|---|---|---|---|
| 时间 | 2025-01 | 2025-01 | 2025-06 论文 / 2025-09 开源 | 2025-10-27 | 2025-12-22 / 2026-02 | 2026-03-18 | 2026-06-01 发布 / 2026-06-12 开源权重 | 2026-10-19 |
| 入口类 / model_type | MiniMaxForCausalLM / minimax | MiniMaxVL01ForConditionalGeneration / minimax_vl_01 | MiniMaxM1ForCausalLM / minimax_m1 | MiniMaxM2ForCausalLM / minimax_m2 | MiniMaxM2ForCausalLM / minimax_m2（与 M2 同） | MiniMaxM2ForCausalLM / minimax_m2（与 M2 同） | MiniMaxM3SparseForConditionalGeneration / minimax_m3_vl | MiniMaxM3SparseForConditionalGeneration / minimax_m3_vl（与 M3 同） |
| 总参 / 激活 | **456B / ≈45.9B（A46B）** | 同 Text-01 | 同 Text-01 | **≈229.9B / ≈9.8B（A10B）** | 同 M2（config 未单列） | 同 M2（config 未单列） | 文本 60 层（总参未单列） | 同 M3（总参未单列） |
| 层数 / hidden | 80 / 6144 | 80 / 6144（文本塔） | 80 / 6144 | **62 / 3072** | 62 / 3072（同 M2） | 62 / 3072（同 M2） | **60 / 6144**（前 3 层 Dense） | 60 / 6144（同 M3） |
| 注意力 | **Lightning 线性 70 层 + Softmax 10 层（7:1）** | 同 Text-01 | 同 Text-01 | **全 Softmax GQA 62 层** | 全 Softmax GQA 62 层（同 M2） | 全 Softmax GQA 62 层（同 M2） | 前 3 层 full + 后 57 层 **MSA 块稀疏**（block 128 / top-16 块 / local 1） | **60 层全部 MSA**（取消前 3 层 full） |
| Q / KV 头 | 64 / 8（GQA） | 64 / 8 | 64 / 8 | 48 / 8 | 48 / 8（同 M2） | 48 / 8（同 M2） | 64 / 4（16:1） | 64 / 4（同 M3） |
| head_dim / rotary_dim | 128 / 64 | 128 / 64 | 128 / 64 | 128 / 64 | 128 / 64（同 M2） | 128 / 64（同 M2） | 128 / 64（partial 0.5） | 128 / 64（同 M3） |
| rope_theta | 1e7 | 1e7 | 1e7 | **5e6** | 5e6（同 M2） | 5e6（同 M2） | 5e6 | **1e7** |
| 路由专家 / top-k | 32 / 2 | 32 / 2 | 32 / 2 | **256 / 8** | 256 / 8（同 M2） | 256 / 8（同 M2） | **128 / 4 + 1 共享**（routed_scaling 2.0） | 128 / 4 + 1 共享（同 M3） |
| 专家中间维 | 9216 | 9216 | 9216 | **1536** | 1536（同 M2） | 1536（同 M2） | 3072（dense 层 12288） | 3072 / dense 12288（同 M3） |
| 共享专家 | **无**（shared=0） | 无 | 无 | 无 | 无（同 M2） | 无（同 M2） | **有（1 个常驻）** | 有（同 M3） |
| 路由打分 | softmax top-2 | softmax | softmax | **sigmoid + e_score_correction_bias** | sigmoid + 偏置（同 M2） | sigmoid + 偏置（同 M2） | sigmoid + 校正偏置 | sigmoid + 校正偏置（同 M3） |
| QK-Norm | 无 | 无 | 无 | **有（per-layer q/k RMSNorm）** | 有 per-layer（同 M2） | 有 per-layer（同 M2） | **有（per-head）+ GemmaNorm** | per-head + GemmaNorm（同 M3） |
| 残差结构 | **门控残差**（α≈3.5566·残差 + β·子层） | 同 Text-01 | 同 Text-01 | **朴素残差**（取消 α 门控） | 朴素残差（同 M2） | 朴素残差（同 M2） | —（GemmaNorm 体系） | —（同 M3） |
| 上下文 | **1M 训练 / 4M 外推**（max_pos 10,240,000） | **8192/块**（文本塔） | **1M** | 196608（192K，公开口径 204K） | 196608（同 M2） | **204800（204K）** ← config 显式扩窗 | **1,048,576（1M）** | **1,048,576（1M，保留）** |
| 词表 | 200064 | 200064（+`<image>`=200025） | 200064 | 200064（bos 200034 / eos 200020） | 200064（bos/eos 未单列，待确认） | 200064（bos/eos 未单列，待确认） | 200064（image 200025 / video 200026） | 200064（同 M3） |
| MTP | 无 | 无 | 无 | **×3 模块** | ×3（同 M2） | ×3（同 M2） | **×7 模块** | **无（字段取消）** |
| 权重格式 | — | — | — | **FP8 e4m3fn** | FP8 e4m3fn（同 M2） | FP8 e4m3fn（同 M2） | bf16 开放权重 | **MxFP8 + MoE NVFP4**（block [1,32]，group 16） |
| 视觉塔 | 无 | **CLIP**：1024/24 层/16 头/336px/patch14/投影 6144 | 无 | 无 | 无 | 无 | **CLIP**：1280/32 层/**2016px**/patch14/投影 6144 + 视频 + 3D 轴向 RoPE + patch_merge（空间×2/时间×2） | **CLIP**：1280/32 层/**672px**/patch14/投影 6144（其余同 M3） |
| 训练特点 | 预训练基座 | 512B 视觉-语言 token 续训 | **大规模 RL（CISPO）长链推理**，40K/80K 思维预算 | 智能体/代码（发布口径） | 多语言编程专家 / 生产力场景 SOTA（release notes 口径） | 「开启模型的自我迭代」（release notes 口径） | 原生多模态（图像+视频） | 效率导向：约 3× 加速、约 70% agentic 成本下降（博客口径） |
| 服务变体 | — | — | — | — | base / highspeed（M2.1、M2.5） | base / highspeed（M2.7） | — | — |
| transformers 模块 | 原生 `minimax` | 无独立模块（官方实现） | 无独立模块 | 原生 `minimax_m2` | 原生 `minimax_m2`（同 M2） | 原生 `minimax_m2`（同 M2） | 原生 `minimax_m3_vl` | 原生 `minimax_m3_vl`（同 M3） |
| 主要论文/公告 | arXiv:2501.08313 | arXiv:2501.08313 | arXiv:2506.13585 | 官方发布 2025-10-27 | 官方 release notes 2025-12-22 / 2026-02 | 官方 release notes 2026-03-18 | 官方发布 2026-06-01，minimaxi.com/blog/minimax-m3 | 官方博客 2026-10-19（minimaxai.co） |

> 说明：① M1 与 Text-01 的 config **逐项 diff 后网络字段全部相同**（hidden/层数/头/专家/rope_theta/rotary_dim/postnorm_alpha/attn_type_list），「80k」指 RL 生成长度而非网络差异。② M2 的 `mlp_intermediate_size=8192` 字段本地原生模块不消费（远端代码字段，**细节待确认**）。③ M3 的 total 参数未在源文档单列。④ **M2.1/M2.5 的 config 与 M2 逐字段 diff 后网络字段完全一致**（仅 transformers_version 4.57.1→4.46.1 与省略若干默认字段），两版能力差异为训练口径；**M2.7 仅 2 处真实差异**：max_position 196608→204800、新增 dtype=bfloat16 标注字段。⑤ M3.1-Flash 与 M3 的 config diff 见 `minimax_m3_1_flash/index.md` 第二节；其 image_grid_pinpoints 仍保留到 2016 的档位表，与 vision_config.image_size=672 的关系**待确认**。

---

## 二、代际演进详解

### 2.1 MiniMax-Text-01（2025-01）：Lightning 线性注意力基座

- **7:1 混合注意力**：80 层中 70 层 Lightning 线性注意力 + 10 层全 Softmax（落在索引 7/15/…/79，每 8 层最后一层）。
- **Lightning Attention 三段式**（`MiniMaxLightningAttention.forward`）：
  1. 块内（block_size=256）精确因果注意力（带对角衰减掩码）；
  2. 块间固定状态 $S\in\mathbb{R}^{B\times H\times d\times d}$ 递归：$S\leftarrow S\cdot\gamma_{\mathrm{block}}+(K\cdot\mathrm{key\_decay})^\top V$；
  3. 逐头指数衰减（`slope_rate_h = base^h`，层越深衰减越慢），`γ_block = exp(-slope_rate·block_size)`。
- Lightning 层不挂 RoPE（RoPE 只在 10 个全注意力层，rotary_dim=64、θ=1e7）；qkv 过 SiLU + sigmoid 输出门。
- **MoE**：32 专家 softmax top-2，`shared_intermediate_size=0` 无共享专家。
- **门控残差**：`hidden = residual·α + sublayer·β`（α=3.5565588200778455，非朴素 x+sublayer）。
- 上下文：训练 1M、推理外推 4M（max_position 10,240,000，无滑动窗口）。

### 2.2 MiniMax-VL-01（2025-01）：加 CLIP 视觉塔

| 维度 | Text-01 | VL-01 |
|---|---|---|
| 文本塔 | 80 层 / 6144 / 7:1 / 32 专家 | **逐字段复用**（`text_config` 同源） |
| 视觉塔 | 无 | CLIP ViT：hidden 1024 · 24 层 · 16 头 · inter 4096 · patch 14 · image 336 · 投影 6144（gelu） |
| 上下文 | 1M 外推 | **8192/块**（按图像块切分，长图分块送塔） |
| 动态分辨率 | — | **grid pinpoints**：基准 336，向上到 672/1008/1344/1680/2016，多瓦片 576 token/瓦 |
| 特殊 token | — | `<image>`=200025 |
| 训练 | 预训练 | 512B 视觉-语言 token 续训 |

- VL-01 是 MiniMax 开源 VLM 起点：CLIP 塔 + 动态瓦片 + 块式 8192 上下文，为后续 M3 的原生多模态提供基础。

### 2.3 MiniMax-M1（2025-06 论文 / 2025-09 开源）：同构权重 + RL 推理训练

- **架构零改动**：与 Text-01 config 网络字段逐项相同；差异全在训练目标——大规模 RL（**CISPO**：裁剪重要性采样权重而非 token 更新，512×H800 三周）。
- 产物：世界首个开源大规模混合注意力长链推理模型，40K / 80K 两档思维预算。
- 效率卖点：长输入被 Lightning 状态近线性吸收，100K 生成长度 FLOPs ≈ DeepSeek-R1 的 25%。

### 2.4 MiniMax-M2（2025-10-27）：路线反转（全 Softmax + 256 专家）

| 维度 | Text-01 / M1 | M2 |
|---|---|---|
| 注意力 | Lightning 7:1 | **62 层全标准 GQA Softmax**（attn_type_list 全 1，无线性分支） |
| 规模 | 456B-A46B | **≈230B-A10B**（hidden 3072 / 62 层） |
| 专家 | 32 top-2，inter 9216 | **256 top-8**，inter 1536（小专家换容量） |
| 路由 | softmax | **sigmoid + e_score_correction_bias** |
| QK-Norm | 无 | **q/k 投影后 per-layer RMSNorm** |
| 残差 | 门控 α·h+β·s | **朴素 x+sublayer**（取消 α） |
| MTP | 无 | **×3** |
| 上下文 | 1M | 192K（公开口径 204K） |
| 权重 | — | **FP8 e4m3fn** |

- M2 的定位：放弃线性注意力对长上下文的成本优势，**用 256 专家压激活量、QK-Norm 稳定长文本**，主打智能体与代码。

### 2.5 M2 系迭代三连：M2.1（2025-12-22）/ M2.5（2026-02）/ M2.7（2026-03-18）

三版均为 API 系模型（各提供 base / highspeed 双变体），但官方 HF 仓库已开放 config：

| 维度 | M2 | M2.1 / M2.5 | M2.7 |
|---|---|---|---|
| config 网络字段 | 基准 | **逐字段相同**（62 层 / 3072 / 256 专家 top-8 / 全 Softmax / QK-Norm / MTP×3 / FP8） | 与 M2 相同，**仅 2 处差异** |
| `max_position_embeddings` | 196608 | 196608 | **204800（204K）** |
| 其他 config 差异 | — | transformers_version 4.57.1→4.46.1；省略若干默认字段 | 同左 + 新增 `dtype=bfloat16` 标注字段 |
| 发布口径 | 智能体/代码 | M2.1 多语言编程专家；M2.5 编程/工具调用/搜索/办公 SOTA | M2.7「开启模型的自我迭代」 |

- 这三连是家族第二次「**同架构换训练**」样本（第一次是 M1）：config 零改动或近乎零改动，能力与定位靠后训练/数据迭代，highspeed 变体是部署侧加速（config 无法核验）。
- M2.7 的 204800 把 M2 时代「公开口径 204K」落成了 config 显式值。

### 2.6 MiniMax-M3（2026-06-01 发布 / 2026-06-12 开源权重）：MSA 块稀疏 + 原生多模态

| 维度 | M2 | M3（M3-VL） |
|---|---|---|
| 注意力 | 全 Softmax | 前 3 层 full + 后 57 层 **MSA 块稀疏**（block 128 / topk_blocks 16 / local 1 / index 4 头 / index_head_dim 128 / sparse_attention_freq 3） |
| MoE | 256 top-8，无共享 | **128 top-4 + 1 常驻共享专家**（routed_scaling 2.0），前 3 层 Dense（inter 12288） |
| 激活 | — | **SwiGLU-OAI**（α=1.702，limit 7.0） |
| Norm | per-layer QK-Norm | **per-head QK-Norm + GemmaNorm** |
| MTP | ×3 | **×7** |
| 上下文 | 192K | **1M** |
| 多模态 | 无 | **CLIP 1280/32 层/2016px + 视频**：3D 轴向 RoPE、patch_merge 空间×2/时间×2、image 200025 / video 200026、每图 576 token |
| 效率 | — | 1M 上下文单 token 算力约 M2 的 1/20，预填充约 9× |

- MSA 机制（index 先选块、main 再精算）在两阶段稀疏注意力中**保留了 Softmax 精度**，同时把 1M 上下文成本拉回近线性——这是对 Text-01 线性化与 M2 全 Softmax 两条路线的折中收敛。

### 2.7 MiniMax-M3.1-Flash（2026-10-19）：M3 底座的效率瘦身

| 维度 | M3 | M3.1-Flash |
|---|---|---|
| 文本塔 / MoE | 60 层 / 6144 / 128 专家 top-4 + 1 共享 | **完全继承** |
| 注意力排布 | 前 3 层 full + 后 57 层 MSA | **60 层全部 MSA** |
| MTP | ×7 | **无（字段取消）** |
| rope_theta | 5e6 | **1e7** |
| 视觉最大分辨率 | 2016px | **672px** |
| 权重 | bf16 开放权重 | **MxFP8 + MoE NVFP4**（block [1,32]，group 16） |
| 上下文 / 多模态 | 1M / 图像+视频 | **保留 1M / 保留多模态输入** |

- 官方博客口径：约 3× 推理加速、约 70% agentic 成本下降，API 约 $0.15 输入 / $0.60 输出每 1M token，开放权重。
- 与 M2 系「同架构换训练」不同，M3.1-Flash 是**同架构换部署形态**：网络骨架不动，靠去 MTP、全层稀疏、视觉降分辨率与低比特量化把推理成本打下来。

---

## 三、关键机制演进一图流

```
注意力：     Lightning 线性 7:1（T01/VL-01/M1）→ 全 Softmax GQA（M2/M2.1/M2.5/M2.7）
             → MSA 块稀疏（index top-16 块 + main 精算）（M3）
             → 全 60 层 MSA（M3.1-Flash）
MoE：        32 top-2 无共享（T01/VL-01/M1）→ 256 top-8 无共享（M2 系全部）
             → 128 top-4 + 1 共享 + 前 3 层 dense（M3 / M3.1-Flash）
路由：       softmax（T01）→ sigmoid + 校正偏置（M2 系 / M3 系）
残差：       门控 α·h+β·s（T01/VL-01/M1）→ 朴素 x+sublayer（M2 系）
QK-Norm：    无（T01）→ per-layer（M2 系）→ per-head + GemmaNorm（M3 系）
MTP：        无（T01）→ ×3（M2 系）→ ×7（M3）→ 取消（M3.1-Flash）
上下文：     1M/4M 外推（T01/M1）→ 192K（M2/M2.1/M2.5）→ 204K（M2.7）→ 1M（M3 / M3.1-Flash）
模态：       文本（T01/M1/M2 系）→ +图像（VL-01）→ +图像+视频（M3 系）
权重：       bf16（T01/M1/VL-01）→ FP8 e4m3fn（M2 系）→ bf16（M3）→ MxFP8 + NVFP4（M3.1-Flash）
训练：       预训练（T01）→ 续训（VL-01）→ RL 推理（M1）→ 智能体/代码（M2）
             → 编程/生产力迭代（M2.1/M2.5/M2.7）→ 原生多模态（M3）→ agentic 效率（M3.1-Flash）
```

演进主线解读：
1. **注意力三连换**：线性化（T01）→ 全 Softmax（M2 系）→ 块稀疏折中（M3），是家族最核心的路线变量；M3.1-Flash 只是把全层统一到 MSA，不改路线。
2. **MoE 配置两轮调整**：32 专家大中间维（9216）→ 256 专家小中间维（1536，M2 系整代沿用）→ 128 专家 + 共享专家 + 前 3 层 dense（M3 系）。
3. **稳定化逐步加码**：门控残差（T01）→ QK-Norm（M2 系）→ per-head QK-Norm + GemmaNorm + SwiGLU-OAI（M3 系）。
4. **两种迭代范式并存**：M1 / M2.1 / M2.5 / M2.7 证明"同架构换训练"（config diff 为零或近乎零）；VL-01 / M3 是"同塔加视觉"；M3.1-Flash 则是"同架构换部署形态"（去 MTP + 低比特量化）。

---

## 四、对源码研读 / debug 的启示

1. **按 model_type 分流**：`minimax`（T01 原生模块）、`minimax_vl_01`（无本地模块，官方实现）、`minimax_m1`（无本地模块，但文本塔与 `models\minimax\` 一一对应）、`minimax_m2`（原生 `models\minimax_m2\`）、`minimax_m3_vl`（原生 `models\minimax_m3_vl\`）——三代注意力分别有独立实现文件。
2. **重点调试对象**：
   - T01/M1 的 Lightning：`MiniMaxLightningAttention` 的块内因果 + `linear_cache [B,H,d,d]` 递归状态、逐头/逐层衰减率计算、SiLU + sigmoid 门；
   - M2 的 `MiniMaxM2BlockSparseTop2MLP`（256 专家 sigmoid top-8 + bias）与 QK-Norm（q/k 后 RMSNorm）、MTP 头堆叠；
   - M3 的 `MiniMaxM3SparseForConditionalGeneration`：MSA 索引头（4 头 × 128 维）选 top-16 块、GQA 组内局部块、patch_merge 时空池化与 3D 轴向 RoPE 的位置表构造。
3. **同架构对照实验**：T01 与 M1 共用 `models\minimax\` 代码（config diff 为零），是做「同架构不同训练」的标准样本；M2 与 M3 是「同家族不同注意力路线」的对照样本。
4. **本地可跑性**：456B/230B 主模型本地不可跑；本地双 16G 卡适合调试 `models\minimax\` / `models\minimax_m2\` / `models\minimax_m3_vl\` 的纯前向/单层单元（无权重或小权重）。

---

## 参考来源

1. `E:\transformers\bayon-note\MiniMax\minimax_text_01\index.md`（Text-01：官方 config + 本地 minimax 源码 + arXiv:2501.08313）
2. `E:\transformers\bayon-note\MiniMax\minimax_visual_01\index.md`（VL-01：官方 config + arXiv:2501.08313）
3. `E:\transformers\bayon-note\MiniMax\minimax_m1\index.md`（M1：官方 config 逐项 diff + arXiv:2506.13585）
4. `E:\transformers\bayon-note\MiniMax\minimax_m2\index.md`（M2：官方 config + 本地 minimax_m2 源码 + 官方发布）
5. `E:\transformers\bayon-note\MiniMax\minimax_m2_1\index.md`（M2.1：官方 config + 官方 release notes 2025-12-22）
6. `E:\transformers\bayon-note\MiniMax\minimax_m2_5\index.md`（M2.5：官方 config + 官方 release notes 2026-02）
7. `E:\transformers\bayon-note\MiniMax\minimax_m2_7\index.md`（M2.7：官方 config + 官方 release notes 2026-03-18）
8. `E:\transformers\bayon-note\MiniMax\minimax_m3_vl\index.md`（M3：官方 config + 本地 minimax_m3_vl 源码 + 官方博客）
9. `E:\transformers\bayon-note\MiniMax\minimax_m3_1_flash\index.md`（M3.1-Flash：官方 config 与 M3 逐项 diff + 官方博客 2026-10-19）
10. 官方仓库/卡片：https://huggingface.co/MiniMaxAI/MiniMax-Text-01 、MiniMax-VL-01 、MiniMax-M1 、MiniMax-M2 、MiniMax-M2.1 、MiniMax-M2.5 、MiniMax-M2.7 、MiniMax-M3 、MiniMax-M3.1-Flash
11. 官方 release notes（模型发布日历）：https://platform.minimaxi.com/docs/release-notes/models
12. 论文/公告：https://arxiv.org/abs/2501.08313 ；https://arxiv.org/abs/2506.13585 ；https://www.minimaxi.com/blog/minimax-m3 ；M3.1-Flash 官方博客（minimaxai.co，2026-10-19）

> 标注约定：全部数字来自各版本目录 index.md 所引官方 config / 本地源码 / 官方论文与公告；未核实项（M2 的 mlp_intermediate_size 消费、M3 总参、M2.1/M2.5/M2.7 的 bos/eos、M3.1-Flash 的 image_grid_pinpoints 与 image_size 关系）按源文档标「待确认」或留空。访问与核验日期：2026-10-31（Text-01~M3 各版本文档），2026-11-03（本对比文档更新 + M2.1/M2.5/M2.7/M3.1-Flash config 拉取）。

> 日期修正记录（2026-11-03）：MiniMax-M3 官方发布日以官方 release notes 为准修正为 **2026-06-01**（06-12 开源权重），原「2026-05」表述已全文更正。

---

*页脚：MiniMax 全家族版本对比与演进 · 数据来源：bayon-note 各版本 index.md（config + transformers 源码 + 官方论文）· 仅供技术学习与论文研究*
