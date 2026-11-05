# Qwen 全家族版本对比与演进 — Qwen1 → 1.5 → 2 → 2.5 → 3 → 3.5 → 3.6 → 3.8

> 整理日期：2026-11-02
> 数据来源：各版本目录 `index.md`（其事实全部源自本地/远端官方 config（hf-mirror / ModelScope 拉取留档）、transformers 本地源码（`models\qwen2\` / `qwen2_moe\` / `qwen2_vl\` / `qwen2_5_vl\` / `qwen3\` / `qwen3_moe\` / `qwen3_vl\` / `qwen3_5\` / `qwen3_5_moe\`）、官方 model card / 博客与论文，详见文末来源）
> 覆盖版本：qwen1、qwen1_5、qwen2、qwen2_5、qwen3、qwen3_5、qwen3_6、qwen3_8（含 2-VL / 2.5-VL / 3-VL 等视觉变体）
> 用途：配合 `E:\transformers` 源码研读与断点 debug 的家族横向速查
> 已核实对比文件：`Qwen\Qwen-27B架构对比-3.5-3.6-3.8.md`（27B 三代逐字段一致）已纳入本表

---

## 结论速览

Qwen 家族的演进主线是一条「**三代注意力路线 + 两轮词表跃迁 + 多模态逐步原生化**」的路线：

1. **Qwen1（2023-08-03 开源 / 2023-09-28 报告）**：remote-code 首代底座——纯 MHA + 融合 QKV 投影（c_attn）+ LogN/动态 NTK 把 8K 训练外推到 32K，词表 151936/152064。
2. **Qwen1.5（2024-02-04 发布）**：**架构换血到 Qwen2 模板**——SwiGLU 全门控维、分立 q/k/v/o 投影、静态 RoPE θ=1e6、全系原生 32K、32B 首引 GQA、ChatML 模板。
3. **Qwen2（2024-06-07 开源 / 2024-07-15 报告）**：**GQA 全系标配**（0.5B 即 14Q·2KV）+ SWA 滑窗字段 + 首个 MoE（57B-A14B，64 专家选 8 + 共享）+ 2-VL 多模态（patch14 · M-RoPE · 动态分辨率）。
4. **Qwen2.5（2024-09-19 开源 / 2024-12-19 报告）**：文本塔**完全延续 qwen2**（0.5B config 与 Qwen2-0.5B 同规格）；2.5-VL 上下文扩 128K、视觉塔改 window attention（窗 112）。
5. **Qwen3（2025-04-29 发布 / 2025-05-14 报告）**：**注意力回归全注意力**（sliding_window=null）+ 逐头 QK-Norm + head_dim 显式 128 + **原生 thinking**（`/think` `/no_think` 软切换）+ MoE 128 选 8（无共享专家）+ 3-VL（patch16、文本 θ 5e6、262K）。
6. **Qwen3.5（2026-02-16 发布）**：**架构大改**——**Gated DeltaNet 线性注意力 + Gated Attention 3:1 混合**（每 4 层 3 线性 + 1 全注意力），词表 151936→**248320**，上下文 262144→YaRN 1M，多模态原生化，MoE 256 选 8 + 显式共享专家。
7. **Qwen3.6（2026-04-22）**：**架构零变化**（27B config 与 3.5 逐字段相同）——换代在稳定化 + Thinking Preservation（历史思考块可选保留）。
8. **Qwen3.8（2026-08-05 开源 / 2026-08-13 FP8）**：**架构零变化**——**Flexible Thinking Control**（思考默认开可关、reasoning_effort 三档、preserve_thinking 默认全开）+ 首次开放 Max 级纯文本 **2.4T-A95B** + 首个官方 FP8 量化版 + TokenSpeed 引擎。

一句话：**Qwen1–Qwen3 走"标准 Transformer 系"（MHA→GQA、加滑窗字段又回归全注意力、MoE 由 57B 试水到 235B 定型），Qwen3.5 起转向"混合注意力"（Gated DeltaNet + Gated Attention 3:1，近 O(n) 长上下文），3.5/3.6/3.8 三代图纸一致、靠训练与行为换代**——这是家族最清晰的演进分水岭。

---

## 一、全家族核心规格对比表

| 项目 | Qwen1 | Qwen1.5 | Qwen2 | Qwen2.5 | Qwen3 | Qwen3.5 | Qwen3.6 | Qwen3.8 |
|---|---|---|---|---|---|---|---|---|
| 时间 | 2023-08-03 开源 / 2023-09-28 报告 | 2024-02-04 发布 | 2024-06-07 开源 / 2024-07-15 报告 | 2024-09-19 开源 / 2024-12-19 报告 | 2025-04-29 发布 / 2025-05-14 报告 | 2026-02-16 发布 | 2026-04-22 | 2026-08-05 开源 / 2026-08-13 FP8 |
| 入口类 / model_type | QWenLMHeadModel / qwen（**remote code**，无原生模块） | Qwen2ForCausalLM / qwen2（原生） | qwen2 / qwen2_moe / qwen2_vl | qwen2 / qwen2_5_vl | qwen3 / qwen3_moe / qwen3_vl | qwen3_5 / qwen3_5_moe | 同 3.5 | qwen3_5 / qwen3_5_moe（2.4T） |
| 代表性尺寸 | 1.8B/7B/14B/72B | 0.5B–72B（6 档） | 0.5B/1.5B/7B/57B-A14B/72B | 0.5B–72B（6 档） | Dense 0.6B–32B + MoE 30B-A3B/235B-A22B | 0.8B/9B/27B/35B-A3B/122B-A10B/397B-A17B | 27B / 35B-A3B | 27B / 2.4T-A95B / 6B-Flash（**待确认**） |
| 注意力 | **MHA**（Q=KV），融合 c_attn，全维 RoPE | MHA（**仅 32B GQA** 5:1），分立投影 | **GQA 全系**（0.5B 7:1 … 72B 8:1）+ SWA 字段 | GQA 全系（14B/32B 5:1） | **GQA 全系 + 逐头 QK-Norm + head_dim 显式 128**；全注意力（sliding null） | **GDN 线性 + GA 3:1**（每 4 层 1 GA）；GA head_dim 256 / partial RoPE 64 / 输出门 swish | 同 3.5 | 同 3.5（2.4T：GA 64Q·4KV / GDN V128） |
| KV 头 | =Q（MHA） | =Q（32B：8） | 2–8（全系分组） | 2–8 | Dense 8 / MoE 4 | GA 2–4 / GDN K16·V16-48 | 同 | GA 4 / GDN V48-128 |
| MoE | 无 | 无 | **57B-A14B**：64 专家 top-8 + 共享 | 无（2.5-Max 闭源） | 30B-A3B / 235B-A22B：128 top-8，**无共享** | 35B-A3B 等：256 top-8 + **显式共享**（aux 0.001） | 同 3.5 | **2.4T-A95B**：512 top-10 + 共享（inter 2048） |
| 词表 | 151936/152064 | 同 | 同 | 同 | 151936 | **248320**（padded） | 同 | 同 |
| 文本上下文 | 8K 训练 / 32K 外推（LogN+NTK） | 32768 | 32768 | 32768 | 40960 | 262144 → **YaRN 约 1M** | 同（约 1,010,000） | 同（约 1,000,000） |
| rope_theta | 1e4 + 动态 NTK（72B 静态 1e6） | 1e6 | 1e6 | 1e6 | 1e6（3-VL 文本 5e6） | 1e7 | 1e7 | 1e7 |
| 多模态 | 无 | 无 | 2-VL：patch14 · M-RoPE[16,24,24] · 动态分辨率 · 32K | 2.5-VL：**128K** · 视觉塔 window attn（窗 112）· 原生动态分辨率 | 3-VL：patch16 · M-RoPE[24,20,20] · 262K | **全系原生多模态**（early fusion · patch16） | 同 | 27B 多模态；2.4T 纯文本 |
| thinking | 无 | 无 | 无 | 无 | **原生**：`/think` `/no_think` + enable_thinking | 默认常开、取消软切换 | interleaved + preserve_thinking 可选 | **Flexible Thinking Control**：可关 + effort 三档 + preserve 默认开 |
| MTP | 无 | 无 | 无 | 无 | qwen3_next（投机头） | multi-steps（mtp_num_hidden_layers=1） | 同 | 同 |
| 官方量化 | 无 | 无 | 无 | 无 | 无 | 无 | 无 | **首个官方 FP8**（27B-FP8，block 128，约 29GB） |
| tie_word_embeddings | 全 false | 仅 0.5B true | 小尺寸 true | 小尺寸 true | ≤4B true，8B 起 false | 0.8B true | false | false |
| 主要论文/公告 | arXiv:2309.16609 | qwen1.5 博客 | 2407.10671 / 2409.12191 | 2412.15115 / 2502.13923 / 2409.12186 | 2505.09388 / 2511.21631 | qwen3.5 博客 | 官方 model card | 2026-08 博客 + model card |

> 说明：① Qwen1.5/2/2.5 的文本塔共用 `qwen2` model_type 与 `models\qwen2\` 代码；Qwen3.5/3.6/3.8 共用 `qwen3_5` / `qwen3_5_moe`（27B 三代 config 逐字段一致，见 `Qwen-27B架构对比-3.5-3.6-3.8.md`）。② 3.8-6B-Flash 的 HF/ModelScope 仓库暂不可达，全部规格标「待确认」，禁止编造。

---

## 二、代际演进详解

### 2.1 Qwen1（2023-08-03 开源 / 2023-09-28 报告）：remote-code 首代底座

- 形态：纯文本 Dense，MHA（Q=KV 恒等），**融合 QKV 投影 c_attn**（d→3d 沿特征维三等分），kv_channels=128，RoPE 全维旋转（rotary_pct=1.0）。
- 长上下文技巧：**LogN 注意力**（推理超长时 query × log 系数补偿 softmax 稀释）+ **动态 NTK**（按真实序列长度即时重算基频 `base' = base × ntk_alpha^{d/(d-2)}`）；72B 例外——直接静态 θ=1e6 并关闭两开关。
- MLP：`w1·silu(w2)`，内维 = intermediate/2（与后续标准 SwiGLU 不同）。
- 分词：QWenTokenizer（tiktoken BPE），对话格式在仓库 `qwen_generation_utils.py` 的 chat()（无内置 chat_template）。
- 规模：1.8B/7B/14B/72B，全系 tie=False。

### 2.2 Qwen1.5（2024-02-04 发布）：架构换血到 qwen2 模板

| 维度 | Qwen1 | Qwen1.5 |
|---|---|---|
| 代码形态 | remote code（trust_remote_code） | **transformers≥4.37 原生**（Qwen2ForCausalLM） |
| QKV 投影 | 融合 c_attn（d→3d，无 bias） | **分立 q/k/v_proj（带 bias）+ o_proj（无 bias）** |
| 头结构 | 纯 MHA | 小尺寸 MHA；**32B 首次 GQA（40Q·8KV）** |
| RoPE | 1e4 + 动态 NTK / LogN | **静态 θ=1e6**，无 NTK/logn 字段 |
| MLP | 内维=inter/2 | **标准 SwiGLU**（inter 即全门控维） |
| 上下文 | 8K 训练扩 32K | **全系原生 32768**（sliding_window 字段实为遗留，未启用） |
| 对话 | 仓库 chat() | **ChatML 内置**（im_start/im_end） |

### 2.3 Qwen2（2024-06-07 开源 / 2024-07-15 报告）：GQA 全系 + 首个 MoE + 2-VL

| 维度 | Qwen1.5 | Qwen2 |
|---|---|---|
| GQA | 仅 32B | **全系标配**（0.5B 14Q·2KV 7:1 … 72B 64Q·8KV 8:1） |
| 滑窗 | 无 | **SWA 字段**（sliding_window 32768/131072 + max_window_layers）；实际 use_sliding_window=false 未启用 |
| MoE | 无 | **首个 MoE 57B-A14B**（64 专家 top-8 + 共享专家，qwen2_moe） |
| 多模态 | 无 | **2-VL**（patch14 · temporal 2 · spatial_merge 2 · M-RoPE[16,24,24] · 动态分辨率）+ 2-Audio |
| 规模 | 0.5B–72B | 0.5B/1.5B/7B/57B-A14B/72B（MoE 与 7B 同维 3584/28 层） |

- 注意：Qwen2 的 `sliding_window` 字段在 v5 `__post_init__` 中因 `use_sliding_window=false` 被置 None，**当前 v5 推理全层走 full_attention**（SWA 机制保留在源码供开关使用）。

### 2.4 Qwen2.5（2024-09-19 开源 / 2024-12-19 报告）：文本延续 + VL 升级

- 文本塔：**与 Qwen2 完全同构**（0.5B config 同规格，仅 max_window_layers 24→21；大尺寸 70）。
- 2.5-VL：上下文 **128K**；视觉塔改 **window attention**（窗 112，全局块 7/15/23/31，32 层中仅 4 块全局）+ 原生动态分辨率（patch 14 直切任意分辨率）。
- 变体：2.5-Coder / 2.5-Omni / 2.5-Max（闭源 API）。

### 2.5 Qwen3（2025-04-29 发布 / 2025-05-14 报告）：全注意力回归 + thinking + QK-Norm

| 维度 | Qwen2.5 | Qwen3 |
|---|---|---|
| 注意力 | GQA + 滑窗字段 | **回归全注意力**（sliding_window=null）+ **逐头 Q/K RMSNorm（QK-Norm）** + head_dim 显式 128 |
| 推理行为 | 无原生思维 | **原生 thinking**（enable_thinking 默认开 + `/think` `/no_think` 软切换；`<think>…</think>` 文本内编码） |
| MoE | 无 | 30B-A3B / 235B-A22B：**128 专家 top-8，无共享专家**，decoder_sparse_step=1 |
| 上下文 | 32K | 40960（文本）；3-VL **262144** |
| 3-VL | — | patch16 · temporal2 · spatial_merge2 · text rope_theta **5e6** · M-RoPE [24,20,20] · deepstack [5,11,17] |
| MTP | 无 | qwen3_next（投机解码） |

### 2.6 Qwen3.5（2026-02-16 发布）：混合注意力大改（Gated DeltaNet + Gated Attention 3:1）

| 维度 | Qwen3 | Qwen3.5 |
|---|---|---|
| 注意力 | 全注意力（head_dim 128） | **3/4 层 Gated DeltaNet（线性，递归状态，近 O(n)）+ 1/4 层 Gated Attention（GQA，head_dim 256）**；partial_rotary 0.25（RoPE 64 维）；attn_output_gate=swish |
| 词表 | 151936 | **248320（padded）** |
| 上下文 | 40960 / 3-VL 262144 | **262144 原生 → YaRN 约 1M** |
| 多模态 | 可选 3-VL 变体 | **全系原生视觉语言**（early fusion 进主 config，全系 Dense/MoE 带视觉塔） |
| MoE | 128 top-8 无共享 | **256 top-8 + 显式共享专家**（shared_inter 512，router_aux_loss_coef 0.001） |
| thinking | 默认开 + 软切换 | **默认常开、取消软切换** |

- GDN 单元（`Qwen3_5GatedDeltaNet`）：QK16 head_dim 128 / V16-48；causal conv1d（kernel 4）局部归纳；chunk 门控 delta rule 训练、recurrent 解码（状态 `S` 与 conv_state 双缓存）。
- 代表尺寸：0.8B/9B/27B（Dense）/ 35B-A3B/122B-A10B/397B-A17B（MoE）。

### 2.7 Qwen3.6（2026-04-22）：架构零变化，稳定化 + Thinking Preservation

- **27B config 与 3.5 逐字段相同**（64 层 / hidden 5120 / inter 17408 / GA 24Q·4KV·head256 / GDN QK16·V48 / 词表 248320 / 262144→YaRN 约 1,010,000）。
- 行为换代：interleaved thinking 默认**只保留最新一条用户消息的思考块**；新增可选 `preserve_thinking=True` 保留全部历史思考块（agent 长程任务：决策一致、减少重复推理）。
- 不官方支持 `/think` `/nothink`；关闭思考用 `enable_thinking=False`；官方建议上下文 ≥128K 以保思考。

### 2.8 Qwen3.8（2026-08-05 开源 / 2026-08-13 FP8）：架构零变化，Flexible Thinking Control + 2.4T-A95B

| 维度 | Qwen3.6 | Qwen3.8 |
|---|---|---|
| 架构图纸 | qwen3_5 | **完全一致**（官方原文「Built on the architectural foundation of Qwen3.5」；27B 逐字段相同） |
| 思考模式 | 默认开；interleaved | **Flexible Thinking Control**：默认开**可关** + reasoning_effort（xhigh/medium/low）+ preserve_thinking **默认全开** |
| 旗舰开放模型 | 35B-A3B（MoE 多模态） | **2.4T-A95B（纯文本超大 MoE，首次开放 Max 级）**：hidden 8192 / 92 层（23×4）/ GDN V128 / MoE 512 top-10 + 共享 / 强制思考不可关 |
| 官方量化 | 无 | **首个官方 FP8**（27B-FP8，fine-grained block 128，约 29GB） |
| 推理引擎 | transformers/vLLM/SGLang/KTransformers | 新增并列推荐 **TokenSpeed** |
| Agentic 能力 | 基线 | 跨代跃升（DeepSWE 1.1：13.3→42.2；SWE-bench Pro：53.5→61.7；Agents' Last Exam：10.6→20.4） |

---

## 三、关键机制演进一图流

```
注意力：     MHA + 融合 c_attn（1）→ MHA / 32B-GQA · 分立投影（1.5）
             → GQA 全系 + SWA 字段（2/2.5）→ 全注意力 + QK-Norm + head_dim=128（3）
             → Gated DeltaNet 线性 + Gated Attention 3:1（3.5 起恒定，3.6/3.8 图纸不变）
投影：       融合 c_attn d→3d（1）→ 分立 q/k/v(带bias) + o(无bias)（1.5 起恒定）
长上下文：   LogN + 动态 NTK（1）→ 静态 θ=1e6 + 原生 32K（1.5/2/2.5）
             → 40960（3）→ 262144 → YaRN 约 1M（3.5/3.6/3.8）
MoE：        无（1/1.5）→ 57B-A14B 64 选 8 + 共享（2）
             → 128 选 8 无共享（3）→ 256 选 8 + 共享（3.5/3.6）→ 512 选 10 + 共享（3.8 2.4T-A95B）
词表：       151936/152064（1–3）→ 248320（3.5 起）
多模态：     无（1/1.5）→ 2-VL patch14·M-RoPE（2）→ 2.5-VL 128K + 视觉塔 window attn（2.5）
             → 3-VL patch16·262K（3）→ 全系原生多模态（3.5 起）
thinking：   无（1–2.5）→ 原生 + /think 软切换（3）→ 默认常开（3.5）
             → interleaved + preserve 可选（3.6）→ Flexible Control 可关/三档/保留默认开（3.8）
训练/行为：  预训练→对齐（1–3）→ 稳定化 + 实用化（3.6）→ 行为控制 + agentic 跃升（3.8）
```

演进主线解读：
1. **两大分水岭**：Qwen1.5 完成「remote code → 原生 qwen2 模板」的换血；Qwen3.5 完成「标准 Transformer 注意力 → 混合注意力（GDN+GA）」的换芯。此后 3.5/3.6/3.8 三代图纸冻结。
2. **注意力路线三连转**：MHA → GQA（+SWA 字段又回归全注意力）→ 线性/全注意力 3:1 混合，与 MiniMax 的 Lightning→Softmax→MSA 形成对照。
3. **MoE 持续扩面**：64→128→256→512 专家，且 3.5 起显式引入共享专家（2 代的共享专家在 qwen2_moe 已出现但 3 代一度去掉）。
4. **多模态逐步原生化**：可选变体（2-VL/3-VL）→ 全系主 config 原生（3.5 起）；3.8 的 2.4T 反转为纯文本（多模态仅 27B 承担）。
5. **行为层代际主题**：thinking 从无到有（3）→ 默认开（3.5）→ 可保留历史（3.6）→ 全维可控（3.8）。

---

## 四、对源码研读 / debug 的启示

1. **按 model_type 分流**：
   - `qwen`（Qwen1）：无原生模块，须 `trust_remote_code` 加载官方 `configuration_qwen.py / modeling_qwen.py`；
   - `qwen2`（Qwen1.5/2/2.5 共用）：`models\qwen2\`；`qwen2_moe`（57B-A14B）、`qwen2_vl`（2-VL）、`qwen2_5_vl`（2.5-VL）、`qwen2_5_omni`（2.5-Omni）；
   - `qwen3` / `qwen3_moe` / `qwen3_vl` / `qwen3_vl_moe` / `qwen3_next`（MTP 投机头）；
   - `qwen3_5` / `qwen3_5_moe`（**Qwen3.5/3.6/3.8 三代共用**，`models\qwen3_5\` + `modular_qwen3_5.py`）。
2. **重点调试对象**：
   - Qwen1：`c_attn` 融合投影三等分、`get_ntk_alpha` 动态基频重算、logn 缓冲表；
   - Qwen2/2.5：`Qwen2Config.__post_init__` 的 `layer_types` 展开（use_sliding_window=false → sliding_window=None 的坑）、`repeat_kv` 分组复制；
   - Qwen3：`q_norm/k_norm`（逐头 RMSNorm，head_dim 128）与显式 head_dim 的投影维推算；
   - Qwen3.5/3.6/3.8：GDN 的 chunk（`torch_chunk_gated_delta_rule`）与 recurrent（`torch_recurrent_gated_delta_rule`）双路径、conv_state 与 recurrent_state 双缓存、GA 输出门（swish）、partial_rotary 0.25（只转前 64 维）、`layer_types` 由 `__post_init__` 预生成（非层循环判断）。
3. **同架构对照实验**：Qwen1.5/2/2.5 共用 `models\qwen2`（config 差异在 GQA/滑窗/尺寸）；Qwen3.5/3.6/3.8-27B config 逐字段一致——是「同架构不同训练/推理行为」的标准样本；Qwen3 vs Qwen3.5 是「全注意力 vs 混合注意力」路线对照。
4. **本地可跑性**：0.5B–7B 级可本地断点；27B（BF16 约 55.6GB / FP8 约 29GB）需较大显存；2.4T-A95B / 397B-A17B / 235B-A22B 本地不可跑，仅可断点其纯前向单层。

---

## 参考来源

1. `E:\transformers\bayon-note\Qwen\qwen1\index.md`（Qwen1：本地 Qwen-1_8B config + 官方 remote code + ModelScope config + arXiv:2309.16609）
2. `E:\transformers\bayon-note\Qwen\qwen1_5\index.md`（Qwen1.5：本地 0.5B config + transformers qwen2 源码 + qwen1.5 博客）
3. `E:\transformers\bayon-note\Qwen\qwen2\index.md`（Qwen2：本地 0.5B/2-VL-2B config + qwen2/qwen2_vl 源码 + arXiv:2407.10671 / 2409.12191）
4. `E:\transformers\bayon-note\Qwen\qwen2_5\index.md`（Qwen2.5：本地 0.5B/2.5-VL-3B config + qwen2_5_vl 源码 + arXiv:2412.15115 / 2502.13923 / 2409.12186）
5. `E:\transformers\bayon-note\Qwen\qwen3\index.md`（Qwen3：本地 0.6B/3-VL-2B config + qwen3/qwen3_moe/qwen3_vl 源码 + arXiv:2505.09388 / 2511.21631）
6. `E:\transformers\bayon-note\Qwen\qwen3_5\index.md`（Qwen3.5：本地 0.8B/35B-A3B config + qwen3_5/qwen3_5_moe 源码 + 官方博客）
7. `E:\transformers\bayon-note\Qwen\qwen3_6\index.md`（Qwen3.6：官方 27B/35B-A3B model card + qwen3_5 源码 + 27B 对比文件）
8. `E:\transformers\bayon-note\Qwen\qwen3_8\index.md`（Qwen3.8：官方 27B/2.4T-A95B/27B-FP8 model card + qwen3_5 源码 + 官方博客）
9. 已核实对比文件：`E:\transformers\bayon-note\Qwen\Qwen-27B架构对比-3.5-3.6-3.8.md`
10. 官方组织与仓库：https://huggingface.co/Qwen ；https://modelscope.cn/organization/qwen ；https://qwenlm.github.io/blog/qwen3/ ；https://qwen.ai/blog?id=qwen3.5 ；https://qwen.ai/blog?id=qwen3.8

> 标注约定：全部数字来自各版本目录 index.md 所引官方 config / 本地源码 / 官方论文与 model card；无法核实项（Qwen3.8-6B-Flash 全部规格、122B-A10B / 397B-A17B 精确 hidden/层数）按源文档标「待确认」不臆造。访问与核验日期：2026-10-26 至 2026-11-02。

---

*页脚：Qwen 全家族版本对比与演进 · 数据来源：bayon-note 各版本 index.md（config + transformers 源码 + 官方论文/model card）· 仅供技术学习与论文研究*
