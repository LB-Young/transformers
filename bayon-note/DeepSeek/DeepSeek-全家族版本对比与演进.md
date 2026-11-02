# DeepSeek 全家族版本对比与演进 — V1 → V2 → V3 → V3.1 → V3.2 → V4（附 R1）

> 整理日期：2026-11-02
> 数据来源：各版本目录 `index.md`（其事实全部源自远端官方 config.json（hf-mirror 拉取留档 `_tools\_cfgpull\`）、transformers 本地源码 `E:\transformers\src\transformers\models\`、官方论文与官方仓库，详见文末来源）
> 覆盖版本：deepseek_v1（DeepSeek-LLM 一代）、deepseek_v2、deepseek_v3、deepseek_v3_1（无独立文档，并入 V3 说明）、deepseek_v3_2（V3.2-Exp）、deepseek_v4（V4-Flash-Base）、deepseek_r1
> 用途：配合 `E:\transformers` 源码研读与断点 debug 的家族横向速查

---

## 结论速览

DeepSeek 家族的演进主线是一条「**低成本推理 + 稀疏化**」的连续路线：

1. **V1（一代 DeepSeek-LLM）是纯 Dense**：LLaMA 式主干（`model_type=llama`），67B 档首次引入 GQA，是后续 MLA 的前置探索。
2. **V2 一举引入两大核心**：MLA 多头潜在注意力（KV cache 压到 576 维/层/token）与 DeepSeekMoE 细粒度专家（160 选 6 + 2 共享），总参 236B / 激活 21B；负载均衡仍用辅助损失。
3. **V3 把 MoE 扩到 671B-T37B**（256 选 8 + 1 共享）、**路由去辅助损失**（sigmoid + bias 修正 + 分组限选 noaux_tc）、引入 **MTP** 多 token 预测头与 **FP8 e4m3** 混合精度训练、词表扩到 129280；MLA 维度与 V2 一致。
4. **V3.1 是同架构改版**（2025-08）：model_type 仍为 `deepseek_v3`，仅量化格式加 `scale_fmt=ue8m0`，不单独成代。
5. **V3.2-Exp（2025-09）架构上唯一的大改动在注意力**：新增 **DSA 稀疏注意力**（Lightning Indexer 先选 top-2048 token 再算注意力），FFN 侧完全继承 V3。
6. **V4-Flash-Base 是全面重构的小尺寸预览**：MQA 单 KV 头 + DSA(top-512) + 滑窗 128、**mHC 超连接**替换标准残差、前 3 层 **Hash MoE**、上下文直达 **1M**；完整尺寸 V4 主仓官方未发布（待确认）。
7. **R1 不是新架构**：与 V3 config 逐字段一致（唯一差异 `transformers_version`），差异全在训练——冷启动推理 SFT + 大规模 RL（GRPO 类、规则奖励）激发长链推理，并蒸馏出 R1-Distill 系列。

一句话：**V2 定"骨架"（MLA+MoE），V3 定"规模与无损失均衡"，V3.2/V4 做"注意力稀疏化与长上下文"，R1 做"训练范式"**。

---

## 一、全家族核心规格对比表

| 项目 | V1 一代（DeepSeek-LLM） | V2（236B-A21B） | V3（671B-T37B） | V3.1 | V3.2-Exp | V4-Flash-Base | R1（671B-T37B） |
|---|---|---|---|---|---|---|---|
| 官方论文/发布时间 | arXiv:2401.02954（2024-01） | arXiv:2405.04434（2024-05） | arXiv:2412.19437（2024-12） | 2025-08 | 2025-09 | 未公开（待确认） | arXiv:2501.12948（2025-01） |
| 入口类 / model_type | LlamaForCausalLM / `llama` | DeepseekV2ForCausalLM / `deepseek_v2` | DeepseekV3ForCausalLM / `deepseek_v3` | 同 V3 | DeepseekV32ForCausalLM / `deepseek_v32` | DeepseekV4ForCausalLM / `deepseek_v4` | DeepseekV3ForCausalLM / `deepseek_v3` |
| 形态 | **Dense**（7B / 67B） | **MoE** | **MoE** | MoE | MoE | **MoE** | **MoE** |
| 总参 / 激活 | 7B / 67B（全激活） | 236B / 21B | 671B / ≈37B | ≈V3 | ≈V3 | 预览版（未给总参） | 671B / ≈37B |
| hidden_size | 4096（7B）/ 8192（67B） | 5120 | 7168 | 7168 | 7168 | **4096** | 7168 |
| 层数 | 30（7B）/ 95（67B） | 60 | 61（前 3 层 Dense） | 61 | 61 | **43** | 61 |
| 注意力 | MHA（7B）/ **GQA**（67B：64Q/8KV） | **MLA** | **MLA** | MLA | **MLA + DSA（top-2048）** | **MQA（64Q/1KV, head_dim 512）+ DSA（top-512）+ 滑窗 128** | MLA |
| Q 低秩 | 无 | q_lora=1536 | q_lora=1536 | 同 | 同 | q_lora=o_lora=1024 | 同 V3 |
| KV cache（每层/token） | 7B: 32×128×2 / 67B: 8×128×2 | **576 维**（kv_lora 512 + rope 64） | 576 维 | 同 | 同 | 未单列（MQA 1 KV 头 + 稀疏） | 同 V3 |
| MoE 专家配置 | 无 | 160 路由选 6 + **2 共享**，moe_inter 1536 | **256 路由选 8 + 1 共享**，moe_inter 2048 | 同 V3 | 同 V3 | **256 路由选 6 + 1 共享，前 3 层 Hash MoE** | 同 V3 |
| 路由打分 | — | softmax + group_limited_greedy（8 组→3 组→6） | **sigmoid + noaux_tc**（分组 8→4） | 同 V3 | 同 V3 | sqrtsoftplus + noaux_tc（o_groups=8） | 同 V3 |
| 负载均衡 | — | **辅助损失** aux_loss_alpha=0.001 | **aux-loss-free**（bias 修正） | 同 V3 | 同 V3 | aux-loss-free | 同 V3 |
| routed_scaling / norm_topk | — | 16.0 / false | 2.5 / true | 同 | 同 | 1.5 / true | 同 V3 |
| 残差结构 | 标准 RMSNorm 前置残差 | 标准 | 标准 | 标准 | 标准 | **mHC 超连接**（hc_mult=4，Sinkhorn 20 迭代） | 标准 |
| 上下文 | 4096（无外推） | 163840（YaRN ×40） | 163840（YaRN ×40） | 同 | 同 | **1048576（1M）**（YaRN ×16 + HCA/CSA 压缩） | 163840 |
| 词表 | 102400 | 102400 | **129280** | 同 | 同 | 129280 | 同 V3 |
| 训练精度 | bf16 | bf16 | **FP8 e4m3** | FP8 + scale_fmt ue8m0 | FP8 + ue8m0 | FP8 e4m3（expert_dtype=fp8） | 同 V3（训练侧重 RL） |
| MTP 多 token 预测 | 无 | 无 | **×1** | ×1 | ×1 | ×1 | ×1 |
| tie_word_embeddings | false | false | false | false | false | false | false |
| 特殊机制 | — | — | aux-loss-free、MTP | — | **Lightning Indexer**（index 64 头 / dim 128 / topk 2048） | Hash MoE、HCA/CSA 压缩、compress_rope_theta=160000 | `<think>...</think>` 思维链标记 |

> 说明：V3.1 无独立文档（目录内仅 `_tools`），其与 V3 的差异记录在 `deepseek_v3_2\index.md` §7：model_type 仍为 `deepseek_v3`，config 逐字段一致，仅量化加 `scale_fmt=ue8m0` 与 transformers_version 不同。V4 主仓（deepseek-ai/DeepSeek-V4）2026-10-31 拉取 404，**待确认**；表中 V4 数字一律以 V4-Flash-Base 官方 config 为准。

---

## 二、代际演进详解

### 2.1 V1（一代）→ V2：从 Dense 到 MLA + DeepSeekMoE（骨架确立）

| 维度 | DeepSeek-LLM 67B（V1） | DeepSeek-V2 |
|---|---|---|
| 形态 | Dense（95 层全激活） | **MoE：1 dense + 59 MoE 层，160 选 6 + 2 共享** |
| 注意力 | GQA（64Q/8KV，8:1） | **MLA**：KV 潜在压缩 576 维，缓存 -93.3%（论文口径） |
| Q 路径 | 直接投影 | Q 低秩分解（q_lora=1536）+ 分量 RMSNorm |
| 路由 | 无 | group_limited_greedy（8 组→3 组→6）+ 16.0 缩放 |
| 上下文 | 4096 | 163840（YaRN factor=40） |
| 负载均衡 | 无 | 辅助损失 aux_loss_alpha=0.001 |

- V2 是家族**架构分水岭**：MLA 与 DeepSeekMoE 两个核心机制在此确立，后续 V3/V3.2/R1 全部沿用其框架。
- 训练成本较 67B 基线省 42.5%、最大生成吞吐 5.76×（论文口径）。

### 2.2 V2 → V3：规模翻倍 + 路由去辅助损失（工艺定型）

| 维度 | V2 | V3 |
|---|---|---|
| 规模 | 236B-A21B | **671B-T37B** |
| MoE | 160 选 6 + 2 共享，moe_inter 1536 | **256 选 8 + 1 共享**，moe_inter 2048 |
| 负载均衡 | 辅助损失 | **aux-loss-free**：sigmoid + e_score_correction_bias + 分组 noaux_tc |
| Dense 起始层 | first_k_dense_replace=1 | **=3**（前 3 层 Dense） |
| 词表 | 102400 | **129280** |
| 训练精度 | bf16 | **FP8 e4m3** 混合精度 |
| MTP | 无 | **×1** |
| 上下文 | 163840 | 163840（YaRN 同口径） |
| MLA 维度 | kv_lora 512 / q_lora 1536 / nope 128 / rope 64 | **完全一致**（MLA 不是 V3 新引入） |

- V3 相对 V2 的关键升级：① MoE 扩规模；② 路由去掉辅助损失（省掉训练配损失项）；③ MTP 投机预测头；④ FP8 压低 671B 训练成本；⑤ 扩词表、Dense 起始层 1→3。

### 2.3 V3 → V3.1 → V3.2-Exp：同架构改版 → 注意力稀疏化

- **V3.1（2025-08）**：model_type 仍是 `deepseek_v3`，config 与 V3 逐字段一致，仅量化格式加 `scale_fmt=ue8m0` + transformers_version 差异——**同架构改版，不单独成代**。
- **V3.2-Exp（2025-09）**：主干字段（hidden/层数/MLA/MoE/词表/上下文）与 V3 **完全一致**，唯一架构改动是注意力换 **DSA 稀疏注意力**：
  - 新增 `DeepseekV32Indexer`（64 头 / 每头 128 维），对每个查询从历史中打分选 top-2048 token，折叠成稀疏掩码后主 MLA 只在子集上算注意力；
  - 效果：注意力从 O(seq²) 降到每查询 O(2048)，长上下文推理显著省算力，同时保留召回关键上下文的能力；
  - model_type 从 `deepseek_v3` → `deepseek_v32`（新模块）。

### 2.4 V3.2 → V4-Flash-Base：全面重构（MQA + mHC + Hash MoE + 1M 上下文）

| 维度 | V3 / V3.2 | V4-Flash-Base |
|---|---|---|
| 尺寸 | 7168 / 61 层 | **4096 / 43 层（小尺寸预览）** |
| 注意力 | MLA / MLA+DSA(2048) | **MQA（64Q/1KV，head_dim 512）+ DSA(top-512) + 滑窗 128** |
| Q/O 低秩 | q_lora 1536（无 o_lora） | q_lora = o_lora = 1024 |
| 残差 | 标准 RMSNorm 残差 | **mHC 超连接**（hc_mult=4 通道扩展 + Sinkhorn 20 迭代门控） |
| MoE | 256 选 8 + 1 共享 | 256 选 6 + 1 共享；**前 3 层 Hash MoE**（免可学习路由器） |
| 打分 | sigmoid / noaux_tc | **sqrtsoftplus / noaux_tc**，routed_scaling 1.5，o_groups=8 |
| 上下文 | 163840 | **1,048,576（1M）**：YaRN ×16（orig 65536）+ 逐层 HCA/CSA 压缩（compress_ratios 交替 4/128）+ compress_rope_theta=160000 |
| 词表 | 129280 | 129280 |

- V4-Flash-Base 代表家族向**超长上下文 + 全链路稀疏化**的方向：注意力（MQA+DSA+滑窗）、残差（mHC）、路由（Hash+稀疏）三层都做了结构改动。
- **待确认**：完整尺寸 DeepSeek-V4 主仓官方未公开发布，V4 数字仅代表 Flash-Base 预览版。

### 2.5 R1：同 V3 底座，训练范式创新（附蒸馏外溢）

- **架构**：与 V3 config 逐字段一致（唯一顶层差异 `transformers_version`：R1=4.46.3，V3=4.33.1），入口类同为 `DeepseekV3ForCausalLM`。
- **训练管线**：
  ```
  DeepSeek-V3 : 通用预训练 + 通用对话 SFT/RLHF
  DeepSeek-R1 : 同一 V3 权重底座
                 ├─ 冷启动：少量推理样本监督微调（reasoning SFT，避免 RL 初期乱输出）
                 └─ 大规模 RL：基于规则的奖励（数学/代码准确率、格式、重复惩罚，GRPO 类）
                    → 激发长链推理、反思、验证、回溯行为
  ```
- **输出行为**：生成回答前先输出 `<think>...</think>` 包裹的思维链（tokenizer_config 定义，非新增架构模块）。
- **R1-Zero**：从 V3-Base 直接上 RL、不经冷启动 SFT 的对照实验，共用同一架构。
- **R1-Distill**：用 R1 思维链做监督微调蒸馏到 Qwen2.5（1.5B/7B/14B/32B）与 Llama3（8B/70B）Dense 底座——这些变体**不共享 V3 的 MLA/MoE**，架构随各自底座。

---

## 三、关键机制演进一图流

```
注意力：  MHA/GQA(V1) → MLA(V2) → MLA(V3/R1) → MLA+DSA(V3.2) → MQA+DSA+滑窗(V4)
MoE：     —(V1 dense) → 160选6+共享2+aux-loss(V2) → 256选8+共享1+aux-free(V3)
          → 同V3(V3.2) → 256选6+共享1+Hash前3层(V4)
残差：    标准 RMSNorm 残差(V1-V3.2) → mHC 超连接(V4)
上下文：  4096(V1) → 163840(V2/V3/V3.1/V3.2) → 1048576(V4, 1M)
词表：    102400(V1/V2) → 129280(V3/V3.1/V3.2/V4/R1)
训练精度：bf16(V1/V2) → FP8 e4m3(V3/V3.1/V3.2/V4/R1)
投机预测：无(V1/V2) → MTP×1(V3+)
```

演进主线解读：
1. **MLA 维度从 V2 起冻结**（kv_lora 512 / q_lora 1536 / nope 128 / rope 64 / v_head 128），V3/R1/V3.2 全部沿用，V4 才改为 MQA + 更小头维。
2. **MoE 沿「细粒度专家 + 共享专家 + 组间限选」框架放大**，路由策略从辅助损失（V2）演进到无辅助损失（V3+），V4 引入 Hash 路由做免学习粗筛。
3. **稀疏化是 V3.2/V4 的主题**：先稀疏注意力（DSA），再叠加稀疏残差（mHC）与稀疏路由（Hash MoE），目标直指 1M 上下文成本。
4. **R1 证明架构不变也能通过训练范式带来代际能力跃升**，并借蒸馏把推理能力下放给小模型。

---

## 四、对源码研读 / debug 的启示

1. **按 model_type 分流加载**：`llama`（V1，无原生 deepseek_v1 模块，trust_remote_code 不需要，直接走 `models\llama\`）；`deepseek_v2` / `deepseek_v3`（R1 也走它）/ `deepseek_v32` / `deepseek_v4` 各是独立模块——打断点的文件路径完全不同。
2. **重点调试对象**：
   - MLA 的压缩潜变量 KV cache（V2/V3：`kv_a_proj_with_mqa` → `expand_kv`，缓存 576 维）——与 MHA/GQA 差异最大的地方；
   - aux-loss-free 路由的 `e_score_correction_bias` 缓冲与分组限选（V3 `TopkRouter`）；
   - DSA 索引器的 top-k 稀疏掩码如何折进 attention mask（V3.2 `DeepseekV32Attention`）；
   - V4 的 mHC 超连接（Sinkhorn 归一化）与 HCA/CSA 压缩器的 `compress_ratios` 逐层生效逻辑。
3. **V3 与 R1 同构**：本地只有一份 `deepseek_v3` 代码可同时加载 V3 与 R1，config diff 只剩 transformers_version——适合做「同架构不同训练」的对照实验。
4. **本地可跑性**：671B 主模型本地双 16G 卡跑不了，R1-Distill（Qwen2.5/Llama3 底座）与 V4-Flash-Base 小尺寸适合本地断点调试。

---

## 参考来源

1. `E:\transformers\bayon-note\DeepSeek\deepseek_v1\index.md`（V1 事实：远端官方 config + arXiv:2401.02954）
2. `E:\transformers\bayon-note\DeepSeek\deepseek_v2\index.md`（V2 事实：远端官方 config + arXiv:2405.04434）
3. `E:\transformers\bayon-note\DeepSeek\deepseek_v3\index.md`（V3 事实：远端官方 config + arXiv:2412.19437）
4. `E:\transformers\bayon-note\DeepSeek\deepseek_v3_2\index.md`（V3.1 同构说明 + V3.2-Exp 事实：官方 config + 本地 `models\deepseek_v32\` 源码）
5. `E:\transformers\bayon-note\DeepSeek\deepseek_v4\index.md`（V4-Flash-Base 事实：官方 config + 本地 `models\deepseek_v4\` 源码）
6. `E:\transformers\bayon-note\DeepSeek\deepseek_r1\index.md`（R1 事实：官方 config + arXiv:2501.12948）
7. 官方仓库：https://huggingface.co/deepseek-ai （DeepSeek-V3 / V3.2-Exp / V4-Flash-Base / R1 / R1-Distill 系列）
8. 官方论文：https://arxiv.org/abs/2401.02954 ；https://arxiv.org/abs/2405.04434 ；https://arxiv.org/abs/2412.19437 ；https://arxiv.org/abs/2501.12948

> 标注约定：全部数字来自各版本目录 index.md 所引官方 config / 本地源码 / 官方论文；未核实项（如 V4 主仓规格）统一标「待确认」。访问与核验日期：2026-10-31（各版本文档），2026-11-02（本对比文档）。

---

*页脚：DeepSeek 全家族版本对比与演进 · 数据来源：bayon-note 各版本 index.md（官方 config + transformers 源码 + 官方论文）· 仅供技术学习与论文研究*
