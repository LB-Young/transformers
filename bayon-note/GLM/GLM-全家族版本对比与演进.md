# GLM 全家族版本对比与演进 — GLM-130B → ChatGLM-6B → ChatGLM2/3 → GLM-4 → GLM-4V → GLM-Z1 → GLM-4.5/4.6 → GLM-4.6V/4.7 → GLM-5/5.2 → GLM-5.3-Flash

> 整理日期：2026-11-02；2026-11-03 更新（补全 GLM-4.6V → 4.7 → 4.7-Flash → GLM-5 → 5.2 → 5.3-Flash 版本线）
> 数据来源：各版本目录 `index.md`（其事实全部源自远端官方 config（hf-mirror 拉取留档 `_tools\_cfgpull\`）、transformers 本地源码 `E:\transformers\src\transformers\models\glm*/`、官方论文与官方仓库，详见文末来源）；GLM-5 系事实以 arXiv:2602.15763 技术报告 + glm5.ai + docs.bigmodel.cn + z.ai/blog + NVIDIA NIM + 本地 glm5_next 源码交叉核验为准
> 覆盖版本：glm_130b、chatglm_6b、chatglm2_3（2 与 3 同构合册）、glm_4（9B + 32B-0414）、glm_4v（GLM-4V-9B）、glm_z1（Z1-9B/32B）、glm_4_5（4.5 / 4.5-Air / 4.6）、glm_4_6 系（4.6V / 4.7 / 4.7-Flash）、glm_5 系（GLM-5 / GLM-5.2 / GLM-5.3-Flash；本地源码模块 `glm5_next` 对应 GLM-5.3-Flash 架构，非 GLM-5 本体）
> 用途：配合 `E:\transformers` 源码研读与断点 debug 的家族横向速查

---

## 结论速览

GLM 家族的演进主线是一条「**骨干从双向走向因果、规模从稠密走向 MoE、能力从对话走向推理与多模态**」的路线：

1. **GLM-130B（2022-08）**：GLM 原始论文思想的最大落地——**双向注意力 + 自回归 blank infilling** 骨干，DeepNorm Post-LN 压 70 层深度，GeGLU（8/3·hidden）、icetk 词表；无公开 HF 权重。
2. **ChatGLM-6B（2023-03）**：压缩到 6B 的对话化改造——引入 **2D-RoPE**（位置 + span 双维度）、prefix-LM 双支路掩码、LayerNorm、130528 词表；INT4 可消费级部署。
3. **ChatGLM2（2023-06）→ ChatGLM3（2023-10）**：**架构换代**——去 2D-RoPE（回 1D）、LayerNorm→RMSNorm(Post-LN)、MHA→**MQA（32Q/2KV）**、FFN→SwiGLU(13696)、词表 65024、上下文 2K→32K；**两代 config 逐字段同构**，仅 `seq_length`（32768 vs 8192）不同，GLM3 的升级在 function call / Agent 等后训练。
4. **GLM-4（2024-01 9B / 2025-04 32B-0414）**：Dense 定型代——**GQA + partial rotary 0.5 + 4×RMSNorm 双残差 + 融合 gate_up 的 SiGLU**，词表扩到 151552，9B 推理扩 128K。
5. **GLM-4V-9B（2024）**：多模态扩展——在 GLM-4-9B 文本塔上挂 **63 层 / hidden 1792 / 1120px 的深视觉塔**，动态分辨率多 tile、8×8 空间合并、`<boi>/<eoi>` 注入，跨模态融合发生在共享文本解码器自注意力内。
6. **GLM-Z1（2025-04）**：**架构零改动**——Z1-9B/32B 与 GLM-4-0414 两个 Dense 版 config 逐项相同，差异全在长思维链（CoT）强化学习训练，与 GLM-4.5 的 MoE 路线并行。
7. **GLM-4.5（2025-07）→ 4.5-Air → 4.6**：**Dense→MoE 大切换**（`glm4_moe`）——160 路由专家 + 1 共享专家、top-8、**sigmoid 路由**（去 softmax）、QK-Norm、归一化从 4×RMSNorm 简化为 2×RMSNorm 前置、新增 **MTP**；4.6 与 4.5 同构仅上下文扩到 202752（200K）。
8. **GLM-4.6V（2025-12-08）**：4.5 系的视觉推理线——**视觉上下文 128K**（约 150 页文档 / 200 页 PPT / 1 小时视频）。
9. **GLM-4.7（2025-12-22）/ 4.7-Flash（2026-01-19）**：4.6 的推理与轻量双线——4.7 旗舰 358B（HF 标；glm-ai.chat 口径 355B 级/32B 激活）+ 200K，Preserved / Turn-level Thinking；4.7-Flash 压到 **30B-A3B** 轻量线，月下载 180 万+。
10. **GLM-5（2026-02-12）**：**架构换代**——**744B-A40B**（= 4.5 总参翻倍）、**78 层**、**MLA + DSA 稀疏注意力**（indexer top-k 2048，长序列算力降 1.5–2×）、**256 专家 top-8**、**MTP**（训练共享 3 层/部署 1 层，接受长度 2.76）、**slime 异步 RL** 框架、**Agentic Engineering**（10,000+ 可验证 SWE 环境）；28.5T token；词表 154880。
11. **GLM-5.2（2026-06-16）→ GLM-5.3-Flash（2026-08-25）**：5.2 把上下文推到 **1M**，**IndexShare**（每 4 层共享 indexer，per-token FLOPs 降 2.9×）+ KVShare，MTP 接受长度 +20%；5.3-Flash 转 **320B-A18B 轻量原生多模态线**——**34 KDA 线性 + 11 DSA 每 4 层交错**、**mHC 流形约束超连接**、**IndexPool**、**288 路由专家 top-8**、vision encoder + **30T 多模态语料**，是 GLM-5 系列首个原生多模态模型。

一句话：**GLM-130B 立"双向"根基，ChatGLM-6B 缩体量开对话，ChatGLM2/3 换因果+MQA，GLM-4 定 GQA+partial rotary 血统，GLM-4.5 切 MoE 并引入 QK-Norm/MTP，Z1 走"同结构+推理训练"并行线；GLM-5 换 MLA+DSA 稀疏注意力并上 Agentic Engineering，5.2 推 1M+IndexShare，5.3-Flash 走 KDA 线性+DSA 混合+mHC+原生多模态轻量线**。

---

## 一、全家族核心规格对比表

| 项目 | GLM-130B | ChatGLM-6B | ChatGLM2-6B | ChatGLM3-6B | GLM-4-9B | GLM-4-32B-0414 | GLM-4V-9B | GLM-Z1（9B/32B） | GLM-4.5 | GLM-4.5-Air | GLM-4.6 | GLM-4.6V | GLM-4.7 | GLM-4.7-Flash | GLM-5 | GLM-5.2 | GLM-5.3-Flash |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 时间 | 2022-08 | 2023-03 | 2023-06 | 2023-10 | 2024-01 | 2025-04 | 2024 | 2025-04 | 2025-07 | 2025-07 | 2025-09-30 | 2025-12-08 | 2025-12-22 | 2026-01-19 | 2026-02-12 | 2026-06-16 | 2026-08-25(HF)/08-26(API) |
| 入口类 / model_type | —（无 HF 权重） | ChatGLMModel / chatglm | ChatGLMModel / chatglm | 同左 | ChatGLMModel→Glm4ForCausalLM / glm4 | Glm4ForCausalLM / glm4 | ChatGLMForConditionalGeneration / chatglm | Glm4ForCausalLM / glm4 | **Glm4MoeForCausalLM / glm4_moe** | 同左 | 同左 | glm4_moe 系 + 视觉塔（细节未公开，待核实） | glm4_moe 系（待核实） | glm4_moe 系（待核实） | —（本地无 GLM-5 本体模块；技术报告 arXiv:2602.15763） | 同 GLM-5（待核实） | **glm5_next / Glm5NextForConditionalGeneration**（本地源码，对应 5.3-Flash 架构） |
| 形态 | **Dense 双向** | Dense | Dense | Dense | **Dense** | Dense | Dense + ViT | Dense（推理训练） | **MoE 355B-A32B** | **MoE 106B-A12B** | MoE 同 4.5（HF 标 357B） | MoE 同 4.5 + 视觉推理 | MoE 旗舰（HF 标 358B；glm-ai.chat 口径 355B 级/32B 激活） | **MoE 30B-A3B**（约 31B 总参） | **MoE 744B-A40B**（glm5.ai 口径 745B/44B，口径差异见 2.8） | 同 GLM-5（744B/40B） | **MoE 320B-A18B + 原生多模态**（vision encoder） |
| hidden / 层数 | 12288 / 70 | 4096 / 28 | 4096 / 28 | 4096 / 28 | 4096 / 40 | 6144 / 61 | 4096 / 40（文本塔） | 4096·40 / 6144·61 | 5120 / 92 | 4096 / 46 | 5120 / 92 | 同 4.6（5120/92，待核实） | —（未公开/待核实） | —（未公开） | hidden **6144** / **78 层**（Table 10：3 dense+75 MoE；报告文字口径 80，注释） | 同 GLM-5（待核实） | hidden **4096** / **45 层**（3 dense+42 MoE） |
| Q / KV 头 | 96 / 96（MHA） | 32 / 32（MHA） | 32 / 2（**MQA**） | 同左 | 32 / 2（GQA） | 48 / 2（GQA） | 32 / 2（文本塔） | 同 GLM-4-0414 | 96 / 8（GQA） | 96 / 8 | 96 / 8 | 同 4.6（待核实） | —（待核实） | —（未公开） | **MLA 64 头**（Q LoRA 2048 / KV LoRA 512；QK dim 192 / V dim 256） | 同 GLM-5（待核实） | 64 头 = 64 KV（强制相等）；MLA q_lora 1536 / kv_lora 512 |
| head_dim | 128 | 128 | kv_channels 128 | 128 | 128 | 128 | 128 | 128 | 128 | 128 | 128 | 同 4.6 | — | — | MLA latent（QK 192 / V 256） | 同左 | v_head_dim 256 / qk_nope 256 |
| 注意力骨干 | 双向 GLM blank infilling | prefix-LM 双支路 | 因果 1D | 因果 1D | 因果 GQA | 因果 GQA | 同 GLM-4 + 视觉 token | 同 GLM-4 | 因果 GQA + **QK-Norm** | 同 4.5（无 QK-Norm） | 同 4.5 | 同 4.5/4.6（+视觉 token） | 同 4.5 系（待核实） | —（待核实） | **MLA + DSA 稀疏注意力**（indexer 32 头×128，top-k 2048；长序列算力降 1.5–2×） | 同 GLM-5 + **IndexShare**（每 4 层共享 indexer） | **MLA + KDA 线性（34 层）+ DSA（11 层，每 4 层交错）混合** |
| 位置编码 | RoPE 1D | **2D-RoPE** | 1D-RoPE（original） | 同左 | **partial rotary 0.5**（θ=1e4） | 同左 | 同 GLM-4 | 同 GLM-4 | partial 0.5 + **θ=1e6** | 同 4.5 | 同 4.5 | 同 4.6 | 同 4.5 系 | — | —（未公开/待核实） | 同 GLM-5 | **文本 NoPE**（qk_rope_head_dim=0 强制）；视觉 axial 2D RoPE |
| 归一化 | **DeepNorm Post-LN** | LayerNorm | RMSNorm Post-LN | 同左 | **4×RMSNorm（pre+post）** | 同左 | 同 GLM-4 | 同 GLM-4 | **2×RMSNorm 前置 + QK-Norm** | 同 4.5 | 同 4.5 | 同 4.6 | 同 4.5 系 | — | —（未公开/待核实） | 同 GLM-5 | **mHC 流形约束超连接**（hc_mult=4 流，非标准 pre-norm） |
| FFN | GeGLU 32768（8/3·d） | inner 16384（4·d） | SwiGLU 13696 | 13696 | SiGLU 13696（gate_up 融合） | 23040 | 13696（文本塔） | 13696 / 23040 | dense 12288 + **专家 moe_inter 1536** | dense 10944 + 1408 | 同 4.5 | 同 4.6 | — | — | MoE intermediate 2048 | 同 GLM-5 | dense inter 12288（前 3 层）+ MoE moe_inter 2048 |
| MoE 专家 | 无 | 无 | 无 | 无 | 无 | 无 | 无 | 无 | **160 路由 + 1 共享，top-8**，first_k_dense 3 | **128 + 1，top-8**，first_k_dense 1 | 同 4.5 | 同 4.5 | —（待核实） | —（未公开） | **256 路由 + 1 共享，top-8** | 同 GLM-5 | **288 路由 + 1 共享，top-8** |
| 路由策略 | — | — | — | — | — | — | — | — | **sigmoid + norm_topk + scaling 2.5** | sigmoid + scaling 1.0 | 同 4.5 | 同 4.5 | —（待核实） | — | —（报告未展开） | 同 GLM-5 | **sigmoid + norm_topk + scaling 2.5**（同 4.5 血统） |
| 上下文 | 2048 | 2048 | **32768**（config） | 8192（config，官方 32K 口径） | 8192 基座 / 推理 128K | 32768 | 8192（文本塔） | 32768 | 131072 | 131072 | **202752** | 200K 文本 + **视觉上下文 128K**（约 150 页文档/200 页 PPT/1 小时视频） | **200K** | **200K** | **202,752**（SFT/DSA 适配；glm5.ai 约 202,000） | **1M** | **1M（1,048,576）**；最大输出 128K |
| 词表 | icetk 150000 | 130528 | padded 65024 | 65024 | **151552** | 151552 | 151552 | 151552 | 151552 | 151552 | 151552 | 151552（待核实） | 151552（待核实） | —（待核实） | **154880** | 同 GLM-5 | **154880** |
| 训练 | ≈400B token | ≈1T token 中英 | 后训练 + MQA 提速 | 42 基准登顶 / Agent | 预训练 + SFT | 同 | 多模态对齐 | **长思维链 RL** | 预训练 + MoE | 同 | 同 4.5 | 视觉推理对齐 | 推理优化线（Preserved / Turn-level Thinking）；SWE-bench 73.8%、Multilingual 66.7%、Terminal-Bench 2.0 41.0% | 轻量线（月下载 180 万+） | **28.5T token**；**slime 异步 RL**；**Agentic Engineering**（10,000+ SWE 环境） | 同 GLM-5；**slime 并行 OPD 约两天合并十余专家模型**；MTP 接受长度 +20% | **30T 多模态语料**；相比 GLM-5.3 注意力算力/KV cache 降 3.01×/4.44× |
| MTP / 特殊机制 | MIP 多任务指令 | [MASK]/[gMASK] | — | function call（模板层） | — | — | boi=151339/eoi=151340 | CoT 标记（待确认） | **MTP ×1** | MTP ×1 | MTP ×1 | MTP×1（待核实） | Preserved/Turn-level Thinking | —（待核实） | **MTP**：训练共享 3 层/部署 1 层（接受长度 2.76 vs DSV3.2 2.55） | **MTP 接受长度 +20%**（消融 4.56→5.47）+ IndexShare/KVShare | 1 个 MTP 层（checkpoint）；**transformers glm5_next 未实现 MTP**（权重显式忽略）；IndexPool（索引器 4 缓存向量→1） |
| transformers 模块 | 无 | 无（trust_remote_code） | 无 | 无 | 原生 `glm4` | 原生 `glm4` | 无（远程；原生 glm4v 是 4.1V/4.5V） | 原生 `glm4` | 原生 `glm4_moe` | 同左 | 同左 | glm4_moe / glm4v（待核实） | glm4_moe 系（待核实） | —（待核实） | —（本地无 GLM-5 本体模块） | 同左 | **glm5_next**（ForConditionalGeneration 多模态入口；SDPA only 不支持 FA） |
| 主要论文/来源 | arXiv:2210.02414 | arXiv:2406.12793 / 2103.10360 | arXiv:2406.12793 | 同左 | zai-org config / card | zai-org config | zai-org config | zai-org config | zai-org config / card | 同左 | 同左 | docs.bigmodel.cn new-releases / zai-org/GLM-4.6V | HF zai-org/GLM-4.7 / new-releases | new-releases | arXiv:2602.15763 / glm5.ai / github zai-org/GLM-5 | z.ai/blog/glm-5.2 / zai-org/GLM-5.2 | 智谱文档 glm-5.3-flash / NVIDIA NIM / 本地 glm5_next 源码 |

> 说明：① GLM-4V-9B 的视觉塔（63 层 / hidden 1792 / image 1120 / patch 14 / 16 头 / inter 15360 / GELU / scaling_factor 8 → 100 token/tile）单独存在，不属于文本主干；transformers 原生 `models\glm4v\` 对应的是后续 GLM-4.1V/4.5V（24 层 / 1536 / 336 / merge 2），**勿混淆**。② ChatGLM2/3 同构合册（唯一 config 差异 `seq_length`）；Z1 与 GLM-4-0414 config 逐项相同。③ GLM-5 主口径 744B/40B/78 层/202,752（技术报告 Table 10 + docs.bigmodel.cn + 智谱研究页一致）；glm5.ai 官网营销口径 745B/44B/5.9% 稀疏度/80 层为口径差异，仅作注释引述。④ 本地 transformers 的 `glm5_next` 模块对应 **GLM-5.3-Flash 架构**（320B/18B、45 层、288 专家、4096 hidden、文本 NoPE），**不是 GLM-5 本体（744B/40B、78 层、256 专家、6144 hidden）的实现**；GLM-5/5.2 本体在本地无对应模块。⑤ 4.6V/4.7/4.7-Flash 的 config 细节（层数、专家数、词表）未从公开 config 逐字段核验，表中以「待核实」标注。

---

## 二、代际演进详解

### 2.1 GLM-130B → ChatGLM-6B：双向骨干 → 对话化 2D-RoPE

| 维度 | GLM-130B | ChatGLM-6B |
|---|---|---|
| 规模 | 130B | 6.2B（INT4 可消费级部署） |
| 骨干 | 双向 GLM blank infilling（span 重建） | prefix-LM：上下文双向 + 生成支路因果 |
| 位置编码 | RoPE 1D | **2D-RoPE**（序列位置 + span 内相对位置双旋转） |
| 归一化 | DeepNorm Post-LN（α=(2N)^(1/4)） | LayerNorm |
| FFN | GeGLU 32768 | inner 16384 |
| 词表 | icetk 150000 | ice tokenizer 130528 |
| 稳定化 | DeepNorm + Embedding Gradient Shrink | — |

- ChatGLM-6B 是承上启下的一代：继承 GLM 双向/prefix 思想与 mask 标记（[MASK]/[gMASK]），但把规模压到可部署、位置编码升级为 2D。

### 2.2 ChatGLM-6B → ChatGLM2/3：架构换代（MQA + RMSNorm + SwiGLU + 1D-RoPE）

| 维度 | ChatGLM-6B | ChatGLM2-6B / ChatGLM3-6B |
|---|---|---|
| 注意力 | MHA 32 头 | **MQA 32Q/2KV**（KV cache 1/16，综述口径提速 ≈42%） |
| 位置编码 | 2D-RoPE | **1D-RoPE（original_rope=true）** |
| 归一化 | LayerNorm | **RMSNorm（post_layer_norm）** |
| FFN | inner 16384 | **SwiGLU 13696**（≈10/3·hidden） |
| 上下文（config） | 2048 | **32768（GLM2）/ 8192（GLM3）** |
| 词表 | 130528 | **padded 65024** |
| 其他 | — | add_qkv_bias=true、softmax fp32、tie=false |

- **ChatGLM2 与 ChatGLM3 架构超参逐字段完全一致**，唯一 config 差异是 `seq_length`（32768 vs 8192）；GLM3 的差异在数据与后训练（function call、代码解释器、Agent 任务）。
- 这是 GLM 家族唯一一次"自我否定式"大改：**丢弃 2D-RoPE、换掉 LayerNorm 与 MHA**，从此确立 1D-RoPE + MQA + RMSNorm 血统。

### 2.3 ChatGLM3 → GLM-4：Dense 定型（GQA + partial rotary + 4×RMSNorm）

| 维度 | ChatGLM3-6B | GLM-4-9B | GLM-4-32B-0414 |
|---|---|---|---|
| 规模 | 6B | ≈9B | ≈32B |
| 注意力 | MQA 32Q/2KV | **GQA 32Q/2KV（16:1）** | GQA 48Q/2KV（24:1） |
| 位置编码 | 1D-RoPE | **partial rotary 0.5**（128 维只转前 64 维） | 同左 |
| 归一化 | RMSNorm Post-LN | **4×RMSNorm「pre+post」双残差**（每层 4 个） | 同左 |
| FFN | SwiGLU 13696 | **融合 gate_up 的 SiGLU**（单投影 chunk 拆 gate/up） | 23040 |
| 词表 | 65024 | **151552** | 151552 |
| 上下文 | 8192 | 8192 基座 / **推理扩 128K** | 32768 |
| QKV bias | true | **true** | **false** |

- GLM-4 的主线：① 把 ChatGLM 系 4×RMSNorm 双残差、partial rotary、GQA 定型为 Dense 两代；② 词表固定 151552；③ 32B-0414 去掉 QKV bias（更大的模型更不需要 bias）。

### 2.4 GLM-4 → GLM-4V-9B：多模态扩展（深视觉塔）

| 维度 | GLM-4-9B（纯文本） | GLM-4V-9B |
|---|---|---|
| 文本塔 | 40 层 / 4096 / GQA 32·2 | **逐字段复用 GLM-4-9B** |
| 视觉塔 | — | ViT **63 层 / hidden 1792 / patch 14 / image 1120 / 16 头 / inter 15360 / GELU** |
| 位置槽 | — | num_positions=6401（80×80 patch + 1） |
| 空间合并 | — | scaling_factor=8（8×8 → **100 token/tile**） |
| 注入 | — | `<boi>=151339 / <eoi>=151340` 包裹视觉 token 进文本序列 |
| 融合 | — | 共享文本解码器自注意力内全连接融合（无独立融合层） |

- GLM-4V-9B 开创 GLM 视觉线；后续 GLM-4.1V/4.5V（原生 `glm4v` 模块）改为**更浅更高效**视觉塔（24 层 / 1536 / 336 / merge 2，支持视频），两代视觉塔不通用。

### 2.5 GLM-4 → GLM-Z1：同架构推理线（零改动）

- Z1-9B / Z1-32B 的 config 与 GLM-4-9B-0414 / GLM-4-32B-0414 **逐项相同**（含 attention_bias true/false、rope_theta 1e4、max 32768）。
- 差异全在训练：在 GLM-4 Dense 权重上用**长思维链（CoT）强化学习**（reasoning RL），使模型先长段自推理再给答案；不新增任何网络算子。
- 定位：与 GLM-4.5 的 MoE 结构升级**并行**的能力扩展线——Z1 验证"test-time compute / 同结构+推理训练"，4.5 验证"结构稀疏化"。

### 2.6 GLM-4 → GLM-4.5：Dense → MoE 大切换（+ QK-Norm + MTP）

| 维度 | GLM-4（Dense） | GLM-4.5 | GLM-4.5-Air | GLM-4.6 |
|---|---|---|---|---|
| 形态 | Dense | **MoE：160 路由 + 1 共享，top-8（≈355B-A32B）** | MoE：128 + 1，top-8（≈106B-A12B） | 同 4.5 |
| hidden / 层 | 4096·40 / 6144·61 | 5120 / 92 | 4096 / 46 | 5120 / 92 |
| 注意力 | GQA + partial rotary | GQA 96Q/8KV + **QK-Norm** + θ=1e6 | 同 4.5（**无 QK-Norm**，θ=1e6） | 同 4.5 |
| 归一化 | 4×RMSNorm（pre+post） | **2×RMSNorm 前置** | 同 4.5 | 同 4.5 |
| 路由 | 无 | **sigmoid + norm_topk_prob + scaling 2.5** | sigmoid + scaling 1.0 | 同 4.5 |
| 专家参数 | — | moe_inter 1536 / first_k_dense 3 | moe_inter 1408 / first_k_dense 1 | 同 4.5 |
| MTP | 无 | **num_nextn_predict_layers=1** | 1 | 1 |
| 上下文 | 8192–32768 | 131072 | 131072 | **202752** |

- GLM-4.5 是家族首个 MoE：sigmoid 路由（非 softmax）+ 共享专家 + top-8；归一化从 4×RMSNorm 双残差简化为 2×RMSNorm 前置；QK-Norm 稳定长上下文；MTP 提升训练密度与投机解码。
- GLM-4.6 与 4.5 config 逐项相同，唯一差异 `max_position_embeddings=202752`（上下文再扩）。

### 2.7 GLM-4.6 → 4.6V → 4.7 → 4.7-Flash：4.5 系的上下文扩展与视觉/推理优化线

| 维度 | GLM-4.6 | GLM-4.6V | GLM-4.7 | GLM-4.7-Flash |
|---|---|---|---|---|
| 时间 | 2025-09-30 | 2025-12-08 | 2025-12-22 | 2026-01-19 |
| 形态 | MoE 同 4.5（HF 标 357B） | 同 4.6 + 视觉推理 | MoE 旗舰（HF 标 358B；glm-ai.chat 口径 355B 级/32B 激活） | **MoE 30B-A3B**（约 31B 总参） |
| 上下文 | 200K（202752） | 200K 文本 + **视觉上下文 128K** | 200K | 200K |
| 定位 | 4.5 同构、上下文再扩 | 视觉推理：约 150 页文档 / 200 页 PPT / 1 小时视频 | 推理优化：**Preserved Thinking / Turn-level Thinking**；SWE-bench 73.8%、Multilingual 66.7%、Terminal-Bench 2.0 41.0% | 轻量线：月下载 180 万+ |
| 开源 | MIT zai-org/GLM-4.6 | MIT zai-org/GLM-4.6V | MIT zai-org/GLM-4.7（+FP8） | MIT |

- 4.6V / 4.7 / 4.7-Flash 的 config 细节（层数、专家数、词表）本次未从公开 config 逐字段核验，表中以「待核实」标注；架构均沿 GLM-4.5 系（glm4_moe）血统。
- 这条线是 GLM-5 之前的「过渡优化」：4.6 扩上下文、4.6V 扩视觉、4.7 扩推理（SWE/Agent）、4.7-Flash 做端侧轻量，与 GLM-5 的架构换代并行。

> 来源：docs.bigmodel.cn/cn/update/new-releases、huggingface.co/zai-org/GLM-4.7、glm-ai.chat/docs/is-glm-open-source/（访问 2026-11-03）

### 2.8 GLM-4.7 → GLM-5：架构换代（MLA + DSA + MTP + slime 异步 RL + Agentic Engineering）

| 维度 | GLM-4.7 系（4.5 血统） | GLM-5（arXiv:2602.15763） |
|---|---|---|
| 时间 | 2025-12-22 | 2026-02-12（API 上线） |
| 规模 | HF 标 358B / 32B 激活级 | **744B 总参 / 40B 激活**（= 4.5 的 355B 翻倍；含 MTP 层参数、不含 embedding 与输出层） |
| hidden / 层数 | 同 4.5 系（待核实） | hidden **6144** / **78 层**（Table 10：3 dense + 75 MoE） |
| 注意力 | GQA 96/8 + QK-Norm | **MLA**（Q LoRA 2048 / KV LoRA 512 / 64 头 / QK dim 192 / V dim 256）+ **DSA 稀疏注意力**（indexer 32 头 × 128，top-k 2048；长序列算力降 1.5–2×；「90% 长上下文注意力条目冗余」） |
| MoE | 160 路由 + 1 共享（4.5 血统，待核实） | **256 路由 + 1 共享，每 token 路由 8 个**；MoE intermediate 2048 |
| 上下文 | 200K | mid-training 32K(1T)→128K(500B)→200K(50B)；DSA 适配与 SFT **202,752 tokens** |
| 词表 | 151552（待核实） | **154880** |
| MTP | —（待核实） | **MTP：训练参数共享 3 层 / 部署 1 层**；接受长度 **2.76**（vs DeepSeek-V3.2 的 2.55，同 4 步投机） |
| 训练预算 | — | **28.5T token**：通用 18T + 代码&推理 9T（=27T 预训练）→ 长代码&推理 1T（32K）→ 长上下文&Agent 500B/50B（128K/200K）→ DSA 适配 20B（200K） |
| 后训练 | Preserved / Turn-level Thinking | **slime 异步 RL 框架**：rollout/训练全异步解耦、Multi-Task Rollout Orchestrator 1k+ 并发 rollout、TITO Gateway、Direct Double-sided Importance Sampling、DP-aware routing；Post-training 流水 Overall SFT → Reasoning RL → Agentic RL → General RL + On-Policy Cross-Stage Distillation |
| Agentic | SWE-bench 73.8% | **Agentic Engineering**：10,000+ 可验证 SWE 环境（9 种语言）、Harbor terminal（Docker 准确率 >90%）、基于 200 万+ 网页的 Web Knowledge Graph |
| 评测亮点 | HLE 24.8、AA 指数 42 | HLE **30.5**（w/ Tools **50.4**，超 Claude Opus 4.5 / Gemini 3 Pro）；HMMT Feb/Nov 2025 **97.9 / 96.9**；SWE-bench Verified **77.8**（开源第一）；LongBench v2 **64.5**（开源最高）；BrowseComp **75.9**；**AA Intelligence Index v4.0 = 50（首个开源权重模型达 50）** |

- **口径差异（必须并列标注）**：主口径 744B/40B/78 层/202,752 来自技术报告 Table 10 + docs.bigmodel.cn + 智谱研究页三方一致；**glm5.ai 官网营销口径为 745B / 44B / 5.9% 稀疏度 / 80 层**（报告 §2.1 文字与智谱研究页文字亦称 80 层）；报告未给 5.9% 稀疏度数字（仅 k=2048、算力降 1.5–2×）。
- **多模态边界**：GLM-5 技术报告无多模态章节、文本 API 输入=文本；glm5.ai「统一处理图像/文档/文本」为营销表述；GLM-5 系列首个原生多模态是 5.3-Flash（见 2.10）。
- 部署侧：INT4 QAT、FP8 rollout + PD 分离；首日适配昇腾/摩尔线程/海光/寒武纪/昆仑芯/沐曦/燧原 7 国产 GPU 平台；昇腾推理优化 Lightning Indexer / Sparse Flash Attention / MLAPO。

> 来源：arXiv:2602.15763（§2.1 / §2.1.1 / §2.3 / §3.2 / §3.6 / §4 / Table 2/7/8/9/10）、glm5.ai、docs.bigmodel.cn/cn/guide/models/text/glm-5、zhipuai.cn/zh/research/153、github.com/zai-org/GLM-5（访问 2026-11-03）

### 2.9 GLM-5 → GLM-5.2：1M 上下文 + IndexShare/KVShare + MTP +20%

| 维度 | GLM-5 | GLM-5.2 |
|---|---|---|
| 时间 | 2026-02-12 | 2026-06-16 |
| 上下文 | 202,752（200K） | **1M-token context**（由 200K 扩展） |
| 稀疏注意力 | DSA：每层独立 indexer，top-k 2048 | **IndexShare：同一份 indexer 跨每 4 层稀疏注意力复用**——1M 上下文下 per-token FLOPs 降 **2.9×**（+ KVShare） |
| MTP | 接受长度 2.76（训练共享 3 层/部署 1 层） | **MTP 接受长度 +20%**；消融链：base 4.56 → +IndexShare+KVShare 5.10 → +Rejection Sampling 5.29 → +E2E TV Loss 5.47 |
| 训练合并 | slime 异步 RL | **slime 并行 OPD：约两天合并十余专家模型**；「Pure Open」MIT 无区域限制 |
| 其余规格 | 744B/40B、256 专家、154880 | 同 GLM-5（744B/40B；MIT zai-org/GLM-5.2 + FP8） |

- GLM-5.2 是「同架构扩上下文 + 稀疏索引共享」的效率优化：IndexShare 把每 4 层一个 indexer 的成本摊薄，KVShare 进一步压 KV cache，换来 1M 上下文下的吞吐与 MTP 接受率双升。

> 来源：z.ai/blog/glm-5.2、docs.bigmodel.cn/cn/update/new-releases（访问 2026-11-03）

### 2.10 GLM-5.2 → GLM-5.3-Flash：320B-A18B 轻量旗舰（KDA 线性+DSA 混合 + mHC + 原生多模态）

| 维度 | GLM-5.2 | GLM-5.3-Flash |
|---|---|---|
| 时间 | 2026-06-16 | 2026-08-25（HF）/ 08-26（API） |
| 规模 | 744B-A40B | **320B 总参 / 18B 激活** |
| hidden / 层数 | 6144 / 78（3 dense+75 MoE） | hidden **4096** / **45 层**（3 dense + 42 MoE） |
| 注意力 | MLA + DSA（indexer top-k 2048） | **MLA + KDA 线性注意力（34 层）+ DSA（11 层）混合，每 4 层交错**；indexer top-k **2048** + index_kpool **16**；相比 GLM-5.3 注意力算力 / KV cache 分别降 **3.01× / 4.44×** |
| 超连接/归一化 | 标准 pre-norm（待核实） | **mHC 流形约束超连接（hc_mult=4 流）**——非标准 pre-norm，残差多流混合 |
| 索引压缩 | IndexShare（每 4 层共享 indexer） | **IndexPool：索引器 4 个缓存向量压缩为 1 个向量** |
| MoE 专家 | 256 路由 + 1 共享，top-8 | **288 路由 + 1 共享，top-8**；sigmoid + norm_topk + scaling 2.5 |
| 上下文 | 1M | **1M（1,048,576）**；最大输出 128K |
| 词表 | 154880 | 154880 |
| 模态 | 文本（报告无多模态章节） | **原生多模态（GLM-5 系列首个）**：输入视频/图像/文本/文件；vision encoder 24 层 / hidden 1024 / 16 头 / 336px / patch 14 / spatial_merge 2 / axial 2D RoPE；≈144 token/图 |
| 训练 | —（GLM-5 为 28.5T） | **30T-token 多模态语料** |
| MTP | 接受长度 +20% | 1 个 MTP 层（checkpoint 口径）；**transformers glm5_next 未实现 MTP**（MTP/shared_head 权重显式忽略） |
| 入口 | —（本地无 GLM-5/5.2 本体模块） | **glm5_next / Glm5NextForConditionalGeneration**（本地源码 `models\glm5_next\`，对应 5.3-Flash 架构，非 GLM-5 本体） |
| 评测 | — | AA 智能指数 **57**（与 Claude Opus 4.8 持平） |

- 5.3-Flash 是 GLM-5 家族的「轻量原生多模态」分支：从 744B 砍到 320B、激活 18B，用 KDA 线性注意力（34 层）承担大部分层、DSA 稀疏注意力每 4 层插一层（11 层），叠加 mHC 超连接与 IndexPool 索引压缩，在 1M 上下文下把算力和 KV cache 同时压到 1/3–1/4。
- **注意**：本地 transformers 的 `glm5_next` 模块源码全部对应此 5.3-Flash 架构（45 层 / 288 专家 / 4096 hidden / 文本 NoPE），**不是 GLM-5 本体（78 层 / 256 专家 / 6144 hidden）的实现**。

> 来源：docs.bigmodel.cn/cn/guide/models/vlm/glm-5.3-flash、docs.api.nvidia.com/nim/reference/z-ai-glm-5-3-flash、本地 `E:\transformers\src\transformers\models\glm5_next\configuration_glm5_next.py / modeling_glm5_next.py`（访问 2026-11-03）

---

## 三、关键机制演进一图流

```
骨干：      双向 GLM blank infilling(130B) → prefix-LM 2D-RoPE(6B) → 因果 1D-RoPE+MQA(2/3)
            → 因果 GQA+partial rotary(4/4V/Z1) → 因果 GQA+QK-Norm(4.5/4.6)
KV 头：     96/96 MHA(130B) → 32/32 MHA(6B) → 32/2 MQA(2/3) → 32·2 / 48·2 GQA(4/Z1)
            → 96/8 GQA(4.5/4.6)
归一化：    DeepNorm Post-LN(130B) → LayerNorm(6B) → RMSNorm Post-LN(2/3)
            → 4×RMSNorm pre+post(4/4V/Z1) → 2×RMSNorm 前置 + QK-Norm(4.5/4.6)
FFN：       GeGLU 32768(130B) → inner 16384(6B) → SwiGLU 13696(2/3) → SiGLU gate_up(4/Z1)
            → dense 12288 + MoE 专家(4.5)
容量：      Dense 130B → Dense 6B → Dense 6B → Dense 9B/32B → **MoE 355B-A32B**(4.5)
上下文：    2048(130B/6B) → 32768(2) / 8192(3) → 8192~128K(4) → 131072(4.5) → 202752(4.6)
词表：      icetk 150000(130B) → 130528(6B) → 65024(2/3) → 151552(4+)
模态：      文本(130B~4/Z1/4.7) → 视觉推理(4.6V：128K 视觉上下文) → 原生多模态(5.3-Flash：vision encoder + 30T 多模态语料)
训练：      400B 预训练(130B) → 1T 中英(6B) → SFT/对齐(2/3) → 长思维链 RL(Z1)
稀疏化：    稠密 GQA(4.5系) → DSA 稀疏注意力(GLM-5：MLA base，indexer top-k 2048)
            → IndexShare 每4层共享 indexer(5.2，1M 下 per-token FLOPs 2.9×↓)
            → KDA 线性(34)+DSA(11) 每4层交错 + IndexPool(5.3-Flash，算力/KV 3.01×/4.44×↓)
超连接：    2×RMSNorm 前置(4.5系) → mHC 流形约束超连接 hc_mult=4 流(5.3-Flash)
位置编码：  partial rotary 0.5(4.5系) → GLM-5 —(待核实) → 文本 NoPE + 视觉 axial 2D RoPE(5.3-Flash)
MTP：       MTP×1(4.5系) → 训练共享3层/部署1层·接受长度2.76(GLM-5) → 接受长度+20%(5.2) → 1层但 transformers 未实现(5.3-Flash)
容量：      MoE 355B-A32B(4.5) → 744B-A40B·256专家(5) → 320B-A18B·288专家(5.3-Flash)
```

演进主线解读：
1. **归一化五连换**：DeepNorm Post-LN → LayerNorm → RMSNorm Post-LN → 4×RMSNorm pre+post → 2×RMSNorm 前置 + QK-Norm（MoE 化时大幅简化）。
2. **KV 效率一路加码**：MHA → MQA(1/16) → GQA（9B 16:1、32B 24:1、4.5 12:1）。
3. **位置编码两次转向**：1D → 2D → 1D（original）→ partial rotary 0.5（血统，延续到 4.5）；GLM-5 待核实，5.3-Flash 直接**文本 NoPE**（qk_rope_head_dim=0 强制）。
4. **容量路线**：130B 稠密 → 6B 稠密 → 9B/32B 稠密 → **4.5 的 MoE 160 专家**，激活 32B、总参 355B → **GLM-5 的 256 专家/744B-A40B** → **5.3-Flash 的 288 专家/320B-A18B**（轻量多模态回摆）。
5. **Z1 是"训练换能力"的支线**：同 GLM-4 架构，靠长思维链 RL 获得推理能力，与 4.5 的架构升级线并行。
6. **稀疏化时代（GLM-5 起）**：从稠密 GQA 一步跨到 MLA + DSA（indexer top-k 2048），5.2 用 IndexShare/KVShare 把 1M 上下文成本再压 2.9×，5.3-Flash 用 KDA 线性 + DSA 混合（每 4 层交错）+ mHC + IndexPool 把算力/KV 压到 1/3–1/4——这是 GLM 家族继 4.5 Dense→MoE 之后第二次架构换代。

---

## 四、对源码研读 / debug 的启示

1. **按 model_type 分流**：`chatglm`（6B/2/3/4-9B 原始版/4V-9B，需 `trust_remote_code` 加载官方仓库 modeling_chatglm.py，transformers v5 无原生模块）→ 原生 `glm4`（GLM-4-0414 / Z1）→ 原生 `glm4_moe`（4.5/Air/4.6/4.6V/4.7 系，后者待核实）→ 原生 `glm4v`（**注意对应 4.1V/4.5V，不是 4V-9B**）→ 原生 **`glm5_next`**（GLM-5 系，**对应 GLM-5.3-Flash 架构**，ForConditionalGeneration 多模态入口）。
2. **重点调试对象**：
   - ChatGLM2/3 的 MQA repeat_kv 与 fp32 softmax（`attention_softmax_in_fp32=true`）；
   - GLM-4 的 4×RMSNorm 双残差（`Glm4DecoderLayer` 每子层 pre+post 两个 norm）与 partial rotary 的 `dim = head_dim × partial_rotary_factor = 64`；
   - GLM-4.5 的 `Glm4MoeMoE`：sigmoid TopkRouter（含 norm_topk + routed_scaling）与共享专家并联，及 MTP 头；QK-Norm 作用于 Q/K 各自 RMSNorm(head_dim)；
   - 同架构对照：GLM-4 与 Z1 共用 `models\glm4\` 代码，适合做「同架构不同训练」实验；ChatGLM2 vs 3 的 config diff 只有 seq_length。
3. **本地可跑性**：ChatGLM-6B / 2 / 3（INT4 亦可）、GLM-4-9B、Z1-9B、4.5-Air（≈106B-A12B）适合本地；GLM-4.5 旗舰 355B 与 GLM-130B（无权重）不适合。
4. **易踩坑**：GLM-4V-9B 的视觉塔（63/1792/1120）与本地 `glm4v` 模块（24/1536/336）不是同一个——按 config 的 vision_config 判断，别用错加载路径。
5. **glm5_next 模块要点（对应 GLM-5.3-Flash 架构，非 GLM-5 本体）**：
   - **入口**：`Glm5NextForConditionalGeneration(PreTrainedModel, GenerationMixin)`，统一多模态（image/video/text），**无文本专用 ForCausalLM 类**；组合 Glm5NextModel（视觉 + 语言模型）+ lm_head。
   - **文本 NoPE**：`qk_rope_head_dim=0` 强制（[cfg-S:223-226]），与 4.5 系 partial rotary 血统彻底切断；视觉侧用 axial 2D RoPE。
   - **MLA**：`kv_lora_rank=512` / `q_lora_rank=1536` / `v_head_dim=256` / `qk_nope_head_dim=256`；cache 在 MLA 压缩 latent 层读写（kv_a_proj_with_mqa → 512 → expand_kv）。
   - **KDA+DSA 每 4 层交错**：运行时 `idx%4!=3` = linear_attention（KDA，34 层），`idx%4==3` = indexed_attention（DSA，11 层）；KDA 带 causal conv1d(kernel=4) + forget gate + 门控 RMSNorm。
   - **DSA 索引器**：`index_topk=2048`、`index_head_dim=128`、`index_n_heads=32`、`index_kpool=16`（每 16 token 一组 softmax 加权 pool，topk 选 128 池展开回 2048 原始 token，再 append_visible_tail 尾池）；`indexer_types[layer]=="shared"` 的层 indexer=None、**复用前层 top-k**（跨层共享，对应 IndexShare 思路）。
   - **mHC**：`hc_mult=4` 多流残差（非标准 pre-norm），Sinkhorn 归一化 comb 权重，`Glm5NextTextHyperHead` 4 流 mean 折叠。
   - **MTP 未实现**：`_keys_to_ignore_on_load_unexpected = [r"layers\.45\.", r"layers\.\d+\.shared_head\."]`，checkpoint 的 MTP 层权重被显式忽略；主输出仅 lm_head。
   - **SDPA only**：`_supports_flash_attn=False`、`_supports_sdpa=True`；DSA 为 block/pool（16 token）级 3D per-query 稀疏，非 row-wise 独立 token 稀疏，FA kernel 需自定义。
   - **MoE 路由**：`logits=linear(x)` → `sigmoid(logits)+e_score_correction_bias` → 组内 top2 → topk(8) → `norm_topk_prob` → ×`routed_scaling_factor(2.5)`——**sigmoid 打分（非 softmax），与 GLM-4.5 血统一致**；**288 路由专家 + 1 共享（inter 2048）**；前 3 层 dense、后 42 层 sparse；aux loss = 0.001 × Switch 式 load balancing。
   - **易踩坑**：勿把 glm5_next 当成 GLM-5 本体实现——它对应 5.3-Flash（320B/18B、45 层、288 专家、4096 hidden、文本 NoPE）；GLM-5 本体（744B/40B、78 层、256 专家、6144 hidden）在本地无对应模块。

---

## 参考来源

1. `E:\transformers\bayon-note\GLM\glm_130b\index.md`（GLM-130B：论文 arXiv:2210.02414 附录 Table 11 + 官方仓库 THUDM/GLM-130B）
2. `E:\transformers\bayon-note\GLM\chatglm_6b\index.md`（ChatGLM-6B：官方 config hf-mirror 留档 + arXiv:2406.12793 / 2103.10360）
3. `E:\transformers\bayon-note\GLM\chatglm2_3\index.md`（ChatGLM2/3：两份官方 config 逐字段比对 + arXiv:2406.12793）
4. `E:\transformers\bayon-note\GLM\glm_4\index.md`（GLM-4：官方 config 留档 + 本地 `models\glm4\` 源码）
5. `E:\transformers\bayon-note\GLM\glm_4v\index.md`（GLM-4V-9B：官方 config 留档 + 原生 glm4v 模块对照）
6. `E:\transformers\bayon-note\GLM\glm_z1\index.md`（GLM-Z1：官方 config 留档 + 与 GLM-4-0414 逐字段比对）
7. `E:\transformers\bayon-note\GLM\glm_4_5\index.md`（GLM-4.5/Air/4.6：官方 config 留档 + 本地 `models\glm4_moe\` 源码）
8. 官方仓库/卡片：https://huggingface.co/zai-org/glm-4-9b 、GLM-4.5、GLM-Z1-32B-0414 ；https://huggingface.co/THUDM/chatglm-6b 、chatglm2-6b 、glm-4v-9b
9. 官方论文：https://arxiv.org/abs/2210.02414 ；https://arxiv.org/abs/2103.10360 ；https://arxiv.org/abs/2406.12793
10. GLM-5 技术报告：https://arxiv.org/pdf/2602.15763（《GLM-5: from Vibe Coding to Agentic Engineering》，Zhipu AI & Tsinghua University）
11. GLM-5 官网（口径差异注释源）：https://glm5.ai/
12. 智谱文档 GLM-5：https://docs.bigmodel.cn/cn/guide/models/text/glm-5 ；版本发布动态：https://docs.bigmodel.cn/cn/update/new-releases
13. 智谱研究页（Lightning Indexer / Sparse Flash Attention / MLAPO 推理优化）：https://www.zhipuai.cn/zh/research/153
14. GLM-5.2 博客：https://z.ai/blog/glm-5.2
15. GLM-5.3-Flash 智谱文档：https://docs.bigmodel.cn/cn/guide/models/vlm/glm-5.3-flash
16. NVIDIA NIM（GLM-5.3-Flash 架构核对：45 层 / 34 KDA + 11 DSA / mHC / 288 专家 / 320B-18B / 30T）：https://docs.api.nvidia.com/nim/reference/z-ai-glm-5-3-flash
17. 开源状态汇总：https://glm-ai.chat/docs/is-glm-open-source/ ；GLM-4.7 HF：https://huggingface.co/zai-org/GLM-4.7 ；GLM-5 开源仓库：https://github.com/zai-org/GLM-5
18. 本地源码：`E:\transformers\src\transformers\models\glm5_next\`（configuration_glm5_next.py / modeling_glm5_next.py / processing_glm5_next.py / image_processing_glm5_next.py / video_processing_glm5_next.py；对应 GLM-5.3-Flash 架构）

> 标注约定：全部数字来自各版本目录 index.md 所引官方 config / 本地源码 / 官方论文；未核实项（GLM-Z1 CoT 特殊 token id）按源文档标「待确认」；GLM-4.6V/4.7/4.7-Flash 的 config 细节、GLM-5 本体路由算法与归一化细节标「待核实」。访问与核验日期：2026-10-31（各版本文档），2026-11-02（本对比文档），**2026-11-03（GLM-4.6V → GLM-5.3-Flash 版本线补全：arXiv:2602.15763 技术报告、glm5.ai、docs.bigmodel.cn、zhipuai.cn/zh/research/153、z.ai/blog/glm-5.2、glm-ai.chat、NVIDIA NIM、本地 glm5_next 源码）**。

---

*页脚：GLM 全家族版本对比与演进 · 数据来源：bayon-note 各版本 index.md（官方 config + transformers 源码 + 官方论文）· 仅供技术学习与论文研究*
