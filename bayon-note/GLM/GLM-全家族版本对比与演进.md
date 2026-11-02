# GLM 全家族版本对比与演进 — GLM-130B → ChatGLM-6B → ChatGLM2/3 → GLM-4 → GLM-4V → GLM-Z1 → GLM-4.5/4.6

> 整理日期：2026-11-02
> 数据来源：各版本目录 `index.md`（其事实全部源自远端官方 config（hf-mirror 拉取留档 `_tools\_cfgpull\`）、transformers 本地源码 `E:\transformers\src\transformers\models\glm*/`、官方论文与官方仓库，详见文末来源）
> 覆盖版本：glm_130b、chatglm_6b、chatglm2_3（2 与 3 同构合册）、glm_4（9B + 32B-0414）、glm_4v（GLM-4V-9B）、glm_z1（Z1-9B/32B）、glm_4_5（4.5 / 4.5-Air / 4.6）
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
7. **GLM-4.5（2025-07）→ 4.5-Air → 4.6**：**Dense→MoE 大切换**（`glm4_moe`）——160 路由专家 + 1 共享专家、top-8、**sigmoid 路由**（去 softmax）、QK-Norm、归一化从 4×RMSNorm 简化为 2×RMSNorm 前置、新增 **MTP**；4.6 与 4.5 同构仅上下文扩到 202752。

一句话：**GLM-130B 立"双向"根基，ChatGLM-6B 缩体量开对话，ChatGLM2/3 换因果+MQA，GLM-4 定 GQA+partial rotary 血统，GLM-4.5 切 MoE 并引入 QK-Norm/MTP，Z1 走"同结构+推理训练"并行线**。

---

## 一、全家族核心规格对比表

| 项目 | GLM-130B | ChatGLM-6B | ChatGLM2-6B | ChatGLM3-6B | GLM-4-9B | GLM-4-32B-0414 | GLM-4V-9B | GLM-Z1（9B/32B） | GLM-4.5 | GLM-4.5-Air | GLM-4.6 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 时间 | 2022-08 | 2023-03 | 2023-06 | 2023-10 | 2024-01 | 2025-04 | 2024 | 2025-04 | 2025-07 | 2025-07 | 2025 年中 |
| 入口类 / model_type | —（无 HF 权重） | ChatGLMModel / chatglm | ChatGLMModel / chatglm | 同左 | ChatGLMModel→Glm4ForCausalLM / glm4 | Glm4ForCausalLM / glm4 | ChatGLMForConditionalGeneration / chatglm | Glm4ForCausalLM / glm4 | **Glm4MoeForCausalLM / glm4_moe** | 同左 | 同左 |
| 形态 | **Dense 双向** | Dense | Dense | Dense | **Dense** | Dense | Dense + ViT | Dense（推理训练） | **MoE 355B-A32B** | **MoE 106B-A12B** | MoE 同 4.5 |
| hidden / 层数 | 12288 / 70 | 4096 / 28 | 4096 / 28 | 4096 / 28 | 4096 / 40 | 6144 / 61 | 4096 / 40（文本塔） | 4096·40 / 6144·61 | 5120 / 92 | 4096 / 46 | 5120 / 92 |
| Q / KV 头 | 96 / 96（MHA） | 32 / 32（MHA） | 32 / 2（**MQA**） | 同左 | 32 / 2（GQA） | 48 / 2（GQA） | 32 / 2（文本塔） | 同 GLM-4-0414 | 96 / 8（GQA） | 96 / 8 | 96 / 8 |
| head_dim | 128 | 128 | kv_channels 128 | 128 | 128 | 128 | 128 | 128 | 128 | 128 | 128 |
| 注意力骨干 | 双向 GLM blank infilling | prefix-LM 双支路 | 因果 1D | 因果 1D | 因果 GQA | 因果 GQA | 同 GLM-4 + 视觉 token | 同 GLM-4 | 因果 GQA + **QK-Norm** | 同 4.5（无 QK-Norm） | 同 4.5 |
| 位置编码 | RoPE 1D | **2D-RoPE** | 1D-RoPE（original） | 同左 | **partial rotary 0.5**（θ=1e4） | 同左 | 同 GLM-4 | 同 GLM-4 | partial 0.5 + **θ=1e6** | 同 4.5 | 同 4.5 |
| 归一化 | **DeepNorm Post-LN** | LayerNorm | RMSNorm Post-LN | 同左 | **4×RMSNorm（pre+post）** | 同左 | 同 GLM-4 | 同 GLM-4 | **2×RMSNorm 前置 + QK-Norm** | 同 4.5 | 同 4.5 |
| FFN | GeGLU 32768（8/3·d） | inner 16384（4·d） | SwiGLU 13696 | 13696 | SiGLU 13696（gate_up 融合） | 23040 | 13696（文本塔） | 13696 / 23040 | dense 12288 + **专家 moe_inter 1536** | dense 10944 + 1408 | 同 4.5 |
| MoE 专家 | 无 | 无 | 无 | 无 | 无 | 无 | 无 | 无 | **160 路由 + 1 共享，top-8**，first_k_dense 3 | **128 + 1，top-8**，first_k_dense 1 | 同 4.5 |
| 路由策略 | — | — | — | — | — | — | — | — | **sigmoid + norm_topk + scaling 2.5** | sigmoid + scaling 1.0 | 同 4.5 |
| 上下文 | 2048 | 2048 | **32768**（config） | 8192（config，官方 32K 口径） | 8192 基座 / 推理 128K | 32768 | 8192（文本塔） | 32768 | 131072 | 131072 | **202752** |
| 词表 | icetk 150000 | 130528 | padded 65024 | 65024 | **151552** | 151552 | 151552 | 151552 | 151552 | 151552 | 151552 |
| 训练 | ≈400B token | ≈1T token 中英 | 后训练 + MQA 提速 | 42 基准登顶 / Agent | 预训练 + SFT | 同 | 多模态对齐 | **长思维链 RL** | 预训练 + MoE | 同 | 同 4.5 |
| MTP / 特殊机制 | MIP 多任务指令 | [MASK]/[gMASK] | — | function call（模板层） | — | — | boi=151339/eoi=151340 | CoT 标记（待确认） | **MTP ×1** | MTP ×1 | MTP ×1 |
| transformers 模块 | 无 | 无（trust_remote_code） | 无 | 无 | 原生 `glm4` | 原生 `glm4` | 无（远程；原生 glm4v 是 4.1V/4.5V） | 原生 `glm4` | 原生 `glm4_moe` | 同左 | 同左 |
| 主要论文/来源 | arXiv:2210.02414 | arXiv:2406.12793 / 2103.10360 | arXiv:2406.12793 | 同左 | zai-org config / card | zai-org config | zai-org config | zai-org config | zai-org config / card | 同左 | 同左 |

> 说明：① GLM-4V-9B 的视觉塔（63 层 / hidden 1792 / image 1120 / patch 14 / 16 头 / inter 15360 / GELU / scaling_factor 8 → 100 token/tile）单独存在，不属于文本主干；transformers 原生 `models\glm4v\` 对应的是后续 GLM-4.1V/4.5V（24 层 / 1536 / 336 / merge 2），**勿混淆**。② ChatGLM2/3 同构合册（唯一 config 差异 `seq_length`）；Z1 与 GLM-4-0414 config 逐项相同。

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
模态：      文本(130B~4/Z1) → 文本+图像(4V-9B：63层深塔) → 4.1V/4.5V 轻塔(24层)
训练：      400B 预训练(130B) → 1T 中英(6B) → SFT/对齐(2/3) → 长思维链 RL(Z1)
```

演进主线解读：
1. **归一化五连换**：DeepNorm Post-LN → LayerNorm → RMSNorm Post-LN → 4×RMSNorm pre+post → 2×RMSNorm 前置 + QK-Norm（MoE 化时大幅简化）。
2. **KV 效率一路加码**：MHA → MQA(1/16) → GQA（9B 16:1、32B 24:1、4.5 12:1）。
3. **位置编码两次转向**：1D → 2D → 1D（original）→ partial rotary 0.5（血统，延续到 4.5）。
4. **容量路线**：130B 稠密 → 6B 稠密 → 9B/32B 稠密 → **4.5 的 MoE 160 专家**，激活 32B、总参 355B。
5. **Z1 是"训练换能力"的支线**：同 GLM-4 架构，靠长思维链 RL 获得推理能力，与 4.5 的架构升级线并行。

---

## 四、对源码研读 / debug 的启示

1. **按 model_type 分流**：`chatglm`（6B/2/3/4-9B 原始版/4V-9B，需 `trust_remote_code` 加载官方仓库 modeling_chatglm.py，transformers v5 无原生模块）→ 原生 `glm4`（GLM-4-0414 / Z1）→ 原生 `glm4_moe`（4.5/Air/4.6）→ 原生 `glm4v`（**注意对应 4.1V/4.5V，不是 4V-9B**）。
2. **重点调试对象**：
   - ChatGLM2/3 的 MQA repeat_kv 与 fp32 softmax（`attention_softmax_in_fp32=true`）；
   - GLM-4 的 4×RMSNorm 双残差（`Glm4DecoderLayer` 每子层 pre+post 两个 norm）与 partial rotary 的 `dim = head_dim × partial_rotary_factor = 64`；
   - GLM-4.5 的 `Glm4MoeMoE`：sigmoid TopkRouter（含 norm_topk + routed_scaling）与共享专家并联，及 MTP 头；QK-Norm 作用于 Q/K 各自 RMSNorm(head_dim)；
   - 同架构对照：GLM-4 与 Z1 共用 `models\glm4\` 代码，适合做「同架构不同训练」实验；ChatGLM2 vs 3 的 config diff 只有 seq_length。
3. **本地可跑性**：ChatGLM-6B / 2 / 3（INT4 亦可）、GLM-4-9B、Z1-9B、4.5-Air（≈106B-A12B）适合本地；GLM-4.5 旗舰 355B 与 GLM-130B（无权重）不适合。
4. **易踩坑**：GLM-4V-9B 的视觉塔（63/1792/1120）与本地 `glm4v` 模块（24/1536/336）不是同一个——按 config 的 vision_config 判断，别用错加载路径。

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

> 标注约定：全部数字来自各版本目录 index.md 所引官方 config / 本地源码 / 官方论文；未核实项（GLM-Z1 CoT 特殊 token id）按源文档标「待确认」。访问与核验日期：2026-10-31（各版本文档），2026-11-02（本对比文档）。

---

*页脚：GLM 全家族版本对比与演进 · 数据来源：bayon-note 各版本 index.md（官方 config + transformers 源码 + 官方论文）· 仅供技术学习与论文研究*
