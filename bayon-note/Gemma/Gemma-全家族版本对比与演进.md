# Gemma 全家族版本对比与演进 — Gemma 1 → 2 → 3 → 3n → 4

> 整理日期：2026-11-02
> 数据来源：各版本目录 `index.md`（其事实全部源自本地 config.json、ModelScope 官方镜像 config、transformers 本地源码 `modeling_gemma*.py`、官方技术报告与模型卡，详见文末来源）
> 覆盖版本：gemma-1（2024-02）、gemma-2（2024-06）、gemma-3（2025-03）、gemma-3n（2025-05，端侧）、gemma-4（26B-A4B MoE 等五尺寸）
> 用途：配合 `E:\transformers` 源码研读与断点 debug 的家族横向速查

---

## 结论速览

Gemma 家族的演进主线是一条「**从小模型打样 → 深度网络稳定化 → 多模态 + 超长上下文 → 端侧效率化 → 收敛为真 MoE**」的路线：

1. **Gemma 1（2024-02）**：pre-norm + GeGLU + RoPE(θ=1e4) + RMSNorm 的 LLaMA 式稠密母版；2B 用 MQA 压 KV、7B 保留完整 MHA；词表 256K、上下文 8192。
2. **Gemma 2（2024-06）**：四大结构增量——GQA 全家族统一（groups=2）、滑窗 4096 与全局 8192 逐层 1:1 交替、**双 soft-capping**（attn 50 / final 30）、pre+post 双 RMSNorm（每层 4 个）；并新增 `query_pre_attn_scalar` 把 Q 缩放与 head_dim 解耦。
3. **Gemma 3（2025-03）**：跨入多模态（4B/12B/27B 挂 SigLIP 视觉塔）；**双频率 RoPE**（global θ=1e6 / local θ=1e4）+ 5:1 滑窗/全局交错；**以 QK-Norm 取代 soft-capping**；上下文 8K → 32K/128K；词表 256K→262144，训练扩到 14T token / 140+ 语言。
4. **Gemma 3n（2025-05）**：端侧专版——MatFormer 嵌套（逐层 FFN 数组）+ AltUp 四路隐藏态 + LAuReL 低秩残差 + PLE 逐层嵌入 + 前 10 层高斯 top-k 激活稀疏 + 末段 KV 共享；首次原生并入音频（Conformer）；上下文回收到 32K。
5. **Gemma 4**：把 3n 的效率思路收敛为**真正的 MoE**（26B-A4B：128 专家 Top-8 + dense MLP 并行融合）；注意力升级为 **per-layer 维度 + Q/K/V head-wise RMSNorm**（移除 query_scalar，scaling=1.0）；full 层 K=V 绑定；上下文最高 256K；12B 走 encoder-free 无独立视觉塔。

一句话：**Gemma 1 定母版，Gemma 2 加固稳定性（GQA/滑窗/softcap/双 norm），Gemma 3 开多模态与长上下文（QK-Norm/双 RoPE），Gemma 3n 做端侧效率（MatFormer/稀疏/KV 共享），Gemma 4 收敛为 MoE + 分层注意力（QKV Norm/per-layer dim）**。

---

## 一、全家族核心规格对比表

| 项目 | Gemma 1 | Gemma 2 | Gemma 3 | Gemma 3n | Gemma 4 |
|---|---|---|---|---|---|
| 发布时间 | 2024-02 | 2024-06 | 2025-03 | 2025-05 | 2026（官方口径） |
| 入口类 | GemmaForCausalLM | Gemma2ForCausalLM | Gemma3ForCausalLM（270M/1B）/ Gemma3ForConditionalGeneration（4B+） | Gemma3nForConditionalGeneration | Gemma4ForConditionalGeneration / Gemma4UnifiedForConditionalGeneration（12B） |
| 尺寸 | 2B / 7B | 2B / 9B / 27B | 270M / 1B / 4B / 12B / 27B | E2B / E4B（有效≈2B/4B，原始≈5B/8B） | E2B / E4B / 12B / **26B-A4B（MoE）** / 31B |
| 形态 | Dense | Dense | Dense | Dense（+稀疏激活） | Dense / **MoE（26B-A4B：128 专家 Top-8 + dense 并行）** |
| 模态 | 纯文本 | 纯文本 | 文本 + 图像（SigLIP） | 文本 + 图像（MobileNetV5）+ **音频（Conformer）** | 文本 + 图像 + 音频 + 视频（12B encoder-free） |
| 注意力 | 2B MQA（1KV）/ 7B MHA（16KV） | **GQA 全系**（groups=2） | GQA（小尺寸 4Q/1KV MQA 式，大尺寸 2:1） | GQA（8Q/2KV，组比 4:1） | GQA（sliding 层与 full 层 KV 头数可不同） |
| 滑窗/全局交错 | 无（全全局） | **1:1 逐层交替**，窗 4096 / 全局 8192 | **5:1**（每 6 层第 6 层 full），窗 512/1024 | **4:1**（每 5 层第 5 层 full），窗 512 | E2B 每 5 层；其余每 6 层 1 full；窗 512/1024 |
| RoPE | θ=1e4（统一） | θ=1e4（统一） | **双频**：global 1e6 / local 1e4 | **双频**：global 1e6 / local 1e4 | **按层类型双频**（sliding θ=1e4 / full θ=1e6，full 另加 proportional 0.25 partial rotary） |
| 注意力稳定化 | 无（scaling=head_dim^-0.5） | attn softcap 50 + final softcap 30 + query_scalar | **QK-Norm**（无 softcap，query_scalar 256/168） | QKV Norm（scaling=1.0）+ final softcap 30 | **Q/K/V head-wise RMSNorm**（scaling=1.0）+ final softcap 30 |
| 归一化布局 | pre-norm only（每层 2 个） | pre+post（每层 4 个） | pre+post（延续） | pre+post（延续） | pre+post（延续） |
| KV cache 优化 | 2B MQA 1/8 | hybrid（滑窗层回收） | hybrid（延续） | hybrid + **末段 KV 共享**（后 10/15 层） | hybrid + E2B/E4B 末段 KV 共享（20/18 层）+ E2B double-wide MLP |
| 上下文 | 8192 | 8192 | 32K（小）/ 128K（大，linear ×8） | 32K | 128K（E2B/E4B）/ 256K（12B/26B/31B） |
| 词表 | 256000（论文 256128） | 256000 | 262144 | 262400（+256 多模态 token） | 262144 |
| head_dim | 256 | 256/256/128 | 256（27B=128） | 256 | sliding 256 / full **512**（per-layer） |
| 训练量 | 2B≈3T / 7B≈6T | 2B 2T（蒸馏）/ 9B 8T（蒸馏）/ 27B 13T（从零） | ≈14T / 140+ 语言 | 官方口径（端侧效率优先） | 官方口径 |
| 特殊机制 | 嵌入缩放 √d、权重绑定 | query_pre_attn_scalar 与 head_dim 解耦 | SigLIP 视觉塔 256 soft token/图 | **MatFormer / AltUp / LAuReL(64) / PLE / 激活稀疏(前10层0.95) / KV 共享** | **MoE 128×Top-8 + dense 并行**、K=V 绑定、per-layer 维度 |
| 技术报告 | arXiv:2403.08295 | arXiv:2408.00118 | arXiv:2503.19786 | arXiv:2505.05146 | arXiv:2607.02770 |

> 说明：Gemma 4 发布年份与技术报告编号（arXiv:2607.02770）以 gemma-4 目录 index.md 标注为准。Gemma 3 的 270M/1B 为纯文本分支（无视觉塔），Gemma 4 的 12B 为 encoder-free（无独立视觉/音频编码器）。

---

## 二、代际演进详解

### 2.1 Gemma 1 → Gemma 2：从 MQA/MHA 到 GQA + 稳定化四件套

| 维度 | Gemma 1 | Gemma 2 |
|---|---|---|
| 注意力 | 2B MQA（1 KV）/ 7B MHA（16 KV） | **GQA 全系**：2B 8Q/4KV · 9B 16Q/8KV · 27B 32Q/16KV，groups 恒为 2 |
| 局部滑窗 | 无 | **窗 4096 与全局 8192 逐层 1:1 交替** |
| soft-capping | 无 | attn $\tanh(s/50)\cdot50$ + final $\tanh(s/30)\cdot30$ |
| query 缩放 | 隐含 $1/\sqrt{\text{head\_dim}}$ | **显式 `query_pre_attn_scalar`**（2B/9B=256，27B=144），与 head_dim 解耦 |
| 归一化 | pre-norm only（每层 2 个 RMSNorm） | **pre+post 双归一化**（每层 4 个），支撑更深网络（18→26/42/46 层） |
| FFN | 2B inter 16384 | 更窄 FFN × 更多层（9216/14336/36864） |
| 训练 | 2B≈3T / 7B≈6T 从零 | **2B/9B 知识蒸馏**（2T/8T），27B 从零（13T） |

- Gemma 2 的口号是 "competitive alternatives to models that are 2-3 times bigger"——靠 GQA 降推理成本、滑窗降 KV、softcap + 双 norm 稳定深网络。

### 2.2 Gemma 2 → Gemma 3：多模态 + 双 RoPE + QK-Norm

| 维度 | Gemma 2 | Gemma 3 |
|---|---|---|
| 模态 | 纯文本 | **4B/12B/27B 挂 SigLIP 视觉塔**（patch 14 / 896px / 256 soft token/图），270M/1B 仍纯文本 |
| 滑窗/全局 | 1:1 逐层交替 | **5:1**（每 6 层第 6 层 full），窗 4096→512/1024 |
| RoPE | 统一 θ=1e4 | **双频率**：full 层 θ=1e6（长程）、sliding 层 θ=1e4（局部） |
| 注意力稳定化 | softcap 50/30 + query_scalar | **全部移除，以 QK-Norm 取代**（Q/K 逐头 RMSNorm，RoPE 前） |
| 上下文 | 8192 | 32K（小尺寸）/ **128K**（大尺寸，linear RoPE ×8） |
| 词表 | 256000 | **262144** |
| 训练 | 英语为主、8T 级 | **14T token、140+ 语言** |
| 27B | head_dim 128 · scalar 144 | head_dim 128 · scalar 168（唯一特殊尺寸） |

- 关键转折：Gemma 3 用 **QK-Norm 替代 soft-capping**（官方明确口径），并把上下文与语言覆盖大幅扩表——这是家族从"实用小模型"走向"通用多模态"的一代。

### 2.3 Gemma 3 → Gemma 3n：端侧效率工程（MatFormer 全家桶）

| 维度 | Gemma 3 | Gemma 3n（E2B/E4B） |
|---|---|---|
| 定位 | 云端/通用，1B–27B | **端侧**（手机/笔记本），有效≈2B/≈4B（原始 5B/8B） |
| FFN | 标准 dense GeGLU | **MatFormer 逐层中间维数组**（E2B [8192]×30 / E4B [16384]×35）+ **前 10 层高斯 top-k 激活稀疏**（cutoff 0.95） |
| 残差技巧 | 标准残差 | **AltUp 四路线**（altup_num_inputs=4）+ **LAuReL 低秩残差**（rank 64）+ **PLE 逐层嵌入**（256 维/层） |
| KV | hybrid | hybrid + **末段 KV 共享**（E2B 后 10 层 / E4B 后 15 层，无 k/v 权重） |
| 交错 | 每 6 层 1 full | 每 5 层 1 full；窗 512 |
| 视觉 | SigLIP ViT 系 | **轻量 MobileNetV5-300M（CNN）**，256 soft token/图 |
| 音频 | 无 | **原生音频**（Conformer×12，128 维 mel → SSCP 下采样） |
| 上下文 | 128K | **32K**（端侧回退） |
| 词表 | 262144 | 262400（+256 多模态特殊 token） |

- Gemma 3n 的本质：用 MatFormer 嵌套 + 激活稀疏 + KV 共享，在约 2B/4B 有效参数下逼近更大模型质量，代价是上下文从 128K 回收至 32K。

### 2.4 Gemma 3n → Gemma 4：收敛为真 MoE + 分层注意力

| 维度 | Gemma 3n | Gemma 4 |
|---|---|---|
| 规模 | E2B/E4B | E2B / E4B / 12B / **26B-A4B** / 31B |
| 稀疏方案 | 激活稀疏（95% cutoff） | **真 MoE**：26B-A4B 每层 dense MLP 与 **128 专家并行**、Top-8（6.25% 激活）、per_expert_scale |
| 注意力归一化 | 无 QKV Norm（query_scalar 式） | **Q/K/V head-wise RMSNorm** + scaling=1.0（移除 query_scalar） |
| 每层维度 | 统一 head_dim | **per-layer**：sliding 层 head_dim 256 / full 层 global_head_dim 512；KV 头也可按层类型不同 |
| K=V 绑定 | 无 | 12B/26B/31B 的 full 层 **attention_k_eq_v=true**（v_proj=None，V=K） |
| KV 共享 | 后 10/15 层 | E2B 后 20 层（+double-wide MLP）/ E4B 后 18 层；12B/26B/31B 无 |
| 上下文 | 32K | **128K（E2B/E4B）/ 256K（12B/26B/31B）** |
| 视觉 | MobileNetV5 | ViT 风格塔（16/27 层，280 soft token/图）或 **encoder-free（12B）** |
| 音频 | Conformer（有） | E2B/E4B/12B 有；**26B/31B 移除音频塔** |
| PLE | 有 | 仅 E2B/E4B 保留 |

- Gemma 4 的 26B-A4B 是家族首个 MoE：**dense MLP 与 128 专家并行、输出相加**（非二选一路由），router 输入取残差（MLP 输入前的 hidden）；激活比 6.25% + dense 全量计算，推理计算量接近 4B 稠密模型而容量 26B——"A4B（Active 4 Billion）"名称来源。

---

## 三、关键机制演进一图流

```
注意力 KV 头：  MQA(2B)/MHA(7B) [G1] → GQA groups=2 全系 [G2] → GQA + 双 RoPE [G3]
               → GQA 4:1 + KV 共享 [G3n] → per-layer GQA（sliding 256 / full 512）+ K=V 绑定 [G4]
滑窗/全局：     无 [G1] → 1:1 逐层交替 [G2] → 5:1 [G3] → 4:1 [G3n] → 5:1/6:1 [G4]
注意力稳定化：  scaling=head_dim^-0.5 [G1] → softcap 50/30 + query_scalar [G2]
               → QK-Norm（去 softcap）[G3] → QKV Norm + scaling=1.0 [G3n/G4]
上下文：        8192 [G1/G2] → 32K/128K [G3] → 32K [G3n] → 128K/256K [G4]
词表：          256000 [G1/G2] → 262144 [G3] → 262400 [G3n] → 262144 [G4]
模态：          文本 [G1/G2] → 文本+图像 [G3] → 文本+图像+音频 [G3n] → 文本+图像+音频+视频 [G4]
稀疏/容量：     dense [G1-G3] → 激活稀疏 [G3n] → MoE 128×Top-8+dense 并行 [G4]
```

演进主线解读：
1. **注意力稳定化三次换挡**：G1 裸缩放 → G2 softcap + query_scalar → G3 起 QK/QKV-Norm 并逐步删掉显式缩放与 softcap（G4 scaling=1.0）。
2. **KV 效率持续加码**：MQA（G1）→ GQA（G2）→ hybrid cache（G2/G3）→ 末段 KV 共享（G3n/G4）+ double-wide MLP（G4 E2B）→ per-layer 维度与 K=V 绑定（G4）。
3. **容量路径从 dense 到 MoE**：G1–G3 纯 dense，G3n 用激活稀疏模拟稀疏 FFN，G4 一步到位上 128 专家真 MoE。
4. **模态与上下文同步扩张**：文本 → 文本+图像 → 文本+图像+音频 → 文本+图像+音频+视频；上下文 8K → 128K/256K（端侧 3n 例外回退 32K）。

---

## 四、对源码研读 / debug 的启示

1. **按入口类分流**：`GemmaForCausalLM`（G1）→ `Gemma2ForCausalLM`（G2）→ `Gemma3ForCausalLM` / `Gemma3ForConditionalGeneration`（G3）→ `Gemma3nForConditionalGeneration`（G3n）→ `Gemma4ForConditionalGeneration` / `Gemma4UnifiedForConditionalGeneration`（G4）——五套独立 modeling 文件，打断点路径完全不同。
2. **重点调试对象**：
   - G2 的滑窗/全局双 mask 按 `layer_types[i]` 逐层分发 + HybridCache 回收逻辑；
   - G3 的 QK-Norm（RoPE 前逐头归一）与双 RoPE 逆频率表按层类型取用；
   - G3n 的 MatFormer 链：AltUp predict/correct → LAuReL → 高斯 top-k 稀疏门控 → PLE 注入，以及末段共享 KV 的 `shared_kv_states[layer_type]` 复用；
   - G4 的 MoE 并行融合（dense ‖ 128 专家相加）与 per-layer 维度构造（`configuration_gemma4.py` 的 `per_layer_config`）。
3. **跨代对照线索**：G2 的 `query_pre_attn_scalar` 到 G3 的 QK-Norm 到 G4 的 QKV-Norm 是一条"注意力缩放如何被归一化取代"的清晰演进线，适合做数值对照实验。
4. **本地可跑性**：G4 26B-A4B 每 token 只激活约 3.8B，是家族里唯一在双 16G 卡上有戏的 MoE；G3n E2B/E4B 与 G4 E2B/E4B 是端侧尺寸，最适合断点调试。

---

## 参考来源

1. `E:\transformers\bayon-note\Gemma\gemma-1\index.md`（Gemma 1 事实：本地 config + arXiv:2403.08295）
2. `E:\transformers\bayon-note\Gemma\gemma-2\index.md`（Gemma 2 事实：本地/ModelScope config + arXiv:2408.00118）
3. `E:\transformers\bayon-note\Gemma\gemma-3\index.md`（Gemma 3 事实：本地/ModelScope config + arXiv:2503.19786 + Google 博客）
4. `E:\transformers\bayon-note\Gemma\gemma-3n\index.md`（Gemma 3n 事实：本地 config + ModelScope E4B config + arXiv:2505.05146）
5. `E:\transformers\bayon-note\Gemma\gemma-4\index.md`（Gemma 4 事实：ModelScope 各变体 config + arXiv:2607.02770 + Google 官方资料）
6. 官方模型卡：https://ai.google.dev/gemma/docs/core/model_card_1 、model_card_2 、model_card_3 、model_card_3n 、model_card_4
7. 技术报告：https://arxiv.org/abs/2403.08295 ；https://arxiv.org/abs/2408.00118 ；https://arxiv.org/abs/2503.19786 ；https://arxiv.org/abs/2505.05146

> 标注约定：全部数字来自各版本目录 index.md 所引 config / 本地源码 / 官方论文；未核实项（如 Gemma 4 发布年份细节）以 gemma-4 目录标注为准。访问与核验日期：2026-10-22（各版本文档），2026-11-02（本对比文档）。

---

*页脚：Gemma 全家族版本对比与演进 · 数据来源：bayon-note 各版本 index.md（config + transformers 源码 + 官方技术报告）· 仅供技术学习与论文研究*
