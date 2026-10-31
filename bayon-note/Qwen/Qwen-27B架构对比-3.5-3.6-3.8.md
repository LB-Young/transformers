# Qwen3.5-27B / Qwen3.6-27B / Qwen3.8-27B 架构对比

> 整理日期：2026-10-26
> 数据来源：各模型官方 Hugging Face / ModelScope model card（文末附链接）
> 用途：配合 `E:\transformers` 源码研读与断点 debug 的架构速查

---

## 结论速览

三代 27B 共享**同一套架构基座**（transformers 中 `model_type = qwen3_5`）：

- **Dense 原生视觉语言模型**（Causal LM + Vision Encoder，图像/视频理解）
- 64 层、Hidden 5120、词表 248,320（padded）
- **Gated DeltaNet（线性注意力）与 Gated Attention（全注意力）3:1 混合**
- 原生 262,144 token 上下文，YaRN 可扩至 1M
- 训练时多步 MTP（Multi-Token Prediction）

**核心结构配置三代完全一致**，代际差异全部体现在：训练/后处理、推理行为控制（thinking）、量化与生态支持。换句话说：换代的不是"图纸"，而是"工艺与用法"。

---

## 一、核心规格对比表

| 项目 | Qwen3.5-27B | Qwen3.6-27B | Qwen3.8-27B |
|---|---|---|---|
| 发布 | 2026-02 起（Qwen3.5 系列） | 2026-04-22 | 2026-08 |
| 类型 | Causal LM + Vision Encoder | 同左 | 同左 |
| 总参数 | 27B（HF 显示约 28B params 含 embedding） | 27B | 27B |
| model_type | `qwen3_5` | `qwen3_5` | `qwen3_5` |
| 层数 | 64 | 64 | 64 |
| Hidden Dimension | 5120 | 5120 | 5120 |
| Token Embedding | 248,320 (padded) | 248,320 (padded) | 248,320 (padded) |
| Hidden Layout | 16×(3×GDN→FFN → 1×GA→FFN) | 相同 | 相同 |
| Gated DeltaNet | V 48 头 / QK 16 头，head_dim 128 | 相同 | 相同 |
| Gated Attention | Q 24 头 / KV 4 头（GQA），head_dim 256 | 相同 | 相同 |
| RoPE 维度 | 64 | 64 | 64 |
| FFN Intermediate | 17,408 | 17,408 | 17,408 |
| MTP | trained with multi-steps | 相同 | 相同 |
| 上下文 | 262,144 → YaRN 1,010,000 | 262,144 → 1,010,000 | 262,144 → 1,000,000 |
| 官方量化版 | — | — | FP8（2026-08-13，约 29GB） |
| 兼容框架 | transformers / vLLM / SGLang / KTransformers | 同左 | 同左 + TokenSpeed |
| BF16 体积 | ~55 GB | ~55.6 GB | ~55 GB |

> 说明：Qwen3.5-27B 部分第三方页面（如社区 GGUF 转换）标 Hidden=4096，与官方 model card 的 5120 不符，本文以官方 ModelScope/HF model card 为准。

---

## 二、共享架构基座详解

### 1. 整体结构

```
输入（文本 / 图像 / 视频）
  → Vision Encoder（多模态 token 融合，early fusion 训练）
  → Embedding（词表 248,320）
  → 64 × DecoderLayer
  → LM Head（输出 248,320）
```

### 2. 64 层 = 16 组 × 4 层的混合注意力

每组内部（重复 16 次）：

```
[Gated DeltaNet → FFN] × 3   ← 线性注意力（低成本、长上下文）
[Gated Attention → FFN] × 1   ← 标准全注意力（精确 recall）
```

3/4 的层用线性注意力，1/4 保留标准注意力——这是"复杂度接近 O(n) 但仍保注意力精度"的混合设计，也是三代模型最核心的架构特征。

### 3. 两种注意力单元

| 单元 | 类型 | 头配置 | 头维度 | 位置编码 |
|---|---|---|---|---|
| Gated DeltaNet | 线性注意力（递归状态） | V 48 / QK 16 | 128 | 内置门控 |
| Gated Attention | 标准 softmax 注意力（GQA） | Q 24 / KV 4 | 256 | RoPE 64 |

- **Gated DeltaNet**：用可学习的门控 + 递归状态（社区 GGUF 元数据可见 `state_size 128`、`time_step_rank 48`）把注意力复杂度从 O(n²) 降到接近 O(n)，是长上下文吞吐的关键。
- **Gated Attention**：GQA 压缩 KV 头（24→4），RoPE dim 64，负责需要精确定位/回忆的路径。

### 4. 长上下文与 MTP

- 原生上下文 262,144；官方建议至少 128K 以保留思考能力；超过后用**静态 YaRN**（`rope_type: yarn`、`factor 4.0`）扩展到 1M。
- MTP：训练时预测多个后续 token，推理时由 vLLM（`qwen3_next_mtp`）/ SGLang（`NEXTN`）做投机解码加速。

### 5. Thinking 模式

默认开启思考（reasoning）输出，且三代都不支持 Qwen3 的 `/think`、`/nothink` 软切换（3.6 起明确写入 model card）。

---

## 三、代际差异

### Qwen3.5 → Qwen3.6（稳定化 + 实用化）

- 官方定位：基于 Qwen3.5 系列与社区反馈，优先**稳定性与现实可用性**。
- 能力：Agentic Coding 强化（前端工作流、仓库级推理），通用多模态小幅提升（MMMU 82.3 → 82.9）。
- 引入 **Thinking Preservation**：新选项保留历史消息中的推理上下文（`preserve_thinking`），默认只保留最新一条用户消息的思考块（interleaved thinking 模式）。
- 架构数字：与 3.5-27B 完全一致。

### Qwen3.6 → Qwen3.8（能力代际跃升 + 生态扩展）

- 官方定位：明确写着"**Built on the architectural foundation of Qwen3.5**"，是当时"最强大"的开源代际。
- 能力（相对 Qwen3.6-27B，官方 benchmark）：
  - Terminal-Bench 2.1：63.4 → **73.0**
  - SWE-bench Pro：53.5 → **61.7**
  - DeepSWE 1.1：13.3 → **42.2**
  - QwenSWEBench：49.3 → **79.0**
  - MathVision（With CI）：85.1 → **94.6**
- **Flexible Thinking Control**：thinking 默认开、可按请求关闭；`reasoning_effort` 调节推理深度；`preserve_thinking` 默认**保留全部历史思考块**（与 3.6 的默认策略相反，对长程 agent 任务更有利，也改善 KV cache 复用）。
- 生态：官方 FP8 量化版（面向 vLLM 与 RTX 50 系 / Hopper 的 FP8 内核）；新增 TokenSpeed 引擎支持；云服务默认 1M 上下文。
- 架构数字：与 3.5/3.6-27B 完全一致。

---

## 四、对源码研读 / debug 的启示

1. **一份源码打三代**：三者 `model_type` 均为 `qwen3_5`，在 `E:\transformers\src\transformers\models\qwen3_5*.py` 同一套 modeling 文件内可加载，config 差异极小——打断点的路径完全相同。
2. **重点调试对象**：
   - Gated DeltaNet 的递归状态更新（`delta_state` 形状 128，`time_step_rank` 48）——与 Gemma 的 MHA 差异最大；
   - Gated Attention 的 GQA（24Q/4KV）+ RoPE 64；
   - MTP 头（多 token 预测分支）与投机解码接口；
   - Vision Encoder 的多模态 token 融合（early fusion）。
3. **注意**：27B 是 **Dense**（无 MoE）；同系列 MoE 路径（256 专家路由）需看 35B-A3B 版本，源码入口相同但多了专家路由分支。
4. **量化路径**：三代 BF16 权重（55GB）本地双 16GB 卡装不下；要本地跑，3.8 用官方 FP8（~29GB，仍需双卡+短上下文）或社区 GGUF 量化；3.5/3.6 无官方量化，需社区 GGUF/AWQ。

---

## 参考来源

- Qwen/Qwen3.5-27B model card（Hugging Face）：https://huggingface.co/Qwen/Qwen3.5-27B
- Qwen/Qwen3.6-27B model card（ModelScope）：https://modelscope.cn/models/qwen/Qwen3.6-27B
- Qwen/Qwen3.8-27B model card（Hugging Face）：https://huggingface.co/Qwen/Qwen3.8-27B
- Qwen3.8-27B 部署资料（NVIDIA NGC FP8 校验 28.77GB）：https://catalog.ngc.nvidia.com/orgs/nim/qwen/models/qwen3.8-27b/hf-017b9c7-fp8-nim-checksum/version-history
