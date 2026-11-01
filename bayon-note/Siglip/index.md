# SigLIP vs SigLIP 2：Google 视觉语言编码器两代架构详解

> 研读对象：`E:\transformers\src\transformers\models\siglip\`（v1）与 `E:\transformers\src\transformers\models\siglip2\`（v2）
> transformers 版本：5.21.0.dev0（本笔记所有行号/字段均以此为准）
> 结论口径：源码逐字段核验 + 官方论文（arXiv:2303.15343 / arXiv:2502.14786）+ 官方模型卡交叉验证；标注「待确认」处未编造

---

## 1. 一句话定位

**SigLIP（Sigmoid Loss for Language Image Pre-Training）** 是 Google 提出的视觉语言双塔编码器，核心是用**逐对 sigmoid 损失**替代 CLIP 的 softmax 对比损失，训练更稳、收敛更快，视觉塔被大量多模态 LLM 直接借用。

- **SigLIP 1**（2023，arXiv:2303.15343）：固定分辨率（默认 224×224 → 14×14=196 patch），learned 位置嵌入。
- **SigLIP 2**（2025，arXiv:2502.14786）：**NaFlex 原生灵活分辨率**（单 checkpoint 支持任意宽高比/分辨率，≤256 patch），训练目标叠加 decoder loss + global-local / masked prediction，多语言 + 去偏数据。

---

## 2. 源码模块与类关系（先看清组织）

两个版本是**独立模块**，但 v2 的类全部继承 v1：

| 模块 | 视觉类 | 文本类 | 双塔类 | 分类头 |
|---|---|---|---|---|
| `models/siglip/` | `SiglipVisionModel`（modeling_siglip.py:553） | `SiglipTextModel`（:471） | `SiglipModel`（:647） | `SiglipForImageClassification`（:835） |
| `models/siglip2/` | `Siglip2VisionModel`（modeling_siglip2.py:499，**继承 SiglipVisionModel**） | `Siglip2TextModel`（:587，**继承 SiglipTextModel**） | `Siglip2Model`（:712，**继承 SiglipModel**） | `Siglip2ForImageClassification`（:915） |

- `modeling_siglip2.py` 由 `modular_siglip2.py` 自动生成（文件头有警示，改动要改 modular）。
- 文本侧 v2 与 v1 **结构完全一致**（继承、无重写）——差异全部集中在视觉塔。

---

## 3. 配置字段对比（configuration_siglip.py vs configuration_siglip2.py）

### 3.1 文本侧：完全相同

| 字段 | SiglipTextConfig | Siglip2TextConfig |
|---|---|---|
| model_type | `siglip_text_model` | `siglip2_text_model` |
| vocab_size | 32000 | 32000 |
| hidden_size | 768 | 768 |
| intermediate_size | 3072 | 3072 |
| num_hidden_layers | 12 | 12 |
| num_attention_heads | 12 | 12 |
| max_position_embeddings | 64 | 64 |
| hidden_act | gelu_pytorch_tanh | gelu_pytorch_tanh |
| projection_size | 默认 = hidden_size | 默认 = hidden_size |

### 3.2 视觉侧：**关键分界点**

| 字段 | SiglipVisionConfig | Siglip2VisionConfig |
|---|---|---|
| model_type | `siglip_vision_model` | `siglip2_vision_model` |
| hidden_size | 768 | 768 |
| intermediate_size | 3072 | 3072 |
| num_hidden_layers | 12 | 12 |
| num_attention_heads | 12 | 12 |
| patch_size | 16 | 16 |
| **image_size** | **224（固定，可配 list）** | **❌ 无此字段** |
| **num_patches** | **❌ 无此字段** | **256（NaFlex：patch 数上限）** |

> 一句话：v1 的视觉配置靠 `image_size` 定死输入；v2 删掉 `image_size`，改为「最多 256 个 patch」的容量上限——这是 NaFlex 的配置层体现。

---

## 4. 视觉塔实现对比（核心机制）

### 4.1 位置嵌入：固定扁平 vs 16×16 网格插值

**SigLIP 1**（`SiglipVisionEmbeddings`，modeling_siglip.py:116）
- `patch_embedding = nn.Linear(3×16×16, 768)`
- `position_embedding = nn.Embedding(196, 768)`（224/16=14 → 14×14=196，**扁平索引**）
- forward：`embeddings = inputs_embeds + position_embeddings`（纯加性，位置网格固定）

**SigLIP 2**（`Siglip2VisionEmbeddings`，modeling_siglip2.py:114）
- `patch_embedding` 同上
- `position_embedding = nn.Embedding(256, 768)` —— 把它看作 **16×16 的 2D 位置网格**（`position_embedding_size = sqrt(256) = 16`）
- forward 时调用 `resize_positional_embeddings`（:130）：
  1. 按该图实际 patch 网格 `(height, width)`，用 **双线性插值**（bilinear + antialias）把 16×16 网格缩放到目标尺寸（保持宽高比）；
  2. 不足 `max_length=256` 的部分，**用插值结果的第一行 `resized_embeddings[0]` 复制填充**（不是零填充）；
  3. 最终输出 `(batch, 256, 768)` 固定序列。

### 4.2 forward 签名与 padding 掩码

| | SiglipVisionModel.forward | Siglip2VisionModel.forward |
|---|---|---|
| 输入 | `pixel_values`（固定 (B,3,H,W)） | `pixel_values`（**已 patchify**，(B,256,…)）+ `pixel_attention_mask` + `spatial_shapes` |
| 掩码 | 无需 | `create_bidirectional_mask` 按像素掩码遮掉 padding patch |
| 输出 | last_hidden_state + pooler_output | 同左 |

### 4.3 共享结构（两版相同）

- 双向注意力 Encoder（`SiglipEncoderLayer` / `Siglip2EncoderLayer`，均 12 层）
- `post_layernorm`
- **MHA pooling head**：两版都有 `SiglipMultiheadAttentionPoolingHead` / `Siglip2MultiheadAttentionPoolingHead`（`use_head` 可关，`vision_use_head` 控制），视觉塔输出用多头注意力池化聚合，而非 CLIP 式的取 CLS。

---

## 5. 训练方法对比（论文层面）

### SigLIP 1（arXiv:2303.15343）
- **逐对 sigmoid 损失**：对 batch 内 (图像, 文本) 每对独立算 `log(1 + exp(-y·(sim·t + b)))`，**不做 softmax 归一化**——摆脱负样本 pair 之间的耦合，batch 可更大、收敛更快；
- 可学习温度 `t` 初始化为 `log(10)`；
- 可学习偏置 `b` 初始化为 `-10`（负样本对远多于正样本，先压低所有 logit，避免初期大步纠偏）——即 **BiC（Bias Correction）初始化**。

### SigLIP 2（arXiv:2502.14786）
在 SigLIP 对比目标之上叠加：
- **decoder loss**（解码器重建图像 patch，辅助密集特征）；
- **global-local 与 masked prediction loss**（全局-局部一致性 + 掩码预测，提升定位/局部语义，如检测、分割这类任务）；
- **NaFlex 训练**：训练时混用多分辨率/多宽高比，位置嵌入插值；
- **多语言 + 去偏过滤**：WebLI 数据上多语言训练 + debiasing 过滤，跨文化语义更均衡；
- 官方主力变体 **so400m**（如 `google/siglip2-so400m-patch16-naflex`），还有 base/large 系列（224 / 384 / naflex 后缀）。

> 注意：decoder / masked 分支是**训练期目标**，HF 发布的推理模型只有「文本塔 + 视觉塔 + 对比头」结构，不含这些训练分支。

---

## 6. 模型家族速览

| 版本 | 官方检查点示例 | 配置要点 |
|---|---|---|
| SigLIP 1 | `google/siglip-base-patch16-224`、`google/siglip-so400m-patch14-384` | 固定分辨率 224/384，patch 16/14 |
| SigLIP 2 | `google/siglip2-base-patch16-naflex`、`google/siglip2-so400m-patch16-naflex`、`google/siglip2-base-patch16-224/384` | naflex = NaFlex 变体；非 naflex = 固定分辨率变体 |

---

## 7. 与多模态 LLM 视觉塔的关系（研读背景）

- **Gemma 3**：视觉编码器即 **SigLIP-2** 变体（400M 级），Gemma 3 把视觉 token 经投影并入语言模型。
- **InternVL**：视觉塔 `InternViT` 与 SigLIP 同源思路（patch 化 ViT + 双塔预训练），SigLIP 2 的 MHA pooling 与 dense/局部特征思路被后续视觉塔沿袭。
- **Qwen2-VL / Qwen3-VL**：视觉塔为自研 ViT（patch 16 + 2D-RoPE 位置编码 + RMSNorm），**不是** SigLIP 实现，但同属「patch 化视觉编码器 + 可变分辨率」家族；其可变分辨率（任意宽高比、位置编码插值）与 SigLIP 2 的 NaFlex 思路一致。SigLIP 2 的 NaFlex 与 Qwen 视觉塔的 2D 位置编码插值互为印证（都是让 ViT 摆脱固定分辨率）。
- 后续工作：SigLIP-HD（用 SigLIP 2 多尺度教师蒸馏标准分辨率学生）等。

---

## 8. 一分钟结论

| 维度 | SigLIP 1 | SigLIP 2 |
|---|---|---|
| 分辨率 | 固定（224/384） | **NaFlex 灵活（≤256 patch，保宽高比）** |
| 位置嵌入 | 固定扁平 learned（加性） | **16×16 网格 learned + 双线性插值 + 边界复制 pad** |
| 输入 | 原始图 (B,3,H,W) | patchify 序列 + pixel mask + spatial_shapes |
| 训练目标 | sigmoid 对比 + t/b 初始化 | + decoder loss + masked prediction + 多语言去偏 |
| 文本塔 | 12L/768/12H | 完全相同（继承） |
| 输出池化 | MHA pooling head | MHA pooling head（同） |

---

## 9. 参考来源

- SigLIP 论文：Sigmoid Loss for Language Image Pre-Training，arXiv:2303.15343（Zhai et al., 2023）
- SigLIP 2 论文：SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features，arXiv:2502.14786（Tschannen et al., 2025）
- HF 模型卡：`google/siglip2-so400m-patch16-naflex`（训练目标/数据说明）
- transformers 源码（本机 editable）：`src/transformers/models/siglip/`、`src/transformers/models/siglip2/`

**待确认/边界**：SigLIP 2 具体训练分辨率集合与 so400m 精确参数以官方权重为准（未在本机下载验证权重）；SigLIP 2 论文中 decoder 分支的 HF 复现细节未展开。
