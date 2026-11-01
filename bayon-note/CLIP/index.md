# CLIP：双塔对比学习奠基之作（Contrastive Language-Image Pre-training）

> 研读对象：`E:\transformers\src\transformers\models\clip\`
> transformers 版本：5.23.0.dev0（本笔记所有行号/字段均以此为准）
> 结论口径：源码逐字段核验 + 官方论文（arXiv:2103.00020）+ HF 模型卡交叉验证；标注「待确认」处未编造

---

## 1. 一句话定位

**CLIP（Contrastive Language-Image Pre-training）** 是 OpenAI 2021 年提出的双塔视觉语言模型（论文 arXiv:2103.00020），核心主张：**图像和文本各自编码到同一个向量空间，用对比学习对齐**，以 4 亿（400M）网络图文对为监督，训练出可**零样本迁移**的视觉模型——零样本分类、图文检索、图文相似度的标准范式。现代多模态 LLM 的视觉塔大多直接或间接继承自 CLIP。

---

## 2. 源码模块与类关系（先看清组织）

| 文件 | 职责 |
|---|---|
| `configuration_clip.py` | `CLIPTextConfig` / `CLIPVisionConfig` / `CLIPConfig` |
| `modeling_clip.py` | 全部模型类（文本塔 / 视觉塔 / 双塔 / 任务头） |
| `tokenization_clip.py` | CLIP 专用 BPE 分词器（49408 词表） |
| `processing_clip.py` | Processor（分词 + 图像预处理合一） |
| `image_processing_clip.py` / `image_processing_pil_clip.py` | 图像 resize → center crop → 归一化 |
| `convert_clip_original_pytorch_to_hf.py` | OpenAI 官方权重 → HF 格式 |

`modeling_clip.py` 类清单：

| 类 | 行号 | 职责 |
|---|---|---|
| `CLIPVisionEmbeddings` | :138 | patch 化嵌入 + CLS + 位置编码 |
| `CLIPTextEmbeddings` | :221 | token + 位置编码（因果注意力的文本嵌入） |
| `CLIPAttention` / `CLIPMLP` | :280 / :338 | 注意力 / FFN（激活 `quick_gelu`） |
| `CLIPEncoderLayer` | :353 | 单层 Transformer（继承 `GradientCheckpointingLayer`） |
| `CLIPEncoder` | :446 | 12 层编码器 |
| `CLIPTextModel` | :485 | 文本塔（池化取 **[EOS]** 状态） |
| `CLIPVisionModel` | :585 | 视觉塔（池化取 **[CLS]** 状态） |
| `CLIPModel` | :651 | 双塔完整模型（对比损失 + 相似度） |
| `CLIPTextModelWithProjection` | :826 | 文本塔 + 投影层（检索用） |
| `CLIPVisionModelWithProjection` | :890 | 视觉塔 + 投影层（检索用） |
| `CLIPForImageClassification` | :959 | 分类头（HF 为有监督微调加的线性头，非原论文结构） |

---

## 3. 配置字段

### 3.1 文本塔（CLIPTextConfig，configuration_clip.py:27）

| 字段 | 默认值 | 说明 |
|---|---|---|
| model_type | `clip_text_model` | |
| vocab_size | **49408** | BPE 词表（含 startoftext/endoftext 等特殊 token） |
| hidden_size | 512 | |
| intermediate_size | 2048 | |
| projection_dim | 512 | 投影目标维度 |
| num_hidden_layers | 12 | |
| num_attention_heads | 8 | |
| max_position_embeddings | **77** | 文本最长 77 token |
| hidden_act | **quick_gelu** | CLIP 特有激活 |
| bos / eos / pad | 49406 / 49407 / 1 | |

### 3.2 视觉塔（CLIPVisionConfig，:77）

| 字段 | 默认值 | 说明 |
|---|---|---|
| model_type | `clip_vision_model` | |
| hidden_size | 768 | |
| intermediate_size | 3072 | |
| projection_dim | 512 | |
| num_hidden_layers | 12 | |
| num_attention_heads | 12 | |
| image_size | 224 | 输入分辨率 |
| patch_size | 32 | → 7×7 = 49 patch + CLS = 50 token |
| hidden_act | quick_gelu | |

### 3.3 双塔（CLIPConfig，:122）

| 字段 | 默认值 | 说明 |
|---|---|---|
| text_config / vision_config | 子配置 | 可分别覆盖 |
| **logit_scale_init_value** | **2.6592** | 可学习温度初始值 ≈ ln(1/0.07) |

---

## 4. 核心机制（对照代码）

### 4.1 双塔各自编码

- **视觉塔**（`CLIPVisionModel`，:585）：`CLIPVisionEmbeddings` 把 224×224 切成 32×32 patch（49 个）拼上 CLS token → 12 层双向 Transformer → 池化取 **[CLS]** 状态（:630 docstring「pooled CLS states」）。
- **文本塔**（`CLIPTextModel`，:485）：BPE 分词 → 位置编码 → 12 层**因果** Transformer（GPT-2 式）→ 池化取 **[EOS]**（最后一个 token）状态（:524 docstring「pooled (EOS token) states」）。

### 4.2 投影与相似度（CLIPModel，:651）

```
image_embeds = visual_projection(pooler_output)   # 768 → 512
text_embeds  = text_projection(pooler_output)     # 512 → 512
两者 L2 归一化 → 余弦相似度矩阵 × logit_scale
```

- `logit_scale` 是**可学习参数**，初始 `2.6592 ≈ ln(1/0.07)`（对比 OpenAI 原实现）。

### 4.3 对比损失（InfoNCE）

对 batch 内每张图：拉近正样本对 $(I_i, T_i)$，推远 batch 内其余文本 $(I_i, T_j)$ 作为负样本；对称地也以文本为锚。**不做显式 hard-negative 挖掘**，靠大 batch 提供负样本。

### 4.4 图像预处理

`image_processing_clip.py`：`resize`（保持比例）→ `center_crop(224)` → ImageNet 均值/方差归一化——与训练口径严格一致。

---

## 5. 训练方法（论文层面）

- **数据**：4 亿（400M）从互联网收集的图文对，天然带噪声但规模巨大；
- **目标**：对称对比损失（InfoNCE），batch 极大（论文以 32768 级 batch 训练）；
- **可学习温度**：`logit_scale` 随训练调整，避免人为设定 softmax 温度；
- **零样本迁移**：分类时把类别名拼成「a photo of a {label}」模板输入文本塔，取与图像特征相似度最高者。

---

## 6. 模型家族速览

| 变体 | 视觉塔 | 说明 |
|---|---|---|
| `openai/clip-vit-base-patch32` | ViT-B/32 | 默认配置口径 |
| `openai/clip-vit-base-patch16` | ViT-B/16 | |
| `openai/clip-vit-large-patch14` | ViT-L/14 | |
| `openai/clip-vit-large-patch14-336` | ViT-L/14@336 | 336 分辨率微调 |
| `openai/clip-vit-large-patch14-336` 系列之外 | RN50 / RN101 / RN50x4 / RN50x16 / RN50x64 | ResNet 视觉塔变体（EfficientNet 缩放） |

---

## 7. 与多模态 LLM 的关系（研读背景）

- **标准视觉塔**：LLaVA、BLIP-2、InstructBLIP、MiniGPT-4 等第一代多模态 LLM 直接借用 CLIP 视觉塔（通常是 ViT-L/14@336）作为视觉编码器。
- **损失演进**：SigLIP（同仓库 `models/siglip/`）用逐对 sigmoid 损失替代 CLIP 的 softmax 对比损失，摆脱 batch 内负样本耦合；BLIP 的 ITC 目标即 CLIP 式对比。
- **同族对比**：Qwen2-VL / InternVL 等自研视觉塔虽非 CLIP 实现，但同属「图文对比预训练视觉编码器」谱系，CLIP 是这条线的原点。

---

## 8. 一分钟结论

| 维度 | 要点 |
|---|---|
| 架构 | 视觉塔 ViT-B/32（12L/768/12H）+ 文本塔 Transformer（12L/512/8H）双塔 |
| 对齐 | 投影到 512 维公共空间，L2 归一化 + 余弦相似度 × logit_scale |
| 损失 | 对称 InfoNCE 对比损失，正对拉近 / 负对推远 |
| 池化 | 视觉取 [CLS]，文本取 [EOS] |
| 预训练数据 | 4 亿网络图文对 |
| 代码入口 | `CLIPModel` / `CLIPTextModel` / `CLIPVisionModel` / `*WithProjection` / `CLIPForImageClassification` |

---

## 9. 参考来源

- 论文：Learning Transferable Visual Models From Natural Language Supervision，arXiv:2103.00020（Radford et al., 2021, PMLR）
- HF 模型卡：`openai/clip-vit-base-patch32` 等
- transformers 源码（本机 editable）：`src/transformers/models/clip/`

**待确认/边界**：论文训练超参（batch、epochs）与各变体精确参数以论文/模型卡为准；`CLIPForImageClassification` 是 HF 为有监督微调额外加的线性头，不属于原论文结构。
