# YOLOS vs CLIP vs BLIP：三个视觉语言模型横向对比

> 研读对象：`E:\transformers\src\transformers\models\yolos\`、`...\models\clip\`、`...\models\blip\`
> transformers 版本：5.23.0.dev0（本笔记所有行号/字段均以此为准）
> 结论口径：源码逐字段核验 + 三篇官方论文（arXiv:2106.00666 / 2103.00020 / 2201.12086）+ HF 模型卡交叉验证；标注「待确认」处未编造
> 配套笔记：`bayon-note/YOLOS/index.md`、`bayon-note/CLIP/index.md`、`bayon-note/BLIP/index.md`

---

## 1. 为什么把这三个放在一起比

它们都躺在 `src/transformers/models/` 下、都处理「视觉 + 语言（或视觉序列）」问题，但代表了**三条完全不同的设计路线**：

| | YOLOS | CLIP | BLIP |
|---|---|---|---|
| 领域 | **目标检测** | **图文对齐** | **统一理解 + 生成** |
| 处理文本？ | ❌ 纯视觉 | ✅ 文本塔 | ✅ 文本塔（编/解码） |
| 一句话 | 纯 ViT 就能做检测 | 双塔对比，零样本分类 | 同一套参数通吃检索/字幕/VQA |

三者不是替代关系，而是「检测 / 对齐 / 生成」三个方向的代表；把它们的差异看清楚，就理解了视觉语言模型设计空间的大半。

---

## 2. 架构对比

```
YOLOS                 CLIP                          BLIP
┌──────────┐          ┌──────────┐ ┌──────────┐     ┌──────────┐  ┌───────────────┐
│ 图像      │          │ 图像塔    │ │ 文本塔    │     │ 视觉塔    │  │ 文本 Transformer│
│ patch 化  │          │ ViT-B/32 │ │ GPT-2式  │     │ ViT-B    │  │ is_decoder 可切换│
│ 1728     │          │ 池化 CLS │ │ 池化 EOS │     │ 池化 CLS │  │ 编码/解码/跨注意力│
└────┬─────┘          └────┬─────┘ └────┬─────┘     └────┬─────┘  └───────┬───────┘
     │  +CLS +100 det      │   I(512)   T(512)           │                │
     ▼                     └─────┬──────┘                ▼                ▼
┌─────────────┐                 ▼ 对比损失           ┌────────────────────────────┐
│ ViT Encoder │          ┌──────────────┐           │ ITC 对比 · ITM 匹配 · ITG 生成 │
│ 12 层       │          │ InfoNCE 拉近  │           └────────────────────────────┘
└─────┬───────┘          └──────────────┘
      ▼ 100 个 det token → 分类 + 框回归
```

| 维度 | YOLOS | CLIP | BLIP |
|---|---|---|---|
| 整体结构 | 单 ViT encoder（纯视觉） | 视觉塔 + 文本塔（双塔） | 视觉塔 + 文本 Transformer（MED 三模式） |
| 视觉塔 | ViT-Base（12L/768/12H） | ViT-B/32（12L/768/12H） | ViT-Base（12L/768/12H） |
| 文本塔 | 无 | 12L/512/8H（因果，GPT-2 式） | 12L/768/8H（is_decoder 可切换） |
| 特殊 token | [CLS] + **100 个 detection token** | 视觉 [CLS] / 文本 [EOS] | 视觉 [CLS] |
| 池化方式 | 不用池化（直接取末尾 det token） | 视觉取 CLS / 文本取 EOS | 视觉取 CLS + post_layernorm |
| 位置编码 | 可插值 + 中间层再注入 | 固定 learned（224 输入） | 固定 learned（384 输入） |
| 激活 | gelu | **quick_gelu** | gelu |

---

## 3. 核心配置对比（各 configuration 默认值）

| 字段 | YolosConfig | CLIPConfig | BlipConfig |
|---|---|---|---|
| image_size | (512, 864) | 224 | 384 |
| patch_size | 16 | 32 | 16 |
| 视觉 hidden / 层 / 头 | 768 / 12 / 12 | 768 / 12 / 12 | 768 / 12 / 12 |
| 文本 hidden / 层 / 头 | — | 512 / 12 / 8 | 768 / 12 / 8 |
| 文本词表 | — | 49408（≤77 token） | 30524（≤512 token） |
| 投影维度 | — | 512（双侧） | 视觉 512 / 文本 768 |
| logit_scale 初始 | — | 2.6592 | 2.6592 |
| 融合层 | — | — | 256（ITM 头） |
| 检测相关 | 100 det tokens；损失 1/5/2，eos 0.1 | — | — |

> 共同点：视觉塔都是「ViT-Base 规模 + patch 化 + gelu 家族激活」；CLIP 与 BLIP 的 ITC 共享同一 logit_scale 口径（≈ ln(1/0.07)）。

---

## 4. 训练目标与损失对比

| | YOLOS | CLIP | BLIP |
|---|---|---|---|
| 目标 | 集合式检测 | 对比对齐 | 对比 + 匹配 + 生成 |
| 损失 | 匈牙利匹配 + CE + L1 + GIoU（1/5/2，eos 0.1） | InfoNCE（对称） | ITC（InfoNCE）+ ITM（二分类）+ ITG（LM） |
| 负样本 | 匹配时隐式 | batch 内负样本 | ITC 同 CLIP；ITM 用负图文对 |
| 数据 | COCO 类检测数据（人工标注） | 4 亿网络图文对 | 14M 级标注 + 网络图（CapFilt 增强） |
| 辅助技巧 | 逐层辅助损失（可选）、中间层位置编码 | 可学习温度 logit_scale | CapFilt 数据引导、label_smoothing 可选 |

---

## 5. 任务与代码入口对比

| 任务 | YOLOS | CLIP | BLIP |
|---|---|---|---|
| 目标检测 | ✅ `YolosForObjectDetection` | ❌ | ❌ |
| 零样本分类 | ❌ | ✅ `CLIPModel`（特征比对） | ❌（可微调） |
| 图文检索/相似度 | ❌ | ✅ `CLIPModel` / `*WithProjection` | ✅ `BlipModel` / `BlipForImageTextRetrieval` |
| 图像描述生成 | ❌ | ❌ | ✅ `BlipForConditionalGeneration` |
| VQA | ❌ | ❌ | ✅ `BlipForQuestionAnswering` |
| backbone 复用 | ✅ `YolosModel` | ✅ `CLIPVisionModel`（被 LLaVA 等借用） | ✅ `BlipVisionModel` |
| 分类微调 | ❌ | ✅ `CLIPForImageClassification`（HF 加的线性头） | ❌ |

---

## 6. 关系网络与演进（研读背景）

- **CLIP 是 BLIP 的零件**：BLIP 的 ITC 目标就是 CLIP 式对比学习（连 logit_scale 初始值都相同）；CLIP 的视觉塔被 BLIP-2、LLaVA、InstructBLIP 等直接借用。
- **SigLIP 是对 CLIP 的改进**：同仓库 `models/siglip/` 用逐对 sigmoid 损失替换 softmax 对比，训练更稳（详见 `bayon-note/Siglip/index.md`）。
- **YOLOS 与多模态无关、但同属「ViT 可视化」主线**：它证明纯 ViT 可做检测；其「位置编码随分辨率插值」与 Qwen2-VL、SigLIP-2 NaFlex 的可变分辨率思路同源。
- **BLIP → BLIP-2 → InstructBLIP**：BLIP 是统一图文框架的主线；BLIP-2 用 Q-Former 桥接冻结视觉塔与 LLM（独立模块 `models/blip_2/`）。
- **两代对比的工程启示**：如果只需要「判断图文是否匹配」→ CLIP/SigLIP 够用且轻；需要「看图说话 / 问答」→ BLIP 系列；需要「检测物体位置」→ YOLOS 或 DETR 系列。

---

## 7. 一分钟结论（选型速查）

| 你的需求 | 首选 | 备选 | 理由 |
|---|---|---|---|
| 检测图中物体 + 画框 | **YOLOS** / DETR | — | 唯一带检测头的三兄弟 |
| 零样本分类 / 图文相似度 | **CLIP**（或 SigLIP） | — | 双塔对比最直接，推理轻 |
| 图像描述 / VQA / 统一框架 | **BLIP** | BLIP-2 | 理解与生成通吃 |
| 给多模态 LLM 当视觉塔 | CLIP 视觉塔 | SigLIP | 生态最成熟，HF 权重多 |

---

## 8. 参考来源

- YOLOS：You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection，arXiv:2106.00666（Fang et al., 2021, NeurIPS）
- CLIP：Learning Transferable Visual Models From Natural Language Supervision，arXiv:2103.00020（Radford et al., 2021, PMLR）
- BLIP：BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation，arXiv:2201.12086（Li et al., 2022, ICML）
- transformers 源码（本机 editable）：`src/transformers/models/yolos/`、`clip/`、`blip/`、`siglip/`、`blip_2/`

**待确认/边界**：三个模型的预训练数据精确规模与超参以各自论文/模型卡为准（本机未复现训练）；对比表中的「最佳选型」为研读性建议，实际选择需结合任务指标与部署成本。
