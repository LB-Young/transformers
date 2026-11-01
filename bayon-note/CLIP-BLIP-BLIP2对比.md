# CLIP vs BLIP vs BLIP-2：多模态模型三代演进横向对比

> 研读对象：`E:\transformers\src\transformers\models\clip\`、`...\models\blip\`、`...\models\blip_2\`
> transformers 版本：5.23.0.dev0（本笔记所有行号/字段均以此为准）
> 结论口径：源码逐字段核验 + 三篇官方论文（arXiv:2103.00020 / 2201.12086 / 2301.12597）+ HF 模型卡交叉验证；标注「待确认」处未编造
> 配套笔记：`bayon-note/CLIP/index.md`、`bayon-note/BLIP/index.md`

---

## 1. 为什么把这三个放在一起比

CLIP → BLIP → BLIP-2 是**视觉语言模型一条清晰的主线**，每一代解决上一代的关键瓶颈：

| 代际 | 模型 | 解决了什么 | 遗留问题 |
|---|---|---|---|
| 1 | **CLIP**（OpenAI 2021） | 用对比学习对齐图文，零样本分类 | 只做「对齐」，不能生成；推理要双塔全跑 |
| 2 | **BLIP**（Salesforce 2022） | 同一套参数理解+生成通吃，CapFilt 治噪声数据 | 视觉塔与文本塔要端到端联合训练，难复用大语言模型 |
| 3 | **BLIP-2**（Salesforce 2023） | 用 Q-Former 桥接**冻结**视觉塔与**冻结** LLM，低成本复用超大规模单模态模型 | 需要两阶段训练；Q-Former 是额外瓶颈 |

一句话：**CLIP 定范式（对比对齐）→ BLIP 加能力（理解+生成+数据引导）→ BLIP-2 换打法（冻结桥接，白嫖大模型）**。

---

## 2. 架构对比

```
CLIP                      BLIP                            BLIP-2
┌─────────┐ ┌─────────┐   ┌─────────┐ ┌──────────────┐     ┌────────────┐   ┌────────────────┐
│ 视觉塔   │ │ 文本塔   │   │ 视觉塔   │ │ 文本 Transformer│     │ 视觉塔(冻结) │   │ Q-Former(可训)  │
│ ViT-B/32│ │ GPT-2式 │   │ ViT-B   │ │ is_decoder    │     │ ViT-L/14   │   │ 12L/768/12H    │
│ 池化CLS │ │ 池化EOS │   │ 池化CLS │ │ 编码/解码/跨注意力│     │ (源自CLIP权 │   │ +32个query token│
└────┬────┘ └────┬────┘   └────┬────┘ └──────┬───────┘     │  重,冻结)   │   └───────┬────────┘
     │     I     │ T           │             │             └─────┬──────┘           │
     └─────┬─────┘             │             ▼                   ▼                  ▼
           ▼ 对比               └── ITC对比·ITM匹配·ITG生成 ──┐   语言投影(Linear)     │
    InfoNCE 拉近正对                     (MED 三模式)        │        │              │
                                                           │        ▼              │
                                                           │   ┌────────────┐       │
                                                           └──▶│ LLM(冻结)   │◀──────┘
                                                               │ OPT / FlanT5│ 生成
                                                               └────────────┘
```

| 维度 | CLIP | BLIP | BLIP-2 |
|---|---|---|---|
| 整体结构 | 视觉塔 + 文本塔（双塔） | 视觉塔 + 文本 Transformer（MED） | 冻结视觉塔 + Q-Former + 冻结 LLM |
| 视觉塔 | ViT-B/32（12L/768/12H） | ViT-Base（12L/768/12H） | **ViT-L/14（39L/1408/16H，默认配置）** |
| 文本侧 | 12L/512/8H（因果，GPT-2 式） | 12L/768/8H（is_decoder 可切换） | Q-Former 12L/768/12H + 外部 LLM |
| 桥接机制 | 无（直接余弦比对） | 无（跨注意力直接融合） | **Q-Former 的 32 个 query token + 线性投影** |
| 特殊 token | 视觉 [CLS] / 文本 [EOS] | 视觉 [CLS] | Q-Former query tokens（可学习） |
| 是否生成文本 | ❌ | ✅（ITG 解码器） | ✅（LLM 自回归/seq2seq） |
| 冻结策略 | 全量可训 | 全量可训 | **视觉塔 + LLM 冻结**（官方检查点口径，见 §4） |

---

## 3. 核心配置对比（各 configuration 默认值）

| 字段 | CLIPConfig | BlipConfig | Blip2Config |
|---|---|---|---|
| 视觉 image_size | 224 | 384 | **224** |
| 视觉 patch_size | 32 | 16 | **14** |
| 视觉 hidden / 层 / 头 | 768 / 12 / 12 | 768 / 12 / 12 | **1408 / 39 / 16** |
| 视觉激活 | quick_gelu | gelu | gelu |
| 文本/Q-Former hidden / 层 / 头 | 512 / 12 / 8 | 768 / 12 / 8 | **768 / 12 / 12** |
| 词表 | 49408（≤77 token） | 30524（≤512 token） | Q-Former 30522（≤512 token） |
| 投影维度 | 512（双侧） | 视觉 512 / 文本 768 | 768 → LLM hidden（`language_projection`） |
| logit_scale 初始 | 2.6592 | 2.6592 | 阶段一 ITC 沿用（源码无独立字段） |
| 融合层 | — | 256（ITM 头） | 256（ITM 头，`image_text_hidden_size`） |
| 特有字段 | — | — | **num_query_tokens=32**、cross_attention_frequency=2、use_decoder_only_language_model 自动判定 |

> BLIP-2 的 Q-Former 配置（configuration_blip_2.py:64）：`cross_attention_frequency=2` 表示**每 2 层插入一次跨注意力**（不是每层都做），这是 Q-Former 控制计算量的关键；`encoder_hidden_size` 在 `Blip2Config.__post_init__` 里自动 = 视觉塔 hidden（1408）。
> BLIP-2 的 LLM 默认是 **OPT**（`sub_configs_defaults` 中 `text_config` 默认 `model_type="opt"`），可换成任何因果/seq2seq 配置；FlanT5 变体靠 checkpoint 配置驱动。

---

## 4. 训练方法对比

| | CLIP | BLIP | BLIP-2 |
|---|---|---|---|
| 阶段数 | 单阶段 | 单阶段 | **两阶段** |
| 目标 | 对称 InfoNCE | ITC + ITM + ITG | 阶段一：Q-Former 三目标（同 BLIP）；阶段二：LLM 生成 LM loss |
| 冻结 | 无 | 无 | 阶段一冻视觉塔；阶段二冻视觉塔 + LLM（只训 Q-Former + 投影） |
| 数据 | 4 亿网络图文对 | 14M 级标注 + 网络图（CapFilt 增强） | 阶段一图文数据；阶段二需要「图像-文本」语料（部分来自 CapFilt 思路） |
| 关键技巧 | 可学习温度 logit_scale | CapFilt 数据引导、MED 参数共享 | **冻结复用**：视觉塔白嫖 CLIP/SigLIP 的 ViT-L/14 权重，LLM 白嫖 OPT/FlanT5 |

**BLIP-2 两阶段细节**（源码视角：`modeling_blip_2.py`）
- `Blip2ForConditionalGeneration`（:1460）组成：`vision_model` + `query_tokens`（1×32×768）+ `qformer` + `language_projection`（768→text hidden）+ `language_model`（:1471-1482）。
- LLM 自动选择：`config.use_decoder_only_language_model` 为真时用 `AutoModelForCausalLM`（OPT），否则 `AutoModelForSeq2SeqLM`（FlanT5）（:1477-1480）。
- **注意**：本仓库 modeling 层**没有代码级冻结**（无 `requires_grad_(False)`），「视觉塔/LLM 冻结」是**官方检查点的参数状态**（发布权重即冻结）；若要复现论文两阶段训练需自行按阶段冻结。

---

## 5. 任务与代码入口对比

| 任务 | CLIP | BLIP | BLIP-2 |
|---|---|---|---|
| 零样本分类 / 图文相似度 | ✅ `CLIPModel` / `*WithProjection` | ❌ | ❌ |
| 图文检索 | ✅ | ✅ `BlipModel` / `BlipForImageTextRetrieval` | ✅ `Blip2ForImageTextRetrieval`（:1832，opt 变体） |
| 图像描述生成 | ❌ | ✅ `BlipForConditionalGeneration` | ✅ `Blip2ForConditionalGeneration`（:1460，FlanT5/OPT） |
| VQA | ❌ | ✅ `BlipForQuestionAnswering` | ❌（本模块无 QA 类；由 `InstructBLIP` 承接） |
| 视觉特征 backbone | ✅ `CLIPVisionModel` | ✅ `BlipVisionModel` | ✅ `Blip2VisionModel`（:481，ViT-L/14 口径） |
| 桥接特征 | — | — | ✅ `Blip2QFormerModel`（:848）/ `Blip2Model`（:957） |
| 文本/视觉投影检索 | ✅ `CLIPTextModelWithProjection` 等 | — | ✅ `Blip2TextModelWithProjection`（:1274）/ `Blip2VisionModelWithProjection`（:1359） |

> VQA 说明：早期 transformers 版本有 `Blip2ForQuestionAnswering`，**当前 5.23.0.dev0 已移除**，VQA 任务由 `models/instructblip/` 模块（InstructBLIP）承担（待确认：移除具体版本号）。

---

## 6. 关系网络与演进（研读背景）

- **BLIP-2 的视觉塔 = CLIP/SigLIP 的 ViT-L/14**：阶段一直接以冻结的 CLIP（或 SigLIP）视觉塔为输入，不重新训练视觉特征——「对比预训练视觉塔」成为可复用资产。
- **BLIP-2 阶段一 = BLIP 的三目标**：ITC/ITM/ITG 原样继承，只是把「文本 Transformer」换成「Q-Former」，并冻住视觉塔。
- **BLIP-2 阶段二 = ITG 的规模化升级**：BLIP 的 ITG 是自家 12 层解码器生成；BLIP-2 换成冻结的 OPT（几十亿参）或 FlanT5，生成能力上了一个量级。
- **InstructBLIP**：在 BLIP-2 基础上做指令微调，承接 VQA/对话；当前仓库 `models/instructblip/` 独立成模块。
- **SigLIP**：用 sigmoid 损失改进 CLIP（同仓库 `models/siglip/`），已被 BLIP-2 后续变体选作视觉塔（详见 `bayon-note/Siglip/index.md`）。

---

## 7. 一分钟结论（选型速查）

| 你的需求 | 首选 | 备选 | 理由 |
|---|---|---|---|
| 零样本分类 / 图文相似度 | **CLIP**（或 SigLIP） | — | 双塔对比最直接、推理轻 |
| 轻量统一理解+生成、可全量微调 | **BLIP** | — | 模型小（~2 亿级）、三任务通吃 |
| 高质量生成、GPU 有限、只想训桥接层 | **BLIP-2** | InstructBLIP | 冻结大模型，只训 Q-Former + 投影 |
| 给多模态 LLM 当视觉塔 | CLIP ViT-L/14 | SigLIP | 生态最成熟；BLIP-2 默认就是它 |

---

## 8. 参考来源

- CLIP：Learning Transferable Visual Models From Natural Language Supervision，arXiv:2103.00020（Radford et al., 2021, PMLR）
- BLIP：BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation，arXiv:2201.12086（Li et al., 2022, ICML）
- BLIP-2：BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models，arXiv:2301.12597（Li et al., 2023, ICML）
- transformers 源码（本机 editable）：`src/transformers/models/clip/`、`blip/`、`blip_2/`、`siglip/`、`instructblip/`

**待确认/边界**：三个模型预训练数据精确规模与超参以各自论文/模型卡为准（本机未复现训练）；「冻结」为官方检查点参数状态而非本仓库代码强制，复现两阶段训练需自行处理；`Blip2ForQuestionAnswering` 的移除版本号待确认；对比表中的「选型」为研读性建议，实际选择需结合任务指标与部署成本。
