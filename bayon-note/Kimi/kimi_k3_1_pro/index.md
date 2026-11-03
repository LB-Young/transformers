# Kimi K3.1-Pro 家族架构详解

> 官方来源：Moonshot AI 官网首页（https://www.moonshot.ai，2026-11-03 抓取核验，页面展示日期 2026-10-24）；**官方 HF 仓库：无**（hf-mirror 组织列表 / 搜索 0 命中，2026-11-03 核验，组织者已联合核验）；**无官方 config 可留档**（本目录无 `_tools/_cfgpull/config.json`——仓库不存在，拉取无目标）；transformers 模块：无（本地 E:\transformers\src\transformers\models\ 无对应模块，model_type 未知）。公开信息尚少，规格表多数字段标「待确认」，**不臆造任何数字**。

## 1. 家族全景

Kimi K3.1-Pro 是 Moonshot AI 于 **2026-10-24** 在官网首页展示的**最新旗舰**（"Latest Research" 栏首位，继 2026-07-16 Kimi K3 之后）。官网口径：

> "The frontier of intelligence, one step further. A **3-trillion-parameter-class** model with a **1-million-token context window**—built for **long-horizon coding, 3D creation, financial research, and complex agentic tasks**."（来源：moonshot.ai 首页，2026-11-03 抓取）

已知事实仅上述三句：3 万亿参数级、1M 上下文、面向长程 coding / 3D 创作 / 金融研究 / 复杂 agent 任务。是否继承 K3 的 KDA + AttnRes + Stable LatentMoE 架构、激活参数、专家配置、权重格式、协议、API 定价——**全部待确认**（无技术报告、无模型卡、无 config、无官方公告全文可核验）。

### 1.1 已知/待确认规格表

| 字段 | 取值 | 来源 |
|---|---|---|
| 发布日期 | 2026-10-24（官网首页展示） | 官方官网（moonshot.ai，2026-11-03 抓取） |
| 总参规模 | 3 万亿参数级（3-trillion-parameter-class） | 官方官网 |
| 上下文 | 1M token | 官方官网 |
| 定位 | 长程 coding / 3D 创作 / 金融研究 / 复杂 agent 任务 | 官方官网 |
| 官方 HF 仓库 | **无**（hf-mirror 搜索 0 命中） | 核验记录（2026-11-03） |
| 官方 config | 无（无可拉取对象） | 核验记录 |
| 入口类 / model_type | 待确认 | — |
| 是否继承 K3 架构（KDA/AttnRes/Stable LatentMoE） | 待确认 | — |
| hidden / 层数 | 待确认 | — |
| 注意力布局（KDA / Gated MLA 层数比例） | 待确认 | — |
| 路由专家 / 共享 / 激活数 | 待确认 | — |
| 激活函数（SiTU-GLU?） | 待确认 | — |
| 视觉塔（MoonViT-V2?）与多模态 | 待确认（官网未提模态，不补） | — |
| 词表 / 特殊 token | 待确认 | — |
| 权重格式（MXFP4?） | 待确认 | — |
| 协议（License） | 待确认 | — |
| API 名 / 定价 | 待确认 | — |
| 技术报告 / 论文 | 待确认（未检索到） | — |

### 1.2 端到端总览

待确认（无 config 与技术报告可锚定）。参照 K3 旗舰路线（见 `kimi_k3/index.md`）推测可能为「视觉塔 + 投影器 + KDA/MLA 混合文本塔」构形，但**仅为路线推测，不作为事实记录**。

## 2. 总体设计

待确认。官网仅说明定位与规模量级，未披露架构细节。

## 3. 解码层解剖

待确认。

## 4. 核心机制

待确认。

## 5. 注意力

待确认。

## 6. 多模态

待确认（官网未提及图像/视频模态；K3 为原生多模态，K3.1-Pro 是否保留不臆测）。

## 7. Token 与词表

待确认。

## 8. 演进

- **Kimi K3（2026-07，见 `kimi_k3` 目录）**：上一代旗舰，2.8T/104B，KDA + AttnRes + Stable LatentMoE，1M 上下文，MXFP4 QAT 发布。
- **Kimi K3.1-Pro（2026-10-24，本目录）**：官网展示的最新旗舰，3 万亿参数级、1M 上下文，面向长程 coding/3D/金融/复杂 agent；公开信息尚少，全部架构字段待确认。
- **同代兄弟**：Kimi-Linear-48B-A3B（KDA 混合线性注意力前身，`models/kimi_linear/`）、K2.8 Preview（2026-09-11 接管 kimi-for-coding；HF 无官方权重仓库）、Kimi-VL-A3B（2025 上代小 VL）。

## 9. 参考文献

1. Moonshot AI 官网首页：https://www.moonshot.ai （2026-11-03 抓取；K3.1-Pro 展示日期 2026-10-24；3T-class / 1M context / 定位原文）
2. 核验记录：hf-mirror 组织列表与搜索对 "Kimi-K3.1" / "Kimi-K3.1-Pro" 0 命中（2026-11-03，组织者已联合核验）
3. 前代对照：`kimi_k3/index.md`（K3 技术报告 arXiv:2607.24653，仅作路线参照，非 K3.1-Pro 事实来源）
