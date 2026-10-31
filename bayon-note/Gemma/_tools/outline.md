# outline.md — Gemma 1/2/3/3n/4 家族架构详解 → Word 批量转换计划

## 任务
把 5 份论文级架构笔记（公众号版 `index.md`，内容源于各 `index.html` 家族全架构版）各写为一份 Word 文档（docx），风格参考用户认可的
`E:\transformers\bayon-note\Gemma\gemma-4-26B-A4B-it-param-breakdown.docx`（编号章节 + 符号速查表 + 文本公式 + 汇总表 + 一句话总结）。

## 交付物（5 份，均放各自模型目录）
1. `gemma-1\Gemma-1-家族架构详解.docx` — 内容源 gemma-1\index.md，图片 _arch\01/02
2. `gemma-2\Gemma-2-家族架构详解.docx` — 内容源 gemma-2\index.md，图片 _arch\01/02
3. `gemma-3\Gemma-3-家族架构详解.docx` — 内容源 gemma-3\index.md，图片 _arch\01/02
4. `gemma-3n\Gemma-3n-家族架构详解.docx` — 内容源 gemma-3n\index.md，图片 _arch\01/02
5. `gemma-4\Gemma-4-家族架构详解.docx` — 内容源 gemma-4\index.md，图片 _arch\01/02/03

## 内容来源与角色
| 材料 | 角色 | 使用方式 |
|---|---|---|
| 5 份 index.md（公众号版） | 内容素材（主体） | 逐节转写；表格、公式、正文原样保留 |
| _arch\*.png（11 张） | 事实/图片素材 | 全部嵌入对应章节，图下 Caption，等比缩放不裁剪 |
| 各 index.html | 内容母本（md 已完整反映） | 仅在 md 有缺口时回查 |
| gemma-4-26B-A4B-it-param-breakdown.docx | 格式参考（非模板底稿） | 沿用：编号章节、文本公式、速查表、汇总表、总结段 |

## 每份文档结构（对齐 md/HTML 9 节，示例风格）
- Title（黑体18pt居中）：Gemma X 家族架构详解
- Subtitle（楷体12pt居中）：入口类 · 尺寸列表 · 关键机制（如 "GemmaForCausalLM · 2B/7B · MQA 与 MHA"）
- 依据段（宋体12pt）：数据来源 = 本机 config.json + transformers 源码 + 官方论文/博客，所有数字标注来源
- 1 家族全景（config 对照大表，速查表风格）
- 2 端到端数据流（插图 + 关键路径说明 + 公式）
- 3 解码层解剖（插图 + 公式）
- 4 核心机制专节（公式 + 源码行号）
- 5 注意力机制（表 + 公式）
- 6 多模态（表）
- 7 特殊 Token（表）
- 8 演进视角（对比表）
- 9 来源（列表：config 路径 / 源码行号 / arXiv / 官方 URL）
- 一句话总结段（示例第 7 节风格）

## 格式规范（general.md + writing-taxonomy.md 一般格式要求）
- A4，页边距上下左右 2.5cm；页脚居中阿拉伯数字页码；无页眉；不加目录（跟随用户认可的示例风格）
- 字体：Title 黑体18pt / Subtitle 楷体12pt / H1 黑体16pt / H2 黑体14pt / H3 黑体12pt（标题黑色、无下划线）/ Normal 宋体12pt 西文 Times New Roman（两端对齐、1.5 倍行距、首行缩进 2 字符）/ Caption 宋体10.5pt 居中（图下）/ TableText 宋体10.5pt
- 表格：三线表风（Table Grid 边框 0.5pt）、表头浅灰 #D9D9D9、垂直居中、首列居中、数字列右对齐、文字列左对齐、行 cantSplit、表头 tblHeader 重复、单元格无缩进
- 公式：文本公式（Unicode × · √ 上标等，示例风格），独立公式居中；LaTeX 记号转可读文本
- 图片：嵌入型、居中、宽 ≤15cm 等比缩放、图下 Caption "图 N：xxx"、与图同页
- 代码块：等宽（Consolas 西文/宋体中文）、缩进段、浅灰底纹可选

## 引用核查表
| 引用 | 说明 | 状态 |
|---|---|---|
| ModelScope 各变体 config.json | 本机已下载核验（E:\models\model\...） | 已核实（沿用 md/HTML 结论） |
| transformers 源码 modeling_gemma*/configuration_gemma* | 本地 E:\transformers 可编辑安装 | 已核实（行号沿用 md） |
| Gemma 1 arXiv:2403.08295 | 官方技术报告 | 已核实（沿用 HTML §9） |
| Gemma 2 arXiv:2408.00118 | 官方技术报告 | 已核实 |
| Gemma 3 arXiv:2503.19786 | 官方技术报告 | 已核实 |
| Gemma 3n arXiv:2505.05146 | 官方技术报告 | 已核实 |
| Gemma 4 arXiv:2607.02770 + ai.google.dev/model_card_4 + blog.google | 官方资料 | 已核实 |
| 数字（hidden/层数/头数/参数量等） | 全部来自 config 原文与源码核算 | 已核实（逐项沿用 md） |

## 校验计划
- audit.py 逐份 audit 至通过（字体/表格/标题/结构）
- render.py 渲染每份 → 逐页 PNG 目检（标题、表格、图片、公式、分页）
- 回读 docx 核对 9 节齐全、图片 2-3 张、表格数一致、无占位符
