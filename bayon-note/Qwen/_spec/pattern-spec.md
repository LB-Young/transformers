# Qwen 主系列 8 代架构详解笔记 — 统一模式规格（v1，硬约束）

本文件是 Qwen 主系列 8 代（Qwen1 / Qwen1.5 / Qwen2 / Qwen2.5 / Qwen3 / Qwen3.5 / Qwen3.6 / Qwen3.8）家族架构详解笔记（index.md + HTML + DOCX 三产物）的统一模式规格。
8 个版本相互独立并行制作，但产物风格、章节结构、图例、演进叙事必须统一。**动手前必须先完整读取本规格、下方体例标杆文件、以及 Skill 必读清单中的全部文件**。

## 0. 交付结构与命名（硬约束）

- 输出根目录：`E:\transformers\bayon-note\Qwen\`
- 每代一个子文件夹，交付物固定为：
  - `qwen1\index.md` + `index.html` + `Qwen-1-家族架构详解.docx`
  - `qwen1_5\index.md` + `index.html` + `Qwen-1.5-家族架构详解.docx`
  - `qwen2\index.md` + `index.html` + `Qwen-2-家族架构详解.docx`
  - `qwen2_5\index.md` + `index.html` + `Qwen-2.5-家族架构详解.docx`
  - `qwen3\index.md` + `index.html` + `Qwen-3-家族架构详解.docx`
  - `qwen3_5\index.md` + `index.html` + `Qwen-3.5-家族架构详解.docx`
  - `qwen3_6\index.md` + `index.html` + `Qwen-3.6-家族架构详解.docx`
  - `qwen3_8\index.md` + `index.html` + `Qwen-3.8-家族架构详解.docx`
- 辅助产物（截图、临时脚本、素材、图）放各自子文件夹内 `_shots/ _arch/ _tools/` 等，不属于交付物。
- **范围边界**：只做自己负责的版本目录；不得改动其他版本目录、`_spec` 目录，**不得改动/删除 Qwen 目录下既有文件**：
  `debug_qwen.py`、`make_qwen_test_image.py`、`qwen_test_image.png`、`validate_qwen_debug.py`、`Qwen-27B架构对比-3.5-3.6-3.8.md`。

## 1. 体例标杆（动手前必须打开理解）

- **内容结构标杆（最重要）**：`E:\transformers\bayon-note\Gemma\gemma-3\index.md`（家族全景对照表 + 端到端总览 + 解码层解剖 + 核心机制 + 注意力 + 多模态 + 特殊 token + 演进视角 + 来源列表 的完整章节骨架）
- **html 布局/配色/图例体例**：`E:\transformers\bayon-note\Gemma\gemma-3\index.html`（内联 SVG 架构图 `.dg` 类体系、比较表格 `.cmp`、`.tip` 提示、来源列表 `ol.src`、顶部 sticky 导航、favicon 内联 SVG、打印样式）
- **多模态章节体例（VL 变体参考）**：`E:\transformers\bayon-note\InternVL\internvl-3\InternVL-3-家族架构详解.html`（视觉编码器细节、tile/动态分辨率机制、token 流图的写法）
- **docx 章节/表格/图例体例**：`E:\transformers\bayon-note\Gemma\gemma-1\Gemma-1-家族架构详解.docx`（用 word skill 的 `scripts/read.py` 读取其结构：Heading 1 = 1/2/3…，Heading 2 = 1.1/1.2…，A4 版面，统一边框表格，正文嵌架构图）
- **docx 公式风格**：`E:\transformers\bayon-note\Gemma\gemma-4-26B-A4B-it-param-breakdown.docx`（文本公式 + “符号/含义/取值”三列表，不用 OMML 公式对象，可渲染）

## 2. docx 体例（冻结）

- Python：`C:\Users\LBY\.conda\envs\transformers-gemma\python.exe`（python-docx 等）。matplotlib 画图用 `C:\Users\LBY\.conda\envs\common312\python.exe`。
- **docx 由 md2docx.py 生成**：把 `E:\transformers\bayon-note\Gemma\_tools\md2docx.py` 复制到自己的版本目录 `_tools\`（或一个共享 Qwen `_tools\`），修改顶部 `BASE` 与 `MODELS` 元组（dir / 标题 / 副标题）适配本代，**严禁改动原文件**。它读 `<dir>\index.md` → 输出 `<dir>\Qwen-X-家族架构详解.docx`，图片从 `<dir>\_arch\*.png` 嵌入（图宽 14.5cm + keep_with_next）。
- 严禁 emoji；编号章节；正文与标题字体字号全文统一；单行中文正文默认单引号包裹；表格配色克制统一（表头灰底 D9D9D9、Table Grid）。
- 公式：文本公式 + 符号/含义/取值三列表（可复算），风格对齐 param-breakdown 标杆（`P_embed`、`base^(-0.5)`、`±`、`≈`、`K^ᵀ`）。
- 架构图：PNG 嵌入，每图带编号图注（图 1、图 2…）；图完整渲染、文字与图不重叠；**至少 2 张图**：
  1. 端到端/家族架构总览图（该代家族主干：Embedding → N×DecoderLayer → LM Head；VL 代含视觉塔路径）
  2. 核心机制图（该代特色：如 Qwen1 的 logn-attn、Qwen2 的 GQA+滑窗、Qwen3 的 thinking/MoE 路由、Qwen3.5+ 的 Gated DeltaNet 与 Gated Attention 3:1 混合）
  - 多模态代（qwen2/qwen2_5/qwen3/qwen3_5/qwen3_6/qwen3_8）建议加第 3 张：视觉塔 → 投影 → 文本融合的 token 流图。
- 校验（唯一允许的校验方式）：word skill 的 `scripts/audit.py audit <docx>` 直至退出码 0、0 错误；`scripts/render.py render <docx> --output-dir <临时目录>` 后**逐张**查看全部页面 PNG（分页、断行、表格、图片、题注分离等）。**禁止** verifier-hub / artifact-preview / 其他渲染方式。

## 3. html 体例（冻结）

- 单文件自包含（CSS/JS 内联、原生 JS）；ECharts 走 `https://fastly.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js`；不用 emoji；有 favicon（内联 SVG）；时间轴用 grid 三列（时间/轴/内容，圆点与轴线 0px 对齐）；375px 优先设计、桌面 `@media (min-width:768px)` 扩展；多列表格 `table-layout:fixed` 显式列宽；**每页至少 1 个用真实数据算出的 ECharts 图**（如家族参数量对比、上下文长度/词表演进）。
- **统一配色（8 个版本完全一致，冻结如下——逐字复制 gemma-3/index.html 的 `:root` 变量与 `.dg` 类 CSS）**：
  `--bg:#faf9f6; --panel:#ffffff; --ink:#23272f; --sub:#66707b; --line:#e5e4df; --accent:#1d4ed8; --accent-ink:#1e40af; --accent-soft:#e2eafb; --accent-line:#c0d0f4;` 及该文件第 10 行起的全部其余变量。`.dg` 图例：node 白底灰描边；dim 用 accent-soft；core 用 hl-soft + hl 描边；edge 灰箭头；重点边用 accent/hl 色；`min-width:880px` 的 SVG 放横向滚动容器。
- 章节 id（按此顺序）：`family`（家族全景与本地 config 对照）、`overview`（端到端架构总览+整体架构图）、`layer`（解码层解剖）、`mech`（核心机制专节）、`attn`（注意力机制）、`mm`（多模态；**纯文本代如 qwen1/qwen1_5 省略此节**）、`tokens`（特殊 token 与词表）、`evo`（演进视角，相对上一代 + 演进图）、`sources`（参考文献）。标题编号 1…9 对应显示。
- 校验（唯一允许）：html skill 的 `scripts/shot.py <html路径>`，读取 stdout 摘要与 `_shots/report.json`，error=0，桌面（1440×900）+ 移动（390×844）截图人工核对（图表移动端 ≥ 视口宽 65%、无文字重叠、图不截断）。**禁止** artifacts-preview / browser-use 等。

## 4. Skill 必读清单（动手前完整 Read，按各 skill 自身规则）

- word skill 根：`C:\Users\LBY\AppData\Local\Doubao\User Data\Default\.doubao\agent_mode\workspace\.skills\word`：SKILL.md、`references\style\writing-taxonomy.md`、`references\workflows\audit.md`、`references\workflows\windows-compat.md`（Windows 必读）。
- html skill 根：`C:\Users\LBY\AppData\Local\Doubao\User Data\Default\.doubao\agent_mode\workspace\.skills\html`：SKILL.md、`references\design\frontend-design.md`（必读）、`references\windows-compat.md`（Windows 必读）。
- 架构图可视化可参考 `doubao-visualization`：`C:\Users\LBY\AppData\Local\Doubao\User Data\Default\.doubao\agent_mode\workspace\.skills\doubao-visualization`。

## 5. 本地事实锚点（config 已由组织者读取核对，制作时必须再自行 Read 对应 `config.json` 全文；一切数值以 config 原文为准，禁止脑补）

本地模型目录（config.json 路径 = 目录 + `\config.json`）：
- Qwen1：`E:\models\qwen\Qwen-1_8B`
- Qwen1.5：`E:\models\qwen\Qwen1.5-0.5B-Chat`
- Qwen2：`E:\models\qwen\Qwen2-0.5B-Instruct`、`E:\models\qwen\Qwen2-VL-2B-Instruct`
- Qwen2.5：`E:\models\qwen\Qwen2.5-0.5B-Instruct`、`E:\models\qwen\Qwen2.5-VL-3B-Instruct`
- Qwen3：`E:\models\qwen\Qwen3-0.6B`、`E:\models\qwen\Qwen3-VL-2B-Instruct`
- Qwen3.5：`E:\models\qwen\Qwen3.5-0.8B-Instruct`（真实仓库 `Qwen/Qwen3.5-0.8B`）、`E:\models\qwen\Qwen3.5-35B-A3B-GPTQ-Int4`
- Qwen3.6 / Qwen3.8：无本地权重，27B 架构事实见第 8 节（已核实）

已核对的关键事实（组织者 2026-10-26 逐字段读取本地 config 所得，版本内可复算，引用时标注「本地 config：<路径>」）：

| 模型（本地目录） | model_type / 入口类 | hidden | 层数 | Q/KV 头 | head_dim | intermediate | vocab | max_pos | 关键特征 |
|---|---|---|---|---|---|---|---|---|---|
| Qwen-1_8B | qwen / QWenLMHeadModel | 2048 | 24 | 16 头（无 GQA 字段） | kv_channels 128 | 11008 | 151936 | 8192 | tie=False · use_logn_attn=True · tokenizer QWenTokenizer |
| Qwen1.5-0.5B-Chat | qwen2 / Qwen2ForCausalLM | 1024 | 24 | 16 / 16（MHA） | — | 2816 | 151936 | 32768 | sliding=32768 · max_window_layers=21 · rope_theta 1e6 · tie=True |
| Qwen2-0.5B-Instruct | qwen2 / Qwen2ForCausalLM | 896 | 24 | 14 / 2（GQA） | — | 4864 | 151936 | 32768 | sliding=32768 · max_window_layers=24 · tie=True |
| Qwen2-VL-2B-Instruct | qwen2_vl / Qwen2VLForConditionalGeneration | 1536 | 28 | 12 / 2 | — | 8960 | 151936 | 32768 | vision_start/end 151652/151653 · image 151655 · video 151656 · vision_config: hidden 1536 / patch 14 / temporal 2 |
| Qwen2.5-0.5B-Instruct | qwen2 / Qwen2ForCausalLM | 896 | 24 | 14 / 2 | — | 4864 | 151936 | 32768 | sliding=32768 · max_window_layers=21 · tie=True |
| Qwen2.5-VL-3B-Instruct | qwen2_5_vl / Qwen2_5_VLForConditionalGeneration | 2048 | 36 | 16 / 2 | — | 11008 | 151936 | 128000 | max_window_layers=70 · vision_config: hidden 1280 / patch 14 / temporal 2 |
| Qwen3-0.6B | qwen3 / Qwen3ForCausalLM | 1024 | 28 | 16 / 8 | 128 | 3072 | 151936 | 40960 | sliding=None（全注意力）· max_window_layers=28 · rope_theta 1e6 · tie=True |
| Qwen3-VL-2B-Instruct | qwen3_vl / Qwen3VLForConditionalGeneration | text 2048 | 28 | 16 / 8 | 128 | 6144 | 151936 | 262144 | text rope_theta 5e6 · vision_config: hidden 1024 / patch 16 / temporal 2 / gelu_pytorch_tanh · 特殊 token 151652-151656 |
| Qwen3.5-0.8B | qwen3_5 / Qwen3_5ForConditionalGeneration | text 1024 | 24 | 8 / 2 | 256 | 3584 | 248320 | 262144 | attn_output_gate=True · 线性注意力：linear_num_key_heads 16 / linear_num_value_heads 16 / key·value head_dim 128 / linear_conv_kernel_dim 4 · vision_config: hidden 768 / patch 16 / temporal 2 · image_token 248056 · eos 248044 |
| Qwen3.5-35B-A3B | qwen3_5_moe / Qwen3_5MoeForConditionalGeneration | text 2048 | 40 | 16 / 2 | 256 | — | 248320 | 262144 | MoE：num_experts 256 / num_experts_per_tok 8 / moe_intermediate 512 / shared_expert_intermediate 512 / router_aux_loss_coef 0.001 · 线性注意力：key 16 头 / value 32 头 / head_dim 128 / conv 4 · vision_config: hidden 1152 / patch 16 / temporal 2 · tie=False |

> 注：Qwen3.5 两个 config 顶层未显式列 `rope_theta`/`sliding_window`（取类默认值），制作时按 `configuration_qwen3_5.py` 类默认值核验后标注。Qwen3.5-0.8B 无 `rope_scaling` 字段；Qwen3 文本层 `rope_scaling=None`。

## 6. transformers 源码依据（架构实现细节权威依据，`E:\transformers\src\transformers\models\` 下）

- Qwen1：**v5 无原生 qwen 模块**——需 `trust_remote_code=True` 加载仓库自带 modeling（本地 `E:\models\qwen\Qwen-1_8B\` 内有对应文件），笔记中写明「官方仓库实现 + 本地 config」双来源，禁止凭记忆编造类结构。
- Qwen1.5 / Qwen2 / Qwen2.5（文本）：全部走 `qwen2\modeling_qwen2.py`（Qwen2ForCausalLM），1.5 无独立模块。
- Qwen2-MoE：`qwen2_moe\`（Qwen2MoeForCausalLM，57B-A14B）。
- Qwen2-VL：`qwen2_vl\`（Qwen2VLForConditionalGeneration + processing/video_processing/image_processing）。
- Qwen2.5-VL：`qwen2_5_vl\`；Qwen2.5-Omni：`qwen2_5_omni\`。
- Qwen3：`qwen3\`（Qwen3ForCausalLM）；Qwen3-MoE：`qwen3_moe\`；Qwen3-VL：`qwen3_vl\`（Qwen3VLForConditionalGeneration + processing/video_processing）；Qwen3-VL-MoE：`qwen3_vl_moe\`；Qwen3-Omni-MoE：`qwen3_omni_moe\`；Qwen3-ASR：`qwen3_asr\`；MTP：`qwen3_next\`。
- Qwen3.5 / Qwen3.6 / Qwen3.8：Dense 与 27B 全走 `qwen3_5\`（configuration/modeling/modular/tokenization）；MoE（35B-A3B 及更大）走 `qwen3_5_moe\`；MTP 投机解码走 `qwen3_next\`。
- 多模态类名注意：**transformers v5 中 VL 模型必须用 `AutoModelForImageTextToText`**（`AutoModelForVision2Seq` 已移除）。

## 7. 每代内容范围（覆盖该代代表尺寸与变体，数值全部官方 model card / config 核实并标注来源）

- **qwen1**：Qwen-1_8B / 7B / 14B / 72B（QWenLMHeadModel、qwen model_type、tiktoken 分词、use_logn_attn、kv_channels 128、NTK/dynamic RoPE）；官方技术报告 arXiv 2309.16609（Qwen Technical Report）。
- **qwen1_5**：0.5B / 1.8B / 7B / 14B / 32B / 72B（Qwen2ForCausalLM、qwen2 model_type、MHA→GQA 分尺寸、32K 上下文、Chat/Base 双版）。
- **qwen2**：0.5B / 1.5B / 7B / 57B-A14B（MoE）/ 72B，+ 2-VL / 2-Audio / 2-Math / 2-Coder 变体（GQA 全系、滑窗 32768、max_window_layers、2-VL：patch 14 / M-RoPE / 动态分辨率、2-Audio：qwen2_audio）。
- **qwen2_5**：0.5B / 1.5B / 7B / 14B / 32B / 72B，+ 2.5-VL（3B/7B/32B/72B，128K 上下文、patch 14、window attention）/ 2.5-Coder / 2.5-Max（闭源 API）。
- **qwen3**：Dense 0.6B / 1.7B / 4B / 8B / 14B / 32B，MoE 30B-A3B / 235B-A22B，+ 3-VL（2B/4B/8B/30B-A3B/32B）、qwen3_asr / omni；thinking 模式（/think /nothink 软切换、thinking tokens 编码）、head_dim 128、全注意力、MTP（qwen3_next）。
- **qwen3_5**：0.8B / 9B / 27B / 35B-A3B / 122B-A10B / 397B-A17B（qwen3_5 + qwen3_5_moe；多模态原生、Gated DeltaNet + Gated Attention 3:1 混合、词表 248320、262144→1M、MTP multi-steps、thinking 默认开且无软切换）。
- **qwen3_6**：27B / 35B-A3B（架构与 3.5 完全一致；Thinking Preservation / preserve_thinking、interleaved thinking 默认只保留最新用户消息思考块、稳定性优先）。
- **qwen3_8**：6B-Flash / 27B / 2.4T-A95B（基于 qwen3_5 架构基座；Flexible Thinking Control（reasoning_effort、可关 thinking、preserve_thinking 默认保留全部）；官方 FP8 量化；TokenSpeed 引擎）。

## 8. 已核实事实与踩坑（必须保持一致，避免与既有产物矛盾）

**三代 27B 官方配置完全一致**（来源：`E:\transformers\bayon-note\Qwen\Qwen-27B架构对比-3.5-3.6-3.8.md`，已标注官方 model card 链接；制作 qwen3_5/3_6/3_8 时可直接引用该文件与官方 card，数字不得互相矛盾）：
- 64 层、hidden 5120、词表 248320 padded、16×(3×Gated DeltaNet→FFN → 1×Gated Attention→FFN)、GDN V48/QK16 头 head_dim128、GA Q24/KV4 头 head_dim256、RoPE dim64、FFN intermediate 17408、MTP multi-steps、上下文 262144→YaRN 1M（3.8 为 1,000,000）、model_type 均为 qwen3_5、Dense（无 MoE）、多模态（vision encoder，early fusion）。
- 代际差异：3.5→3.6 = 稳定化 + Thinking Preservation（preserve_thinking 默认只保留最新一条用户消息思考块）；3.6→3.8 = Flexible Thinking Control（reasoning_effort、可关 thinking、preserve_thinking 默认保留全部）+ 能力跃升（SWE-bench Pro 53.5→61.7 等，benchmark 数字从官方 card 核实）+ FP8 量化（~29GB）+ TokenSpeed。

**踩坑（写作时须规避，勿与下载报告矛盾）**：
1. Qwen-1_8B 为旧版架构（QWenLMHeadModel，需 trust_remote_code，tokenizer 需 tiktoken 包）；v5 无原生 qwen1 模块。
2. Qwen3.5 系列官方仓库名已去掉 `-Instruct` 后缀（如 `Qwen/Qwen3.5-0.8B`；`-Base` 为无对话版）；本地目录名 `Qwen3.5-0.8B-Instruct` 是落盘别名。
3. transformers v5 中多模态类名为 `AutoModelForImageTextToText`。
4. Qwen3.5-35B-A3B-GPTQ-Int4 加载需 gptqmodel（Windows 无 wheel，权重已逐字节校验）——笔记中提及量化加载时按此口径写。
5. Qwen3.5-27B 部分社区页面（GGUF 转换）标 Hidden=4096，与官方 model card 5120 不符——以官方 card 为准。
6. hf-mirror 大文件偶发「假死」，重试即可；ModelScope 为备用通道。

## 9. 版本演进叙事（8 份必须一致；写入各自文档前逐条用来源核实，禁止编造）

- Qwen1 → Qwen1.5：代码代际上**没有独立 Qwen1.5 模块**——全面切换到 Qwen2 架构（qwen2 model_type / Qwen2ForCausalLM：SwiGLU、RMSNorm、RoPE 1e6、tie_embeddings、MHA→大尺寸 GQA）；分词换为基于 151936 词表的新 tokenizer；上下文 2048/8192 → 32768；大模型引入 GQA、attention bias 移除等。
- Qwen1.5 → Qwen2：GQA 全系标配（最小 0.5B 即 14Q/2KV）；引入滑窗注意力（sliding_window 32768 + max_window_layers）；MoE 首次（Qwen2-MoE 57B-A14B：细粒度专家 + 共享专家）；多模态与音频变体（2-VL：patch 14 + M-RoPE + 动态分辨率；2-Audio）。
- Qwen2 → Qwen2.5：文本架构延续 qwen2 model_type（0.5B 级 config 与 Qwen2 同规格）；2.5-VL 扩展上下文至 128K、视觉塔优化（window attention、原生动态分辨率、多模态 RoPE）；新增 2.5-Coder、2.5-Omni、2.5-Max（闭源）。
- Qwen2.5 → Qwen3：原生 thinking（/think /nothink 软切换、thinking 块用特殊 token 编码进序列）；注意力回归全注意力（Qwen3 文本无 sliding_window，max_position 40960，head_dim 128，QKV bias False）；MoE 两档（30B-A3B / 235B-A22B：128 专家、top-8 纯路由、**无共享专家**——2026-10-26 由 qwen3/_tools/_cfgpull 官方 config 核实，勘误此前"shared+non-shared 混合"假设；共享专家直到 Qwen3.5 MoE 才引入）；3-VL 视觉塔 patch 14→16、文本 RoPE theta 5e6、上下文 262144；MTP（qwen3_next）、ASR/Omni 变体。
- Qwen3 → Qwen3.5：**架构大改**——Gated DeltaNet（线性注意力，递归状态）+ Gated Attention（GQA 全注意力）3:1 混合（model_type qwen3_5）；词表 151936 → 248320（padded）；上下文 40960 → 262144（YaRN 1M）；视觉原生化（early fusion、patch 16）；MTP multi-steps；thinking 默认开且移除 /think 软切换；MoE 变体 qwen3_5_moe（256 专家 / 8 活跃，共享专家）。
- Qwen3.5 → Qwen3.6：架构数字不变（同 qwen3_5）；定位「稳定化 + 实用化」；Thinking Preservation（preserve_thinking 选项，interleaved thinking 默认只保留最新用户消息思考块）；Agentic Coding 强化。
- Qwen3.6 → Qwen3.8：官方明示 "Built on the architectural foundation of Qwen3.5"（架构数字不变）；Flexible Thinking Control（thinking 默认开、可按请求关、reasoning_effort 调节、preserve_thinking 默认保留全部历史思考块）；能力代际跃升（benchmark 从官方 card 核实）；FP8 官方量化 + TokenSpeed；超大 MoE 2.4T-A95B 与 6B-Flash。
- 通用主线（各代文档的演进章节必须与相邻代说法互相咬合）：词表 151936（Qwen1–Qwen3，1.5 起含 Chat）→ 248320（3.5+）；上下文 8192（Qwen1）→ 32768（1.5/2）→ 128K（2.5-VL / 3-VL）→ 40960（Qwen3 文本）→ 262144（3.5+，YaRN 1M）；注意力 MHA（Qwen1/小 1.5）→ GQA（1.5 大尺寸起 / Qwen2 全系）→ GQA + 滑窗（Qwen2/2.5）→ 全注意力 head_dim 128（Qwen3）→ Gated DeltaNet + Gated Attention 混合（3.5+）。

## 10. 参考文献（编号已联网核实后引用，禁止编造编号/URL）

- 每份文档的参考文献表须含：官方 model card（HF + ModelScope 对应 repo 链接）、transformers 源码文件路径、本地 config.json 绝对路径、官方技术报告/博客（联网核实后引用）。
- 官网/仓库：https://github.com/QwenLM/Qwen 、https://qwenlm.github.io/blog/ 、Qwen 官方 HF 组织页 https://huggingface.co/Qwen 、ModelScope 组织页 https://modelscope.cn/organization/qwen 。
- 主要论文候选（写作前核实编号）：Qwen Technical Report（arXiv:2309.16609）、Qwen2（arXiv:2407.10671）、Qwen2.5 技术报告（arXiv:2412.15115）、Qwen3（arXiv:2505.09388）、Qwen2-VL（arXiv:2409.12191）、Qwen2.5-VL（arXiv:2502.13923）、Qwen3-VL（arXiv:2506.13779）、Qwen3 Technical Report（arXiv:2507.01401）、Qwen3.5（arXiv:2510.03611）等——**每条都必须核实再写入，核实不到就删或标「待确认」**。
- 所有数字必须来自官方 model card / config / 已核实来源并标注出处；未核实信息不得编造，统一标「待确认」。

## 11. 验收清单（交付前逐条自检，并在最终报告中逐条回报证据）

1. 本代目录下三产物存在且可打开（index.md 文本完整、index.html 可被浏览器解析、docx 可被 python-docx 打开），绝对路径回报。
2. docx：audit 0 错误、退出码 0（回报错误计数与最后退出码）；render 成功且逐页核验（回报页数与核验结论）；架构图完整不截断。
3. html：shot.py 运行，error=0（回报 stdout 摘要、截图路径）；桌面+移动截图人工核对（重点：ECharts 图移动端宽度 ≥ 视口 65%、时间轴对齐、无文字重叠）。
4. 必备章节齐全：家族全景（含本地 config 实参表）/ 端到端总览 / 解码层与核心机制 / 注意力（+多模态，如适用）/ 演进视角 / 参考文献；章节 id 与第 3 节一致。
5. 全部架构图完整渲染，文字与图不重叠。
6. config 实参表逐字段标注来源（config.json 绝对路径 / 官方 model card URL）；参考文献条目逐条标注 URL；无来源支撑的结论明确标注「待确认」或删除。
7. 演进叙事与本规格第 9 节一致（相邻代说法咬合）。
8. 最终报告回报：三产物绝对路径、audit/render/shot 证据、本代「相对上一代演进要点」3–5 条（供组织者交叉核对）、参考文献最终清单、未核实「待确认」项清单。

## 12. 运行环境（Windows 本地）

- 唯一可用 shell：PowerShell（Bash 禁用）；复杂逻辑写 .py 文件再执行；文件读写显式 UTF-8；Windows 下 python 写入文本用 `newline=''` 或显式编码。
- docx/脚本 python：`C:\Users\LBY\.conda\envs\transformers-gemma\python.exe`（torch 2.11.0+cu128、transformers 5.21.0.dev0 editable @ E:\transformers、PIL 13.0.0、huggingface_hub 2.2.0）；matplotlib 用 `C:\Users\LBY\.conda\envs\common312\python.exe`。
- 网络：中国大陆，HF 直连可能慢，可用 `HF_ENDPOINT=https://hf-mirror.com` 或 ModelScope 拉 config / model card。
- 禁止 sudo/apt/docker；无需请求任何权限确认；只在自己的版本目录内写文件。
- 严禁改动/删除第 0 节列出的既有 Qwen 文件与 `Gemma\_tools\md2docx.py` 原件；docx 生成一律用复制件。
