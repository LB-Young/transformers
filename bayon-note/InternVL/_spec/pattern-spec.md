# InternVL 系列架构详解笔记 — 统一模式规格（v1，硬约束）

本文件是 InternVL 1.5 / 2.0 / 2.5 / 3.0 / 3.5 五个版本架构详解笔记（HTML + DOCX 双产物）的统一模式规格。
五个版本相互独立并行制作，但产物风格、章节结构、图例、演进叙事必须统一。**动手前必须先完整读取本规格与下方体例标杆文件**。

## 0. 交付结构与命名（硬约束）

- 输出根目录：`E:\transformers\bayon-note\InternVL\`
- 每版本一个子文件夹，产物文件名固定为：
  - `internvl-1_5\InternVL-1_5-家族架构详解.html` 与 `InternVL-1_5-家族架构详解.docx`
  - `internvl-2\InternVL-2-家族架构详解.html` 与 `InternVL-2-家族架构详解.docx`
  - `internvl-2_5\InternVL-2_5-家族架构详解.html` 与 `InternVL-2_5-家族架构详解.docx`
  - `internvl-3\InternVL-3-家族架构详解.html` 与 `InternVL-3-家族架构详解.docx`
  - `internvl-3_5\InternVL-3_5-家族架构详解.html` 与 `InternVL-3_5-家族架构详解.docx`
- 每份文档覆盖**该版本家族的完整架构**（不只本地最小尺寸模型）：视觉编码器（InternViT 300M/6B、448px、各代变体）、投影层（MLP、像素 shuffle、动态分辨率 tile）、语言底座（InternLM2/Qwen2/Qwen2.5/Qwen3/GPT-OSS、MoE 变体）、训练管线（预训练/SFT/RLHF/Cascade RL）、该版本相对上一代的架构演进点。
- 辅助产物（截图、临时脚本、素材、_arch 等）放各自子文件夹内 `_shots/ _arch/ _tools/` 等，不属于交付物。
- **范围边界**：只做自己负责的版本；不得改动其他版本目录与 `_spec` 目录。

## 1. 体例标杆（动手前必须打开理解）

- docx 章节/表格/图例体例：`E:\transformers\bayon-note\Gemma\gemma-1\Gemma-1-家族架构详解.docx`（用 word skill 的 `scripts/read.py` 读取其结构：Heading 1 = 1/2/3…，Heading 2 = 1.1/1.2…，A4 版面，统一边框表格，正文嵌架构图）
- docx 公式风格：`E:\transformers\bayon-note\Gemma\gemma-4-26B-A4B-it-param-breakdown.docx`（文本公式 + “符号/含义/取值”三列表，不用 OMML 公式对象，可渲染）
- html 布局/配色/图例体例：`E:\transformers\bayon-note\Gemma\gemma-1\index.html`（内联 SVG 架构图 `.dg` 类体系、比较表格 `.cmp`、`.tip` 提示、来源列表 `ol.src`、顶部 sticky 导航、favicon 内联 SVG、打印样式）
- 注意：Gemma 系列是纯文本模型，InternVL 是多模态模型，图与章节按第 2/3 节扩展，不要照搬 Gemma 的解码层解剖章节。

## 2. docx 体例（冻结）

- Python：`C:\Users\LBY\.conda\envs\transformers-gemma\python.exe`（python-docx 等）。matplotlib 画图用 `C:\Users\LBY\.conda\envs\common312\python.exe`。
- 严禁 emoji；编号章节；正文与标题字体字号全文统一；单行中文正文默认单引号包裹；表格配色克制统一。
- 公式：文本公式 + 符号/含义/取值三列表（可复算）。
- 架构图：PNG 嵌入，每图带编号图注（图 1、图 2…）；图完整渲染、文字与图不重叠；至少 5 类图：
  1. 整体架构图（图像→ViT→投影→LLM 端到端）
  2. 视觉编码器细节图（InternViT 层结构）
  3. 动态分辨率 / tile 机制图（像素 tiling、thumbnail、max_dynamic_patch）
  4. token 流图（图像→视觉特征→MLP→LLM 的拼接位置，含 special token）
  5. 版本演进对比图（相对上一代）
- 校验（唯一允许的校验方式）：word skill 的 `scripts/audit.py audit <docx>` 直至退出码 0、0 错误；`scripts/render.py render <docx> --output-dir <临时目录>` 后**逐张**查看全部页面 PNG（分页、断行、表格、图片、题注分离等）。**禁止** verifier-hub / artifact-preview / 其他渲染方式。

## 3. html 体例（冻结）

- 单文件自包含（CSS/JS 内联、原生 JS）；ECharts 走 `https://fastly.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js`；不用 emoji；有 favicon（内联 SVG）；时间轴用 grid 三列（时间/轴/内容，圆点与轴线 0px 对齐）；375px 优先设计、桌面 `@media (min-width:768px)` 扩展；多列表格 `table-layout:fixed` 显式列宽；图优先 ECharts（每页至少 1 个用真实数据算出的 ECharts 图，如家族参数量/视觉 token 数对比）。
- **统一配色（五个版本完全一致，冻结如下）**：
  - `--bg:#f6f7f9; --panel:#ffffff; --ink:#24292f; --sub:#6b7280; --line:#e3e6ea;`
  - `--accent:#0e7490; --accent-ink:#155e75; --accent-soft:#e0f2f5; --accent-line:#b7dbe4;`
  - `--hl:#b45309; --hl-soft:#f7ead6; --hl-line:#e8cfae; --mono:#6d5a2e; --radius:12px;`
  - 图例 `.dg`：node 白底灰描边；dim 用 accent-soft；core 用 hl-soft + hl 描边；edge 灰箭头；重点边用 accent/hl 色；`min-width:880px` 的 SVG 放横向滚动容器。
- 章节 id（按此顺序）：`family`（家族全景与本地 config 对照）、`overview`（端到端架构总览+整体架构图）、`vit`（视觉编码器 InternViT 细节+图）、`projector`（投影层与语言底座）、`mech`（核心机制专节：像素 shuffle/tile/训练管线关键机制）、`pipeline`（训练管线，用 grid 三列时间轴）、`evo`（版本演进对比，相对上一代 + 演进图）、`sources`（参考文献）。
- 校验（唯一允许）：html skill 的 `scripts/shot.py <html路径>`，读取 stdout 摘要与 `_shots/report.json`，error=0，桌面（1440×900）+ 移动（390×844）截图人工核对（图表移动端 ≥ 视口宽 65%，无文字重叠、图不截断）。**禁止** artifacts-preview / browser-use 等。

## 4. Skill 必读清单（动手前完整 Read，按各 skill 自身规则）

- word skill 根：`C:\Users\LBY\AppData\Local\Doubao\User Data\Default\.doubao\agent_mode\workspace\.skills\word`：SKILL.md、`references\style\writing-taxonomy.md`、`references\workflows\audit.md`、`references\workflows\windows-compat.md`（Windows 必读）；写作场景命中 academic 时读 `references\style\academics.md`。
- html skill 根：`C:\Users\LBY\AppData\Local\Doubao\User Data\Default\.doubao\agent_mode\workspace\.skills\html`：SKILL.md、`references\design\frontend-design.md`（必读）、`references\windows-compat.md`（Windows 必读）。
- 架构图可视化可参考 `doubao-visualization`：`C:\Users\LBY\AppData\Local\Doubao\User Data\Default\.doubao\agent_mode\workspace\.skills\doubao-visualization`。

## 5. 本地事实锚点（config 已由组织者读取核对，制作时必须再自行 Read 对应 config.json 全文；一切数值以 config 原文为准，禁止脑补）

本地模型目录（config.json 路径 = 目录 + `\config.json`）：
- 1.5：`E:\models\model\Mini-InternVL-Chat-2B-V1-5`
- 2.0：`E:\models\model\InternVL2-1B`
- 2.5：`E:\models\model\InternVL2_5-1B`
- 3.0：`E:\models\model\InternVL3-1B-hf`
- 3.5：`E:\models\model\InternVL3_5-1B-HF`

已核对的关键事实（组织者已读取，版本内可复算）：
- 1.5/2.0/2.5 为 **legacy `internvl_chat` 格式**：顶层含 `llm_config`（嵌套底座 config）+ `vision_config`（嵌套 `intern_vit_6b` 型 config）+ `dynamic_image_size:true / force_image_size:448 / downsample_ratio:0.5 / max_dynamic_patch:12 / min_dynamic_patch:1 / ps_version:v2 / select_layer:-1 / use_thumbnail:true`。**注意**：本地小模型 vision_config 的数值（hidden 1024、24 层、16 头、patch 14、448px、intermediate 4096、qkv_bias true、qk_normalization false）是 **InternViT-300M 规格**，但 model_type 字符串仍为 `intern_vit_6b`——文中须按论文/博客事实写清「1.5 家族视觉塔有 300M 与 6B 两个规格，本地 2B 模型用 300M/448px 规格」。
- 1.5 底座：InternLM2-1.8B（hidden 2048、24 层、16 头、8 KV 头、intermediate 8192、vocab 92553、max_pos 32768、rope_theta 1e6、dynamic rope×2、rms_norm_eps 1e-5）；template `internlm2-chat`。
- 2.0 底座：Qwen2-0.5B（hidden 896、24 层、14 头、2 KV 头、intermediate 4864、vocab 151655、max_pos 32768、sliding_window 32768）；template `Hermes-2`。
- 2.5 底座：Qwen2.5-0.5B（hidden 896、24 层、14 头、2 KV 头、intermediate 4864、vocab 151674、max_window_layers 21）；template `internvl2_5`。
- 3.0 为 **transformers 原生 `internvl` 格式**：顶层 `text_config`（架构名 Qwen2ForCausalLM，hidden 896、24 层、14 头、2 KV 头、intermediate 4864、vocab 151674、max_pos 32768、rope dynamic×2、max_window_layers 70——以 config 原文为准，不脑补为其他底座）+ `vision_config`（model_type `internvl_vision`：hidden 1024、24 层、16 头、intermediate 4096、patch [14,14]、image_size [448,448]、attention_bias true、layer_scale_init_value 0.1、use_mean_pooling true、use_absolute_position_embeddings true、initializer_factor 0.1）+ `image_seq_length:256 / image_token_id:151667 / projector_hidden_act:gelu / vision_feature_layer:-1 / vision_feature_select_strategy:default`。
- 3.5 为 **transformers 原生 `internvl` 格式**：`text_config` 为 Qwen3-0.6B（Qwen3ForCausalLM，hidden 1024、28 层、16 头、8 KV 头、head_dim 128、intermediate 3072、vocab 151936、max_pos 40960、全 full_attention、无 rope_scaling）+ `vision_config` 同 3.0 的 `internvl_vision`（300M 规格）+ `image_seq_length:256 / image_token_id:151671`。

## 6. transformers 源码依据（架构实现细节权威依据）

- `E:\transformers\src\transformers\models\internvl\`：仅支持 2.x/3.x/3.5 的 `InternVLForConditionalGeneration`（`modeling_internvl.py` 中 741 行起；`get_image_features` 543/755 行；`pixel_shuffle` 670 行，`scale_factor=downsample_ratio=0.5`）、`configuration_internvl.py`、`processing_internvl.py`（图像 tile 预处理）、`video_processing_internvl.py`。
- 1.5 的 legacy 架构（`InternVLChatModel` = vision_model + mlp1 + llm）不在 transformers 内，参考官方仓库代码结构与 config：`https://github.com/OpenGVLab/InternVL`（internvl_chat 目录，modeling_internvl_chat.py / configuration_internvl_chat.py），在线核实后引用，禁止凭记忆编造结构细节。

## 7. 参考文献（编号已联网核实，写作时可直接引用；每份文档的参考文献表须含以下对应条目 + transformers 源码 + 本地 config 路径）

- InternVL 1.0（背景）：arXiv:2312.14238
- InternVL 1.5：arXiv:2404.16821「How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites」
- InternVL 2.0：**无独立 arXiv 论文**，引用官方博客「InternVL2: Better than the Best—Expanding Performance Boundaries of Open-Source Multimodal Models with the Progressive Scaling Strategy」https://internvl.github.io/blog/2024-07-02-InternVL-2.0/（2024-07-02）
- InternVL 2.5：arXiv:2412.05271（Revisiting InternVL / Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling）
- InternVL 3.0：arXiv:2504.10479「InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models」（2025-04-14）
- InternVL 3.5：arXiv:2508.18265「InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency」
- 其他论文/博客引用（如 VisionLLMv2、MM-NIAH、原生多模态预训练相关）必须联网核实后再写入，禁止编造编号。

## 8. 版本演进叙事（五份必须一致；写入各自文档前逐条用来源核实，禁止编造）

- 1.5 → 2.0：视觉塔连续学习升级；语言底座从 InternLM2 扩展到 Qwen 系列（1B 用 Qwen2-0.5B）；动态分辨率像素 tile + thumbnail 保留；多模态输入（图/视频/医疗）+ 多任务输出（VisionLLMv2 解码器）。
- 2.0 → 2.5：原生多模态预训练范式（Native Multimodal Pretraining，去重后的原生多模态语料，全参数训练视觉塔不冻结）；更强的基础模型与推理能力。
- 2.5 → 3.0：原生多模态预训练 + 更优文本基座 + test-time scaling（思维链、多 token 预测）；1B 级 ViT 从 300M 规格演进（以论文/config 为准）；长上下文增强。
- 3.0 → 3.5：Cascade RL（级联强化学习，推理计算负载均衡）；推理效率大幅提升（论文称相对 3.0 最高 4.05×）；GUI 交互与具身智能等新能力；最大 241B-A28B。
- 每份文档的「版本演进对比」章节内容必须与本条一致（相邻版本的说法互相咬合），组织者交付前会交叉核对五份。

## 9. 验收清单（交付前逐条自检，并在最终报告中逐条回报证据）

1. 两份产物存在且可打开（docx 可被 python-docx 打开、html 可被浏览器解析），绝对路径回报。
2. docx：audit 0 错误、退出码 0（回报错误计数与最后退出码）；render 成功且逐页核验（回报页数与核验结论）。
3. html：shot.py 运行，error=0（回报 stdout 摘要、截图路径），桌面+移动截图核对（回报核对结论，重点：ECharts 图移动端宽度 ≥ 视口 65%、时间轴对齐、无文字重叠）。
4. 必备章节齐全：架构总览 / 关键模块细节（视觉编码器、投影、语言模型）/ 训练管线 / 版本演进对比 / 本地 config 实参表 / 参考文献。
5. 全部架构图完整渲染（不出现半截图），文字与图不重叠。
6. config 实参表逐字段标注来源（config.json 绝对路径）；参考文献条目逐条标注 URL；无来源支撑的结论明确标注为推测或删除。
7. 演进叙事与本规格第 8 节一致。
8. 最终报告回报：两份产物绝对路径、audit/render/shot 证据、本版本「相对上一代演进要点」3–5 条（供组织者交叉核对）、参考文献最终清单。

## 10. 运行环境（Windows 本地）

- 唯一可用 shell：PowerShell（Bash 禁用）；复杂逻辑写 .py 文件再执行；文件读写显式 UTF-8；Windows 下 python 写入文本用 `newline=''` 或显式编码。
- docx/脚本：`C:\Users\LBY\.conda\envs\transformers-gemma\python.exe`；matplotlib：`C:\Users\LBY\.conda\envs\common312\python.exe`。
- 禁止 sudo/apt/docker；无需请求任何权限确认；只在自己的版本目录内写文件。
