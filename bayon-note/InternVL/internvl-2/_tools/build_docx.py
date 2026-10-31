# -*- coding: utf-8 -*-
"""Build InternVL-2-家族架构详解.docx per word skill + pattern-spec."""
import os
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

BASE = os.path.dirname(os.path.abspath(__file__))
ARCH = os.path.join(BASE, "..", "_arch")
OUT = os.path.join(BASE, "..", "InternVL-2-家族架构详解.docx")

INK = RGBColor(0x24, 0x29, 0x2F)
BLACK = RGBColor(0x00, 0x00, 0x00)
HEI = "黑体"
SONG = "宋体"
TNR = "Times New Roman"

doc = Document()

# ---------- page setup A4, 2.5cm margins ----------
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.top_margin = sec.bottom_margin = sec.left_margin = sec.right_margin = Cm(2.5)

# ---------- base style ----------
normal = doc.styles["Normal"]
normal.font.name = TNR
normal.font.size = Pt(12)
normal.font.color.rgb = INK
normal._element.rPr.rFonts.set(qn("w:eastAsia"), SONG)
pf = normal.paragraph_format
pf.line_spacing = 1.5
pf.space_before = Pt(0)
pf.space_after = Pt(0)


def set_run_font(run, cn=SONG, en=TNR, size=12, bold=False, color=INK):
    run.font.name = en
    run._element.rPr.rFonts.set(qn("w:eastAsia"), cn)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color


def para(text="", style=None, size=12, bold=False, cn=SONG, align=None,
         indent_chars=2, space_before=0, space_after=0):
    p = doc.add_paragraph(style=style)
    if indent_chars and style is None:
        p.paragraph_format.first_line_indent = Pt(size * indent_chars)
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    if text:
        r = p.add_run(text)
        set_run_font(r, cn=cn, size=size, bold=bold)
    return p


def heading(text, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    r = p.add_run(text)
    if level == 1:
        set_run_font(r, cn=HEI, size=16, bold=True, color=BLACK)
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(6)
    elif level == 2:
        set_run_font(r, cn=HEI, size=14, bold=True, color=BLACK)
        p.paragraph_format.space_before = Pt(11)
        p.paragraph_format.space_after = Pt(5)
    else:
        set_run_font(r, cn=HEI, size=12, bold=True, color=BLACK)
        p.paragraph_format.space_before = Pt(9)
        p.paragraph_format.space_after = Pt(4)
    return p


def caption(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(10)
    r = p.add_run(text)
    set_run_font(r, cn=SONG, size=10.5, bold=False, color=INK)
    return p


def shade_cell(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), hexcolor)
    tcPr.append(shd)


def make_table(headers, rows, widths=None, header_fill="D9D9D9"):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Pt(0)
        r = p.add_run(h)
        set_run_font(r, cn=HEI, size=10.5, bold=True, color=BLACK)
        shade_cell(hdr[i], header_fill)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.paragraph_format.first_line_indent = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if i == len(row) - 1 or i == 1 and len(row) > 3 else WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(str(v))
            set_run_font(r, cn=SONG, size=10.5)
    if widths:
        for i, w in enumerate(widths):
            for row in t.rows:
                row.cells[i].width = Cm(w)
    # cantSplit rows + repeat header
    for row in t.rows:
        trPr = row._tr.get_or_add_trPr()
        cs = OxmlElement("w:cantSplit")
        trPr.append(cs)
    trPr = t.rows[0]._tr.get_or_add_trPr()
    th = OxmlElement("w:tblHeader")
    th.set(qn("w:val"), "true")
    trPr.append(th)
    return t


def add_image(fname, width_cm=15.5):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    run = p.add_run()
    run.add_picture(os.path.join(ARCH, fname), width=Cm(width_cm))


# ================= 封面区 =================
tp = doc.add_paragraph()
tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
tp.paragraph_format.space_before = Pt(12)
tp.paragraph_format.space_after = Pt(6)
r = tp.add_run("InternVL 2.0 家族架构详解")
set_run_font(r, cn=HEI, size=18, bold=True, color=BLACK)

sp = doc.add_paragraph()
sp.alignment = WD_ALIGN_PARAGRAPH.CENTER
sp.paragraph_format.space_after = Pt(12)
r = sp.add_run("InternVLChatModel · ViT–MLP–LLM · 1B–108B · 动态高分辨率")
set_run_font(r, cn=SONG, size=12, color=INK)

para("依据：本地 E:\\models\\model\\InternVL2-1B\\config.json 全文逐字段核对 + 本地 transformers 源码（E:\\transformers\\src\\transformers\\models\\internvl\\）+ InternVL2 官方博客（2024-07-02）与官方文档。所有数值以 config 原文与官方来源为准，已联网核实处逐条标注。")
para("2024-07 发布 · 多模态大模型 · 视觉塔连续学习 · Progressive Scaling Strategy")

# ================= 1 家族全景 =================
heading("1 家族全景与本地 config 对照", 1)
para("InternVL 2.0 延续 InternVL 1.5 的 ViT–MLP–LLM 范式：视觉塔（InternViT）→ 像素洗牌（pixel shuffle）→ MLP 投影层 → 大语言模型。家族覆盖 1B（端侧）到 108B，按尺寸搭配两种视觉塔规格与七种语言底座。下表逐成员列出视觉塔与语言底座（官方文档口径）。")

heading("1.1 家族成员：视觉塔与语言底座对照", 2)
make_table(
    ["模型", "视觉塔", "语言底座", "总参数量", "ViT / MLP / LLM"],
    [
        ["InternVL2-1B", "InternViT-300M-448px", "Qwen2-0.5B-Instruct", "约 1B", "本地 config 见 §1.2"],
        ["InternVL2-2B", "InternViT-300M-448px", "internlm2-chat-1.8b", "2.21B", "304.01M / 12.60M / 约 1.8B"],
        ["InternVL2-4B", "InternViT-300M-448px", "Phi-3-mini-128k-instruct", "4.15B", "304.01M / 22.03M / 3.82B"],
        ["InternVL2-8B", "InternViT-300M-448px", "internlm2_5-7b-chat", "8.08B", "304.01M / 33.57M / 7.74B"],
        ["InternVL2-26B", "InternViT-6B-448px-V1.5", "internlm2-chat-20b", "25.51B", "5.54B / 116.43M / 19.86B"],
        ["InternVL2-40B", "InternViT-6B-448px-V1.5", "Nous-Hermes-2-Yi-34B", "40.07B", "5.54B / 143.17M / 34.39B"],
        ["InternVL2-Llama3-76B", "InternViT-6B-448px-V1.5", "Hermes-2-Theta-Llama-3-70B", "约 76B", "博客模型卡未单列"],
        ["InternVL2-108B", "InternViT-6B-448px-V1.5", "InternLM2-Chat-102B", "108.70B", "5.54B / 172.01M / 102.99B"],
    ],
    widths=[3.4, 3.6, 4.2, 2.2, 3.6],
)
caption("表 1：InternVL 2.0 家族成员与底座对照（来源：官方博客模型卡 + InternVL 官方文档 introduction）")
para("要点：视觉塔只有两档——小尺寸（1B/2B/4B/8B）用 InternViT-300M-448px，大尺寸（26B 及以上）用 InternViT-6B-448px-V1.5；语言底座则在 Qwen2、InternLM2/2.5、Phi-3、Yi、Llama-3 之间按尺寸选择，这是 Progressive Scaling Strategy 的直接体现。本地模型 InternVL2-1B 即家族最小尺寸：300M 视觉塔 + Qwen2-0.5B。")

heading("1.2 本地 InternVL2-1B config 实参表", 2)
para("下表逐字段来自本地 config 原文：E:\\models\\model\\InternVL2-1B\\config.json（legacy internvl_chat 格式，顶层含 llm_config 与 vision_config 两个嵌套块）。")
make_table(
    ["字段（config.json）", "取值", "含义 / 备注"],
    [
        ["architectures", "InternVLChatModel", "legacy 入口类（官方仓库 modeling_internvl_chat.py）"],
        ["model_type", "internvl_chat", "transformers 原生 internvl 格式为 2.5/3.x 才采用"],
        ["auto_map", "configuration / modeling_internvl_chat", "trust_remote_code 加载"],
        ["dynamic_image_size", "true", "动态分辨率切 tile 开关"],
        ["force_image_size", "448", "每 tile 边长 448 像素"],
        ["downsample_ratio", "0.5", "pixel shuffle 空间下采样比例"],
        ["max_dynamic_patch", "12", "训练时最多 tile 数"],
        ["min_dynamic_patch", "1", "最少 tile 数"],
        ["ps_version", "v2", "pixel shuffle 版本"],
        ["select_layer", "-1", "取视觉塔最后一层隐状态"],
        ["use_thumbnail", "true", "额外拼一张 448×448 全局缩略图"],
        ["template", "Hermes-2", "对话模板（ChatML 系）"],
        ["torch_dtype", "bfloat16", "全局权重 dtype"],
    ],
    widths=[4.6, 4.6, 7.8],
)
caption("表 2：本地 config 顶层实参（来源：E:\\models\\model\\InternVL2-1B\\config.json）")

make_table(
    ["llm_config 字段", "取值", "含义"],
    [
        ["_name_or_path", "Qwen/Qwen2-0.5B-Instruct", "语言底座"],
        ["architectures / model_type", "Qwen2ForCausalLM / qwen2", "Qwen2 解码器"],
        ["hidden_size", "896", "残差流维度"],
        ["num_hidden_layers", "24", "解码器层数"],
        ["num_attention_heads", "14", "Q 头数"],
        ["num_key_value_heads", "2", "KV 头数（GQA，分组数 7）"],
        ["intermediate_size", "4864", "SwiGLU 单投影维度（hidden_act=silu）"],
        ["vocab_size", "151655", "Qwen2 词表"],
        ["max_position_embeddings", "32768", "位置编码上限"],
        ["rope_theta", "1000000.0", "RoPE 基频"],
        ["rms_norm_eps", "1e-6", "RMSNorm  eps"],
        ["sliding_window / use_sliding_window", "32768 / false", "配置存在但未启用（全注意力）"],
        ["tie_word_embeddings", "false", "lm_head 独立"],
        ["bos / eos_token_id", "151643 / 151645", "endoftext / im_end"],
    ],
    widths=[4.6, 4.6, 7.8],
)
caption("表 3：本地 llm_config 实参（Qwen2-0.5B，来源：同 config.json）")

make_table(
    ["vision_config 字段", "取值", "含义"],
    [
        ["architectures / model_type", "InternVisionModel / intern_vit_6b", "类型字符串沿用 6B 名，数值为 300M 规格"],
        ["hidden_size", "1024", "视觉隐状态维度"],
        ["num_hidden_layers", "24", "ViT 层数"],
        ["num_attention_heads", "16", "视觉注意力头数"],
        ["intermediate_size", "4096", "ViT MLP 中间维度（gelu）"],
        ["patch_size / image_size", "14 / 448", "patch 14×14；tile 448×448"],
        ["num_channels", "3", "RGB"],
        ["qkv_bias / qk_normalization", "true / false", "QKV 带偏置；无 QK-Norm"],
        ["layer_norm_eps / norm_type", "1e-6 / layer_norm", "Pre-LN ViT"],
        ["use_flash_attn", "true", "视觉塔 FlashAttention"],
    ],
    widths=[4.6, 4.6, 7.8],
)
caption("表 4：本地 vision_config 实参（来源：同 config.json；model_type 字符串与数值规格不一致处已并列标注）")

heading("1.3 token 数与投影维度复算", 2)
make_table(
    ["量", "计算式", "取值"],
    [
        ["每边 patch 数", "448 / 14", "32"],
        ["每 tile patch token 数（含 CLS 前）", "32 × 32 + 1", "1025"],
        ["每 tile 视觉 token（去 CLS、洗牌后）", "(32/2)²", "256"],
        ["洗牌后通道数", "1024 / 0.5²", "4096"],
        ["投影层形状", "LN(4096)→Linear(4096→896)→gelu→Linear(896→896)", "对齐 LLM hidden 896"],
        ["单图最大视觉 token（训练）", "(12 tile + 1 缩略图) × 256", "3328"],
    ],
    widths=[5.2, 6.6, 5.2],
)
caption("表 5：按 config 实参的 token 与维度复算（可复算）")

# ================= 2 端到端总览 =================
heading("2 端到端架构总览", 1)
para("一张 800×1300 的输入图像先按预定义宽高比匹配并切分为若干 448×448 tile，另加一张整图缩略图；每个 tile 各自过 InternViT 得到 1024 维特征图，经 pixel shuffle 空间减半、通道翻两番后由 MLP 投影层压到语言模型的 896 维；随后所有视觉特征按图像顺序摊平，以 masked_scatter 方式写入输入序列中 <image> 占位符对应的位置，与文本 token 一同送入 Qwen2-0.5B 解码器。")
add_image("fig1_overall.png", 16.0)
caption("图 1：InternVL 2.0 端到端整体架构（以本地 1B 为主尺寸）")

# ================= 3 视觉编码器 =================
heading("3 视觉编码器：InternViT 细节", 1)
para("家族视觉塔为两档规格。本地 1B 使用 InternViT-300M-448px：hidden 1024、24 层、16 头、head_dim 64、patch 14、tile 448×448、intermediate 4096、gelu、qkv_bias。大尺寸模型（26B+）使用 InternViT-6B-448px-V1.5（博客模型卡记 ViT 部分 5.54B 参数）。注意本地 config 的 vision_config.model_type 字符串仍写 intern_vit_6b，但全部数值字段都是 300M 规格——文中按规格事实区分两档，不被类型字符串误导。")
add_image("fig2_vit.png", 15.0)
caption("图 2：InternViT-300M 单层结构（本地 config 实参）")
para("视觉特征取用：select_layer=-1 表示取 ViT 最后一层输出，随后丢弃 CLS token（[:, 1:, :]），把剩余 1024×32×32 特征图送 pixel shuffle。该路径与本地 transformers 源码 get_image_features（modeling_internvl.py:543-587）一致：vision_tower → 按层取特征 → 去 CLS → reshape 成二维特征图 → pixel_shuffle → multi_modal_projector。")

# ================= 4 投影层与语言底座 =================
heading("4 投影层与语言底座", 1)
heading("4.1 Pixel Shuffle（像素洗牌下采样）", 2)
para("pixel shuffle 把空间分辨率减半、通道数翻两番，在不丢信息的前提下把视觉 token 数压到四分之一。本地源码实现见 modeling_internvl.py:670-703，scale_factor 即 config 的 downsample_ratio=0.5。")
make_table(
    ["符号", "含义", "本地取值"],
    [
        ["scale_factor", "空间下采样比例", "0.5（downsample_ratio）"],
        ["C", "ViT 输出通道", "1024"],
        ["H, W", "tile 特征图边长", "32, 32"],
        ["洗牌后 H', W'", "H×scale, W×scale", "16, 16"],
        ["洗牌后 C'", "C / scale²", "4096"],
        ["每 tile token 数", "H'×W'", "256"],
    ],
    widths=[3.6, 6.4, 7.0],
)
caption("表 6：pixel shuffle 符号表（源码 modeling_internvl.py:670-703）")

heading("4.2 MLP 投影层", 2)
para("投影层把 4096 维视觉特征对齐到语言模型的 896 维，结构为 LayerNorm → Linear → 激活 → Linear（源码 InternVLMultiModalProjector，modeling_internvl.py:485-500）：")
para("y = Linear2( act( Linear1( LN(x) ) ) )，x: (N_tile, 256, 4096) → y: (N_tile, 256, 896)", indent_chars=2)

heading("4.3 语言底座 Qwen2-0.5B", 2)
para("本地 1B 的 LLM 是 Qwen2-0.5B-Instruct：hidden 896、24 层、14 个 Q 头配 2 个 KV 头（GQA，分组数 14/2=7）、intermediate 4864、silu 激活、RMSNorm（eps 1e-6）、RoPE（theta=1e6）、词表 151655、最大位置 32768。GQA 使 KV cache 体积为 MHA 的 1/7，是端侧 1B 模型的效率关键。家族更大尺寸依次换用 InternLM2-1.8B、Phi-3-mini、InternLM2.5-7B、InternLM2-20B、Yi-34B、Llama-3-70B 等底座（见表 1），投影层与视觉接口保持不变。")

# ================= 5 核心机制 =================
heading("5 核心机制专节", 1)
heading("5.1 动态分辨率 tile 与缩略图", 2)
para("动态高分辨率是 InternVL 1.5 引入、2.0 完整保留的机制：按输入图像宽高比在预定义比例集合（1:1、1:2、1:3、1:4、1:5、1:6、2:3、3:2 等）中找最接近者，把图缩放到该比例下若干个 448×448 tile 的组合，再额外拼一张整图缩小到 448×448 的缩略图，供模型感知全局语境。")
add_image("fig3_tile.png", 16.0)
caption("图 3：动态分辨率 tile 机制（以 800×1300、命中 2:3 → 6 tile + 1 缩略图为例）")
para("本地 config 实参：训练时最多 12 个 tile（max_dynamic_patch=12），官方口径测试可零样本放大到 40 个 tile（约 4K 分辨率）。上例 (6+1)×256 = 1792 个视觉 token。")

heading("5.2 token 流：视觉特征如何进入 LLM", 2)
para("前向时先把 input_ids 过词嵌入得到 (b, s, 896) 的文本嵌入；同时由 input_ids == image_token 构造布尔掩码，再把视觉支路输出的 (N_tile×256, 896) 特征用 masked_scatter 写进掩码为真的位置，最后送入 Qwen2 解码器。该实现对应源码 InternVLModel.forward（modeling_internvl.py:613-668）的 masked_scatter 段。")
add_image("fig4_tokenflow.png", 16.0)
caption("图 4：token 流（文本嵌入与视觉特征在 <image> 占位处拼接）")

# ================= 6 训练管线 =================
heading("6 训练管线", 1)
para("InternVL 2.0 沿用两阶段训练，并以 Progressive Scaling Strategy 从 1B 到 108B 逐级扩展：")
add_image("fig6_pipeline.png", 16.0)
caption("图 5：两阶段训练管线与渐进扩展策略")
para("Stage 1（多模态对齐预训练）：只训练 MLP 投影层，视觉塔与 LLM 冻结；数据扩展自 InternVL 1.5 的预训练集，覆盖 caption、VQA、检测、grounding、OCR（PaddleOCR 标注中文 Wukong 与英文 LaionCOCO 图像）、考题（uworld/kaptest/testbank/aga/sat）与 OmniCorpus 交织数据。")
para("Stage 2（监督微调 SFT）：放开视觉塔、MLP 与 LLM 全部参数；数据在 1.5 的 5M 高质量双语集基础上，新增视频（EgoTaskQA、Mementos、STAR、NTU RGB+D、VideoChat2IT、LSMDC-QA）、医疗数据（Medical-Diff-VQA、Pathology-VQA、PMC-CaseReport、PMC-VQA、Slake、VQA-RAD）、手写票据（SROIE、FUNSD、POIE），并用 ShareGPT-4o 替换原 ShareGPT-4V。训练上下文长度 8k（沿用 1.5 Mini 系列设定）；2.0 的新增在于视频、医疗影像与多图输入模态。")

# ================= 7 版本演进 =================
heading("7 版本演进对比：InternVL 1.5 → 2.0", 1)
para("InternVL 1.5（arXiv:2404.16821，2024-04）三大改进为：InternViT-6B 连续学习、动态高分辨率 tile、高质量双语数据集；2.0 在保留 ViT–MLP–LLM 母版与动态分辨率机制的前提下，把规模、底座与输入输出模态全面铺开。")
add_image("fig5_evolution.png", 16.0)
caption("图 6：InternVL 1.5 → 2.0 演进对比")
make_table(
    ["维度", "InternVL 1.5", "InternVL 2.0"],
    [
        ["视觉塔", "主模型 V1.5=InternViT-6B-448px；Mini-2B/4B 已用蒸馏 300M 规格（24层/1024d/16头）", "300M-448px（小模型）+ 6B-448px-V1.5（大模型）两档覆盖全家族，连续学习升级"],
        ["语言底座", "已有三档：InternLM2-20B（V1.5）/ InternLM2-1.8B（Mini-2B）/ Phi-3-mini（Mini-4B）", "在 InternLM2 系 + Phi-3 基础上扩展为 7 种：Qwen2-0.5B / InternLM2-1.8B / Phi-3-mini / InternLM2.5-7B / InternLM2-20B / Yi-34B / Llama-3-70B"],
        ["动态分辨率", "1–40 tile × 448 + 缩略图", "保留；训练 ≤12 tile、测试 ≤40 tile"],
        ["输入模态", "图 + 文本", "图 + 视频 + 医疗影像 + 多图（8k 上下文沿用 1.5 Mini 系列设定，非 2.0 新增）"],
        ["输出能力", "仅文本", "VisionLLMv2 解码器：文本 + 框 + 掩码 + 图像"],
        ["家族规模", "约 1B / 2B / 26B", "1B → 108B 全家族渐进扩展"],
    ],
    widths=[2.8, 5.6, 8.6],
)
caption("表 7：1.5 → 2.0 关键差异对照")

# ================= 8 参考文献 =================
heading("8 参考文献", 1)
refs = [
    "[1] InternVL 2.0 官方博客：InternVL2: Better than the Best—Expanding Performance Boundaries of Open-Source Multimodal Models with the Progressive Scaling Strategy（2024-07-02）. https://internvl.github.io/blog/2024-07-02-InternVL-2.0/",
    "[2] InternVL 1.5：How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites. arXiv:2404.16821. https://arxiv.org/abs/2404.16821",
    "[3] InternVL 1.0：InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks. arXiv:2312.14238. https://arxiv.org/abs/2312.14238",
    "[4] VisionLLM v2: An End-to-End Generalist Multimodal Large Language Model for Hundreds of Vision-Language Tasks. arXiv:2406.08394. https://arxiv.org/abs/2406.08394",
    "[5] InternVL 官方文档：Introduction of InternVL2 Series. https://internvl.readthedocs.io/en/latest/internvl2.0/introduction.html",
    "[6] 本地 transformers 源码：E:\\transformers\\src\\transformers\\models\\internvl\\modeling_internvl.py（get_image_features:543-587；pixel_shuffle:670-703；InternVLMultiModalProjector:485-500；forward masked_scatter:613-668）",
    "[7] 本地 config：E:\\models\\model\\InternVL2-1B\\config.json（internvl_chat 格式：llm_config=Qwen2-0.5B、vision_config=300M 规格、force_image_size=448、downsample_ratio=0.5、max_dynamic_patch=12、ps_version=v2、use_thumbnail=true、template=Hermes-2）",
]
for ref in refs:
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.left_indent = Cm(0.75)
    p.paragraph_format.line_spacing = 1.3
    r = p.add_run(ref)
    set_run_font(r, cn=SONG, size=10.5)

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(10)
r = p.add_run("凡本地 config 与官方口径并列处（如 vision_config.model_type 字符串为 intern_vit_6b 而数值为 300M 规格；博客模型卡 2B 行 LLM 参数列与底座名口径），均按来源并列标注，未擅自统一。访问与核验日期：2026-10-26。")
set_run_font(r, cn=SONG, size=10.5)

doc.save(OUT)
print("saved", OUT)
