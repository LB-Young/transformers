# -*- coding: utf-8 -*-
"""生成 InternVL-2_5-家族架构详解.docx（A4、编号章节、嵌入5图、config实参表、符号表）。"""
import os
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ARCH = r"E:\transformers\bayon-note\InternVL\internvl-2_5\_arch"
OUT  = r"E:\transformers\bayon-note\InternVL\internvl-2_5\InternVL-2_5-家族架构详解.docx"

INK = RGBColor(0x24, 0x29, 0x2F)
SUB = RGBColor(0x6b, 0x72, 0x80)
ACC = RGBColor(0x0e, 0x74, 0x90)
HL  = RGBColor(0xb4, 0x53, 0x09)

doc = Document()

# ---------- 页面 A4 + 2.5cm 边距 ----------
sec = doc.sections[0]
sec.page_width = Cm(21.0); sec.page_height = Cm(29.7)
sec.top_margin = sec.bottom_margin = Cm(2.5)
sec.left_margin = sec.right_margin = Cm(2.5)

# ---------- 基础样式 ----------
def set_font(style_or_run, ascii_f="Times New Roman", ea_f="宋体", size=12, bold=False, color=INK):
    f = style_or_run.font
    f.name = ascii_f
    f.size = Pt(size)
    f.bold = bold
    f.color.rgb = color
    rpr = style_or_run.element.get_or_add_rPr() if hasattr(style_or_run, "element") else None
    rFonts = style_or_run.element.rPr.rFonts if (hasattr(style_or_run, "element") and style_or_run.element.rPr is not None) else None

normal = doc.styles["Normal"]
normal.font.name = "Times New Roman"
normal.font.size = Pt(12)
normal.font.color.rgb = INK
_nrpr = normal.element.get_or_add_rPr(); _nrf = _nrpr.get_or_add_rFonts()
_nrf.set(qn("w:ascii"), "Times New Roman"); _nrf.set(qn("w:hAnsi"), "Times New Roman")
_nrf.set(qn("w:eastAsia"), "宋体"); _nrf.set(qn("w:cs"), "Times New Roman")
for attr in ("w:asciiTheme","w:hAnsiTheme","w:eastAsiaTheme","w:csTheme"):
    if _nrf.get(qn(attr)) is not None: del _nrf.attrib[qn(attr)]
pf = normal.paragraph_format
pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
pf.space_before = Pt(0); pf.space_after = Pt(0)

def style_heading(name, size):
    st = doc.styles[name]
    st.font.name = "Times New Roman"
    st.font.size = Pt(size)
    st.font.bold = True
    st.font.color.rgb = INK
    rpr = st.element.get_or_add_rPr()
    rFonts = rpr.get_or_add_rFonts()
    rFonts.set(qn("w:ascii"), "Times New Roman")
    rFonts.set(qn("w:hAnsi"), "Times New Roman")
    rFonts.set(qn("w:eastAsia"), "黑体")
    rFonts.set(qn("w:cs"), "Times New Roman")
    for attr in ("w:asciiTheme","w:hAnsiTheme","w:eastAsiaTheme","w:csTheme"):
        if rFonts.get(qn(attr)) is not None:
            del rFonts.attrib[qn(attr)]
    st.paragraph_format.space_before = Pt(12 if name=="Heading 1" else 9)
    st.paragraph_format.space_after = Pt(6)
    st.paragraph_format.keep_with_next = True

style_heading("Heading 1", 16)
style_heading("Heading 2", 14)
style_heading("Heading 3", 12)

# ---------- 段落 helper ----------
def run_font(r, ascii_f="Times New Roman", ea_f="宋体", size=12, bold=False, color=INK, italic=False):
    r.font.name = ascii_f; r.font.size = Pt(size); r.font.bold = bold; r.font.italic = italic
    r.font.color.rgb = color
    r._element.rPr.rFonts.set(qn("w:eastAsia"), ea_f)
    return r

def body(text, indent=True, size=12):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if indent:
        p.paragraph_format.first_line_indent = Pt(size*2)
    r = p.add_run(text)
    run_font(r, size=size)
    return p

def caption(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4); p.paragraph_format.space_after = Pt(10)
    r = p.add_run(text)
    run_font(r, size=10.5, color=SUB)
    return p

def add_image(fname, width_cm=15.5):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8); p.paragraph_format.space_after = Pt(2)
    run = p.add_run()
    run.add_picture(os.path.join(ARCH, fname), width=Cm(width_cm))

def h1(t): doc.add_heading(t, level=1)
def h2(t): doc.add_heading(t, level=2)
def h3(t): doc.add_heading(t, level=3)

# ---------- 表格 helper ----------
def set_cell_border(cell):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge in ("top","left","bottom","right"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4")
        e.set(qn("w:space"), "0"); e.set(qn("w:color"), "9CA3AF")
        borders.append(e)
    tcPr.append(borders)

def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), hexcolor)
    tcPr.append(sh)

def no_indent(cell):
    for p in cell.paragraphs:
        p.paragraph_format.first_line_indent = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.space_before = Pt(1); p.paragraph_format.space_after = Pt(1)

def make_table(headers, rows, widths=None, header_fill="D9E2E8"):
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    hdr = t.rows[0].cells
    for i, htext in enumerate(headers):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(htext); run_font(r, size=10.5, bold=True, color=INK)
        shade(hdr[i], header_fill); set_cell_border(hdr[i]); no_indent(hdr[i])
        hdr[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    # repeat header on new page
    trPr = t.rows[0]._tr.get_or_add_trPr()
    tblHeader = OxmlElement("w:tblHeader"); tblHeader.set(qn("w:val"), "true"); trPr.append(tblHeader)
    for ridx, row in enumerate(rows):
        cells = t.add_row().cells
        # cantSplit per row
        rPr = t.rows[-1]._tr.get_or_add_trPr()
        cant = OxmlElement("w:cantSplit"); rPr.append(cant)
        for i, val in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if i==0 else WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(str(val)); run_font(r, size=10.5)
            set_cell_border(cells[i]); no_indent(cells[i])
            cells[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if ridx % 2 == 1: shade(cells[i], "F3F6F8")
    if widths:
        for i, w in enumerate(widths):
            for row in t.rows:
                row.cells[i].width = Cm(w)
    return t

def formula_block(lines):
    """文本公式块：等宽、浅底、居中。"""
    for ln in lines:
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Pt(0)
        p.paragraph_format.space_before = Pt(2); p.paragraph_format.space_after = Pt(2)
        r = p.add_run(ln); run_font(r, ascii_f="Consolas", ea_f="宋体", size=10.5, color=ACC)

# ================= 封面标题 =================
title = doc.add_paragraph(); title.alignment = WD_ALIGN_PARAGRAPH.CENTER
title.paragraph_format.space_before = Pt(6); title.paragraph_format.space_after = Pt(4)
r = title.add_run("InternVL 2.5 家族架构详解")
run_font(r, ascii_f="Times New Roman", ea_f="黑体", size=22, bold=True, color=INK)

sub = doc.add_paragraph(); sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = sub.add_run("ViT–MLP–LLM · InternViT-300M/6B × Qwen2.5/InternLM2.5 · 1B–78B")
run_font(r, ea_f="楷体", size=13, color=SUB)

meta = doc.add_paragraph(); meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = meta.add_run("依据本地 config.json（E:\\models\\model\\InternVL2_5-1B\\config.json）+ 本地 transformers 源码 + 论文 arXiv:2412.05271 · 2024-12 发布")
run_font(r, size=10.5, color=SUB)

body("InternVL 2.5 是上海 AI Lab（OpenGVLab）在 InternVL 2.0 基础上推出的开源多模态大模型系列，沿用成熟的 ViT–MLP–LLM 范式，在模型底座、数据规模与测试时推理三方面同时扩展（model / data / test-time scaling）。本文以本地 InternVL2_5-1B 的 legacy internvl_chat 格式 config.json 为实参锚点，结合论文 Table 1/2 与本地 transformers 源码 modeling_internvl.py 逐项复算；家族覆盖 1B–78B 共 7 个尺寸，本地 1B 使用 InternViT-300M-448px-V2.5 视觉塔与 Qwen2.5-0.5B-Instruct 语言底座。")

# ================= 1 家族全景 =================
h1("1 家族全景与本地 config 对照")
body("本地模型目录为 E:\\models\\model\\InternVL2_5-1B，config.json 采用 legacy internvl_chat 格式：顶层为 llm_config（嵌套语言底座）与 vision_config（嵌套视觉塔），外加 dynamic_image_size、force_image_size、downsample_ratio 等多模态预处理超参。注意一个易错点：vision_config.model_type 字符串为 intern_vit_6b，但本地 1B 的实际数值（hidden 1024、24 层、16 头、patch 14、448px）是 InternViT-300M 规格——家族视觉塔有 300M 与 6B 两个规格，本地小模型用 300M/448px。")

h2("1.1 本地 config 顶层与多模态超参实参表")
make_table(
    ["config 字段（顶层）", "取值", "含义 / 来源"],
    [
        ["architectures", "InternVLChatModel", "legacy 入口类（auto_map 指向 modeling_internvl_chat.InternVLChatModel）"],
        ["model_type", "internvl_chat", "legacy 格式；非 transformers 原生 internvl"],
        ["dynamic_image_size", "true", "按图像宽高行动态切 448 tile"],
        ["force_image_size", "448", "每个 tile 的边长（像素）"],
        ["downsample_ratio", "0.5", "pixel unshuffle 下采样比（token 数降为 1/4）"],
        ["max_dynamic_patch", "12", "动态 tile 数量上限"],
        ["min_dynamic_patch", "1", "动态 tile 数量下限"],
        ["use_thumbnail", "true", "额外拼接 1 个整图缩略图 tile"],
        ["ps_version", "v2", "pixel shuffle 版本"],
        ["select_layer", "-1", "取 ViT 最后一层 hidden state"],
        ["template", "internvl2_5", "对话模板"],
        ["torch_dtype", "bfloat16", "训练/推理精度"],
        ["use_backbone_lora / use_llm_lora", "0 / 0", "未启用 LoRA"],
    ],
    widths=[5.2, 4.0, 6.8])
caption("表 1  本地 config 顶层实参（来源：E:\\models\\model\\InternVL2_5-1B\\config.json）")

h2("1.2 语言底座 llm_config 实参表")
make_table(
    ["llm_config 字段", "取值", "含义"],
    [
        ["_name_or_path", "Qwen/Qwen2.5-0.5B-Instruct", "底座来源"],
        ["architectures / model_type", "Qwen2ForCausalLM / qwen2", "Qwen2.5 解码器"],
        ["hidden_size", "896", "残差流维度"],
        ["num_hidden_layers", "24", "Transformer 层数"],
        ["num_attention_heads", "14", "Query 头数"],
        ["num_key_value_heads", "2", "KV 头数（GQA：14Q 共享 2 KV）"],
        ["intermediate_size", "4864", "SwiGLU 中间维"],
        ["hidden_act", "silu", "激活函数"],
        ["vocab_size", "151674", "词表大小（比 2.0 的 Qwen2-0.5B 的 151655 更大）"],
        ["max_position_embeddings", "32768", "最大位置长度"],
        ["max_window_layers", "21", "前 21 层用全局注意力（Qwen2.5 滑窗设计）"],
        ["sliding_window / use_sliding_window", "32768 / false", "滑窗未实际启用"],
        ["rms_norm_eps", "1e-6", "RMSNorm  eps"],
        ["rope_theta", "1000000.0", "RoPE 基频 1e6"],
        ["bos / eos_token_id", "151643 / 151645", "起止 token"],
        ["tie_word_embeddings", "false", "嵌入与 lm_head 不绑定"],
    ],
    widths=[5.6, 4.6, 5.8])
caption("表 2  本地 1B 语言底座实参（来源：同 config.json 的 llm_config 节）")

h2("1.3 视觉塔 vision_config 实参表")
make_table(
    ["vision_config 字段", "取值", "含义"],
    [
        ["model_type", "intern_vit_6b（字符串）", "注意：数值实为 300M 规格"],
        ["architectures", "InternVisionModel", "视觉塔类"],
        ["hidden_size", "1024", "ViT 隐藏维"],
        ["num_hidden_layers", "24", "ViT 层数"],
        ["num_attention_heads", "16", "ViT 注意力头数"],
        ["intermediate_size", "4096", "ViT FFN 中间维"],
        ["patch_size", "14", "切块边长"],
        ["image_size", "448", "训练 tile 分辨率"],
        ["hidden_act", "gelu", "ViT MLP 激活"],
        ["qkv_bias", "true", "QKV 投影带 bias"],
        ["qk_normalization", "false", "300M 不做 QK-Norm（6B 才有）"],
        ["norm_type / layer_norm_eps", "layer_norm / 1e-6", "300M 用标准 LN（6B 用 RMSNorm）"],
        ["use_flash_attn", "true", "FlashAttention"],
    ],
    widths=[5.6, 4.6, 5.8])
caption("表 3  本地 1B 视觉塔实参（来源：同 config.json 的 vision_config 节）")

h2("1.4 家族模型表（论文 Table 2）")
body("InternVL 2.5 家族共 7 个尺寸：1B–8B 用 InternViT-300M-448px-V2.5，26B–78B 用 InternViT-6B-448px-V2.5；语言底座全面升级到 Qwen2.5 与 InternLM2.5。OpenCompass 分为论文本地测试口径。")
make_table(
    ["模型", "参数量", "视觉编码器", "语言模型", "OpenCompass"],
    [
        ["InternVL2.5-1B", "0.9B", "InternViT-300M-448px-V2.5", "Qwen2.5-0.5B-Instruct", "54.5"],
        ["InternVL2.5-2B", "2.2B", "InternViT-300M-448px-V2.5", "internlm2_5-1_8b-chat", "59.8"],
        ["InternVL2.5-4B", "3.7B", "InternViT-300M-448px-V2.5", "Qwen2.5-3B-Instruct", "65.1"],
        ["InternVL2.5-8B", "8.1B", "InternViT-300M-448px-V2.5", "internlm2_5-7b-chat", "68.1"],
        ["InternVL2.5-26B", "25.5B", "InternViT-6B-448px-V2.5", "internlm2_5-20b-chat", "71.3"],
        ["InternVL2.5-38B", "38.4B", "InternViT-6B-448px-V2.5", "Qwen2.5-32B-Instruct", "73.3"],
        ["InternVL2.5-78B", "78.4B", "InternViT-6B-448px-V2.5", "Qwen2.5-72B-Instruct", "75.5"],
    ],
    widths=[3.6, 2.0, 4.6, 4.4, 1.8])
caption("表 4  InternVL 2.5 家族模型构成（来源：论文 arXiv:2412.05271 Table 2）")

# ================= 2 端到端架构总览 =================
h1("2 端到端架构总览")
body("InternVL 2.5 保持与 1.5/2.0 相同的 ViT–MLP–LLM 范式：输入图像先按宽高行动态切成若干 448×448 tile（外加 1 张整图缩略图），每个 tile 经 InternViT 编码为视觉 token，再经 pixel unshuffle 降采样与 2 层 MLP 投影到 LLM 隐空间，最后与文本 embedding 在同一序列中拼接，送入因果语言模型自回归生成。")
add_image("fig1_overall.png", 16.0)
caption("图 1  InternVL 2.5 端到端整体架构（ViT–MLP–LLM 范式，以本地 1B 为主尺寸）")

# ================= 3 视觉编码器 =================
h1("3 视觉编码器 InternViT")
body("InternViT 有 300M 与 6B 两个规格。本地 1B 使用 InternViT-300M-448px-V2.5：它是 6B 教师模型经余弦蒸馏得到的 0.3B 变体，24 层、hidden 1024、16 头、patch 14，用标准 LayerNorm、不做 QK-Norm；之后又在更丰富的数据混合上以 next-token prediction（NTP）损失增量预训练到 V2.5。6B 规格则为 45 层、hidden 3200、25 头、带 QK-Norm 与 RMSNorm，参数量 5.5B（从原始 48 层去掉最后 3 层）。")
add_image("fig2_vit.png", 15.0)
caption("图 2  InternViT 视觉编码器层结构（本地 1B = InternViT-300M-448px-V2.5）")

h2("3.1 InternViT 版本演进与规格（论文 Table 1）")
make_table(
    ["版本", "分辨率", "hidden", "层数", "中间维", "头数", "QK-Norm", "归一化", "训练损失", "参数"],
    [
        ["6B-224px", "固定224", "3200", "48", "12800", "25", "有", "RMS", "CLIP", "5.9B"],
        ["6B-448px-V1.0", "固定448", "3200", "48", "12800", "25", "有", "RMS", "NTP", "5.9B"],
        ["6B-448px-V1.2", "固定448", "3200", "45", "12800", "25", "有", "RMS", "NTP", "5.5B"],
        ["6B-448px-V1.5", "动态448", "3200", "45", "12800", "25", "有", "RMS", "NTP", "5.5B"],
        ["6B-448px-V2.5", "动态448", "3200", "45", "12800", "25", "有", "RMS", "NTP", "5.5B"],
        ["300M-Distill", "固定448", "1024", "24", "4096", "16", "无", "LN", "Cosine", "0.3B"],
        ["300M-448px", "动态448", "1024", "24", "4096", "16", "无", "LN", "NTP", "0.3B"],
        ["300M-448px-V2.5", "动态448", "1024", "24", "4096", "16", "无", "LN", "NTP", "0.3B"],
    ],
    widths=[3.0, 1.6, 1.4, 1.2, 1.6, 1.0, 1.3, 1.2, 1.4, 1.2])
caption("表 5  InternViT 各代规格（来源：论文 arXiv:2412.05271 Table 1；本地 1B 对应末行）")

# ================= 4 投影层与语言底座 =================
h1("4 投影层与语言底座")
h2("4.1 Pixel Unshuffle 与视觉 token 复算")
body("视觉特征在送入 MLP 前做 pixel unshuffle（transformers 源码 modeling_internvl.py pixel_shuffle，scale_factor=downsample_ratio=0.5），把每个 448 tile 的视觉 token 数压到 1/4。以下按本地 config 与源码 forward 路径逐项复算。")
formula_block([
    "patch_grid = image_size / patch_size = 448 / 14 = 32",
    "tokens_raw = 32 × 32 = 1024   （去 CLS 后：[:, 1:, :]）",
    "feat = (b, 32, 32, 1024)  --pixel_shuffle(0.5)-->  (b, 16, 16, 4096)",
    "tokens_tile = 16 × 16 = 256   （每 tile 256 个 token，维度 4096）",
    "MLP projector: 4096 → 896   （对齐 LLM hidden_size）",
    "max_tiles = max_dynamic_patch + thumbnail = 12 + 1 = 13",
    "max_visual_tokens = 13 × 256 = 3328",
])
make_table(
    ["符号", "含义", "取值（本地 1B）"],
    [
        ["image_size", "tile 边长", "448（force_image_size）"],
        ["patch_size", "切块边长", "14"],
        ["tokens_raw", "单 tile patch token 数", "32×32 = 1024"],
        ["downsample_ratio", "pixel shuffle 下采样比", "0.5"],
        ["tokens_tile", "单 tile 投影后 token 数", "16×16 = 256"],
        ["dim_tile", "单 token 维度（shuffle 后）", "1024 / 0.5² = 4096"],
        ["LLM hidden", "投影目标维度", "896"],
        ["max_dynamic_patch", "动态 tile 上限", "12"],
        ["use_thumbnail", "缩略图 tile", "true（+1）"],
        ["max visual tokens", "单图视觉 token 上限", "13×256 = 3328"],
    ],
    widths=[3.6, 5.4, 6.0])
caption("表 6  视觉 token 复算符号表（来源：config.json + modeling_internvl.py pixel_shuffle :670-703、get_image_features :543-587）")

h2("4.2 MLP 投影与 Qwen2.5 底座")
body("投影层是一个随机初始化的 2 层 MLP（论文：a randomly initialized 2-layer MLP projector），把 4096 维视觉特征映射到 LLM 的 896 维隐空间。语言底座在 2.5 中全面升级：小尺寸用 Qwen2.5-0.5B/3B 或 internlm2_5-1.8B/7B，大尺寸用 internlm2_5-20B 或 Qwen2.5-32B/72B。本地 1B 底座 Qwen2.5-0.5B 采用 GQA（14 Q 头共享 2 KV 头）、SwiGLU、RMSNorm、RoPE 基频 1e6，词表 151674。")

# ================= 5 核心机制 =================
h1("5 核心机制")
h2("5.1 动态分辨率与像素 tile 机制")
body("动态分辨率策略从 1.5 沿用：图像先按原始宽高比缩放到最近的 448 tile 整数倍网格，高分辨率图被切成多个 tile，低分辨率图 tile 数少于上限；同时额外拼接一张整图缩略图（thumbnail）以保留全局视野。max_dynamic_patch=12 限制 tile 数，避免高分辨率图产生过长视觉序列。")
add_image("fig3_tile.png", 16.0)
caption("图 3  动态分辨率与像素 tile 机制（dynamic_image_size / max_dynamic_patch=12 / use_thumbnail）")

h2("5.2 token 流与图像占位 token 注入")
body("在 embedding 阶段，视觉特征通过 masked_scatter 替换输入序列中的图像占位 token，使视觉 token 与文本 token 在同一序列内参与因果注意力。占位 token 的数量必须与视觉特征 token 数严格相等（源码 get_placeholder_mask 会校验，不一致直接报错）。")
add_image("fig4_tokenflow.png", 16.0)
caption("图 4  视觉 token 流与图像占位 token 的拼接位置（本地 1B：LLM hidden=896）")

# ================= 6 训练管线 =================
h1("6 训练管线")
body("InternVL 2.5 单模型训练分三个阶段（论文 Figure 4 / Table 3），逐步解锁可训练参数，兼顾对齐稳定性与视觉能力。")
make_table(
    ["阶段", "可训练参数", "数据", "学习率", "目的"],
    [
        ["Stage 1：MLP Warmup", "仅 MLP", "预训练混合（对话格式 ChatML）", "较高（如 2e-4）", "冻结 ViT 与 LLM，先打通跨模态对齐，从本阶段起即用动态高分辨率训练"],
        ["Stage 1.5：ViT 增量学习（可选）", "ViT + MLP", "预训练混合", "较低（如 1e-5）", "增强视觉特征提取（多语言 OCR、数学图表等）；训练一次后可复用于不同 LLM"],
        ["Stage 2：全量指令微调", "ViT + MLP + LLM 全量", "高质量多模态指令数据", "统一（如 4e-5）", "全模型训练，严格数据质量控制，抑制重复输出等异常"],
    ],
    widths=[3.6, 2.8, 3.8, 2.2, 4.6])
caption("表 7  三阶段训练管线（来源：论文 arXiv:2412.05271 §3.2 Figure 4 / Table 3）")
body("配套采用渐进式扩展策略（progressive scaling）：先用较小 LLM 联合训练 ViT，得到的视觉特征可迁移到更大 LLM 而无需重训，从而以较低成本完成大规模对齐。数据规模从 1.5 的 5.1M、2.0 的 7.3M 翻倍到 2.5 的 16.3M，并严格过滤异常与重复样本；InternVL2.5-78B 仅用约 120B token 训练（对比 Qwen2-VL 累计 1.4T）。测试时则引入 CoT 思维链提示实现 test-time scaling，78B 在 MMMU 验证集上成为首个超过 70 分的开源模型。")

# ================= 7 版本演进对比 =================
h1("7 版本演进对比：InternVL 2.0 → 2.5")
body("2.5 保持 2.0 的 ViT–MLP–LLM 架构与动态分辨率机制不变，升级集中在三方面：模型（视觉塔 V2.5 增量 NTP 训练、语言底座换 Qwen2.5/InternLM2.5）、数据（样本翻倍至 16.3M 并严格去重过滤）、测试时扩展（CoT 推理）。家族方面，2.0 开源档覆盖 1B/2B/4B/8B/26B/40B/Llama3-76B，并另含闭源的 InternVL2-Pro 108B；2.5 开源档为 1B/2B/4B/8B/26B/38B/78B（新增 38B 与 78B，未再单列 Pro 档）。")
add_image("fig5_evo.png", 16.0)
caption("图 5  InternVL 2.0 → 2.5 演进对比（架构同构；升级在模型、数据、测试时扩展）")
make_table(
    ["维度", "InternVL 2.0", "InternVL 2.5"],
    [
        ["视觉塔", "InternViT-300M / 6B（V1.5）", "300M/6B 增量 NTP 训练 → V2.5（视觉塔全参数训练，不冻结）"],
        ["语言底座", "Qwen2 / InternLM2 / Phi-3 / Yi / Llama3", "升级 Qwen2.5 / InternLM2.5"],
        ["1B 底座", "Qwen2-0.5B（vocab 151655）", "Qwen2.5-0.5B（vocab 151674）"],
        ["预训练样本", "7.3M", "16.3M（翻倍 + 严格去重过滤异常样本）"],
        ["推理", "直接回答为主", "CoT 思维链 test-time scaling，MMMU 超 70"],
        ["家族规模", "开源 1B–76B（1B/2B/4B/8B/26B/40B/Llama3-76B）+ InternVL2-Pro 108B", "开源 1B–78B（1B/2B/4B/8B/26B/38B/78B，新增 38B / 78B）"],
    ],
    widths=[3.0, 5.5, 7.5])
caption("表 8  InternVL 2.0 → 2.5 演进对照（来源：论文 arXiv:2412.05271；2.0 见官方博客 2024-07-02）")
body("说明：2.5 已采用原生对话格式的多模态指令数据并对视觉塔做全参数增量训练；更彻底的原生多模态预训练（把语言预训练与多模态对齐合并为单阶段）是在后续 InternVL 3.0（arXiv:2504.10479）中正式提出的。")

# ================= 8 参考文献 =================
h1("8 参考文献")
refs = [
    "[1] InternVL 2.5: Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling. arXiv:2412.05271. https://arxiv.org/abs/2412.05271",
    "[2] InternVL 1.5: How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites. arXiv:2404.16821. https://arxiv.org/abs/2404.16821",
    "[3] InternVL2: Better than the Best — Expanding Performance Boundaries of Open-Source Multimodal Models with the Progressive Scaling Strategy（官方博客，2024-07-02）. https://internvl.github.io/blog/2024-07-02-InternVL-2.0/",
    "[4] InternVL 1.0: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks. arXiv:2312.14238. https://arxiv.org/abs/2312.14238",
    "[5] InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models（原生多模态预训练范式出处）. arXiv:2504.10479. https://arxiv.org/abs/2504.10479",
    "[6] 本地 transformers 源码 modeling_internvl.py / configuration_internvl.py（E:\\transformers\\src\\transformers\\models\\internvl\\）：get_image_features :543-587、pixel_shuffle :670-703、masked_scatter 占位替换 :649-652。",
    "[7] 本地 config：E:\\models\\model\\InternVL2_5-1B\\config.json（legacy internvl_chat 格式，2026-10-26 读取核对）。",
]
for rtext in refs:
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.left_indent = Pt(18)
    p.paragraph_format.first_line_indent = Pt(-18)
    p.paragraph_format.space_after = Pt(3)
    run_font(p.add_run(rtext), size=10.5)

# 页脚
footer = sec.footer
fp = footer.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
run_font(fp.add_run("InternVL 2.5 家族架构详解 · 以本地 config.json 与论文 arXiv:2412.05271 逐项核对 · 仅用于技术学习与论文配图"), size=9, color=SUB)

doc.save(OUT)
print("SAVED", OUT)
