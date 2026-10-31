# -*- coding: utf-8 -*-
"""生成 InternVL 3.5 家族架构详解.docx（A4、编号章节、嵌图、config 实参表、文本公式）。"""
import os
from docx import Document
from docx.shared import Pt, Cm, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCH = os.path.join(ROOT, "_arch")
OUT = os.path.join(ROOT, "InternVL-3_5-家族架构详解.docx")

CONFIG_PATH = r"E:\models\model\InternVL3_5-1B-HF\config.json"
SRC_PATH = r"E:\transformers\src\transformers\models\internvl"

INK = RGBColor(0x24, 0x29, 0x2F)
SUB = RGBColor(0x6b, 0x72, 0x80)
ACC = RGBColor(0x0e, 0x74, 0x90)
HL  = RGBColor(0xb4, 0x53, 0x09)

doc = Document()

# ---------- 页面 A4 + 2.5cm 边距 ----------
sec = doc.sections[0]
sec.page_height = Cm(29.7); sec.page_width = Cm(21.0)
sec.top_margin = Cm(2.5); sec.bottom_margin = Cm(2.5)
sec.left_margin = Cm(2.5); sec.right_margin = Cm(2.5)

# ---------- 基础字体工具 ----------
def set_run_font(run, cn="宋体", en="Times New Roman", size=12, bold=False, color=INK):
    run.font.name = en
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    r = run._element
    rPr = r.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts'); rPr.append(rFonts)
    rFonts.set(qn('w:eastAsia'), cn)
    rFonts.set(qn('w:ascii'), en)
    rFonts.set(qn('w:hAnsi'), en)

def style_normal():
    st = doc.styles['Normal']
    st.font.name = 'Times New Roman'; st.font.size = Pt(12)
    st.font.color.rgb = INK
    st.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    pf = st.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    pf.space_before = Pt(0); pf.space_after = Pt(0)
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

def style_heading(name, size, before, after):
    st = doc.styles[name]
    st.font.name = 'Times New Roman'; st.font.size = Pt(size)
    st.font.bold = True; st.font.color.rgb = INK
    rpr = st.element.get_or_add_rPr()
    rFonts = rpr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts'); rpr.append(rFonts)
    rFonts.set(qn('w:eastAsia'), '黑体')
    rFonts.set(qn('w:ascii'), 'Times New Roman')
    rFonts.set(qn('w:hAnsi'), 'Times New Roman')
    # 清除主题字体 token，避免 majorHAnsi/majorEastAsia 覆盖显式字体
    for attr in ('w:asciiTheme','w:eastAsiaTheme','w:hAnsiTheme','w:cstheme'):
        if rFonts.get(qn(attr)) is not None:
            del rFonts.attrib[qn(attr)]
    pf = st.paragraph_format
    pf.space_before = Pt(before); pf.space_after = Pt(after)
    pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
    pf.keep_with_next = True

style_normal()
style_heading('Heading 1', 16, 14, 6)
style_heading('Heading 2', 14, 11, 5)
style_heading('Heading 3', 12, 9, 4)

# ---------- 段落辅助 ----------
def title(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(18)
    p.paragraph_format.space_before = Pt(6)
    set_run_font(p.add_run(text), cn='黑体', size=18, bold=True, color=INK)
    return p

def h1(text):
    return doc.add_heading(text, level=1)

def h2(text):
    return doc.add_heading(text, level=2)

def h3(text):
    return doc.add_heading(text, level=3)

def body(text, indent=True):
    p = doc.add_paragraph()
    if indent:
        p.paragraph_format.first_line_indent = Pt(24)  # 2 字符
    set_run_font(p.add_run(text), size=12)
    return p

def caption(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(10)
    set_run_font(p.add_run(text), cn='宋体', size=10.5, color=INK)
    return p

def add_figure(fname, cap, width_in=6.3):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    run = p.add_run()
    run.add_picture(os.path.join(ARCH, fname), width=Inches(width_in))
    caption(cap)

# ---------- 表格辅助 ----------
def set_cell_border(cell):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement('w:tcBorders')
    for edge in ('top','left','bottom','right'):
        e = OxmlElement(f'w:{edge}')
        e.set(qn('w:val'),'single'); e.set(qn('w:sz'),'4')
        e.set(qn('w:space'),'0'); e.set(qn('w:color'),'999999')
        borders.append(e)
    tcPr.append(borders)

def shade(cell, hexfill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'),'clear'); shd.set(qn('w:color'),'auto'); shd.set(qn('w:fill'),hexfill)
    tcPr.append(shd)

def cell_text(cell, text, bold=False, size=10.5, align='left', cn='宋体', color=INK):
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    p = cell.paragraphs[0]
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.space_before = Pt(1); p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    if align == 'center': p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align == 'right': p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    else: p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    set_run_font(p.add_run(str(text)), cn=cn, size=size, bold=bold, color=color)

def make_table(headers, rows, widths=None, header_fill='F2F2F2'):
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    hdr = t.rows[0].cells
    for i, htext in enumerate(headers):
        cell_text(hdr[i], htext, bold=True, align='center', cn='黑体')
        shade(hdr[i], header_fill)
        set_cell_border(hdr[i])
    # repeat header row
    trPr = t.rows[0]._tr.get_or_add_trPr()
    th = OxmlElement('w:tblHeader'); th.set(qn('w:val'),'true'); trPr.append(th)
    for ridx, row in enumerate(rows):
        cells = t.add_row().cells
        # cantSplit per row
        trPr = cells[0]._tc.getparent().get_or_add_trPr()
        cs = OxmlElement('w:cantSplit'); cs.set(qn('w:val'),'true'); trPr.append(cs)
        for i, val in enumerate(row):
            al = 'center' if i == 0 else 'left'
            cell_text(cells[i], val, align=al)
            set_cell_border(cells[i])
            if ridx % 2 == 1:
                shade(cells[i], 'FAFBFC')
    if widths:
        for i, w in enumerate(widths):
            for row in t.rows:
                row.cells[i].width = Cm(w)
    return t

def table_caption(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6); p.paragraph_format.space_after = Pt(3)
    set_run_font(p.add_run(text), cn='宋体', size=10.5, bold=True)

# ============================================================
# 正文开始
# ============================================================
title("InternVL 3.5 家族架构详解")
sub = doc.add_paragraph(); sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub.paragraph_format.space_after = Pt(16)
set_run_font(sub.add_run("ViT–MLP–LLM 范式 · Cascade RL · 以本地 InternVL3_5-1B-HF 与 arXiv:2508.18265 为准"),
             cn='楷体', size=12, color=SUB)

# ---------- 1 ----------
h1("1 概述与家族全景")
h2("1.1 定位")
body("InternVL 3.5 是 OpenGVLab 于 2025 年 8 月发布的开源多模态大模型家族，沿用 InternVL 系列一贯的 ViT–MLP–LLM 三段式范式：图像经 InternViT 视觉编码器抽取特征，经像素洗牌与 MLP 投影对齐到语言模型维度，再以特殊图像占位符替换的方式混入 Qwen3（或 GPT-OSS）语言模型的输入嵌入，完成自回归解码。")
body("相对上一代 InternVL 3.0，3.5 的核心增量集中在三处：一是提出 Cascade RL 级联强化学习框架以提升推理能力；二是提出视觉分辨率路由 ViR 与解耦部署 DvD，把推理效率相对 3.0 最高提升约 4.05 倍；三是扩展到 GUI 交互、具身智能、SVG 理解与生成等新能力。")

h2("1.2 家族规模")
body("依据论文 Table 1，3.5 家族覆盖从 1B 到 241B 的稠密与混合专家（MoE）模型。视觉编码器分两档：中小尺寸用 InternViT-300M，38B 与 241B 用 InternViT-6B；语言底座主体为 Qwen3 系列，20B-A4B 变体采用 GPT-OSS-20B。")
table_caption("表 1  InternVL 3.5 家族构成（论文 Table 1，A 表示激活参数量）")
make_table(
    ["模型", "视觉编码器", "语言底座", "视觉参", "语言参", "总参"],
    [
        ["InternVL3.5-1B", "InternViT-300M", "Qwen3-0.6B", "0.3B", "0.8B", "1.1B"],
        ["InternVL3.5-2B", "InternViT-300M", "Qwen3-1.7B", "0.3B", "2.0B", "2.3B"],
        ["InternVL3.5-4B", "InternViT-300M", "Qwen3-4B", "0.3B", "4.4B", "4.7B"],
        ["InternVL3.5-8B", "InternViT-300M", "Qwen3-8B", "0.3B", "8.2B", "8.5B"],
        ["InternVL3.5-14B", "InternViT-300M", "Qwen3-14B", "0.3B", "14.8B", "15.1B"],
        ["InternVL3.5-38B", "InternViT-6B", "Qwen3-32B", "5.5B", "32.8B", "38.4B"],
        ["InternVL3.5-20B-A4B", "InternViT-300M", "GPT-OSS-20B", "0.3B", "20.9B", "21.2B (A4B)"],
        ["InternVL3.5-30B-A3B", "InternViT-300M", "Qwen3-30B-A3B", "0.3B", "30.5B", "30.8B (A3B)"],
        ["InternVL3.5-241B-A28B", "InternViT-6B", "Qwen3-235B-A22B", "5.5B", "235.1B", "240.7B (A28B)"],
    ],
    widths=[3.6, 2.8, 3.0, 1.8, 1.8, 2.6]
)
body("本文以本地最小尺寸 InternVL3_5-1B-HF（即表 1 中 1B 档）的 config.json 为实参依据，数值以原文为准；家族级结论以论文 Table 1 为准。", indent=True)

# ---------- 2 ----------
h1("2 端到端架构总览")
body("下图给出本地 1B 模型的端到端数据流：任意比例图像先做动态分辨率 tile 切分，每个 448×448 tile 独立过 InternViT-300M，经像素洗牌压缩空间分辨率、升维通道后由 MLP 投影到 1024 维，最终替换语言模型输入中的图像占位符。")
add_figure("fig1_overall.png", "图 1  InternVL 3.5 端到端架构：图像 → InternViT-300M → 像素洗牌 → MLP 投影 → Qwen3-0.6B")
h2("2.1 关键尺寸换算（文本公式）")
body("单 tile 的视觉 token 数可由 config 参数直接复算，公式如下：")
body("P = (image_size_h / patch_h) × (image_size_w / patch_w) = (448/14)² = 32² = 1024", indent=False)
body("经 pixel_shuffle（scale = downsample_ratio = 0.5）后：", indent=False)
body("T = P × scale² = 1024 × 0.25 = 256，通道 C' = C / scale² = 1024 / 0.25 = 4096", indent=False)
body("该结果与 config 顶层 image_seq_length = 256 自洽：每个 tile 恰好产出 256 个视觉 token。符号说明见表 2。")
table_caption("表 2  视觉 token 换算符号表")
make_table(
    ["符号", "含义", "本地取值"],
    [
        ["image_size", "单个 tile 边长", "[448, 448]"],
        ["patch_size", "ViT patch 边长", "[14, 14]"],
        ["P", "单 tile patch token 数（去 CLS 前 1025）", "1024"],
        ["downsample_ratio", "像素洗牌空间下采样比例", "0.5"],
        ["T", "单 tile 视觉 token 数", "256"],
        ["C", "ViT 输出通道 hidden_size", "1024"],
        ["C'", "洗牌后通道数", "4096"],
        ["image_seq_length", "每 tile 写入的视觉 token 数", "256"],
        ["image_token_id", "图像占位符 token id", "151671"],
    ],
    widths=[3.4, 6.6, 4.0]
)

# ---------- 3 ----------
h1("3 视觉编码器 InternViT")
h2("3.1 规格档位")
body("3.5 家族视觉编码器有两档：中小尺寸（1B/2B/4B/8B/14B、20B-A4B、30B-A3B）共用 InternViT-300M，38B 与 241B-A28B 使用 InternViT-6B。本地 1B 的 vision_config 属 300M 规格，但 model_type 字符串为 internvl_vision（transformers 原生格式），其结构与 InternViT-6B 同族，仅宽度与层数不同。")
h2("3.2 层结构")
body("本地 vision_config 为：hidden_size=1024、num_hidden_layers=24、num_attention_heads=16、intermediate_size=4096、patch_size=[14,14]、image_size=[448,448]、attention_bias=True、layer_scale_init_value=0.1、use_mean_pooling=True、use_absolute_position_embeddings=True、use_qk_norm=False、norm_type=layer_norm、layer_norm_eps=1e-6。每层为 LN→多头自注意力→LayerScale 残差，再接 LN→MLP(GELU)→LayerScale 残差。")
add_figure("fig2_vit.png", "图 2  InternViT-300M 视觉编码器结构（transformers internvl_vision 实现）")
body("输出端 use_mean_pooling=True 时，对 patch token 做均值池化并跳过 CLS；在 get_image_features 中对应 vision_feature_select_strategy=default 时取 vision_features[:, 1:, :]，即去掉 CLS 后再 reshape 为空间网格。")

# ---------- 4 ----------
h1("4 投影层与动态分辨率")
h2("4.1 像素洗牌下采样")
body("modeling_internvl.py 的 pixel_shuffle(scale_factor=downsample_ratio=0.5) 将 (B, 32, 32, 1024) 重塑为 (B, 16, 16, 4096)：空间分辨率减半、通道数变为 4 倍。这与 v2 像素洗牌实现一致，是 1.5 起沿用的下采样方式。")
h2("4.2 MLP 投影器")
body("InternVLMultiModalProjector 由三层构成：LayerNorm(4096) → Linear(4096→1024) → GELU（projector_hidden_act=gelu）→ Linear(1024→1024)。输出维度严格等于语言模型 hidden_size（本地 text_config.hidden_size=1024），使视觉特征可直接替换语言模型的词嵌入。")
h2("4.3 动态高分辨率 tile 机制")
body("沿用 InternVL 1.5 提出的 Dynamic High Resolution：任意比例图像先缩放至边长为 448 的倍数，切分为若干 448×448 tile，再附加一张整图压成的缩略图 tile。每个 tile 独立过同一套 ViT–投影管线，各产出 256 个 token，由 min/max dynamic patch 约束总 tile 数。")
add_figure("fig3_tile.png", "图 3  动态高分辨率 tile 机制与视觉 token 总量计算")
body("3.5 在 tile 之上新增 ViR（Visual Resolution Router）：由 ViCO 轻量训练得到的路由器，按语义为每个 tile 动态选择 1/4 或 1/16 的压缩率，在几乎不损失性能的前提下进一步削减视觉 token（InternVL3.5-Flash 变体）。")

# ---------- 5 ----------
h1("5 token 流与语言底座")
h2("5.1 图像占位符替换")
body("处理侧（processing_internvl.py）把 prompt 中的图像占位符展开为 BOI + 256×N 个占位符 + EOI。前向时 get_placeholder_mask 命中 input_ids == image_token_id(151671) 的位置，用 masked_scatter 把投影后的视觉特征写回 inputs_embeds；并强制校验占位符数等于视觉特征数，否则报错。")
add_figure("fig4_tokenflow.png", "图 4  视觉 token 经占位符替换进入语言模型嵌入的数据流")
h2("5.2 语言底座 Qwen3-0.6B")
body("本地 text_config 为 Qwen3ForCausalLM（_name_or_path 指向 Qwen3-0.6B），全层 full_attention、无滑动窗口、无 rope_scaling，rope_theta=1e6。相比 3.0 本地 1B 使用的 Qwen2.5-0.5B（hidden 896、24 层、14 头、2 KV 头），3.5 升级为 hidden 1024、28 层、16 头、8 KV 头，采用 GQA。")
table_caption("表 3  本地语言底座 text_config 实参（来源：config.json）")
make_table(
    ["字段", "取值", "说明"],
    [
        ["architectures", "Qwen3ForCausalLM", "因果语言模型"],
        ["hidden_size", "1024", "与投影输出维度一致"],
        ["num_hidden_layers", "28", "全部 full_attention"],
        ["num_attention_heads", "16", "查询头数"],
        ["num_key_value_heads", "8", "GQA"],
        ["head_dim", "128", "每头维度"],
        ["intermediate_size", "3072", "FFN 中间维"],
        ["hidden_act", "silu", "激活函数"],
        ["attention_bias", "false", "注意力无偏置"],
        ["vocab_size", "151936", "词表"],
        ["max_position_embeddings", "40960", "最大位置"],
        ["rope_theta", "1000000", "RoPE 基频"],
        ["rope_scaling", "null", "无位置缩放"],
        ["rms_norm_eps", "1e-6", "RMSNorm  eps"],
        ["max_window_layers", "28", "全部层"],
    ],
    widths=[4.6, 3.6, 5.8]
)

# ---------- 6 ----------
h1("6 训练管线与 Cascade RL")
h2("6.1 预训练与监督微调")
body("3.5 延续原生多模态预训练（视觉塔与语言底座联合训练），随后进行监督微调（SFT）作为后训练起点。SFT 模型具备多任务指令跟随能力，是强化学习阶段的初始化。")
h2("6.2 Cascade RL 专节")
body("Cascade RL 是 3.5 相对 3.0 后训练范式的核心改动，分两阶段：先用离线 RL 高效热身，再用在线 RL 精修输出分布，从而在可控 GPU 成本下把推理能力推到更高上限。")
body("离线阶段采用混合偏好优化 MPO，损失为偏好损失、质量损失与生成损失的加权和：L_MPO = w_p·L_p + w_q·L_q + w_g·L_g。其中偏好损失用 DPO、质量损失用 BCO、生成损失用语言模型损失。该阶段训练效率高，为在线阶段产出高质量 rollout。")
body("在线阶段采用 GSPO（无参考模型约束），优势为同一 query 采样响应间的归一化奖励，与 GRPO 同族；对稠密与 MoE 模型均有效。论文报告该框架带来整体推理性能最高 +16.0% 的提升。")
h2("6.3 推理效率：ViR 与 DvD")
body("ViR 为每个 tile 动态选择 1/4 或 1/16 压缩率，削减视觉 token 数；DvD（Decoupled Vision-Language Deployment）把 ViT 与 LLM 部署到不同 GPU，并行视觉编码与文本解码。二者叠加实现最高约 4.05 倍推理加速。")
add_figure("fig5_cascade_rl.png", "图 5  Cascade RL 两阶段框架与 ViR/DvD 推理效率设计")

# ---------- 7 ----------
h1("7 版本演进对比（3.0 → 3.5）")
body("3.5 在保留 ViT–MLP–LLM 与动态分辨率 tile 主干的前提下，主要从后训练与推理效率两条线升级，并扩展能力边界。")
table_caption("表 4  InternVL 3.0 → 3.5 演进要点")
make_table(
    ["维度", "InternVL 3.0", "InternVL 3.5"],
    [
        ["后训练", "常规离线/在线 RL", "Cascade RL：MPO 离线热身 + GSPO 在线精修"],
        ["推理效率", "基线", "ViR 动态压缩 + DvD 解耦部署，最高约 4.05×"],
        ["推理能力", "基线", "整体推理最高 +16.0%"],
        ["1B 语言底座", "Qwen2.5-0.5B（896/24层/2KV头）", "Qwen3-0.6B（1024/28层/8KV头）"],
        ["能力边界", "通用图文、OCR、 grounding", "新增 GUI 交互、具身智能、SVG 理解与生成"],
        ["最大模型", "—", "241B-A28B（InternViT-6B + Qwen3-235B-A22B）"],
    ],
    widths=[2.8, 5.6, 6.2]
)
add_figure("fig6_family_params.png", "图 6  InternVL 3.5 家族参数量构成（论文 Table 1，对数刻度）", width_in=6.0)

# ---------- 8 ----------
h1("8 本地 config.json 实参表")
body("下表逐项列出本地模型 config 的顶层与嵌套字段，数值均来自原文，文件路径为：" + CONFIG_PATH)
table_caption("表 5  InternVL3_5-1B-HF config.json 实参（顶层）")
make_table(
    ["字段", "取值", "说明"],
    [
        ["architectures", "InternVLForConditionalGeneration", "条件生成类"],
        ["model_type", "internvl", "transformers 原生格式"],
        ["transformers_version", "4.55.0", "生成版本"],
        ["torch_dtype", "bfloat16", "权重精度"],
        ["image_seq_length", "256", "每 tile 视觉 token 数"],
        ["image_token_id", "151671", "图像占位符 id"],
        ["downsample_ratio", "0.5", "像素洗牌比例"],
        ["projector_hidden_act", "gelu", "投影器激活"],
        ["vision_feature_layer", "-1", "取最后一层视觉特征"],
        ["vision_feature_select_strategy", "default", "去掉 CLS"],
    ],
    widths=[5.0, 4.2, 5.4]
)
table_caption("表 6  vision_config 实参（InternViT-300M 规格）")
make_table(
    ["字段", "取值"],
    [
        ["architectures / model_type", "InternVisionModel / internvl_vision"],
        ["hidden_size / num_hidden_layers / num_attention_heads", "1024 / 24 / 16"],
        ["intermediate_size", "4096"],
        ["patch_size / image_size", "[14,14] / [448,448]"],
        ["attention_bias / use_qk_norm", "true / false"],
        ["layer_scale_init_value", "0.1"],
        ["use_mean_pooling / use_absolute_position_embeddings", "true / true"],
        ["norm_type / layer_norm_eps", "layer_norm / 1e-6"],
        ["num_channels", "3"],
    ],
    widths=[7.0, 7.6]
)

# ---------- 9 ----------
h1("9 参考文献")
refs = [
    "[1] InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency. arXiv:2508.18265. https://arxiv.org/abs/2508.18265",
    "[2] InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models. arXiv:2504.10479. https://arxiv.org/abs/2504.10479",
    "[3] Qwen3 Technical Report. arXiv:2505.09388. https://arxiv.org/abs/2505.09388",
    "[4] InternVL 1.5: How Far Are We to GPT-4V? arXiv:2404.16821. https://arxiv.org/abs/2404.16821",
    "[5] transformers 源码：InternVLForConditionalGeneration / get_image_features / pixel_shuffle / InternVLMultiModalProjector. 本地路径 " + SRC_PATH,
    "[6] 本地模型配置：" + CONFIG_PATH,
]
for r in refs:
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.left_indent = Pt(18)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    set_run_font(p.add_run(r), size=10.5)

doc.save(OUT)
print("SAVED:", OUT)
