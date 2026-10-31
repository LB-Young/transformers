# -*- coding: utf-8 -*-
"""Build InternVL-1_5-家族架构详解.docx"""
import os
from docx import Document
from docx.shared import Pt, Cm, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")
ARCH = os.path.join(ROOT, "_arch")
OUT = os.path.join(ROOT, "InternVL-1_5-家族架构详解.docx")

INK = RGBColor(0x24, 0x29, 0x2f)
TEAL_INK = RGBColor(0x15, 0x5e, 0x75)
AMBER = RGBColor(0xb4, 0x53, 0x09)

doc = Document()

# page setup A4, margins 2.5cm
sec = doc.sections[0]
sec.page_width = Cm(21.0)
sec.page_height = Cm(29.7)
sec.top_margin = Cm(2.5); sec.bottom_margin = Cm(2.5)
sec.left_margin = Cm(2.5); sec.right_margin = Cm(2.5)

# Normal style
normal = doc.styles["Normal"]
normal.font.name = "Times New Roman"
normal.font.size = Pt(12)
normal.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
normal.paragraph_format.line_spacing = 1.5
normal.paragraph_format.space_after = Pt(0)


def _set_run_font(run, east="宋体", west="Times New Roman", size=12, bold=False, color=INK):
    run.font.name = west
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts"); rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), east)


def title(text, sub=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    _set_run_font(r, east="黑体", size=18, bold=True)
    p.paragraph_format.space_after = Pt(6)
    if sub:
        p2 = doc.add_paragraph()
        p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r2 = p2.add_run(sub)
        _set_run_font(r2, east="楷体", size=12, color=RGBColor(0x6b, 0x72, 0x80))
        p2.paragraph_format.space_after = Pt(12)


def h1(text):
    p = doc.add_heading(level=1)
    r = p.add_run(text)
    _set_run_font(r, east="黑体", size=16, bold=True, color=RGBColor(0, 0, 0))
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    return p


def h2(text):
    p = doc.add_heading(level=2)
    r = p.add_run(text)
    _set_run_font(r, east="黑体", size=14, bold=True, color=RGBColor(0, 0, 0))
    p.paragraph_format.space_before = Pt(11)
    p.paragraph_format.space_after = Pt(5)
    return p


def body(text, indent=True):
    p = doc.add_paragraph()
    r = p.add_run(text)
    _set_run_font(r, size=12)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if indent:
        p.paragraph_format.first_line_indent = Pt(24)
    return p


def caption(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    _set_run_font(r, size=10.5, color=RGBColor(0x24, 0x29, 0x2f))
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(10)
    return p


def add_fig(fname, cap, width_in=6.3):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(os.path.join(ARCH, fname), width=Inches(width_in))
    p.paragraph_format.space_before = Pt(6)
    caption(cap)


def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:fill"), hexcolor)
    tcPr.append(shd)


def set_cell(cell, text, bold=False, size=10.5, align="left", color=INK):
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.first_line_indent = Pt(0)
    if align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align == "right":
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = p.add_run(text)
    _set_run_font(r, size=size, bold=bold, color=color)


def make_table(headers, rows, widths=None, hl_rows=None):
    hl_rows = hl_rows or set()
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for j, htxt in enumerate(headers):
        set_cell(t.rows[0].cells[j], htxt, bold=True, align="center")
        shade(t.rows[0].cells[j], "D9D9D9")
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            align = "center" if j == 0 else "left"
            set_cell(t.rows[i + 1].cells[j], str(val), align=align)
            if i in hl_rows:
                shade(t.rows[i + 1].cells[j], "F7EAD6")
    if widths:
        for j, w in enumerate(widths):
            for row in t.rows:
                row.cells[j].width = Cm(w)
    # cantSplit rows
    for row in t.rows:
        trPr = row._tr.get_or_add_trPr()
        cant = OxmlElement("w:cantSplit"); trPr.append(cant)
    return t


# ============================ CONTENT ============================
title("InternVL 1.5 家族架构详解",
      "InternVLChatModel · ViT–MLP–LLM · 本地 Mini-InternVL-Chat-2B-V1-5 逐字段核对")

body('本文以本地 config.json、官方 HuggingFace 配置与 InternVL 1.5 论文（arXiv:2404.16821）为唯一依据，'
     '讲解 InternVL 1.5 开放多模态模型家族的完整架构：视觉编码器 InternViT、像素重排与 MLP 投影、'
     'InternLM2 语言底座、动态高分辨率分块机制与两阶段训练管线，并对比 InternVL 1.0 到 1.5 的架构演进。'
     '本地模型目录为 E:\\models\\model\\Mini-InternVL-Chat-2B-V1-5，其 config.json 采用 legacy internvl_chat 格式。')

# ---------------- 1 家族全景 ----------------
h1('1 家族全景与本地 config 对照')

h2('1.1 家族模型清单')
body('InternVL 1.5 家族按官方仓库（OpenGVLab/InternVL）与 HuggingFace 发布清单，共三个公开权重模型，'
     '统一采用 ViT–MLP–LLM 架构，仅在视觉塔与语言底座尺寸上做替换：')
make_table(
    ['模型', '视觉编码器', '语言底座', '总参数量'],
    [
        ['InternVL-Chat-V1.5', 'InternViT-6B-448px-V1.5（3200 维·45 层·25 头）', 'InternLM2-Chat-20B', '约 26B'],
        ['Mini-InternVL-Chat-2B-V1-5（本地）', 'InternViT-300M-448px-V1.5（1024 维·24 层·16 头）', 'InternLM2-Chat-1.8B', '约 2B'],
        ['Mini-InternVL-Chat-4B-V1-5', 'InternViT-300M-448px-V1.5（同 2B 视觉塔）', 'Phi-3-mini-128k-instruct', '约 4B'],
    ],
    widths=[4.2, 6.0, 4.0, 2.3],
    hl_rows={1},
)
caption('表 1：InternVL 1.5 家族模型清单（来源：官方文档 internvl1.5/introduction 与 HuggingFace 模型卡）')
body('需要特别说明：任务书给出的家族规模锚点含 1B/8B，但经官方仓库与 HuggingFace 清单核实，'
     'InternVL 1.5 公开发布的为 2B / 4B / 26B 三个尺寸；1B、8B 等尺寸属于其后继 InternVL 2.0 家族，'
     '本文不纳入 1.5 架构统计。')

h2('1.2 本地 Mini-InternVL-Chat-2B-V1-5 顶层 config 实参')
body('下表逐字段取自 config.json 原文，文件绝对路径为 '
     'E:\\models\\model\\Mini-InternVL-Chat-2B-V1-5\\config.json（legacy internvl_chat 格式）：')
make_table(
    ['字段', '取值', '含义'],
    [
        ['architectures', 'InternVLChatModel', 'legacy 入口类（transformers 不内置 1.5）'],
        ['model_type', 'internvl_chat', 'legacy 配置类型'],
        ['template', 'internlm2-chat', '对话模板'],
        ['dynamic_image_size', 'true', '启用动态分辨率分块'],
        ['force_image_size', '448', '基础 tile 边长（像素）'],
        ['downsample_ratio', '0.5', 'pixel shuffle 下采样比例'],
        ['max_dynamic_patch', '12', '训练时最多 tile 数'],
        ['min_dynamic_patch', '1', '最少 tile 数'],
        ['ps_version', 'v2', 'pixel shuffle 实现版本'],
        ['select_layer', '-1', '取视觉塔最后一层特征'],
        ['use_thumbnail', 'true', '额外拼接整图缩略图'],
        ['torch_dtype', 'bfloat16', '训练/推理精度'],
    ],
    widths=[4.5, 4.5, 7.5],
)
caption('表 2：本地 config.json 顶层字段（来源：E:\\models\\model\\Mini-InternVL-Chat-2B-V1-5\\config.json）')

h2('1.3 视觉塔两种规格对照')
body('本地 vision_config 的 model_type 字符串虽为 intern_vit_6b，实际数值是 InternViT-300M 规格。'
     '家族视觉塔有 300M 与 6B 两规格，二者结构同构，差异如下表（6B 列取自 InternVL-Chat-V1.5 官方 config）：')
make_table(
    ['字段', '300M（本地 2B）', '6B（主模型 V1.5）', '备注'],
    [
        ['hidden_size', '1024', '3200', '残差流维度'],
        ['num_hidden_layers', '24', '45', '6B 原始 48 层裁掉最后 3 层'],
        ['num_attention_heads', '16', '25', 'head_dim=64 / 128'],
        ['intermediate_size', '4096', '12800', 'MLP 宽度 = 4×hidden'],
        ['patch_size', '14', '14', 'patch 边长（像素）'],
        ['image_size', '448', '448', 'tile 分辨率'],
        ['qkv_bias', 'true', 'false', '是否对 QKV 加偏置'],
        ['qk_normalization', 'false', 'true', '6B 对 Q/K 做归一化'],
        ['layer_norm_eps', '1e-6', '1e-6', '层归一化 eps'],
        ['hidden_act', 'gelu', 'gelu', 'MLP 激活函数'],
    ],
    widths=[4.0, 3.8, 3.8, 4.9],
)
caption('表 3：InternViT 300M 与 6B 规格对照（来源：本地 config 与 InternVL-Chat-V1.5 官方 config）')

# ---------------- 2 整体架构 ----------------
h1('2 端到端整体架构')
body('InternVL 1.5 采用业界常用的 ViT–MLP–LLM 配置：预训练 InternViT 视觉编码器输出图像特征，'
     '经像素重排压缩 token 数后由随机初始化的 MLP 投影器映射到语言模型嵌入维度，再与文本 token 直接拼接，'
     '送入预训练 InternLM2 做自回归解码。与 1.0 不同，1.5 没有任何中间语言模型桥接模块。')
add_fig('fig1_overall.png', '图 1：InternVL 1.5 端到端整体架构（ViT–MLP–LLM，无 QLLaMA 桥接）')
body('数据路径：输入图像按长宽比切成 448×448 的 tile 并附一张缩略图；每个 tile 经 InternViT 得到 32×32 个 patch token；'
     'pixel shuffle（比例 0.5）把 token 数压到四分之一；MLP 投影后替换 input_ids 中 <image> 占位符位置的词嵌入；'
     '最终与文本嵌入共同进入 InternLM2 因果 Transformer。')

# ---------------- 3 视觉编码器 ----------------
h1('3 视觉编码器 InternViT 细节')
body('InternViT 是标准 ViT 结构（源自 InternImage/EVA/DINOv2 代码体系），无 class token，patch 即序列。'
     '每个 448×448、patch=14 的图像被切为 (448/14)²=32×32=1024 个 patch token。')
add_fig('fig2_vit.png', '图 2：InternViT 视觉编码器层结构（pre-norm ViT 块，300M 与 6B 同构）')

h2('3.1 关键计算')
make_table(
    ['符号', '含义', '取值（本地 2B）'],
    [
        ['H×W', 'tile 边长', '448×448'],
        ['P', 'patch 边长', '14'],
        ['N_patch', '每 tile patch token 数 = (H/P)²', '32² = 1024'],
        ['D_vit', '视觉隐藏维', '1024（6B 为 3200）'],
        ['L_vit', '视觉层数', '24（6B 为 45）'],
        ['r', 'pixel shuffle 比例', '0.5'],
        ['N_out', '每 tile 输出 token 数 = N_patch × r²', '1024 × 0.25 = 256'],
    ],
    widths=[3.5, 6.5, 5.5],
)
caption('表 4：视觉 token 数复算（公式 N_out=(H/(P·(1/r)))²；来源：本地 config + 论文 §3.1）')

h2('3.2 连续学习策略')
body('论文 §3.2 记录了 InternViT-6B 的连续学习过程：V1.2 阶段发现倒数第四层特征对多模态任务最好，'
     '于是直接丢弃最后 3 层权重，把视觉塔从 48 层裁到 45 层，并把分辨率从 224 提到 448；'
     'V1.5 阶段进一步用动态分辨率（1~12 个 448 tile）的图文与 OCR 数据继续训练。'
     'select_layer=-1 即取裁剪后视觉塔的最后一层输出。本地 300M 视觉塔是从 6B-448px-V1.5 蒸馏而来。')

# ---------------- 4 投影与语言底座 ----------------
h1('4 投影层与语言底座')
h2('4.1 MLP 投影器')
body('pixel shuffle 把空间维减半的同时把通道数扩大到 4 倍：视觉特征 (16×16, D_vit) 经投影器变为 (16×16, D_llm)。'
     '投影器为两层 MLP（中间 GELU），随机初始化：')
make_table(
    ['符号', '含义', '本地 2B 取值'],
    [
        ['D_vit', '视觉隐藏维', '1024'],
        ['D_llm', '语言模型隐藏维', '2048'],
        ['mlp1', '第一层 Linear', '4096 → 2048（4×D_vit → D_llm）'],
        ['激活', '中间激活', 'GELU'],
        ['mlp2', '第二层 Linear', '2048 → 2048'],
    ],
    widths=[3.5, 6.0, 6.0],
)
caption('表 5：MLP 投影器维度（来源：论文 §3.1 + 官方 modeling_internvl_chat.py）')

h2('4.2 语言底座 InternLM2-1.8B（本地 llm_config）')
make_table(
    ['字段', '取值', '备注'],
    [
        ['model_type', 'internlm2', 'InternLM2ForCausalLM'],
        ['hidden_size', '2048', '残差流维度'],
        ['num_hidden_layers', '24', '解码层数'],
        ['num_attention_heads', '16', 'Q 头数'],
        ['num_key_value_heads', '8', 'GQA，KV 头为 Q 头一半'],
        ['intermediate_size', '8192', 'SwiGLU 中间维'],
        ['hidden_act', 'silu', 'SwiGLU 激活'],
        ['vocab_size', '92553', '词表大小'],
        ['max_position_embeddings', '32768', '上下文上限'],
        ['rope_theta', '1000000', 'RoPE 基频'],
        ['rope_scaling', 'dynamic, factor=2.0', '动态外推上下文'],
        ['rms_norm_eps', '1e-5', 'RMSNorm eps'],
        ['bos/eos/pad_token_id', '1 / 2 / 2', '特殊 token'],
    ],
    widths=[5.0, 4.5, 6.0],
)
caption('表 6：本地 llm_config 实参（来源：E:\\models\\model\\Mini-InternVL-Chat-2B-V1-5\\config.json）')
body('主模型 InternVL-Chat-V1.5 的语言底座为 InternLM2-Chat-20B（hidden 6144、48 层、48 Q 头、8 KV 头、'
     'intermediate 16384、dynamic rope factor=3.0），与本地 2B 同构，仅规模放大。')

# ---------------- 5 核心机制 ----------------
h1('5 核心机制：动态分辨率、像素重排与缩略图')
h2('5.1 动态长宽比匹配')
body('训练时在 1~12 个 tile 的所有组合中共定义 35 种预设长宽比（如 1:1、1:2、2:1、3:1、2:6 等）。'
     '对每张输入图像计算其长宽差，选最接近且面积不超过原图两倍的预设比例，避免低分辨率图被过度放大。')
add_fig('fig3_tiles.png', '图 3：动态高分辨率分块机制（按长宽比匹配 35 种预设网格）')

h2('5.2 token 数复算')
make_table(
    ['场景', 'tile 数 N', '缩略图', '视觉 token 总数 = (N+1)×256'],
    [
        ['最小输入', '1', '否（单 tile 即全局）', '256'],
        ['训练典型', '1~12', '是', '512 ~ 3328'],
        ['测试上限（4K）', '最多 40', '是', '10496'],
    ],
    widths=[3.5, 3.0, 4.5, 5.0],
)
caption('表 7：视觉 token 数复算（来源：论文 §3.3；训练 256~3328、测试 10496 与论文一致）')
body('像素重排（pixel shuffle / 官方代码 scale_factor=downsample_ratio=0.5）把相邻 2×2 空间位置的特征'
     '折叠进通道维，空间分辨率减半而通道 ×4，token 数变为四分之一，从而在高分辨率下控制序列长度。')

h2('5.3 token 拼接流')
add_fig('fig4_tokenflow.png', '图 4：Token 流——视觉特征替换 <image> 占位符后与文本共享注意力')
body('input_ids 中先按对话模板放入 N×256 个 <image> 占位符；前向时把这些位置的词嵌入整体替换为 MLP 输出的视觉嵌入，'
     '随后视觉 token 与文本 token 在同一条因果序列里走同一套 InternLM2 注意力，不存在跨模态交叉注意力模块。')

# ---------------- 6 训练管线 ----------------
h1('6 训练管线')
body('论文 §4.1 明确训练分两阶段，两阶段上下文长度均为 4096，响应格式提示与 LLaVA-1.5 一致：')
make_table(
    ['阶段', '可训练部分', '冻结部分', '数据'],
    [
        ['阶段一：模型对接预训练', 'InternViT 视觉塔 + MLP 投影器', 'LLM 冻结', 'caption 53.9%、OCR 32.0%、检测 grounding 5.2%、其他 8.9%'],
        ['阶段二：全参数微调', '全部约 26B 参数', '无', '高质量中英双语 SFT 数据（自然场景、图表、文档、对话）'],
    ],
    widths=[3.8, 4.2, 2.5, 5.0],
)
caption('表 8：两阶段训练管线（来源：论文 §3.4 与 §4.1）')
body('Mini 系列（2B/4B）使用与主模型相同的数据，但因训练成本低采用 8K 上下文长度。'
     '阶段一只更新视觉侧、冻结 LLM，保证语言能力不被破坏；阶段二放开全部参数做指令微调。')

# ---------------- 7 演进 ----------------
h1('7 版本演进：InternVL 1.0 → 1.5')
body('InternVL 1.0（arXiv:2312.14238）采用 InternViT-6B + 8B 参数 QLLaMA 中间桥接 + LLaMA 解码器的三段式结构，'
     '桥接模块大且训练复杂，视觉分辨率固定 224px。1.5 在论文中给出三点改进：')
add_fig('fig5_evo.png', '图 5：InternVL 1.0 → 1.5 架构演进')
make_table(
    ['维度', 'InternVL 1.0', 'InternVL 1.5'],
    [
        ['视觉编码器', 'InternViT-6B，224px 固定', 'InternViT-6B/300M，448px 连续学习、动态分辨率'],
        ['桥接方式', '8B QLLaMA 中间语言模型桥接', '去除桥接，pixel shuffle + MLP 直接拼接'],
        ['语言底座', 'LLaMA 系解码器', 'InternLM2（1.8B/20B）直接作为 LLM'],
        ['分辨率', '固定低分辨率', '1~12 tile 训练、40 tile 测试（4K）+ 缩略图'],
        ['数据', '图文对比对齐', '高质量中英双语 OCR/文档/对话数据'],
    ],
    widths=[3.2, 5.5, 6.8],
)
caption('表 9：InternVL 1.0 与 1.5 架构对照（来源：论文 §3 与 arXiv:2312.14238）')

# ---------------- 8 参考文献 ----------------
h1('8 参考文献')
refs = [
    '[1] Chen Z, Wang W, Tian H, et al. How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites. arXiv:2404.16821, 2024. https://arxiv.org/abs/2404.16821',
    '[2] Chen Z, Wu J, Wang W, et al. InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks. CVPR 2024. arXiv:2312.14238. https://arxiv.org/abs/2312.14238',
    '[3] OpenGVLab. InternViT-6B-448px-V1-5 配置（hidden 3200, 45 层, 25 头, intermediate 12800, qk_normalization=true）. https://huggingface.co/OpenGVLab/InternViT-6B-448px-V1-5',
    '[4] OpenGVLab/InternVL 官方仓库，internvl_chat/internvl/model/internvl_chat/modeling_internvl_chat.py（InternVLChatModel = vision_model + mlp1 + llm）. https://github.com/OpenGVLab/InternVL',
    '[5] 本地 config：E:\\models\\model\\Mini-InternVL-Chat-2B-V1-5\\config.json（legacy internvl_chat，vision_config=intern_vit_6b 型 300M 规格，llm_config=InternLM2-1.8B）',
    '[6] 本地 transformers 源码：E:\\transformers\\src\\transformers\\models\\internvl\\modeling_internvl.py（pixel_shuffle 实现，scale_factor=downsample_ratio=0.5，供 2.x/3.x 参考；1.5 legacy 结构不在 transformers 内）',
]
for rtext in refs:
    p = doc.add_paragraph()
    run = p.add_run(rtext)
    _set_run_font(run, size=10.5)
    p.paragraph_format.space_after = Pt(4)

p = doc.add_paragraph()
run = p.add_run('说明：家族 1B/8B 尺寸经核实属于 InternVL 2.0 而非 1.5，已在正文标注；凡本地 config 与论文口径差异处均并列标注，未擅自统一。访问与核验日期 2026-10-26。')
_set_run_font(run, size=10.5, color=RGBColor(0x6b, 0x72, 0x80))

doc.save(OUT)
print("SAVED", OUT)
