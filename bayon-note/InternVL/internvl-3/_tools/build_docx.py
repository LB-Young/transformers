# -*- coding: utf-8 -*-
"""生成 InternVL-3-家族架构详解.docx（A4、编号章节、统一表格、嵌入架构图）"""
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import os

ARCH = r"E:\transformers\bayon-note\InternVL\internvl-3\_arch"
OUT = r"E:\transformers\bayon-note\InternVL\internvl-3\InternVL-3-家族架构详解.docx"

doc = Document()

# ---------- 页面：A4，上下左右 2.5cm ----------
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.top_margin = sec.bottom_margin = sec.left_margin = sec.right_margin = Cm(2.5)

# ---------- 样式 ----------
def set_style_font(style, cn_font, en_font, size, bold=False, color=(0, 0, 0)):
    style.font.name = en_font
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.color.rgb = RGBColor(*color)
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn('w:rFonts'))
    if rfonts is None:
        rfonts = OxmlElement('w:rFonts'); rpr.append(rfonts)
    rfonts.set(qn('w:eastAsia'), cn_font)
    for attr in ('w:asciiTheme', 'w:hAnsiTheme', 'w:eastAsiaTheme', 'w:cstheme'):
        if rfonts.get(qn(attr)) is not None:
            del rfonts.attrib[qn(attr)]

normal = doc.styles['Normal']
set_style_font(normal, '宋体', 'Times New Roman', 12)
normal.paragraph_format.line_spacing = 1.5
normal.paragraph_format.space_before = Pt(0)
normal.paragraph_format.space_after = Pt(0)

for name, size in [('Heading 1', 16), ('Heading 2', 14), ('Heading 3', 12)]:
    st = doc.styles[name]
    set_style_font(st, '黑体', 'Times New Roman', size, bold=True)
    st.paragraph_format.space_before = Pt(14 if size == 16 else 11)
    st.paragraph_format.space_after = Pt(6 if size == 16 else 5)
    st.paragraph_format.keep_with_next = True

# ---------- 辅助函数 ----------
def para(text, indent=True, align=None, size=12, bold=False):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.name = 'Times New Roman'
    r._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    if indent:
        p.paragraph_format.first_line_indent = Pt(24)
    if align:
        p.alignment = align
    return p

def h1(t): doc.add_heading(t, level=1)
def h2(t): doc.add_heading(t, level=2)

def caption(t):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(t)
    r.font.size = Pt(10.5)
    r.font.name = 'Times New Roman'
    r._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(10)

def add_fig(fname, width_cm, cap):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Pt(0)
    run = p.add_run()
    run.add_picture(os.path.join(ARCH, fname), width=Cm(width_cm))
    caption(cap)

def set_cell_border(cell):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement('w:tcBorders')
    for edge in ('top', 'left', 'bottom', 'right'):
        el = OxmlElement(f'w:{edge}')
        el.set(qn('w:val'), 'single'); el.set(qn('w:sz'), '4')
        el.set(qn('w:space'), '0'); el.set(qn('w:color'), '808080')
        borders.append(el)
    tcPr.append(borders)

def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear'); shd.set(qn('w:fill'), hexcolor)
    tcPr.append(shd)

def make_table(headers, rows, widths=None, first_col_center=True):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for j, htext in enumerate(headers):
        c = t.rows[0].cells[j]
        c.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        set_cell_border(c); shade(c, 'F2F2F2')
        p = c.paragraphs[0]
        p.paragraph_format.first_line_indent = Pt(0)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(htext)
        r.font.bold = True; r.font.size = Pt(10.5)
        r.font.name = 'Times New Roman'
        r._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.rows[i + 1].cells[j]
            c.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_border(c)
            if i % 2 == 1:
                shade(c, 'FAFAFA')
            p = c.paragraphs[0]
            p.paragraph_format.first_line_indent = Pt(0)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if (j == 0 and first_col_center) else WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(str(val))
            r.font.size = Pt(10.5)
            r.font.name = 'Times New Roman'
            r._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    if widths:
        for j, w in enumerate(widths):
            for row in t.rows:
                row.cells[j].width = Cm(w)
    # 表头跨页重复 + 行内不跨页
    trPr = t.rows[0]._tr.get_or_add_trPr()
    tblHeader = OxmlElement('w:tblHeader'); tblHeader.set(qn('w:val'), 'true'); trPr.append(tblHeader)
    for row in t.rows:
        trPr = row._tr.get_or_add_trPr()
        cantSplit = OxmlElement('w:cantSplit'); trPr.append(cantSplit)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return t

# ================= 封面标题区 =================
tp = doc.add_paragraph(); tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
tp.paragraph_format.space_before = Pt(24); tp.paragraph_format.space_after = Pt(6)
tr = tp.add_run('InternVL-3 家族架构详解')
tr.font.size = Pt(18); tr.font.bold = True; tr.font.name = 'Times New Roman'
tr._element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')

sp = doc.add_paragraph(); sp.alignment = WD_ALIGN_PARAGRAPH.CENTER
sr = sp.add_run('InternVLForConditionalGeneration · 1B–78B · 原生多模态预训练 + 测试时扩展')
sr.font.size = Pt(12); sr.font.name = 'Times New Roman'
sr._element.rPr.rFonts.set(qn('w:eastAsia'), '楷体')

para('依据：本机 E:\\models\\model\\InternVL3-1B-hf 的 config.json 与 preprocessor_config.json、本地 transformers 源码 '
     'E:\\transformers\\src\\transformers\\models\\internvl\\（modeling_internvl.py / configuration_internvl.py / processing_internvl.py），'
     '以及 InternVL3 论文 arXiv:2504.10479。所有数值以 config 原文与源码为准，论文口径单独标注。', indent=True)

# ================= 1 家族全景 =================
h1('1 家族全景：InternVL3 家族与本地 config 对照')
para('InternVL3 是上海 AI Lab InternVL 系列的第三代开放多模态大模型，共发布 7 个尺寸（1B 到 78B），'
     '统一遵循 ViT–MLP–LLM 三段式架构：视觉编码器取 InternViT-300M 或 InternViT-6B 两种规格，'
     '语言底座取 Qwen2.5 系列预训练 base 模型（9B 型号用 InternLM3-8B），投影层为随机初始化的两层 MLP。'
     '家族规格与 OpenCompass 成绩（本地测试口径）如下表，源自论文 Table 1。')

make_table(
    ['模型', '总参数', '视觉编码器', '语言模型', 'OpenCompass'],
    [
        ['InternVL3-1B', '0.9B', 'InternViT-300M-448px-V2.5', 'Qwen2.5-0.5B', '57.4'],
        ['InternVL3-2B', '1.9B', 'InternViT-300M-448px-V2.5', 'Qwen2.5-1.5B', '63.9'],
        ['InternVL3-8B', '8.1B', 'InternViT-300M-448px-V2.5', 'Qwen2.5-7B', '73.3'],
        ['InternVL3-9B', '9.2B', 'InternViT-300M-448px-V2.5', 'InternLM3-8B', '72.4'],
        ['InternVL3-14B', '15.1B', 'InternViT-300M-448px-V2.5', 'Qwen2.5-14B', '75.5'],
        ['InternVL3-38B', '38.4B', 'InternViT-6B-448px-V2.5', 'Qwen2.5-32B', '77.3'],
        ['InternVL3-78B', '78.4B', 'InternViT-6B-448px-V2.5', 'Qwen2.5-72B', '79.5'],
    ],
    widths=[2.8, 1.8, 5.2, 3.6, 1.8])
caption('表 1：InternVL3 家族规格表（源自 arXiv:2504.10479 Table 1）')

para('本文以本地最小尺寸 InternVL3-1B-hf（对应表中 1B：0.9B，InternViT-300M + Qwen2.5-0.5B）为主尺寸逐字段核对。'
     '需要强调：3.0 是 transformers 原生 internvl 格式——顶层直接嵌套 text_config 与 vision_config，'
     '不再是 1.5/2.0/2.5 时代的 legacy internvl_chat 格式（顶层 llm_config + intern_vit_6b 型 vision_config）。')

h2('1.1 本地 config 实参表')
para('下列三张表逐字段来自本地 config.json（绝对路径：E:\\models\\model\\InternVL3-1B-hf\\config.json，'
     'transformers_version=4.52.0.dev0，model_type=internvl，architectures=InternVLForConditionalGeneration）。')

make_table(
    ['字段（顶层 config.json）', '取值', '含义 / 源码佐证'],
    [
        ['architectures', 'InternVLForConditionalGeneration', '顶层模型类（modeling_internvl.py:741）'],
        ['model_type', 'internvl', 'configuration_internvl.py:108'],
        ['image_seq_length', '256', '每个 448×448 切片展开的视觉 token 数（=32²×0.5²，processing_internvl.py:58）'],
        ['image_token_id', '151667', '图像占位 token 的 id（forward 中 masked_scatter 的掩码依据，:602）'],
        ['downsample_ratio', '0.5', 'Pixel Unshuffle 缩放因子（:578, :670）'],
        ['projector_hidden_act', 'gelu', 'MLP 投影器中间激活（:492）'],
        ['vision_feature_layer', '-1', '取最后一层隐藏态（:559-565）'],
        ['vision_feature_select_strategy', 'default', 'default=丢弃 CLS，保留 patch 特征（:566-567）'],
        ['torch_dtype', 'bfloat16', '权重精度'],
    ],
    widths=[5.2, 3.4, 6.6])
caption('表 2：本地 InternVL3-1B-hf 顶层 config 实参（来源：E:\\models\\model\\InternVL3-1B-hf\\config.json）')

make_table(
    ['字段（text_config）', '取值', '含义 / 源码佐证'],
    [
        ['architectures / model_type', 'Qwen2ForCausalLM / qwen2', 'Qwen2.5-0.5B 在 HF 的架构类名为 Qwen2'],
        ['hidden_size', '896', '语言模型残差流维度'],
        ['num_hidden_layers', '24', 'Decoder 层数'],
        ['num_attention_heads', '14', 'Query 头数'],
        ['num_key_value_heads', '2', 'KV 头数（GQA，分组数 14/2=7）'],
        ['intermediate_size', '4864', 'SwiGLU 前馈维度'],
        ['hidden_act', 'silu', 'SwiGLU 激活'],
        ['vocab_size', '151674', '词表行数（嵌入与 lm_head 绑定）'],
        ['max_position_embeddings', '32768', '最大位置长度'],
        ['rope_theta', '1000000.0', 'RoPE 基频'],
        ['rope_scaling', 'type=dynamic, factor=2.0', '动态位置插值，可外推到 2 倍长度'],
        ['max_window_layers', '70', '全注意力层上限（实际 24 层，全注意力）'],
        ['rms_norm_eps', '1e-6', 'RMSNorm epsilon'],
        ['bos / eos_token_id', '151643 / 151645', '序列开始 / 结束'],
    ],
    widths=[5.2, 3.8, 6.2])
caption('表 3：text_config（Qwen2.5-0.5B）实参（同 config.json 的 text_config 节）')

make_table(
    ['字段（vision_config）', '取值', '含义 / 源码佐证'],
    [
        ['architectures / model_type', 'InternVisionModel / internvl_vision', 'configuration_internvl.py:25'],
        ['hidden_size', '1024', '视觉编码器残差流维度'],
        ['num_hidden_layers', '24', 'ViT 层数'],
        ['num_attention_heads', '16', 'ViT 注意力头数'],
        ['intermediate_size', '4096', 'ViT MLP 维度（GELU）'],
        ['patch_size', '[14, 14]', 'patch 边长像素'],
        ['image_size', '[448, 448]', '切片输入分辨率'],
        ['attention_bias', 'true', 'qkv 投影带 bias（qkv_bias）'],
        ['layer_scale_init_value', '0.1', '每个子层输出前可学习缩放的初始化值（V2.5 新特性）'],
        ['use_mean_pooling', 'true', '分类侧对 patch 特征均值池化'],
        ['use_absolute_position_embeddings', 'true', '使用可学习二维绝对位置编码'],
        ['use_qk_norm', 'false', '未启用 QK-Norm'],
        ['initializer_factor', '0.1', '权重初始化缩放因子'],
        ['layer_norm_eps', '1e-6', 'LayerNorm epsilon'],
    ],
    widths=[5.6, 3.6, 6.0])
caption('表 4：vision_config（InternViT-300M-448px-V2.5）实参（同 config.json 的 vision_config 节）')

h2('1.2 参数量自核算（按 config 复算）')
make_table(
    ['组成', '参数量（约）', '计算式'],
    [
        ['视觉编码器 InternViT-300M', '≈302M', '每层：qkv=1024×3072 + out=1024² + MLP=2×1024×4096 ≈12.6M ×24 层 + patch 嵌入与位置编码'],
        ['语言模型 Qwen2.5-0.5B', '≈494M', '嵌入(绑定)=151674×896≈135.9M；注意力每层≈1.84M；MLP 每层=3×896×4864≈13.06M；×24 层'],
        ['MLP 投影器', '≈4.5M', 'Linear1 4096×896 + Linear2 896² + LayerNorm(4096)'],
        ['合计', '≈0.80B–0.9B', '与论文 Table 1 的 0.9B 同量级'],
    ],
    widths=[4.6, 2.6, 8.0])
caption('表 5：本地 1B 模型参数量自核算（可复算，官方口径 0.9B）')

# ================= 2 端到端架构总览 =================
h1('2 端到端架构总览')
para('端到端数据流：输入图像经动态分辨率切片预处理为若干 448×448 张量，逐切片过 InternViT-300M 视觉编码器；'
     '取最后一层隐藏态并丢弃 CLS 后按 32×32 空间网格 reshape，经 Pixel Unshuffle 把空间维减半、通道维 ×4（每切片 1024 个 patch 特征压缩为 256 个 token、维度 4096）；'
     '再过两层 MLP 投影到 896 维；最后在 LLM 的 embedding 层上，把文本序列中 image_token_id=151667 的占位位置用这些视觉特征 masked_scatter 写回，送入 Qwen2.5-0.5B 自回归解码。')
add_fig('fig1_overview.png', 16.0, '图 1：InternVL-3 端到端架构总览（ViT–MLP–LLM，以本地 1B 为主尺寸）')
para('源码对应路径：InternVLModel.__init__ 组装 vision_tower / multi_modal_projector / language_model（:530-536）；'
     'get_image_features 完成视觉编码与投影（:543-587）；forward 中 embed_tokens 查表后做 masked_scatter 替换（:634-652）；'
     '最后进入 self.language_model（:654-660）。')

# ================= 3 视觉编码器 =================
h1('3 视觉编码器 InternViT-300M-448px-V2.5')
para('视觉编码器为 InternViT-300M 规格（hidden 1024、24 层、16 头、intermediate 4096、patch 14×14、输入 448×448）。'
     '相对 1.5/2.x 时代的 InternViT，3.0 的 V2.5 版本有三个 config 可证的新特性：'
     '其一，每个注意力与 MLP 子层输出前引入可学习 LayerScale，初始化值 0.1（layer_scale_init_value=0.1）；'
     '其二，分类侧使用 mean pooling（use_mean_pooling=true）而非 CLS 全局池化；'
     '其三，使用可学习绝对位置编码（use_absolute_position_embeddings=true），并在长上下文场景叠加论文提出的 V2PE 可变视觉位置编码。')
add_fig('fig2_vit.png', 13.5, '图 2：InternViT-300M-448px-V2.5 层结构（按本地 vision_config 逐字段核对）')
para('一个 448×448 切片 = (448/14)² = 32×32 = 1024 个 patch token，外加 1 个 CLS token。'
     '前向取 vision_feature_layer=-1（最后一层），select_strategy=default 时直接切片去掉 CLS（:566-567），'
     '得到 (B, 1024, 1024) 的 patch 特征序列，再 reshape 回 32×32 空间网格供 Pixel Unshuffle 使用。')

# ================= 4 投影层与语言底座 =================
h1('4 投影层与语言底座')
h2('4.1 Pixel Unshuffle：视觉 token 数压缩到四分之一')
para('沿用 InternVL2.5 的像素重组操作（paper §2.1 称 pixel unshuffle）：把 32×32 的空间网格按 scale_factor=0.5 重组，'
     '空间维减半为 16×16，通道维按 1/0.5²=4 倍扩张（1024→4096）。于是每切片 1024 个 patch 特征被压缩为 256 个视觉 token，'
     '与 config 的 image_seq_length=256 严格相等，也与 processing_internvl.py:58 的口径一致：image_seq_length = (image_size/patch_size)² × scale_factor²。')

make_table(
    ['符号', '含义', '取值（本地 config）'],
    [
        ['H, W', '切片边长（像素）', '448, 448'],
        ['p', 'patch 边长（像素）', '14'],
        ['S', 'patch 网格边长 = H/p', '32'],
        ['r', 'downsample_ratio（空间缩放）', '0.5'],
        ['d_v', '视觉特征维度', '1024'],
        ['n_tok', '每切片视觉 token 数 = (S·r)²', '(32×0.5)² = 256'],
        ['d_out', '像素重组后通道维度 = d_v/r²', '1024/0.25 = 4096'],
    ],
    widths=[2.2, 6.4, 5.6])
caption('表 6：Pixel Unshuffle 符号表（数值均可由本地 config 复算）')

h2('4.2 MLP 投影器结构')
para('投影器为随机初始化的两层 MLP（论文 §2.1：two-layer network with random initialization），源码结构见 modeling_internvl.py:485-500：'
     'LayerNorm(4096) → Linear(4096→896) → GELU → Linear(896→896)。'
     '其中 4096 = d_v / r²，896 = 语言模型 hidden_size；即视觉特征先过归一化再投影进 LLM 的残差流空间。')

h2('4.3 语言底座 Qwen2.5-0.5B')
para('语言部分即 Qwen2.5-0.5B（HF 架构类 Qwen2ForCausalLM）：hidden 896、24 层、14 个 Q 头、2 个 KV 头（GQA 分组数 7）、'
     'head_dim=64、intermediate 4864（SwiGLU）、vocab 151674、RoPE 基频 1e6 且 dynamic 缩放 ×2、rms_norm_eps=1e-6。'
     '论文特别强调：LLM 组件一律用预训练 base 模型初始化，不使用 instruction-tuned 版本，以便在原生多模态预训练中统一塑形语言与多模态能力。')

# ================= 5 核心机制专节 =================
h1('5 核心机制专节')
h2('5.1 动态分辨率切片（tile）与缩略图')
para('为兼顾高分辨率细节与算力开销，图像不强制 resize 到固定尺寸，而是按宽高比动态切分为若干 448×448 切片，'
     '另加一张整图缩略图（thumbnail）补全全局视野。本地 preprocessor_config.json（E:\\models\\model\\InternVL3-1B-hf\\preprocessor_config.json）'
     '记录 max_patches=12、min_patches=1、size=448×448、ImageNet 均值方差归一化；'
     'processing_internvl.py:148-150 在文本中把每个 <image> 占位展开为 256×切片数 个图像 token，并用 BOI/EOI 特殊 token 包裹。')
add_fig('fig3_tile.png', 16.0, '图 3：动态分辨率切片机制（max_patches=12，含缩略图）')

h2('5.2 视觉 token 流与占位替换')
para('文本侧先按 tokenizer 约定写入 BOI、重复 N 次的图像占位 token（context_image_token，id=151667）与 EOI；'
     '模型 forward 时先 embed_tokens 查表，再用占位 id 构造 special_image_mask，把 ViT 输出的 (B, 256N, 896) 特征 masked_scatter 写回占位位置（:646-652）。'
     '源码同时校验占位 token 总数与视觉特征元素数严格相等，不等即抛错（:607-610）。'
     '视觉 token 在预训练中只作条件上下文，不参与 loss 计算（见 5.3）。')
add_fig('fig4_tokenflow.png', 16.0, '图 4：视觉 token 流与占位替换（masked_scatter 路径）')

h2('5.3 原生多模态预训练目标')
para('InternVL3 的核心训练范式是原生多模态预训练：把语言预训练与多模态对齐合并进同一个预训练阶段，'
     '大规模纯文本语料与多模态语料（图文、视频、交错图文）交错混合训练，全部参数联合优化（不再冻结 ViT 或 LLM 的任何部分）。'
     '训练目标仍为标准自回归交叉熵，但 loss 只在文本 token 上计算：')
para('L_text-only(θ) = − Σ_{i: x_i∈Text} w_i · log p_θ(x_i | x_1, …, x_{i−1})', indent=False, align=WD_ALIGN_PARAGRAPH.CENTER)
make_table(
    ['符号', '含义', '取值 / 口径'],
    [
        ['x_i', '样本中第 i 个 token（文本或视觉）', '序列长度 L'],
        ['w_i', 'token 损失权重', 'square averaging：w = 1/l^0.5（l 为样本需算 loss 的 token 数）'],
        ['视觉 token', '仅作条件，不预测、不算 loss', '论文 §2.2 式 (6)'],
        ['语料配比', '文本 : 多模态', '1 : 3；总 token 约 200B（文本 50B + 多模态 150B）'],
    ],
    widths=[2.4, 6.0, 5.8])
caption('表 7：原生多模态预训练符号表（论文 arXiv:2504.10479 §2.2）')

h2('5.4 V2PE 可变视觉位置编码')
para('为支持更长的多模态上下文，InternVL3 在视觉嵌入中引入 Variable Visual Position Encoding（V2PE）：'
     '视觉 token 的位置增量 δ 可调节，论文消融显示短上下文任务取较小 δ 即可，全文实验默认 δ=1；'
     '训练侧配合头并行（head-parallel）与序列并行（sequence-parallel）支持最长 32K token 的序列。'
     '注意本地 config 中 use_absolute_position_embeddings=true，V2PE 是在绝对位置编码之上叠加的机制。')

# ================= 6 训练管线 =================
h1('6 训练管线：原生预训练 → SFT → MPO → 测试时扩展')
para('InternVL3 的完整训练管线分三大阶段，外加推理期的测试时扩展策略：')
make_table(
    ['阶段', '数据规模', '关键技术'],
    [
        ['原生多模态预训练', '约 200B token（文本 50B + 多模态 150B，配比 1:3）', '文本+多模态语料单阶段交错联合训练；全部参数不冻结；loss 仅在文本 token 上；square averaging 加权'],
        ['监督微调 SFT', '21.7M 样本（InternVL2.5 为 16.3M）', '沿用 2.5 的随机 JPEG 压缩、square loss 重加权、多模态数据打包；新增工具调用、GUI、3D、长上下文、视频、科学图表等样本'],
        ['混合偏好优化 MPO', '约 30 万偏好对（基于 MMPR v1.2）', '损失 = DPO 偏好损失 + BCO 质量损失 + LM 生成损失；用 8B/38B/78B 的 SFT 版产出 rollout，缓解训练-推理分布偏移、增强思维链推理'],
        ['测试时扩展（推理期）', 'Best-of-N 采样', 'VisualPRM-8B 作为评审模型：对思维链每一步打分（预测 +/− 的概率），再选最优回答；小尺寸模型也能获得可观推理增益'],
    ],
    widths=[3.2, 4.6, 8.4])
caption('表 8：InternVL3 训练与推理管线（论文 arXiv:2504.10479 §2.2–§2.4）')
para('基础设施侧，团队扩展 InternEVO 框架以支持 ViT/MLP/LLM 三段式解耦切分，重叠通信与计算，'
     '同规模训练吞吐较 InternVL2.5 提升 50%–200%（论文 §2.5）。')

# ================= 7 版本演进对比 =================
h1('7 版本演进对比：InternVL 2.5 → InternVL 3.0')
para('InternVL2.5（arXiv:2412.05271）已首次引入原生多模态预训练的雏形与像素 unshuffle；'
     'InternVL3.0 在其基础上把范式收敛为完整的单阶段原生预训练，并升级视觉塔、语言底座与后训练/测试时策略。核心差异如下图与下表。')
add_fig('fig5_evolution.png', 15.0, '图 5：InternVL 2.5 → 3.0 架构与训练范式演进对比')

make_table(
    ['维度', 'InternVL 2.5', 'InternVL 3.0'],
    [
        ['预训练范式', 'MLP warmup 对齐 + 多阶段微调，语言与多模态训练分离', '语言+多模态语料单阶段原生联合训练，全部参数共同优化，约 200B token'],
        ['视觉编码器', 'InternViT-300M / 6B-448px（基础 ViT 结构）', 'InternViT-V2.5：LayerScale(0.1)、mean pooling、绝对位置编码 + V2PE'],
        ['语言底座', 'Qwen2.5 / InternLM2.5 系列', 'Qwen2.5 全系列 + InternLM3-8B（9B 型号），统一用 base 模型初始化'],
        ['后训练', 'SFT（16.3M 样本）', 'SFT 扩到 21.7M + MPO 混合偏好优化（DPO+BCO+LM，约 30 万偏好对）'],
        ['测试时策略', '常规推理', 'Best-of-N + VisualPRM-8B 分步奖励评分，思维链可叠加选优'],
        ['上下文长度', '4K–32K 常规', 'V2PE + 头/序列并行支持 32K 序列'],
    ],
    widths=[2.6, 5.9, 7.1])
caption('表 9：InternVL 2.5 vs 3.0 逐维度对比')
para('演进主线：InternVL3.0 把原生多模态预训练从可选实验收敛为标准范式——语言与视觉能力在同一阶段联合塑形，'
     '后训练用 MPO 补上思维链推理，推理期用 VisualPRM 做测试时扩展；家族规模从 1B 连续扩展到 78B，'
     '1B 级视觉塔在保留 300M 规格的同时完成了 V2.5 结构升级。')

# ================= 8 来源 =================
h1('8 来源')
refs = [
    '[1] 本地 config：E:\\models\\model\\InternVL3-1B-hf\\config.json（model_type=internvl；text_config=Qwen2ForCausalLM；vision_config=internvl_vision）。',
    '[2] 本地预处理配置：E:\\models\\model\\InternVL3-1B-hf\\preprocessor_config.json（size=448×448，max_patches=12，min_patches=1）。',
    '[3] 本地 transformers 源码：E:\\transformers\\src\\transformers\\models\\internvl\\modeling_internvl.py（InternVLForConditionalGeneration:741；get_image_features:543；pixel_shuffle:670；InternVLMultiModalProjector:485）；configuration_internvl.py；processing_internvl.py（占位展开:148，image_seq_length 口径:58）。',
    '[4] InternVL3 论文：InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models，arXiv:2504.10479 — https://arxiv.org/abs/2504.10479。',
    '[5] InternVL2.5 论文（演进对照）：Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling，arXiv:2412.05271 — https://arxiv.org/abs/2412.05271。',
    '[6] InternVL3 官方博客（2025-04-11）— https://internvl.github.io/blog/2025-04-11-InternVL-3.0/。',
    '[7] InternVL 开源仓库 — https://github.com/OpenGVLab/InternVL。',
]
for r in refs:
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(0)
    run = p.add_run(r)
    run.font.size = Pt(10.5)
    run.font.name = 'Times New Roman'
    run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')

para('凡本地 config 与论文口径并列处均已标注；本文未采用无法核实的传闻性结论。访问与核验日期：2026-10-26。', indent=False)

doc.save(OUT)
print('saved:', OUT)
