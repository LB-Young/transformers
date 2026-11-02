# -*- coding: utf-8 -*-
"""生成 CLIP-家族架构详解.docx（A4、编号章节、统一表格、嵌入架构图）"""
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import os

ARCH = r"E:\transformers\bayon-note\CLIP\_arch"
OUT = r"E:\transformers\bayon-note\CLIP\CLIP-家族架构详解.docx"

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.top_margin = sec.bottom_margin = sec.left_margin = sec.right_margin = Cm(2.5)

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

def code_para(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.size = Pt(10.5)
    r.font.name = 'Consolas'
    r._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    p.paragraph_format.left_indent = Pt(24)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
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
tr = tp.add_run('CLIP 家族架构详解')
tr.font.size = Pt(18); tr.font.bold = True; tr.font.name = 'Times New Roman'
tr._element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')

sp = doc.add_paragraph(); sp.alignment = WD_ALIGN_PARAGRAPH.CENTER
sr = sp.add_run('双塔对比学习奠基之作 · 400M 网络图文对 · 对称 InfoNCE · 零样本迁移范式')
sr.font.size = Pt(12); sr.font.name = 'Times New Roman'
sr._element.rPr.rFonts.set(qn('w:eastAsia'), '楷体')

para('依据：transformers 源码（本机 editable）E:\\transformers\\src\\transformers\\models\\clip\\（版本 5.23.0.dev0，'
     '本文档所有行号/字段均以此为准），官方论文 arXiv:2103.00020（Radford et al., 2021, PMLR），HF 模型卡交叉验证。'
     'CLIP 是 OpenAI 2021 年的双塔视觉语言模型：图像与文本各自编码到同一向量空间，用对比学习对齐，'
     '以 4 亿网络图文对为监督，训练出可零样本迁移的视觉模型。标注「待确认」处未编造。', indent=True)

# ================= 1 定位 =================
h1('1 一句话定位')
para('CLIP（Contrastive Language-Image Pre-training）核心主张：图像和文本各自编码到同一个向量空间，'
     '用对比学习对齐，以 4 亿（400M）网络图文对为监督，训练出可零样本迁移的视觉模型——'
     '零样本分类、图文检索、图文相似度的标准范式。现代多模态 LLM 的视觉塔大多直接或间接继承自 CLIP。')

# ================= 2 模块与类 =================
h1('2 源码模块与类关系')
make_table(
    ['文件', '职责'],
    [
        ['configuration_clip.py', 'CLIPTextConfig / CLIPVisionConfig / CLIPConfig'],
        ['modeling_clip.py', '全部模型类（文本塔 / 视觉塔 / 双塔 / 任务头）'],
        ['tokenization_clip.py', 'CLIP 专用 BPE 分词器（49408 词表）'],
        ['processing_clip.py', 'Processor（分词 + 图像预处理合一）'],
        ['image_processing_clip.py / image_processing_pil_clip.py', '图像 resize → center crop → 归一化'],
        ['convert_clip_original_pytorch_to_hf.py', 'OpenAI 官方权重 → HF 格式'],
    ],
    widths=[6.0, 8.8])
caption('表 1：CLIP 源码文件分工')

h2('2.1 modeling_clip.py 类清单')
make_table(
    ['类', '行号', '职责'],
    [
        ['CLIPVisionEmbeddings', ':138', 'patch 化嵌入 + CLS + 位置编码'],
        ['CLIPTextEmbeddings', ':221', 'token + 位置编码（因果注意力的文本嵌入）'],
        ['CLIPAttention / CLIPMLP', ':280 / :338', '注意力 / FFN（激活 quick_gelu）'],
        ['CLIPEncoderLayer', ':353', '单层 Transformer（继承 GradientCheckpointingLayer）'],
        ['CLIPEncoder', ':446', '12 层编码器'],
        ['CLIPTextModel', ':485', '文本塔（池化取 [EOS] 状态）'],
        ['CLIPVisionModel', ':585', '视觉塔（池化取 [CLS] 状态）'],
        ['CLIPModel', ':651', '双塔完整模型（对比损失 + 相似度）'],
        ['CLIPTextModelWithProjection', ':826', '文本塔 + 投影层（检索用）'],
        ['CLIPVisionModelWithProjection', ':890', '视觉塔 + 投影层（检索用）'],
        ['CLIPForImageClassification', ':959', '分类头（HF 为有监督微调加的线性头，非原论文结构）'],
    ],
    widths=[5.4, 2.6, 6.8])
caption('表 2：modeling_clip.py 类清单')

# ================= 3 配置字段 =================
h1('3 配置字段（configuration_clip.py）')
h2('3.1 文本塔 CLIPTextConfig（:27）')
make_table(
    ['字段', '默认值', '说明'],
    [
        ['model_type', 'clip_text_model', '—'],
        ['vocab_size', '49408', 'BPE 词表（含 startoftext/endoftext 等特殊 token）'],
        ['hidden_size', '512', '—'],
        ['intermediate_size', '2048', '—'],
        ['projection_dim', '512', '投影目标维度'],
        ['num_hidden_layers', '12', '—'],
        ['num_attention_heads', '8', '—'],
        ['max_position_embeddings', '77', '文本最长 77 token'],
        ['hidden_act', 'quick_gelu', 'CLIP 特有激活'],
        ['bos / eos / pad', '49406 / 49407 / 1', '特殊 token'],
    ],
    widths=[4.6, 3.2, 7.0])
caption('表 3：文本塔配置字段')

h2('3.2 视觉塔 CLIPVisionConfig（:77）')
make_table(
    ['字段', '默认值', '说明'],
    [
        ['model_type', 'clip_vision_model', '—'],
        ['hidden_size', '768', '—'],
        ['intermediate_size', '3072', '—'],
        ['projection_dim', '512', '—'],
        ['num_hidden_layers', '12', '—'],
        ['num_attention_heads', '12', '—'],
        ['image_size', '224', '输入分辨率'],
        ['patch_size', '32', '→ 7×7 = 49 patch + CLS = 50 token'],
        ['hidden_act', 'quick_gelu', '—'],
    ],
    widths=[4.6, 3.2, 7.0])
caption('表 4：视觉塔配置字段')

h2('3.3 双塔 CLIPConfig（:122）')
make_table(
    ['字段', '默认值', '说明'],
    [
        ['text_config / vision_config', '子配置', '可分别覆盖'],
        ['logit_scale_init_value', '2.6592', '可学习温度初始值 ≈ ln(1/0.07)'],
    ],
    widths=[4.6, 3.2, 7.0])
caption('表 5：双塔配置字段')

# ================= 4 端到端架构 =================
h1('4 端到端双塔架构总览')
para('视觉塔把 224×224 切成 7×7=49 个 32×32 patch + CLS → 12 层双向 Transformer → 池化取 [CLS]；'
     '文本塔 BPE 分词 → 12 层因果 Transformer（GPT-2 式）→ 池化取 [EOS]。两塔投影到 512 维公共空间做对比。')
add_fig('01_clip端到端双塔架构.png', 15.2, '图 1：CLIP 端到端双塔架构总览')

# ================= 5 核心机制 =================
h1('5 核心机制（对照代码）')
h2('5.1 双塔各自编码')
para('视觉塔（CLIPVisionModel :585）：CLIPVisionEmbeddings 把 224×224 切成 32×32 patch（49 个）拼上 CLS token → '
     '12 层双向 Transformer → 池化取 [CLS] 状态（:630 docstring「pooled CLS states」）。')
para('文本塔（CLIPTextModel :485）：BPE 分词 → 位置编码 → 12 层因果 Transformer（GPT-2 式）→ '
     '池化取 [EOS]（最后一个 token）状态（:524 docstring「pooled (EOS token) states」）。')
h2('5.2 投影与相似度（CLIPModel :651）')
code_para('image_embeds = visual_projection(pooler_output)   # 768 → 512')
code_para('text_embeds  = text_projection(pooler_output)     # 512 → 512')
code_para('两者 L2 归一化 → 余弦相似度矩阵 × logit_scale')
para('logit_scale 是可学习参数，初始 2.6592 ≈ ln(1/0.07)（对比 OpenAI 原实现）。')
h2('5.3 对比损失（InfoNCE）')
para('对 batch 内每张图：拉近正样本对 (I_i, T_i)，推远 batch 内其余文本 (I_i, T_j) 作为负样本；'
     '对称地也以文本为锚。不做显式 hard-negative 挖掘，靠大 batch 提供负样本。')
h2('5.4 图像预处理')
para('image_processing_clip.py：resize（保持比例）→ center_crop(224) → ImageNet 均值/方差归一化——与训练口径严格一致。')

# ================= 6 训练方法 =================
h1('6 训练方法（论文层面）')
make_table(
    ['要素', '要点'],
    [
        ['数据', '4 亿（400M）从互联网收集的图文对，天然带噪声但规模巨大'],
        ['目标', '对称对比损失（InfoNCE），batch 极大（论文以 32768 级 batch 训练）'],
        ['可学习温度', 'logit_scale 随训练调整，避免人为设定 softmax 温度'],
        ['零样本迁移', '分类时把类别名拼成「a photo of a {label}」模板输入文本塔，取与图像特征相似度最高者'],
    ],
    widths=[3.4, 11.4])
caption('表 6：CLIP 训练方法要点')

# ================= 7 家族速览 =================
h1('7 模型家族速览')
make_table(
    ['变体', '视觉塔', '说明'],
    [
        ['openai/clip-vit-base-patch32', 'ViT-B/32', '默认配置口径'],
        ['openai/clip-vit-base-patch16', 'ViT-B/16', '—'],
        ['openai/clip-vit-large-patch14', 'ViT-L/14', '—'],
        ['openai/clip-vit-large-patch14-336', 'ViT-L/14@336', '336 分辨率微调'],
        ['RN 系列', 'RN50 / RN101 / RN50x4 / RN50x16 / RN50x64', 'ResNet 视觉塔变体（EfficientNet 缩放）'],
    ],
    widths=[6.4, 3.4, 5.0])
caption('表 7：CLIP 家族检查点')

h2('一分钟结论')
make_table(
    ['维度', '要点'],
    [
        ['架构', '视觉塔 ViT-B/32（12L/768/12H）+ 文本塔 Transformer（12L/512/8H）双塔'],
        ['对齐', '投影到 512 维公共空间，L2 归一化 + 余弦相似度 × logit_scale'],
        ['损失', '对称 InfoNCE 对比损失，正对拉近 / 负对推远'],
        ['池化', '视觉取 [CLS]，文本取 [EOS]'],
        ['预训练数据', '4 亿网络图文对'],
        ['代码入口', 'CLIPModel / CLIPTextModel / CLIPVisionModel / *WithProjection / CLIPForImageClassification'],
    ],
    widths=[3.0, 11.8])
caption('表 8：一分钟结论')
para('与多模态 LLM 的关系：LLaVA、BLIP-2、InstructBLIP、MiniGPT-4 等第一代多模态 LLM 直接借用 CLIP 视觉塔'
     '（通常 ViT-L/14@336）作为视觉编码器；SigLIP（同仓库 models/siglip/）用逐对 sigmoid 损失替代 CLIP 的 '
     'softmax 对比损失；BLIP 的 ITC 目标即 CLIP 式对比。Qwen2-VL / InternVL 等自研视觉塔虽非 CLIP 实现，'
     '但同属「图文对比预训练视觉编码器」谱系，CLIP 是这条线的原点。')

# ================= 8 来源 =================
h1('8 来源')
refs = [
    '[1] 论文：Learning Transferable Visual Models From Natural Language Supervision，arXiv:2103.00020（Radford et al., 2021, PMLR）',
    '[2] HF 模型卡：openai/clip-vit-base-patch32 等',
    '[3] transformers 源码（本机 editable）：src/transformers/models/clip/（5.23.0.dev0）',
]
for r in refs:
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(0)
    run = p.add_run(r)
    run.font.size = Pt(10.5)
    run.font.name = 'Times New Roman'
    run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
para('待确认/边界：论文训练超参（batch、epochs）与各变体精确参数以论文/模型卡为准；'
     'CLIPForImageClassification 是 HF 为有监督微调额外加的线性头，不属于原论文结构。', indent=False)

doc.save(OUT)
print('saved:', OUT)
