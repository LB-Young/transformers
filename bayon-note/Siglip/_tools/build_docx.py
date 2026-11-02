# -*- coding: utf-8 -*-
"""生成 Siglip-家族架构详解.docx（A4、编号章节、统一表格、嵌入架构图）"""
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import os

ARCH = r"E:\transformers\bayon-note\Siglip\_arch"
OUT = r"E:\transformers\bayon-note\Siglip\Siglip-家族架构详解.docx"

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
tr = tp.add_run('SigLIP vs SigLIP 2 架构详解')
tr.font.size = Pt(18); tr.font.bold = True; tr.font.name = 'Times New Roman'
tr._element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')

sp = doc.add_paragraph(); sp.alignment = WD_ALIGN_PARAGRAPH.CENTER
sr = sp.add_run('逐对 sigmoid 损失 · BiC 偏置校正 · SigLIP 2 NaFlex 原生灵活分辨率 · decoder loss')
sr.font.size = Pt(12); sr.font.name = 'Times New Roman'
sr._element.rPr.rFonts.set(qn('w:eastAsia'), '楷体')

para('依据：transformers 源码（本机 editable）E:\\transformers\\src\\transformers\\models\\siglip\\（v1）与 '
     'E:\\transformers\\src\\transformers\\models\\siglip2\\（v2），版本 5.21.0.dev0（本文档所有行号/字段均以此为准），'
     '官方论文 arXiv:2303.15343（SigLIP 1）/ arXiv:2502.14786（SigLIP 2），官方模型卡交叉验证。'
     'SigLIP 是 Google 的视觉语言双塔编码器，核心用逐对 sigmoid 损失替代 CLIP 的 softmax 对比损失；'
     'SigLIP 2 引入 NaFlex 原生灵活分辨率与更强训练目标。视觉塔被大量多模态 LLM（含 Gemma 3）直接借用。'
     '标注「待确认」处未编造。', indent=True)

# ================= 1 定位 =================
h1('1 一句话定位')
para('SigLIP 1（2023，arXiv:2303.15343）：固定分辨率（默认 224×224 → 14×14=196 patch），learned 位置嵌入，'
     'sigmoid 损失训练更稳、收敛更快。')
para('SigLIP 2（2025，arXiv:2502.14786）：NaFlex 原生灵活分辨率（单 checkpoint 支持任意宽高比/分辨率，≤256 patch），'
     '训练目标叠加 decoder loss + global-local / masked prediction，多语言 + 去偏数据。')

# ================= 2 模块与类 =================
h1('2 源码模块与类关系')
para('两个版本是独立模块，但 v2 的类全部继承 v1；文本侧 v2 与 v1 结构完全一致（继承、无重写）——差异全部集中在视觉塔。')
make_table(
    ['模块', '视觉类', '文本类', '双塔类', '分类头'],
    [
        ['models/siglip/', 'SiglipVisionModel（:553）', 'SiglipTextModel（:471）', 'SiglipModel（:647）', 'SiglipForImageClassification（:835）'],
        ['models/siglip2/', 'Siglip2VisionModel（:499，继承 SiglipVisionModel）', 'Siglip2TextModel（:587，继承 SiglipTextModel）', 'Siglip2Model（:712，继承 SiglipModel）', 'Siglip2ForImageClassification（:915）'],
    ],
    widths=[2.6, 4.0, 3.6, 3.0, 2.8])
caption('表 1：两模块类关系')
para('modeling_siglip2.py 由 modular_siglip2.py 自动生成（文件头有警示，改动要改 modular）。', indent=False)

# ================= 3 配置字段对比 =================
h1('3 配置字段对比（configuration_siglip.py vs configuration_siglip2.py）')
h2('3.1 文本侧：完全相同')
make_table(
    ['字段', 'SiglipTextConfig', 'Siglip2TextConfig'],
    [
        ['model_type', 'siglip_text_model', 'siglip2_text_model'],
        ['vocab_size', '32000', '32000'],
        ['hidden_size', '768', '768'],
        ['intermediate_size', '3072', '3072'],
        ['num_hidden_layers', '12', '12'],
        ['num_attention_heads', '12', '12'],
        ['max_position_embeddings', '64', '64'],
        ['hidden_act', 'gelu_pytorch_tanh', 'gelu_pytorch_tanh'],
        ['projection_size', '默认 = hidden_size', '默认 = hidden_size'],
    ],
    widths=[4.6, 5.0, 5.0])
caption('表 2：文本侧配置字段（两版相同）')

h2('3.2 视觉侧：关键分界点')
make_table(
    ['字段', 'SiglipVisionConfig', 'Siglip2VisionConfig'],
    [
        ['model_type', 'siglip_vision_model', 'siglip2_vision_model'],
        ['hidden_size', '768', '768'],
        ['intermediate_size', '3072', '3072'],
        ['num_hidden_layers', '12', '12'],
        ['num_attention_heads', '12', '12'],
        ['patch_size', '16', '16'],
        ['image_size', '224（固定，可配 list）', '无此字段'],
        ['num_patches', '无此字段', '256（NaFlex：patch 数上限）'],
    ],
    widths=[4.6, 5.0, 5.0])
caption('表 3：视觉侧配置字段（两版差异点）')
para('一句话：v1 的视觉配置靠 image_size 定死输入；v2 删掉 image_size，改为「最多 256 个 patch」的容量上限——'
     '这是 NaFlex 的配置层体现。', indent=False)

# ================= 4 视觉塔实现对比 =================
h1('4 视觉塔实现对比（核心机制）')
h2('4.1 位置嵌入：固定扁平 vs 16×16 网格插值')
para('SigLIP 1（SiglipVisionEmbeddings，modeling_siglip.py:116）：patch_embedding = nn.Linear(3×16×16, 768)；'
     'position_embedding = nn.Embedding(196, 768)（224/16=14 → 14×14=196，扁平索引）；'
     'forward：embeddings = inputs_embeds + position_embeddings（纯加性，位置网格固定）。')
para('SigLIP 2（Siglip2VisionEmbeddings，modeling_siglip2.py:114）：patch_embedding 同上；'
     'position_embedding = nn.Embedding(256, 768) —— 看作 16×16 的 2D 位置网格（position_embedding_size = sqrt(256) = 16）；'
     'forward 时调用 resize_positional_embeddings（:130）：① 按该图实际 patch 网格 (height, width)，'
     '用双线性插值（bilinear + antialias）把 16×16 网格缩放到目标尺寸（保持宽高比）；'
     '② 不足 max_length=256 的部分，用插值结果的第一行 resized_embeddings[0] 复制填充（不是零填充）；'
     '③ 最终输出 (batch, 256, 768) 固定序列。')
add_fig('01_siglip两代视觉塔对比.png', 15.2, '图 1：SigLIP 1 vs SigLIP 2 视觉塔位置编码与 NaFlex 流程')

h2('4.2 forward 签名与 padding 掩码')
make_table(
    ['', 'SiglipVisionModel.forward', 'Siglip2VisionModel.forward'],
    [
        ['输入', 'pixel_values（固定 (B,3,H,W)）', 'pixel_values（已 patchify，(B,256,…)）+ pixel_attention_mask + spatial_shapes'],
        ['掩码', '无需', 'create_bidirectional_mask 按像素掩码遮掉 padding patch'],
        ['输出', 'last_hidden_state + pooler_output', '同左'],
    ],
    widths=[2.4, 6.0, 6.0])
caption('表 4：forward 签名对比')

h2('4.3 共享结构（两版相同）')
para('双向注意力 Encoder（SiglipEncoderLayer / Siglip2EncoderLayer，均 12 层）、post_layernorm、'
     'MHA pooling head（SiglipMultiheadAttentionPoolingHead / Siglip2MultiheadAttentionPoolingHead，'
     'use_head 可关，vision_use_head 控制）——视觉塔输出用多头注意力池化聚合，而非 CLIP 式的取 CLS。')

# ================= 5 训练方法对比 =================
h1('5 训练方法对比（论文层面）')
h2('5.1 SigLIP 1（arXiv:2303.15343）')
code_para('对 batch 内 (图像, 文本) 每对独立计算：')
code_para('loss = log(1 + exp(-y·(sim·t + b)))')
code_para('# 不做 softmax 归一化 —— 摆脱负样本 pair 之间的耦合，batch 可更大、收敛更快')
code_para('# t：可学习温度，初始 log(10)；b：可学习偏置，初始 -10（BiC 偏置校正初始化）')
code_para('# —— 负样本对远多于正样本，先压低所有 logit，避免初期大步纠偏')
h2('5.2 SigLIP 2（arXiv:2502.14786）')
make_table(
    ['训练目标', '说明'],
    [
        ['decoder loss', '解码器重建图像 patch，辅助密集特征'],
        ['global-local / masked prediction', '全局-局部一致性 + 掩码预测，提升定位/局部语义（检测、分割类任务）'],
        ['NaFlex 训练', '训练时混用多分辨率/多宽高比，位置嵌入插值'],
        ['多语言 + 去偏过滤', 'WebLI 数据上多语言训练 + debiasing 过滤，跨文化语义更均衡'],
    ],
    widths=[5.0, 9.8])
caption('表 5：SigLIP 2 叠加训练目标')
para('注意：decoder / masked 分支是训练期目标，HF 发布的推理模型只有「文本塔 + 视觉塔 + 对比头」结构，'
     '不含这些训练分支。', indent=False)

# ================= 6 家族速览 =================
h1('6 模型家族速览')
make_table(
    ['版本', '官方检查点示例', '配置要点'],
    [
        ['SigLIP 1', 'google/siglip-base-patch16-224、google/siglip-so400m-patch14-384', '固定分辨率 224/384，patch 16/14'],
        ['SigLIP 2', 'google/siglip2-base-patch16-naflex、google/siglip2-so400m-patch16-naflex、google/siglip2-base-patch16-224/384', 'naflex = NaFlex 变体；非 naflex = 固定分辨率变体'],
    ],
    widths=[2.6, 8.6, 4.2])
caption('表 6：官方检查点示例')

h2('一分钟结论')
make_table(
    ['维度', 'SigLIP 1', 'SigLIP 2'],
    [
        ['分辨率', '固定（224/384）', 'NaFlex 灵活（≤256 patch，保宽高比）'],
        ['位置嵌入', '固定扁平 learned（加性）', '16×16 网格 learned + 双线性插值 + 边界复制 pad'],
        ['输入', '原始图 (B,3,H,W)', 'patchify 序列 + pixel mask + spatial_shapes'],
        ['训练目标', 'sigmoid 对比 + t/b 初始化', '+ decoder loss + masked prediction + 多语言去偏'],
        ['文本塔', '12L/768/12H', '完全相同（继承）'],
        ['输出池化', 'MHA pooling head', 'MHA pooling head（同）'],
    ],
    widths=[2.6, 5.8, 6.4])
caption('表 7：一分钟结论')
para('与多模态 LLM 的关系：Gemma 3 视觉编码器即 SigLIP-2 变体（400M 级）；InternVL 视觉塔 InternViT 与 SigLIP '
     '同源思路（patch 化 ViT + 双塔预训练）；Qwen2-VL / Qwen3-VL 视觉塔为自研 ViT（patch 16 + 2D-RoPE + RMSNorm），'
     '不是 SigLIP 实现，但其可变分辨率（任意宽高比、位置编码插值）与 SigLIP 2 的 NaFlex 思路一致；'
     '后续工作 SigLIP-HD（用 SigLIP 2 多尺度教师蒸馏标准分辨率学生）等。')

# ================= 7 来源 =================
h1('7 来源')
refs = [
    '[1] SigLIP 论文：Sigmoid Loss for Language Image Pre-Training，arXiv:2303.15343（Zhai et al., 2023）',
    '[2] SigLIP 2 论文：SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features，arXiv:2502.14786（Tschannen et al., 2025）',
    '[3] HF 模型卡：google/siglip2-so400m-patch16-naflex（训练目标/数据说明）',
    '[4] transformers 源码（本机 editable）：src/transformers/models/siglip/、src/transformers/models/siglip2/（5.21.0.dev0）',
]
for r in refs:
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(0)
    run = p.add_run(r)
    run.font.size = Pt(10.5)
    run.font.name = 'Times New Roman'
    run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
para('待确认/边界：SigLIP 2 具体训练分辨率集合与 so400m 精确参数以官方权重为准（未在本机下载验证权重）；'
     'SigLIP 2 论文中 decoder 分支的 HF 复现细节未展开。', indent=False)

doc.save(OUT)
print('saved:', OUT)
