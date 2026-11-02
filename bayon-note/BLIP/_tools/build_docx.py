# -*- coding: utf-8 -*-
"""生成 BLIP-家族架构详解.docx（A4、编号章节、统一表格、嵌入架构图）"""
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import os

ARCH = r"E:\transformers\bayon-note\BLIP\_arch"
OUT = r"E:\transformers\bayon-note\BLIP\BLIP-家族架构详解.docx"

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
tr = tp.add_run('BLIP 家族架构详解')
tr.font.size = Pt(18); tr.font.bold = True; tr.font.name = 'Times New Roman'
tr._element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')

sp = doc.add_paragraph(); sp.alignment = WD_ALIGN_PARAGRAPH.CENTER
sr = sp.add_run('统一视觉语言理解与生成 · MED 同一套参数三模式 · ITC/ITM/ITG · CapFilt 数据引导')
sr.font.size = Pt(12); sr.font.name = 'Times New Roman'
sr._element.rPr.rFonts.set(qn('w:eastAsia'), '楷体')

para('依据：transformers 源码（本机 editable）E:\\transformers\\src\\transformers\\models\\blip\\（版本 5.23.0.dev0，'
     '本文档所有行号/字段均以此为准），官方论文 arXiv:2201.12086（Li et al., 2022, ICML），HF 模型卡交叉验证。'
     'BLIP 是 Salesforce 提出的统一多模态框架：同一套 MED 文本参数以三种模式同时支持图文检索、图像描述、VQA，'
     '并用 CapFilt 从噪声网络图文对中引导出高质量训练集。与 CLIP 只做「对齐/匹配」不同，BLIP 理解与生成通吃。'
     '标注「待确认」处未编造。', indent=True)

# ================= 1 定位 =================
h1('1 一句话定位')
para('BLIP（Bootstrapping Language-Image Pre-training，ICML 2022）是 Salesforce 的统一多模态框架'
     '（论文 arXiv:2201.12086）：同一套参数（MED，Multimodal mixture of Encoder-Decoder）以三种模式同时支持'
     '图文检索、图像描述生成、VQA，并用 CapFilt 机制从噪声网络图文对中「引导（Bootstrapping）」出高质量训练数据。'
     '它是 BLIP-2、InstructBLIP 的前身。')

# ================= 2 模块与类 =================
h1('2 源码模块与类关系')
make_table(
    ['文件', '职责'],
    [
        ['configuration_blip.py', 'BlipTextConfig / BlipVisionConfig / BlipConfig'],
        ['modeling_blip.py', '视觉塔 + 双塔模型 + 任务模型（字幕 / VQA / 检索）'],
        ['modeling_blip_text.py', '文本 Transformer（encoder/decoder 可切换，含跨注意力）'],
        ['processing_blip.py', 'Processor（分词 + 图像预处理合一）'],
        ['image_processing_blip.py / image_processing_pil_blip.py', '图像预处理（resize → 中心裁剪 384 → ImageNet 归一化）'],
        ['convert_blip_original_pytorch_to_hf.py', 'Salesforce 官方权重 → HF 格式'],
    ],
    widths=[5.6, 9.2])
caption('表 1：BLIP 源码文件分工')

h2('2.1 modeling_blip.py 类清单')
make_table(
    ['类', '行号', '职责'],
    [
        ['BlipVisionEmbeddings', ':172', '视觉 patch 嵌入 + CLS + 位置编码'],
        ['BlipTextEmbeddings', ':247', '文本嵌入'],
        ['BlipAttention / BlipMLP', ':285 / :348', '注意力 / FFN'],
        ['BlipEncoderLayer', ':363', '单层 Transformer（继承 GradientCheckpointingLayer）'],
        ['BlipEncoder', ':418', '12 层编码器'],
        ['BlipVisionModel', ':450', '视觉塔（池化取 [CLS] + post_layernorm）'],
        ['BlipModel', ':509', '双塔：ITC 对比 + ITM 匹配头（get_image_features / get_text_features）'],
        ['BlipForConditionalGeneration', ':769', '图像描述生成（ITG，GenerationMixin）'],
        ['BlipForQuestionAnswering', ':942', 'VQA（GenerationMixin）'],
        ['BlipForImageTextRetrieval', ':1165', '图文检索（ITM/ITC 双头，含 use_itm_head 开关）'],
    ],
    widths=[5.2, 2.6, 7.0])
caption('表 2：modeling_blip.py 类清单')

h2('2.2 modeling_blip_text.py 类清单（文本侧）')
make_table(
    ['类', '行号', '职责'],
    [
        ['BlipTextEmbeddings', ':46', '文本嵌入'],
        ['BlipTextSelfAttention / BlipTextAttention', ':93 / :217', '自注意力（decoder 模式含 cross-attention 分支）'],
        ['BlipTextLayer', ':272', '单层：is_decoder 为 True 时启用跨注意力（:280-282）'],
        ['BlipTextEncoder', ':321', '12 层编码器'],
        ['BlipTextPooler', ':372', '池化'],
        ['BlipTextLMHead / BlipTextOnlyMLMHead', ':406 / :423', 'LM / MLM 预测头'],
        ['BlipTextModel', ':460', 'BERT 式文本模型，is_decoder 可切换（:498）'],
        ['BlipTextLMHeadModel', ':582', '因果 LM 模型（GenerationMixin，默认 is_decoder=True）'],
    ],
    widths=[5.4, 2.6, 6.8])
caption('表 3：modeling_blip_text.py 类清单')

# ================= 3 配置字段 =================
h1('3 配置字段（configuration_blip.py）')
h2('3.1 文本塔 BlipTextConfig（:27）')
make_table(
    ['字段', '默认值', '说明'],
    [
        ['model_type', 'blip_text_model', '—'],
        ['vocab_size', '30524', '词表'],
        ['hidden_size', '768', '残差流维度'],
        ['encoder_hidden_size', '768', '跨注意力输入维度（BlipConfig.__post_init__ 强制 = 视觉塔 hidden）'],
        ['intermediate_size', '3072', 'FFN 维度'],
        ['projection_dim', '768', '文本投影维度'],
        ['num_hidden_layers', '12', '层数'],
        ['num_attention_heads', '8', '注意力头'],
        ['max_position_embeddings', '512', '文本最大长度'],
        ['is_decoder', 'True', '默认按解码器初始化（ITG 用）'],
        ['bos / eos / pad / sep', '30522 / 2 / 0 / 102', '特殊 token'],
        ['tie_word_embeddings', 'True', '嵌入与 LM 头绑定'],
    ],
    widths=[4.6, 3.2, 7.0])
caption('表 4：文本塔配置字段')

h2('3.2 视觉塔 BlipVisionConfig（:77）')
make_table(
    ['字段', '默认值', '说明'],
    [
        ['model_type', 'blip_vision_model', '—'],
        ['hidden_size', '768', '—'],
        ['intermediate_size', '3072', '—'],
        ['projection_dim', '512', '视觉投影维度（与文本侧不同，靠 BlipModel 层对齐）'],
        ['num_hidden_layers', '12', '—'],
        ['num_attention_heads', '12', '—'],
        ['image_size', '384', '输入分辨率'],
        ['patch_size', '16', '→ 24×24 = 576 patch + CLS = 577 token'],
        ['hidden_act', 'gelu', '—'],
    ],
    widths=[4.6, 3.2, 7.0])
caption('表 5：视觉塔配置字段')

h2('3.3 双塔 BlipConfig（:112）')
make_table(
    ['字段', '默认值', '说明'],
    [
        ['text_config / vision_config', '子配置', '可分别覆盖'],
        ['projection_dim', '512', 'ITC 公共空间维度'],
        ['logit_scale_init_value', '2.6592', '同 CLIP 口径 ≈ ln(1/0.07)'],
        ['image_text_hidden_size', '256', 'ITM 融合层维度（图文匹配二分类头）'],
        ['label_smoothing', '0.0', '文本生成可配'],
        ['tie_word_embeddings', 'True', '—'],
    ],
    widths=[4.6, 3.2, 7.0])
caption('表 6：双塔配置字段')

# ================= 4 端到端架构 =================
h1('4 端到端架构总览')
para('图像 → 视觉塔（ViT，384×384 / patch 16 → 577 token）→ [CLS] 池化 + post_layernorm → 512 维视觉嵌入；'
     '文本 → 文本塔（12L/768/8H，is_decoder 可切换）→ 512 维文本嵌入。两塔在 BlipModel 层投影对齐，'
     'ITC/ITM/ITG 三头复用同一套特征。')
add_fig('01_blip端到端架构总览.png', 15.2, '图 1：BLIP 端到端架构总览（双塔 + 公共空间 + 三头）')
para('跨注意力（ITM/ITG）路径：图像特征作为 encoder_hidden_states 注入文本塔（见第 5 节 MED 三模式）。')

# ================= 5 MED 三模式 =================
h1('5 MED：同一套文本参数，三种模式')
para('BlipTextLayer（:272）在 is_decoder=True 时加入 cross-attention（:280-282），由此同一套 Transformer 权重可切换为三种模式：')
make_table(
    ['模式', '使用目标', '实现'],
    [
        ['单模态编码器', 'ITC（对比）', 'BlipModel 中 text_model(input_ids)，无跨注意力'],
        ['图像引导编码器', 'ITM / VQA', 'text_model(..., encoder_hidden_states=image_embeds)，启用跨注意力'],
        ['图像引导解码器', 'ITG（生成）', 'text_decoder(..., encoder_hidden_states=image_embeds, labels)，自回归 LM'],
    ],
    widths=[3.4, 3.0, 8.4])
caption('表 7：MED 三模式')
add_fig('02_blip_med三模式.png', 15.2, '图 2：MED 三种模式切换示意')

# ================= 6 三目标 =================
h1('6 三个预训练目标（实现细节）')
h2('6.1 ITC 图文对比（BlipModel :509，代码 :730-744）')
code_para('image_embeds = visual_projection(pooler_output)   # 768 → 512')
code_para('text_embeds  = text_projection(pooler_output)     # 768 → 512')
code_para('两者 L2 归一化 → 余弦相似度矩阵 × logit_scale   # 同 CLIP 对比逻辑')
h2('6.2 ITM 图文匹配（BlipModel / BlipForImageTextRetrieval :1165）')
para('文本编码器以 image_embeds 为 encoder_hidden_states 做跨注意力融合；融合输出过 image_text_hidden_size=256 的'
     '二分类头，判断「图文是否配对」——与 ITC 的全局余弦对齐不同，ITM 是细粒度 token 级匹配。')
h2('6.3 ITG 图像引导文本生成（BlipForConditionalGeneration :769）')
para('text_decoder 以 image_embeds 为跨注意力输入，自回归生成 caption，优化标准 LM loss；'
     'VQA（BlipForQuestionAnswering :942）同构，把问题也作为输入 token。')

# ================= 7 CapFilt =================
h1('7 训练方法：CapFilt（论文层面）')
para('问题：网络图文对（如 LAION）噪声大，直接预训练会被错误文本带偏；人工标注质量高但量小。'
     'CapFilt 用「生成器 + 过滤器」两级引导：')
add_fig('03_blip_capfilt数据引导.png', 15.2, '图 3：CapFilt 数据引导流程')
para('训练数据来源：COCO / Visual Genome / Conceptual Captions / SBU Captions 等 14M 级人工标注数据 + '
     '大规模网络图片（论文整体扩充规模约 1.29 亿张，具体构成以论文为准，待确认）。')

# ================= 8 家族速览 =================
h1('8 模型家族速览')
make_table(
    ['任务', 'HF 检查点示例'],
    [
        ['图像描述（ITG）', 'Salesforce/blip-image-captioning-base / blip-image-captioning-large'],
        ['VQA', 'Salesforce/blip-vqa-base / blip-vqa-large'],
        ['图文检索（ITM+ITC）', 'Salesforce/blip-itm-base-coco / blip-itm-large-coco / blip-itm-base-flickr / blip-itm-large-flickr'],
    ],
    widths=[3.6, 11.2])
caption('表 8：HF 检查点速览')
para('注意：BLIP-2 是独立模块（models/blip_2/），采用 Q-Former 桥接冻结视觉塔与 LLM，不在本文档范围。')

h2('一分钟结论')
make_table(
    ['维度', '要点'],
    [
        ['架构', 'ViT 视觉塔（12L/768/12H）+ 文本 Transformer（12L/768/8H，is_decoder 可切换）'],
        ['MED', '同一套文本参数 = 单模态编码 / 图文编码（+跨注意力）/ 图文解码'],
        ['三目标', 'ITC 图文对比（512 维）· ITM 图文匹配（256 维融合头）· ITG 图像引导生成（LM）'],
        ['数据', 'CapFilt：Captioner 生成 + Filter 过滤，引导高质量训练集'],
        ['池化', '视觉取 [CLS] + post_layernorm'],
        ['代码入口', 'BlipModel（检索/表示）· BlipForConditionalGeneration（字幕）· BlipForQuestionAnswering（VQA）· BlipForImageTextRetrieval（检索）'],
    ],
    widths=[3.0, 11.8])
caption('表 9：一分钟结论')

# ================= 9 来源 =================
h1('9 来源')
refs = [
    '[1] 论文：BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation，arXiv:2201.12086（Li et al., 2022, ICML）',
    '[2] HF 模型卡：Salesforce/blip-vqa-base、Salesforce/blip-image-captioning-base、Salesforce/blip-itm-base-coco 等',
    '[3] transformers 源码（本机 editable）：src/transformers/models/blip/、src/transformers/models/blip_2/（5.23.0.dev0）',
]
for r in refs:
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(0)
    run = p.add_run(r)
    run.font.size = Pt(10.5)
    run.font.name = 'Times New Roman'
    run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
para('待确认/边界：训练数据精确构成与总规模（约 1.29 亿）以论文为准（本机未复现训练流程）；'
     'ITM 头与检索头在 BlipModel / BlipForImageTextRetrieval 中的具体参数共享细节以源码最新版为准；'
     'BLIP-2 为独立模块，未在本文档展开。', indent=False)

doc.save(OUT)
print('saved:', OUT)
