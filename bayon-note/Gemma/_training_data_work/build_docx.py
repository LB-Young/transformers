# -*- coding: utf-8 -*-
"""生成《Gemma 训练数据演进详解.docx》——无模板创建，python-docx。"""
import os
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

OUT = r"E:\transformers\bayon-note\Gemma\Gemma训练数据演进详解.docx"
WORK = r"E:\transformers\bayon-note\Gemma\_training_data_work"
FIG1 = os.path.join(WORK, "fig1_token_budget.png")
FIG2 = os.path.join(WORK, "fig2_pipeline_evolution.png")

BLACK = RGBColor(0x25, 0x25, 0x25)


def set_font(run, cn="宋体", en="Times New Roman", size=12, bold=False, color=BLACK, italic=False):
    run.font.name = en
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:ascii"), en)
    rFonts.set(qn("w:hAnsi"), en)
    rFonts.set(qn("w:eastAsia"), cn)


def para(doc, text, style=None, align=None, first_indent=True, space_before=None,
         space_after=None, line=1.5, size=None, cn=None, bold=False, keep_next=False):
    p = doc.add_paragraph(style=style)
    if align is not None:
        p.alignment = align
    pf = p.paragraph_format
    if first_indent and style in (None, "Normal"):
        pf.first_line_indent = Cm(0.85)
    if line:
        pf.line_spacing = line
    if space_before is not None:
        pf.space_before = Pt(space_before)
    if space_after is not None:
        pf.space_after = Pt(space_after)
    if keep_next:
        pf.keep_with_next = True
    r = p.add_run(text)
    set_font(r, cn=cn or "宋体", size=size or 12, bold=bold)
    return p


def heading(doc, text, level):
    conf = {1: (16, 14, 6, "Heading 1"), 2: (14, 11, 5, "Heading 2"), 3: (12, 9, 4, "Heading 3")}
    size, sb, sa, st = conf[level]
    p = doc.add_paragraph(style=st)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf = p.paragraph_format
    pf.space_before = Pt(sb)
    pf.space_after = Pt(sa)
    pf.line_spacing = 1.5
    pf.keep_with_next = True
    r = p.add_run(text)
    set_font(r, cn="黑体", size=size, bold=True)
    return p


def caption(doc, text, before=4, after=18):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = 1.0
    pf.keep_with_next = False
    r = p.add_run(text)
    set_font(r, cn="宋体", size=10.5, bold=False)
    return p


def src_note(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf = p.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(18)
    pf.line_spacing = 1.5
    r = p.add_run(text)
    set_font(r, cn="宋体", size=10.5, italic=False)
    return p


def add_image(doc, path, width_cm, before=6, after=0):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = 1.0
    run = p.add_run()
    run.add_picture(path, width=Cm(width_cm))
    return p


def _set_cell_shading(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = tcPr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tcPr.append(shd)
    shd.set(qn("w:fill"), fill)


def _set_cell_vertical(cell):
    tcPr = cell._tc.get_or_add_tcPr()
    vAlign = OxmlElement("w:vAlign")
    vAlign.set(qn("w:val"), "center")
    tcPr.append(vAlign)


def _row_cant_split(row):
    trPr = row._tr.get_or_add_trPr()
    cs = OxmlElement("w:cantSplit")
    trPr.append(cs)


def _row_tbl_header(row):
    trPr = row._tr.get_or_add_trPr()
    th = OxmlElement("w:tblHeader")
    th.set(qn("w:val"), "true")
    trPr.append(th)


def add_table(doc, headers, rows, widths_cm, aligns, header_aligns=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    # 列宽
    for i, w in enumerate(widths_cm):
        for cell in table.columns[i].cells:
            cell.width = Cm(w)
    # 表头
    for i, h in enumerate(headers):
        cell = table.cell(0, i)
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pf = p.paragraph_format
        pf.first_line_indent = Cm(0)
        pf.space_before = Pt(2)
        pf.space_after = Pt(2)
        pf.line_spacing = 1.0
        r = p.add_run(h)
        set_font(r, cn="宋体", size=10.5, bold=True)
        _set_cell_shading(cell, "D9D9D9")
        _set_cell_vertical(cell)
    _row_tbl_header(table.rows[0])
    # 数据行
    for ri, row in enumerate(rows):
        tr = table.rows[ri + 1]
        _row_cant_split(tr)
        for ci, val in enumerate(row):
            cell = tr.cells[ci]
            cell.text = ""
            p = cell.paragraphs[0]
            a = aligns[ci] if isinstance(aligns, list) else aligns
            p.alignment = a
            pf = p.paragraph_format
            pf.first_line_indent = Cm(0)
            pf.space_before = Pt(2)
            pf.space_after = Pt(2)
            pf.line_spacing = 1.0
            r = p.add_run(val)
            set_font(r, cn="宋体", size=10.5, bold=False)
            _set_cell_vertical(cell)
    return table


doc = Document()

# 页面设置 A4 边距 2.5cm
sec = doc.sections[0]
sec.page_width = Cm(21.0)
sec.page_height = Cm(29.7)
sec.top_margin = Cm(2.5)
sec.bottom_margin = Cm(2.5)
sec.left_margin = Cm(2.5)
sec.right_margin = Cm(2.5)

# ---- 标题 ----
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(18)
p.paragraph_format.line_spacing = 1.5
r = p.add_run("Gemma 训练数据演进详解")
set_font(r, cn="黑体", size=18, bold=True)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(12)
p.paragraph_format.line_spacing = 1.5
r = p.add_run("——预训练、指令微调与强化学习数据的四代演变")
set_font(r, cn="楷体", size=12, bold=False)

# ---- 1. 概述 ----
heading(doc, "1. 概述与演进主线", 1)
para(doc, "Gemma 系列在四代技术报告中呈现出清晰的数据演进路线：预训练数据从英文单模态文本扩展至多语言与图像、音频多模态，token 预算逐代提升；指令微调（SFT）数据从人工与合成混合转向以大型教师模型蒸馏为主；强化学习对齐从单一人类偏好对走向多奖励信号融合，并在 Gemma 4 引入思维链（thinking）数据。")
para(doc, "四代模型共享同一条数据治理主线：全部执行有害内容过滤、个人信息与敏感数据剔除、评测集去污染、复读风险控制，并持续纳入上下文归因、审慎表达与拒答子集以提升事实性。")
para(doc, "本报告分别以 Gemma 1 至 Gemma 4 四份技术报告为依据（详见第 7 节参考文献），按预训练、指令微调、强化学习三个阶段逐代梳理数据构成与改进，并给出可复用的对比表与图。")

# ---- 2. 预训练 ----
heading(doc, "2. 预训练数据演进", 1)

heading(doc, "2.1 Gemma 1：英文文本基本盘与过滤管道", 2)
para(doc, "Gemma 1 技术报告确立了后续三代沿用的预训练数据框架。2B 与 7B 模型分别使用约 3 万亿与 6 万亿 token 的以英文为主的数据，涵盖网页文档、数学与代码，明确不针对多模态与多语言任务优化。")
para(doc, "数据混合通过 2B 与 7B 模型上的系列消融实验确定，并借鉴 Gemini 采用分阶段训练，在训练后期提高相关高质量数据的权重。")
para(doc, "分词采用 Gemini 分词器的子集（SentencePiece，词表 256k，拆分数字、保留空白、字节级编码）。过滤管道包含启发式规则与模型分类器，用于去除有害或低质量内容、剔除个人信息与敏感数据，并将全部评测集从预训练混合数据中剔除，同时执行针对性的污染分析，通过最小化敏感输出传播降低复读风险。")

heading(doc, "2.2 Gemma 2：规模翻倍与知识蒸馏", 2)
para(doc, "Gemma 2 技术报告将预训练规模大幅提升：27B、9B、2B 模型分别使用 13 万亿、8 万亿与 2 万亿 token，数据来源扩展为网页文档、代码与科学文章，仍以英文为主、非多模态；分词器与过滤技术完全沿用 Gemma 1。")
para(doc, "本代预训练的关键创新是知识蒸馏：以更大模型为教师，将教师对每个 token 的条件概率分布蒸馏给学生模型，最小化教师与学生分布的负对数似然。该做法与 Gemini 1.5 一致。")

heading(doc, "2.3 Gemma 3：多语言、图文混合与质量重加权", 2)
para(doc, "Gemma 3 技术报告首次将图像与文本混合数据引入预训练，token 预算因此上调。27B、12B、4B、1B 分别使用 14 万亿、12 万亿、4 万亿与 2 万亿 token。")
para(doc, "多语言数据量显著增加，同时加入单语与平行语料，并借鉴 Chung 等人的方法处理语种不平衡；分词器更换为与 Gemini 2.0 一致的分词器（词表 262k，对非英语更均衡）。")
para(doc, "过滤管道在既有基础上新增质量重加权步骤（受 Sachdeva 等 2024 年工作启发），降低低质量数据占比。蒸馏细化为按教师概率加权采样每个 token 的 256 个 logits，学生仅在这些采样位置上通过交叉熵学习教师分布，未采样位置的教师概率置零并重归一化。")

heading(doc, "2.4 Gemma 4：多模态扩展", 2)
para(doc, "Gemma 4 技术报告将预训练数据扩展至多模态。数据为大规模、跨域、跨模态集合，包含网页文档、代码、图像与音频（E2B、E4B 与 12B 含音频），截止日期为 2025 年 1 月；分词器与 Gemini 团队 2025 年模型一致（262k）。")
para(doc, "过滤执行基准去污、降低有害或不安全输出与复读风险。需要明确的事实边界：论文未披露各尺寸模型的 token 预算。")

heading(doc, "2.5 预训练数据演进小结", 2)
para(doc, "横向对比显示：token 预算随代际与模型规模整体提升；数据构成从英文单模态扩展至多语言与图文、音视频多模态；过滤管道从启发式加模型分类器演进至质量重加权；Gemma 2 起引入预训练蒸馏，Gemma 3 起采用采样蒸馏。各代对比见表 1 与图 1。")

caption(doc, "表 1　预训练数据演进对比")
add_table(doc,
    ["对比维度", "Gemma 1", "Gemma 2", "Gemma 3", "Gemma 4"],
    [
        ["token 预算", "2B=3T，7B=6T", "2B=2T，9B=8T，27B=13T", "1B=2T，4B=4T，12B=12T，27B=14T", "未披露"],
        ["数据构成", "英文网页、数学、代码", "网页、代码、科学文章（英文为主）", "图文混合、多语言（单语+平行）", "网页、代码、图像、音频"],
        ["分词器", "Gemini 子集，256k", "同左，256k", "Gemini 2.0，262k", "Gemini Team 2025，262k"],
        ["过滤技术", "启发式+模型分类器", "同 Gemma 1", "同前+质量重加权", "去污、安全、复读控制"],
        ["特殊技术", "分阶段训练提升后期数据质量", "预训练知识蒸馏", "每 token 256-logit 采样蒸馏", "类似 Gemma 3"],
    ],
    [2.6, 3.3, 3.4, 3.4, 3.3],
    [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT,
     WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT])
src_note(doc, "数据来源：依据 Gemma 1—4 技术报告（arXiv 2403.08295、2408.00118、2503.19786、2607.02770）整理；T 为万亿 token 的简称，Gemma 4 token 预算论文未披露。")

add_image(doc, FIG1, 12.6)
caption(doc, "图 1　各代 Gemma 模型预训练 token 预算对比")
src_note(doc, "数据来源：依据 Gemma 1—4 技术报告整理；Gemma 4 预算未披露，以斜纹柱示意。")

# ---- 3. SFT ----
heading(doc, "3. 指令微调（SFT）数据演进", 1)

heading(doc, "3.1 Gemma 1：人工与合成混合、逐对评估选数", 2)
para(doc, "Gemma 1 的 SFT 数据为纯文本、纯英文的人工与合成 prompt-response 对混合。数据混合通过基于语言模型的逐对评估确定：在留出提示集上由测试模型与基线模型分别生成回答，随机打乱后请更大、能力更强的模型表达偏好，提示集覆盖指令遵循、事实性、创造力与安全等维度，评审采用思维链、评分规则与宪法等策略。")
para(doc, "合成数据执行多阶段过滤，剔除含个人信息、不安全或有害输出、错误自我识别与重复的样例。借鉴 Gemini，纳入促进上下文归因、审慎表达与拒答的数据子集，在不过度损伤其他指标的前提下提升事实性、降低幻觉。")

heading(doc, "3.2 Gemma 2：行为克隆与外部提示数据", 2)
para(doc, "Gemma 2 在 Gemma 1.1 的 post-training 数据基础上扩展，混合内部数据与外部公开数据，明确采用 LMSYS-chat-1M 的提示部分而不使用其回答。")
para(doc, "SFT 采用行为克隆，在合成与真实提示上学习、回答主要来自更大教师模型的合成输出，并执行从教师到学生分布的蒸馏；所有数据经过与预训练一致的过滤阶段。")

heading(doc, "3.3 Gemma 3：从大型教师蒸馏", 2)
para(doc, "Gemma 3 的 post-training 采用相对此前配方改进的方法，从大型指令微调教师进行知识蒸馏，并与强化学习阶段衔接；论文未披露 SFT 数据的具体规模，数据过滤延续既有管道。")

heading(doc, "3.4 Gemma 4：继承配方与思维链数据", 2)
para(doc, "Gemma 4 的 post-training 与 Gemma 3 类似，显著差异是新增思维模式：模型可在作答前输出推理轨迹。数据过滤继续执行个人信息、有害输出、错误自我识别与重复样例剔除，并保留上下文归因、审慎与拒答子集。")

heading(doc, "3.5 SFT 数据演进小结", 2)
para(doc, "合成数据占比与教师蒸馏权重逐代上升；外部公开提示数据从 Gemma 2 起进入管道；Gemma 4 引入思维链数据。各代对比见表 2。")

caption(doc, "表 2　指令微调（SFT）数据演进对比")
add_table(doc,
    ["对比维度", "Gemma 1", "Gemma 2", "Gemma 3", "Gemma 4"],
    [
        ["数据构成", "纯文本纯英文、人工+合成对", "扩展 Gemma 1.1 数据（内部+公开）", "从大型 IT 教师蒸馏", "类似 Gemma 3 + 思维链数据"],
        ["外部公开数据", "无", "LMSYS-chat-1M（仅提示）", "未披露", "未披露"],
        ["合成策略", "人工+合成对，LM 逐对评估选混合", "行为克隆（教师合成回答）+学生分布蒸馏", "改进版知识蒸馏", "同左"],
        ["数据过滤", "多阶段过滤合成数据", "同左", "同左", "同左"],
        ["事实性子集", "归因、审慎、拒答子集", "同左", "同左", "同左"],
    ],
    [2.6, 3.3, 3.4, 3.4, 3.3],
    [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT,
     WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT])
src_note(doc, "数据来源：依据 Gemma 1—4 技术报告整理；Gemma 3、Gemma 4 的 SFT 数据具体体量论文未披露。")

# ---- 4. RL ----
heading(doc, "4. 强化学习对齐数据演进", 1)

heading(doc, "4.1 Gemma 1：经典 RLHF", 2)
para(doc, "Gemma 1 收集人类评分者的偏好对，在 Bradley-Terry 模型下训练奖励函数（与 Gemini 一致），policy 以一组高质量提示为基础，使用新颖的强化学习算法优化该奖励函数。")
para(doc, "为调优超参数并缓解奖励黑客，依赖高容量模型作为自动评分者，与基线模型执行逐对比较。")

heading(doc, "4.2 Gemma 2：更大、更对话化的奖励模型", 2)
para(doc, "Gemma 2 的 RLHF 算法与 Gemma 1.1 类似，但换用新的奖励模型：规模比 policy 大一个数量级，且更面向对话能力、尤其是多轮对话。各训练阶段获得的模型取平均（模型合并），以提升整体性能。")

heading(doc, "4.3 Gemma 3：多奖励信号融合", 2)
para(doc, "Gemma 3 的强化学习阶段基于 BOND、WARM、WARP 的改进版本，并采用多种奖励函数：基于人类反馈数据训练的权重平均奖励模型、代码执行反馈，以及求解数学问题的真实答案奖励，用以提升有用性、数学、代码、推理、指令遵循与多语言能力，同时最小化有害性。")

heading(doc, "4.4 Gemma 4：继承与推理轨迹", 2)
para(doc, "Gemma 4 的对齐数据与强化学习沿用 Gemma 3 方案，差异在于思维模式引入推理轨迹数据，模型在作答前先生成推理痕迹。")

heading(doc, "4.5 RL 数据演进小结", 2)
para(doc, "偏好信号从单一人类偏好对走向多信号融合；奖励模型规模增大且更对话化；算法从新颖强化学习算法演进至 BOND、WARM、WARP 的改进版本。各代对比见表 3。")

caption(doc, "表 3　强化学习对齐数据演进对比")
add_table(doc,
    ["对比维度", "Gemma 1", "Gemma 2", "Gemma 3", "Gemma 4"],
    [
        ["偏好/奖励数据", "人工英文偏好对", "英文偏好对（奖励模型更大）", "人类反馈数据（权重平均奖励模型）", "类似 Gemma 3"],
        ["奖励模型", "Bradley-Terry", "比 policy 大一个数量级、面向多轮", "权重平均奖励模型", "类似 Gemma 3"],
        ["额外奖励信号", "无", "无", "代码执行反馈、数学真实答案", "同左"],
        ["算法", "新颖强化学习算法", "类似 Gemma 1.1", "BOND、WARM、WARP 改进版", "类似 Gemma 3"],
        ["特色", "高容量模型自动评分", "模型平均（合并）", "多奖励融合", "思维链推理轨迹"],
    ],
    [2.6, 3.3, 3.4, 3.4, 3.3],
    [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT,
     WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT])
src_note(doc, "数据来源：依据 Gemma 1—4 技术报告整理。")

# ---- 5. 共同主线 ----
heading(doc, "5. 贯穿各代的共同数据处理主线", 1)

heading(doc, "5.1 统一过滤管道", 2)
para(doc, "四代模型全部执行统一的预训练数据过滤：去除有害或不安全输出、剔除个人信息与敏感数据、评测集去污染，并通过最小化敏感输出传播控制复读风险。该管道是 Gemini 技术路线的延续，也是四代数据质量一致性的基础。")

heading(doc, "5.2 事实性数据子集", 2)
para(doc, "促进上下文归因、审慎表达与拒答的数据子集自 Gemma 1 起沿用至 Gemma 4，被各代报告反复验证可在不明显损伤其他指标的前提下提升事实性指标、降低幻觉。")

heading(doc, "5.3 格式化控制 token 演进", 2)
para(doc, "指令微调数据的格式化约定随代际调整：Gemma 1 与 Gemma 2 使用 start_of_turn、end_of_turn 及 user、model 控制 token；Gemma 3 起预训练模型输出 eos、指令微调模型输出 end_of_turn，文本需显式添加 BOS；Gemma 4 指令微调模型输出 turn。对应关系见表 4。")

caption(doc, "表 4　格式化控制 token 演进")
add_table(doc,
    ["模型", "对话控制 token", "生成结束标记", "附加要求"],
    [
        ["Gemma 1", "user、model、start_of_turn、end_of_turn", "eos", "沿用 Gemini 格式化风格"],
        ["Gemma 2", "同 Gemma 1 控制 token，schema 不同", "eos；多轮续接时移除", "—"],
        ["Gemma 3", "start_of_turn、end_of_turn", "IT 输出 end_of_turn；PT 输出 eos", "文本需显式 [BOS]"],
        ["Gemma 4", "同 Gemma 3 体系", "IT 输出 turn", "思维链、函数调用显式激活"],
    ],
    [2.5, 5.4, 4.1, 4.0],
    [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT,
     WD_ALIGN_PARAGRAPH.LEFT])
src_note(doc, "数据来源：依据 Gemma 1—4 技术报告格式化章节整理。")

heading(doc, "5.4 数据治理管道演进全景", 2)
para(doc, "将预训练、SFT 与强化学习三个阶段串联，可以完整看到四代数据管道的演进路径，如图 2 所示。")
add_image(doc, FIG2, 14.4)
caption(doc, "图 2　Gemma 1 至 Gemma 4 三阶段训练数据管道演进")
src_note(doc, "数据来源：依据 Gemma 1—4 技术报告整理；Gemma 4 token 预算与 SFT、RL 数据体量论文未披露。")

# ---- 6. 总结 ----
heading(doc, "6. 总结与论文写作建议", 1)
para(doc, "四代演进可归纳为五条主线：规模扩张（token 预算逐代提升）、构成扩展（英文单语到多语言多模态）、合成策略深化（人工混合到教师蒸馏主导）、对齐信号多元（偏好对到多奖励融合）、治理贯穿（过滤与事实性子集全程沿用）。")
para(doc, "论文写作建议：其一，按数据规模、数据构成、合成策略、对齐信号、数据治理五个维度组织演进叙事，比按模型逐代平铺更有辨识度；其二，Gemma 4 的 token 预算及三代模型的 SFT、RL 数据具体体量在论文中未披露，引用时应注明论文未披露而非自行估计；其三，Gemma 3 的 256-logit 采样蒸馏与质量重加权是预训练数据处理的关键技术节点，建议作为演进叙事的支点。")

# ---- 7. 参考文献 ----
heading(doc, "7. 参考文献", 1)
refs = [
    "Gemma Team. Gemma: Open Models Based on Gemini Research and Technology. arXiv:2403.08295, 2024.",
    "Gemma Team. Gemma 2: Improving Open Language Models at a Practical Size. arXiv:2408.00118, 2024.",
    "Gemma Team. Gemma 3 Technical Report. arXiv:2503.19786, 2025.",
    "Gemma Team. Gemma 4 Technical Report. arXiv:2607.02770, 2026.",
]
for i, ref in enumerate(refs):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.first_line_indent = Cm(0)
    pf.space_before = Pt(0)
    pf.space_after = Pt(3)
    pf.line_spacing = 1.5
    r = p.add_run(ref)
    set_font(r, cn="宋体", size=12, bold=False)

doc.save(OUT)
print("saved:", OUT)
