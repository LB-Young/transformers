# -*- coding: utf-8 -*-
"""Gemma 训练算法演进详解 · docx 生成（无模板创建）"""
import docx
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

OUT = r'E:\transformers\bayon-note\Gemma\Gemma训练算法演进详解.docx'
FIG1 = r'E:\transformers\bayon-note\Gemma\_training_algo_work\fig1_algo_matrix.png'
FIG2 = r'E:\transformers\bayon-note\Gemma\_training_algo_work\fig2_distill_evolution.png'

NAVY = RGBColor(0x1A, 0x2B, 0x4A)
MID = RGBColor(0x52, 0x64, 0x7D)
DARK = RGBColor(0x25, 0x25, 0x25)
GREY = RGBColor(0x76, 0x76, 0x76)

doc = Document()

# ---------- 页面 ----------
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.top_margin = sec.bottom_margin = Cm(2.5)
sec.left_margin = sec.right_margin = Cm(2.5)

# ---------- 默认样式 ----------
st = doc.styles['Normal']
st.font.name = 'Times New Roman'
st.font.size = Pt(12)
st.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')

def set_run(r, font='宋体', size=12, bold=False, color=DARK, italic=False):
    r.font.name = 'Times New Roman'
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = color
    r._element.rPr.rFonts.set(qn('w:eastAsia'), font)

def para(text='', size=12, bold=False, font='宋体', color=DARK, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
         indent=0.85, before=0, after=0, line=1.5, keep_next=False):
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    if indent:
        pf.first_line_indent = Cm(indent)
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    if line:
        pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
        pf.line_spacing = line
    if keep_next:
        pf.keep_with_next = True
    if text:
        r = p.add_run(text)
        set_run(r, font=font, size=size, bold=bold, color=color)
    return p

def heading(text, level=1):
    sizes = {1: 16, 2: 14, 3: 12}
    befores = {1: 14, 2: 11, 3: 9}
    afters = {1: 6, 2: 5, 3: 4}
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf = p.paragraph_format
    pf.space_before = Pt(befores[level])
    pf.space_after = Pt(afters[level])
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = 1.5
    pf.keep_with_next = True
    r = p.add_run(text)
    set_run(r, font='黑体', size=sizes[level], bold=True, color=NAVY if level == 1 else DARK)
    return p

def set_cell_bg(cell, color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:fill'), color)
    tcPr.append(shd)

def set_cell_vertical(cell):
    tcPr = cell._tc.get_or_add_tcPr()
    vAlign = OxmlElement('w:vAlign')
    vAlign.set(qn('w:val'), 'center')
    tcPr.append(vAlign)

def set_row_header(row):
    trPr = row._tr.get_or_add_trPr()
    tblHeader = OxmlElement('w:tblHeader')
    trPr.append(tblHeader)

def set_row_cantsplit(row):
    trPr = row._tr.get_or_add_trPr()
    cantSplit = OxmlElement('w:cantSplit')
    trPr.append(cantSplit)

def cell_text(cell, text, size=10.5, bold=False, color=DARK, align=WD_ALIGN_PARAGRAPH.LEFT):
    cell.text = ''
    p = cell.paragraphs[0]
    p.alignment = align
    pf = p.paragraph_format
    pf.first_line_indent = Cm(0)
    pf.left_indent = Cm(0)
    pf.right_indent = Cm(0)
    pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
    pf.space_before = Pt(1)
    pf.space_after = Pt(1)
    r = p.add_run(text)
    set_run(r, font='宋体', size=size, bold=bold, color=color)

def add_table(headers, rows, widths):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = 'Table Grid'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    # 宽度
    for j, w in enumerate(widths):
        for row in t.rows:
            row.cells[j].width = Cm(w)
    # 表头
    hdr = t.rows[0]
    set_row_header(hdr)
    set_row_cantsplit(hdr)
    for j, h in enumerate(headers):
        cell = hdr.cells[j]
        set_cell_bg(cell, 'D9D9D9')
        set_cell_vertical(cell)
        cell_text(cell, h, bold=True, color=NAVY, align=WD_ALIGN_PARAGRAPH.CENTER)
    # 内容
    for i, rowdata in enumerate(rows):
        row = t.rows[1 + i]
        set_row_cantsplit(row)
        for j, val in enumerate(rowdata):
            cell = row.cells[j]
            set_cell_vertical(cell)
            al = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            cell_text(cell, val, align=al)
    return t

def add_fig(path, caption):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run()
    run.add_picture(path, width=Cm(13.5))
    cap = para('图 ' + caption, size=10.5, align=WD_ALIGN_PARAGRAPH.CENTER,
               indent=0, before=4, after=18)
    for r in cap.runs:
        r.font.bold = False
    return cap

def add_table_caption(caption):
    para('表 ' + caption, size=10.5, align=WD_ALIGN_PARAGRAPH.CENTER,
         indent=0, before=6, after=4)

def add_source(text):
    para('数据来源：' + text, size=9, color=GREY, indent=0, before=4, after=12,
         align=WD_ALIGN_PARAGRAPH.LEFT)

# ================= 封面标题 =================
para('Gemma 训练算法演进详解', size=18, bold=True, font='黑体', color=NAVY,
     align=WD_ALIGN_PARAGRAPH.CENTER, indent=0, before=6, after=6)
para('预训练、指令微调与强化学习算法的四代演变', size=12, font='楷体', color=MID,
     align=WD_ALIGN_PARAGRAPH.CENTER, indent=0, before=0, after=18)

# ================= 1 概述 =================
heading('1　概述与演进主线', 1)
para('Gemma 系列四代技术报告记录了一条完整的训练算法演进路线：预训练从标准 next-token 预测起步，'
     '在 Gemma 2 首次引入知识蒸馏，在 Gemma 3 细化为每 token 256-logit 采样蒸馏并叠加量化感知训练（QAT）；'
     '指令微调从以数据选择为核心的经典 SFT，走向行为克隆、on-policy 蒸馏与大型教师蒸馏；'
     '强化学习对齐从经典 RLHF 出发，经更大更对话向的奖励模型与模型平均，演进为 BOND、WARM、WARP 改进版下的多奖励融合，'
     '并在 Gemma 4 引入思维链（thinking mode）推理轨迹训练。')
para('与训练数据演进文档相比，本文聚焦算法本身：训练目标、损失函数、蒸馏对象、奖励信号与工程配套（量化、推测解码、训练基础设施）。'
     '所有事实均取自四代技术报告原文；论文未披露的部分（如 Gemma 1 的强化学习算法名称、各代优化器与学习率超参）在文中如实标注。')

# ================= 2 预训练算法 =================
heading('2　预训练算法演进', 1)

heading('2.1　Gemma 1：标准训练与分阶段数据加权', 2)
para('Gemma 1 的预训练不包含任何蒸馏或特殊学习目标，使用标准的 next-token 预测。'
     '借鉴 Gemini 的预训练经验，团队采用分阶段训练（staged training）：在训练后期提高高质量数据（精选的数学、网页与代码数据）的权重，'
     '以改善最终模型质量。预训练在 TPUv4、TPUv5e 与 TPUv5p 集群上完成，使用 256-way 数据复制、'
     'ZeRO-3 优化器状态分片与 GSPMD、Pathways 编排。')

heading('2.2　Gemma 2：知识蒸馏首次引入', 2)
para('Gemma 2 首次把知识蒸馏引入预训练。训练目标由标准 next-token 预测损失与蒸馏损失共同构成，'
     '蒸馏项最小化教师分布与学生分布之间的交叉熵：')
para('min Σx −P_T(x | x_c) · log P_S(x | x_c)', size=12, align=WD_ALIGN_PARAGRAPH.CENTER,
     indent=0, before=6, after=6)
para('其中 P_T 为教师（较大模型）给出的目标 token 概率，P_S 为学生分布，x_c 为上下文。'
     'Gemini 1.5 使用相同方法。蒸馏使学生模型在保持较低推理成本的同时获得接近大模型的分布知识。')

heading('2.3　Gemma 3：采样蒸馏与量化感知训练', 2)
para('Gemma 3 把蒸馏细化为“每 token 采样 256 logits”：按教师概率加权采样每个位置的 256 个 logits，'
     '未采样位置的概率置零并重归一化，学生只在此采样集上计算交叉熵。该设计大幅降低蒸馏的内存与通信开销，'
     '使蒸馏训练可以规模化。论文还给出小、大教师对比消融：在较短训练视窗内，用小教师蒸馏更优；'
     '训练时间足够长后，大教师的优势反转。Gemma 3 首次引入 QAT：以约 5,000 步训练量化权重，'
     '以未量化 checkpoint 的概率为目标，使模型在预训练与后训练分布下都支持 int4 与 SFP8 推理。')

heading('2.4　Gemma 4：配方继承与工程升级', 2)
para('Gemma 4 的预训练蒸馏配方与 Gemma 3 保持一致，论文未披露新的目标函数设计，'
     '新增的工程层升级集中在量化与解码（见第 5 章）。')

heading('2.5　预训练算法小结', 2)
para('表 1 汇总四代模型在预训练算法上的差异。图 1 给出三阶段训练算法在四代中的完整矩阵。', after=6)
add_table_caption('1　预训练算法演进对比')
add_table(
    ['维度', 'Gemma 1', 'Gemma 2', 'Gemma 3', 'Gemma 4'],
    [
        ['训练目标', '标准 next-token', 'next-token + 全词汇分布蒸馏', 'next-token + 256-logit 采样蒸馏', '同 Gemma 3'],
        ['蒸馏目标', '无', 'min Σx −P_T·log P_S', '按概率加权采样 256 logits', '同 Gemma 3'],
        ['数据加权', '分阶段训练', '分阶段训练', '分阶段训练 + 质量重加权', '同 Gemma 3'],
        ['量化训练', '无（FP 训练）', '无', 'QAT 约 5,000 步，int4 / SFP8', 'QAT 扩展（int2/4/8，含编码器）'],
        ['关键参考', '—', 'Gemini 1.5 同法', 'Sachdeva 2024（质量重加权）', '—'],
    ],
    [2.6, 3.2, 3.5, 3.6, 3.1]
)
add_source('根据 Gemma 1—4 技术报告整理；Gemma 4 未披露蒸馏超参细节，以“同 Gemma 3”标注。')
add_fig(FIG1, '1　三阶段训练算法演进矩阵')
add_source('根据 Gemma 1—4 技术报告整理。')

# ================= 3 SFT =================
heading('3　指令微调算法演进', 1)

heading('3.1　Gemma 1：数据驱动的 SFT', 2)
para('Gemma 1 的指令微调以数据为中心：在人工与合成 prompt-response 对上执行标准监督微调，'
     '并用基于语言模型的逐对评估（以大模型作为 judge，结合思维链、评分规则与安全准则）'
     '从候选数据混合中选出最优配比。此阶段算法本身仍是经典 SFT，创新集中在数据选择机制。')

heading('3.2　Gemma 2：行为克隆与 on-policy 蒸馏', 2)
para('Gemma 2 的 post-training 在 SFT 环节加入两类机制：一是行为克隆，使用内部与公开提示'
     '（如 LMSYS-chat-1M 提示）配合教师合成的响应进行训练；二是学生分布蒸馏（on-policy distillation），'
     '让学生分布匹配教师输出分布，对应 Agarwal 2024 与 Gu et al. 2024 提出的方法。'
     'SFT 数据在 Gemma 1.1 基础上扩展。')

heading('3.3　Gemma 3：大型教师蒸馏', 2)
para('Gemma 3 的 instruction tuning 改为从大型指令微调（IT）教师改进版蒸馏，'
     '蒸馏对象从预训练模型升级为指令微调后的教师，并与后续强化学习阶段衔接。'
     '论文未披露 SFT 数据的具体规模。')

heading('3.4　Gemma 4：继承与思维链数据', 2)
para('Gemma 4 的 SFT 配方与 Gemma 3 一致，新增 thinking mode 数据：'
     '模型在给出最终答案前先输出推理轨迹，该训练数据贯穿指令微调与后续对齐阶段。')

heading('3.5　SFT 算法小结', 2)
para('表 2 汇总指令微调算法在四代中的演进。', after=6)
add_table_caption('2　指令微调算法演进对比')
add_table(
    ['维度', 'Gemma 1', 'Gemma 2', 'Gemma 3', 'Gemma 4'],
    [
        ['核心算法', '标准 SFT + LM 逐对评估选数', '行为克隆 + on-policy 蒸馏', '大型 IT 教师蒸馏', '同 Gemma 3 + thinking 数据'],
        ['数据选择', 'LM judge 逐对比较', '扩展 1.1 数据 + 外部提示', '未披露具体规模', '同 Gemma 3'],
        ['蒸馏角色', '无', '学生分布蒸馏（Agarwal 2024；Gu 2024）', '教师为指令微调模型', '同 Gemma 3'],
        ['阶段衔接', '—', '—', '与 RL 阶段衔接', '推理轨迹贯穿 SFT / RL'],
    ],
    [2.6, 3.5, 3.6, 3.3, 3.4]
)
add_source('根据 Gemma 1—4 技术报告整理；Gemma 3、Gemma 4 的 SFT 数据规模论文未披露。')

# ================= 4 RL =================
heading('4　强化学习对齐算法演进', 1)

heading('4.1　Gemma 1：经典 RLHF', 2)
para('Gemma 1 采用经典 RLHF 管线：收集人工英文偏好对，训练 Bradley-Terry 奖励模型（与 Gemini 一致），'
     '再使用一个论文未命名的新颖强化学习算法优化 policy。为防止奖励黑客（reward hacking），'
     '团队用高容量模型作为自动评分器对回答评级。论文未披露该强化学习算法的名称与实现细节。')

heading('4.2　Gemma 2：更大更对话向的奖励模型与模型平均', 2)
para('Gemma 2 的 RLHF 算法与 Gemma 1.1 保持一致，但奖励模型显著升级：'
     '比 policy 大一个数量级，且更面向多轮对话。此外，post-training 各阶段结束后对模型权重取平均（模型平均），'
     '该做法与 WARP（Ramé 2024）的模型合并思想一致，用于提升最终性能与稳定性。')

heading('4.3　Gemma 3：BOND、WARM、WARP 与多奖励', 2)
para('Gemma 3 的强化学习微调基于 BOND（Sessa 2024）、WARM（Ramé 2024b）与 WARP（Ramé 2024a）的改进版本，'
     '并把奖励信号从单一偏好对扩展为多奖励融合：权重平均奖励模型（基于人类反馈数据）、'
     '代码执行反馈（Gehring 2024）与数学真实答案奖励（DeepSeek-AI 2025；Lambert 2024）。'
     '多奖励设计针对有用性、数学、代码、推理、指令遵循与多语言等目标。')

heading('4.4　Gemma 4：继承与推理轨迹', 2)
para('Gemma 4 的对齐方案沿用 Gemma 3 的算法体系，论文未披露新的强化学习算法细节；'
     '核心变化是 thinking mode：模型先输出推理轨迹再给出最终答案，该行为由 SFT 与 RL 阶段的思维链数据共同塑造。')

heading('4.5　RL 算法小结', 2)
para('表 3 汇总强化学习对齐算法在四代中的演进。', after=6)
add_table_caption('3　强化学习对齐算法演进对比')
add_table(
    ['维度', 'Gemma 1', 'Gemma 2', 'Gemma 3', 'Gemma 4'],
    [
        ['算法', '经典 RLHF（新颖 RL 算法，未命名）', '同 Gemma 1.1', 'BOND / WARM / WARP 改进版', '同 Gemma 3'],
        ['奖励模型', 'Bradley-Terry（与 Gemini 一致）', '大一个数量级、更对话向', '权重平均（WARM）', '同 Gemma 3'],
        ['奖励信号', '人类偏好对', '人类偏好对', '人类 + 代码执行 + 数学真实答案', '同 Gemma 3 + 推理轨迹'],
        ['特色机制', '高容量模型自动评分', '阶段模型平均（WARP 思想）', '多奖励融合', 'thinking mode'],
    ],
    [2.6, 3.6, 3.2, 3.5, 3.1]
)
add_source('根据 Gemma 1—4 技术报告整理；Gemma 1 的 RL 算法未命名、Gemma 4 未披露新算法细节。')

# ================= 5 配套 =================
heading('5　蒸馏、量化与工程配套', 1)

heading('5.1　蒸馏目标函数的两次演进', 2)
para('预训练蒸馏在 Gemma 2 与 Gemma 3 之间完成两次关键演进：'
     '目标从“完整词汇表概率的交叉熵”细化为“每 token 256 logits 加权采样集上的交叉熵”，'
     '核心收益是内存与通信开销的大幅下降，使蒸馏训练可规模化。图 2 对比两种蒸馏的流程与目标。')
add_fig(FIG2, '2　预训练知识蒸馏目标演进')
add_source('根据 Gemma 2、Gemma 3 技术报告整理。')

heading('5.2　量化感知训练', 2)
para('QAT 自 Gemma 3 引入（约 5,000 步、以未量化 checkpoint 的概率为目标、支持 int4 与 SFP8 推理），'
     '到 Gemma 4 扩展为两个量化集合：mobile 量化采用 per-channel 低位宽（权重 int2/int4 + 激活 int8），'
     'Q4_0 采用 blockwise 量化并引入 fp16 标量缩放以稳定推理，同时把 QAT 覆盖到图像与音频编码器。'
     '对 150M 图像编码器，W8A8 量化使前向内存占用减半（400MB 降至 200MB）、端上延迟降低 44%；'
     '音频编码器把激活精度降至 8bit、权重按层簇降至 {2,4,8}bit，磁盘占用从 390MB 降至 87MB（减少 78%）。')

heading('5.3　推测解码与训练基础设施', 2)
para('Gemma 4 引入多 token 预测（MTP）drafter 用于推测解码：模型最后一层激活与 token embedding 输入 drafter head，'
     '由 4 层 Transformer 块（模型维度 256 或 1024，3 个 local 与 1 个 global attention 层）交叉注意力到主模型 KV，'
     '省去 MTP prefill 并支持任意草稿长度；E2B 与 E4B 的 drafter 通过 top-k token cluster 投影，'
     '把词汇投影从 262,000 维降至 4,096 维。训练基础设施方面，Gemma 1—3 使用 TPUv4/v5e/v5p，'
     'Gemma 4 使用 TPUv4 与 TPUv6e，并采用 Slice-Granularity Elasticity：局部故障时用更少 slice 继续训练，'
     '把中断恢复时间从数分钟降到数秒。')

para('表 4 汇总工程配套技术在四代中的演进。', after=6)
add_table_caption('4　工程配套技术演进对比')
add_table(
    ['维度', 'Gemma 1', 'Gemma 2', 'Gemma 3', 'Gemma 4'],
    [
        ['蒸馏', '无', '全词汇分布蒸馏', '256-logit 采样蒸馏', '同 Gemma 3'],
        ['量化', '无', '无', 'QAT 约 5,000 步（int4 / SFP8）', 'mobile（int2/4 + int8）与 Q4_0，含编码器'],
        ['推测解码', '无', '无', '无', 'MTP drafter（4 层、KV 交叉注意力、cluster top-k）'],
        ['基础设施', 'TPUv4 / v5e / v5p', '同左', '同左', 'TPUv4 / v6e + Slice-Granularity Elasticity'],
    ],
    [2.6, 3.4, 3.4, 3.6, 3.4]
)
add_source('根据 Gemma 1—4 技术报告整理。')

# ================= 6 总结 =================
heading('6　总结与论文写作建议', 1)
para('综合四代技术报告，训练算法演进可归纳为三条主线：第一，蒸馏从无到有、从全分布到采样集逐步深化，'
     '是预训练算法演进的主干；第二，后训练从“数据混合选择”走向“算法套件”，'
     '行为克隆、on-policy 蒸馏、BOND/WARM/WARP 改进版与多奖励依次进入管道；'
     '第三，工程配套（QAT、MTP drafter、容错训练基础设施）与算法并行成熟，成为模型可部署性的关键支撑。')
para('写作论文时建议注意三点：一是以“蒸馏深化”作为预训练章节的主线，把 Gemma 2 的公式与 Gemma 3 的采样设计作为重点；'
     '二是后训练部分突出“偏好对 → 多奖励 + 思维链”的信号演进；'
     '三是 Gemma 1 的新颖强化学习算法未命名、Gemma 4 的算法细节未披露，引用时应注明口径，避免过度推断。')

# ================= 7 参考文献 =================
heading('7　参考文献', 1)
refs = [
    'Gemma Team. Gemma: Open Models Based on Gemini Research and Technology. arXiv:2403.08295, 2024.',
    'Gemma Team. Gemma 2: Improving Open Language Models at a Practical Size. arXiv:2408.00118, 2024.',
    'Gemma Team. Gemma 3 Technical Report. arXiv:2503.19786, 2025.',
    'Gemma Team. Gemma 4 Technical Report. arXiv:2607.02770, 2026.',
]
for i, ref in enumerate(refs, 1):
    para(ref, size=10.5, align=WD_ALIGN_PARAGRAPH.LEFT, indent=0, before=0, after=2)

doc.save(OUT)
print('saved:', OUT)
