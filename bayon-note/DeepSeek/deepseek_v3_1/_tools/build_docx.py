# -*- coding: utf-8 -*-
"""生成 DeepSeek-V3.1-家族架构详解.docx（A4、编号章节、统一表格、嵌入架构图）"""
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import os

ARCH = r"E:\transformers\bayon-note\DeepSeek\deepseek_v3_1\_arch"
OUT = r"E:\transformers\bayon-note\DeepSeek\deepseek_v3_1\DeepSeek-V3.1-家族架构详解.docx"

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
tr = tp.add_run('DeepSeek-V3.1 家族架构详解')
tr.font.size = Pt(18); tr.font.bold = True; tr.font.name = 'Times New Roman'
tr._element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')

sp = doc.add_paragraph(); sp.alignment = WD_ALIGN_PARAGRAPH.CENTER
sr = sp.add_run('671B-T37B MoE · 混合推理（思考/非思考）· 840B 长上下文续训 · UE8M0 FP8 Scale')
sr.font.size = Pt(12); sr.font.name = 'Times New Roman'
sr._element.rPr.rFonts.set(qn('w:eastAsia'), '楷体')

para('依据：远端官方 config.json（hf-mirror 拉取 deepseek-ai/DeepSeek-V3.1 与 DeepSeek-V3.1-Base，'
     '2026-11-02 逐字段核对，留档于 _tools\\_cfgpull\\）、本地 transformers 源码 '
     'E:\\transformers\\src\\transformers\\models\\deepseek_v3\\（V3.1 无独立模块，与 V3 同一实现），'
     '以及官方发布公告与模型卡（2025-08-19 开源）。核心结论：V3.1 的 config.json 与 V3 逐字段一致，'
     '版本升级全部发生在训练数据、后训练目标、精度格式与对话模板四个层面。所有数字均标注来源，未编造。', indent=True)

# ================= 1 家族全景 =================
h1('1 家族全景：V3.1 / V3.1-Base 配置对照（与 V3 逐字段同构）')
para('关键结论：DeepSeek-V3.1 的 config.json 与 DeepSeek-V3 逐字段一致（两份远端 config 已逐一比对）。'
     'V3.1 不是新架构，而是「V3 架构 + 长上下文续训 + 混合推理后训练 + 精度格式升级」。下表全部字段来自远端官方 config。')

make_table(
    ['字段', 'DeepSeek-V3.1 / V3.1-Base', '与 V3 对比'],
    [
        ['入口类 / model_type', 'DeepseekV3ForCausalLM / deepseek_v3', '相同'],
        ['形态', 'MoE（671B 总参 / ≈37B 激活）', '相同'],
        ['hidden_size / 层数', '7168 / 61（前 3 层 Dense，后 58 层 MoE）', '相同'],
        ['注意力头', '128 Q / 128 KV（MLA 压缩）', '相同'],
        ['MLA 秩', 'kv_lora 512 / q_lora 1536 / qk_rope 64 / qk_nope 128 / v_head 128', '相同'],
        ['FFN / MoE', 'inter 18432；moe_inter 2048；256 专家选 8 + 1 共享；routed_scaling 2.5', '相同'],
        ['路由', 'sigmoid + noaux_tc（aux-loss-free）；n_group 8 / topk_group 4', '相同'],
        ['词表 / 上下文', 'vocab 129280；max_pos 163840（YaRN factor 40）', '相同（tokenizer 内容重构，见 §4.4）'],
        ['MTP / tie', 'num_nextn_predict_layers=1（~14B 附加）；tie=false', '相同'],
        ['精度格式', 'fp8 e4m3 + scale_fmt=ue8m0（权重+激活）', '不同：V3.1 明确以 UE8M0 scale 训练；V3 config 无 scale_fmt 字段'],
    ],
    widths=[3.6, 6.4, 4.4])
caption('表 1：V3.1 / V3.1-Base 官方 config 对照（来源：_tools\\_cfgpull\\ 两份远端 config）')

para('要点：① V3.1 的「685B」口径 = 671B 主模型 + ~14B MTP 模块（官方仓库 tag 与部分第三方页面），推理时去掉 MTP 即 671B；'
     '② config 相同不代表权重相同——V3.1-Base 是在 V3 权重基础上又续训了 840B tokens，且 tokenizer 文件与 chat template 已重构；'
     '③ mlp.gate.e_score_correction_bias 必须用 FP32 加载与计算（官方使用建议原文），路由行为才正确。')

# ================= 2 端到端数据流 =================
h1('2 端到端数据流总览')
para('文本 token → 词嵌入（129280×7168，与 LM Head 解绑）→ 61 个 DeepseekV3DecoderLayer（前置 RMSNorm：'
     'MLA 注意力子层 + Dense FFN / DeepSeekMoE 子层，均带残差）→ 最终 RMSNorm → LM Head。另有 1 层 MTP 头训练时预测下一未来 token。'
     '该数据流与 V3 完全一致——V3.1 唯一的「模式切换」发生在模板层（见 §4.1）。')
add_fig('01_deepseek_v3_1端到端总览.png', 15.5, '图 1：DeepSeek-V3.1 端到端数据流总览（架构与 V3 同构，图沿用 V3 版）')
add_fig('02_deepseek_v3_1解码层与核心机制.png', 15.5, '图 2：DeepSeek-V3.1 解码层与 MLA/MoE 核心机制（与 V3 同构）')
para('因架构逐字段同构，架构图直接沿用 deepseek_v3 目录的同名版本（仅文件名前缀改为 v3_1）；'
     '本节不重复展开 MLA / MoE 细节，完整解剖见 deepseek_v3 详文与下方 §3 速览。')

# ================= 3 核心机制速览 =================
h1('3 核心机制速览（与 V3 同构）')
make_table(
    ['机制', '一句话', '关键实参'],
    [
        ['MLA 多头潜在注意力', 'Q/KV 双低秩压缩，KV cache 只存 512+64 维潜变量', 'kv_lora 512 / q_lora 1536 / qk_rope 64 / qk_nope 128 / v_head 128'],
        ['DeepSeekMoE', '细粒度路由专家 + 共享专家常开', '256 选 8（+1 共享），moe_inter 2048，前 3 层 Dense'],
        ['aux-loss-free 负载均衡', 'sigmoid 打分 + bias 修正 + 分组限选', 'scoring=sigmoid / topk=noaux_tc / 8 组选 4'],
        ['MTP 多 token 预测', '训练期预测下一未来 token，推理可作投机解码草稿', 'num_nextn_predict_layers=1'],
        ['YaRN 长上下文', '4096 → 163840 外推（config 上限）', 'factor=40 / original_max=4096'],
    ],
    widths=[3.4, 5.6, 5.4])
caption('表 2：核心机制速览（完整解剖见 deepseek_v3 详文，两版为同一模块实现）')

# ================= 4 核心增量 =================
h1('4 V3.1 核心增量（本文档重点）')
para('架构零改动前提下，V3.1 的「版本升级」全部发生在训练数据、后训练目标、精度格式与对话模板四个层面——'
     '这正好是「家族架构详解」里区分 V3 与 V3.1 的全部依据。')

h2('4.1 ① 混合推理：单一权重，思考 / 非思考双模式')
para('官方公告（2025-08-19）：「一个模型同时支持思考模式与非思考模式」。实现完全在 chat template 层'
     '（tokenizer_config.json / assets/chat_template.jinja），权重与推理图无任何分支：')
para('非思考（Non-Thinking）前缀：｜begin▁of▁sentence｜{system}｜User｜{query}｜Assistant｜</think>', indent=False)
para('思考（Thinking）前缀：｜begin▁of▁sentence｜{system}｜User｜{query}｜Assistant｜<think>', indent=False)
para('非思考模式在 Assistant 后直接加 </think>（空思考闭合标记），模型直接输出答案；'
     '思考模式以 <think> 开启，模型先产出显式思维链，再输出可见答案；'
     '多轮对话中每轮保留 </think>，仅最后一轮按模式决定是否追加 <think>。')
para('意义：把 R1 的「推理模型」能力并入 V3 的同一套权重，无需再部署独立的 reasoner 模型；'
     '用户侧通过「深度思考」按钮或 API 的 thinking 开关自由切换。V3.1-Think 在输出 token 数减少 20%–50% '
     '的情况下平均表现与 R1-0528 持平（官方口径）。')

h2('4.2 ② 长上下文续训：840B tokens 两阶段（V3.1-Base 的核心投资）')
make_table(
    ['阶段', '目标长度', '续训数据量', '相对 V3-0324 同阶段'],
    [
        ['Phase 1', '→ 32K', '630B tokens', '10 倍'],
        ['Phase 2', '32K → 128K', '209B tokens', '3.3 倍'],
        ['合计', '—', '≈840B tokens', '—'],
    ],
    widths=[2.4, 3.4, 3.8, 4.8])
caption('表 3：长上下文两阶段续训（官方模型卡 / 公告原文）')
para('数据来源为额外收集的长文档；官方在 128K 全范围做 Needle in a Haystack（NIAH）检索测试核对召回。'
     '与 config 的关系：config 的 max_position_embeddings=163840（YaRN 外推上限）在 V3 就已如此，'
     'V3.1 的 128K 是「训练到达的长度」而非配置变化——即 V3 只外推到 128K 可用，V3.1 是在 128K 长度上真正训练过。')

h2('4.3 ③ UE8M0 FP8 Scale 精度格式（与 V3 的精度差异）')
para('V3 的 quantization_config：{quant_method: fp8, fmt: e4m3, activation_scheme: dynamic, weight_block_size: [128,128]}；'
     'V3.1 在其基础上增加 scale_fmt: ue8m0（已核对远端 config）：权重与激活的 per-block scale 因子用 UE8M0'
     '（8 位指数、0 位尾数的微缩放格式）编码，张量元素本身仍为 E4M3/E5M2 FP8。')
para('目的：使 checkpoint 与 MXFP8 微缩放（microscaling）硬件数值兼容（官方口径「面向即将发布的国产下一代芯片设计」，'
     'DeepGEMM 提供实现）。部署注意（官方使用建议）：e_score_correction_bias 必须 FP32 加载与计算；'
     'FP8 权重与激活须按 UE8M0 scale 格式解析（Terminus 版曾公开注明 self_attn.o_proj 存在 scale 格式不符问题，后续修正）。')

h2('4.4 ④ Tokenizer 与 chat template 重构')
para('官方明确「V3.1 对分词器及 chat template 进行了较大调整，与 DeepSeek-V3 存在明显差异」：')
para('思考标记：新增 <think> / </think>（见 §4.1）；工具调用：非思考模式支持结构化 tool-call 格式——'
     '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_name<｜tool▁sep｜>args<｜tool▁call▁end｜>…<｜tool▁calls▁end｜>，'
     '供 Code Agent（如 SWE 框架）与 Search Agent（思考模式内多轮搜索）使用；'
     'API 兼容：新增对 Anthropic API 格式的支持，可接入 Claude Code 框架；API Beta 支持 strict Function Calling（按 schema 校验输出）。')
para('注：vocab_size=129280 的 config 数值未变，但 tokenizer 词表文件本身（新增特殊标记）随版本重构——'
     '词表大小相同、内容有增改，部署时须用 V3.1 配套 tokenizer。')

h2('4.5 ⑤ 能力提升（后训练优化的效果，官方评测口径）')
make_table(
    ['类别', '基准', 'V3.1 非思考', 'V3-0324', 'V3.1 思考', 'R1-0528'],
    [
        ['通用', 'MMLU-Pro', '83.7', '81.2', '84.8', '85.0'],
        ['通用', 'GPQA-Diamond', '74.9', '68.4', '80.1', '81.0'],
        ['搜索智能体', 'BrowseComp', '—', '—', '30.0', '8.9'],
        ['搜索智能体', 'HLE (Python+Search)', '—', '—', '29.8', '24.8'],
        ['代码智能体', 'Aider-Polyglot', '68.4', '55.1', '76.3', '71.6'],
        ['数学推理', 'AIME 2024', '66.3', '59.4', '93.1', '91.4'],
    ],
    widths=[2.6, 4.2, 2.0, 2.0, 2.0, 2.0])
caption('表 4：官方评测对照（来源：DeepSeek-V3.1 模型卡 Evaluation 节）')
para('口径：Search Agent 用官方内部搜索框架（商业搜索 API + 网页过滤 + 128K 上下文）；SWE 用内部代码 agent 框架；'
     'HLE 为纯文本子集。V3.1 定位「迈向 Agent 时代的第一步」——相比 V3-0324 与 R1-0528 的主要增益集中在 '
     'Code/Search Agent 与思考效率（CoT 压缩）。')

# ================= 5 演进视角 =================
h1('5 演进视角：V3 → V3.1 → V3.1-Terminus → V3.2-Exp')
make_table(
    ['维度', 'V3（2024-12）', 'V3.1（2025-08）', 'V3.1-Terminus（2025-09）', 'V3.2-Exp（2025-09）'],
    [
        ['架构', 'MLA + MoE + MTP（61 层 / 256 专家选 8 / 7168）', '逐字段同构', '同构', '同构'],
        ['参数量', '671B / 37B 激活', '671B / 37B（+~14B MTP = 685B 口径）', '同左', '同左'],
        ['上下文', 'config 163840（YaRN），训练长度较短', '两阶段续训至 128K（840B tokens）', '同左', '同左'],
        ['推理模式', '单一模式', '混合推理（思考/非思考模板切换）', '同左 + 语言一致性修复', '引入 DSA 稀疏注意力'],
        ['精度', 'FP8 e4m3', 'UE8M0 FP8 scale（权重+激活）', '同左（o_proj scale 问题曾注记）', '—'],
        ['定位', '通用基座', 'Agent 时代第一步（工具/搜索增强）', '稳定性 + Agent 再优化', '实验性注意力升级'],
    ],
    widths=[2.2, 3.6, 4.2, 4.0, 3.2])
caption('表 5：V3 → V3.1 系列 → V3.2-Exp 逐维度演进')
para('V3.1 之后：V3.1-Terminus（2025-09-22）修复中英文混杂/异常字符，进一步优化 Code/Search Agent；'
     'V3.2-Exp（2025-09-29）在同一架构上引入 DeepSeek Sparse Attention（DSA），是注意力机制层面的首次改动'
     '（详见 deepseek_v3_2 文档）。V3.1 系列是「V3 架构 → V4 新代际（CSA/HCA）」之间的过渡主线。')

# ================= 6 来源 =================
h1('6 来源')
refs = [
    '[1] 远端官方 config（hf-mirror 拉取，2026-11-02 逐字段核对，存 _tools\\_cfgpull\\）：DeepSeek-V3.1.config.json / DeepSeek-V3.1-Base.config.json（与 V3 逐字段一致，含 scale_fmt: ue8m0）。',
    '[2] 官方发布公告（2025-08-19）：api-docs.deepseek.com/zh-cn/news/news250821/（混合推理、840B 续训、UE8M0、tokenizer/template 重构、Agent 提升）。',
    '[3] 官方版本更新（2025-09-22）：api-docs.deepseek.com/zh-cn/news/news250922/（V3.1-Terminus）。',
    '[4] 官方模型卡：huggingface.co/deepseek-ai/DeepSeek-V3.1；modelscope.cn/models/deepseek-ai/DeepSeek-V3.1（评测表 / 使用建议 / chat template）。',
    '[5] transformers 源码（本机 editable）：E:\\transformers\\src\\transformers\\models\\deepseek_v3\\（与 V3 同一实现，V3.1 无独立模块）；V3 技术报告 arXiv:2412.19437。',
]
for r in refs:
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(0)
    run = p.add_run(r)
    run.font.size = Pt(10.5)
    run.font.name = 'Times New Roman'
    run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')

para('凡 config 字段与官方口径并列处均已标注；本文未采用无法核实的结论。访问与核验日期：2026-11-02。', indent=False)

doc.save(OUT)
print('saved:', OUT)
