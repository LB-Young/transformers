# -*- coding: utf-8 -*-
"""DeepSeek v1/v2 md -> docx（复制自 Qwen/qwen2/_tools/md2docx.py，仅改 BASE/MODELS/OUT_NAME/依据段口径）。"""
import os, re, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from PIL import Image
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

BASE = r"E:\transformers\bayon-note\DeepSeek"
MODELS = [
    ("deepseek_v1", "DeepSeek-LLM（一代）家族架构详解",
     "LlamaForCausalLM · 7B / 67B · RMSNorm + SwiGLU + RoPE(θ=1e4) · 7B MHA / 67B GQA · 词表 102400"),
    ("deepseek_v2", "DeepSeek-V2 家族架构详解",
     "DeepseekV2ForCausalLM · 236B-A21B · MLA(kv_lora 512) + DeepSeekMoE(160选6+共享2) · YaRN 163840"),
]
OUT_NAME = {"deepseek_v1": "DeepSeek-V1", "deepseek_v2": "DeepSeek-V2"}

LATEX_REPL = [
    ("\\top", "ᵀ"), ("\\to", "→"),
    ("\\approx", "≈"), ("\\pm", "±"), ("\\infty", "∞"),
    ("\\partial", "∂"), ("\\in", "∈"),
    ("\\rightarrow", "→"), ("\\Rightarrow", "⇒"),
    ("\\equiv", "≡"), ("\\ne", "≠"), ("\\leq", "≤"), ("\\geq", "≥"),
    ("\\times", "×"), ("\\div", "÷"), ("\\cdot", "·"), ("\\oplus", "⊕"),
    ("\\theta", "θ"), ("\\sigma", "σ"), ("\\alpha", "α"), ("\\beta", "β"),
    ("\\mu", "μ"), ("\\epsilon", "ε"), ("\\gamma", "γ"), ("\\lambda", "λ"),
    ("\\Sigma", "Σ"), ("\\Delta", "Δ"), ("\\Phi", "Φ"),
    ("\\cos", "cos"), ("\\sin", "sin"), ("\\tanh", "tanh"),
    ("\\log", "log"), ("\\exp", "exp"), ("\\max", "max"), ("\\min", "min"),
    ("\\operatorname", ""), ("\\mathrm", ""), ("\\text", ""),
    ("\\left", ""), ("\\right", ""),
    ("\\quad", " "), ("\\;", " "), ("\\:", " "), ("\\,", " "),
    ("\\dots", "…"), ("\\ldots", "…"),
]

def sqrt_convert(s):
    m = re.search(r"\\sqrt\{", s)
    while m:
        i = m.end(); depth = 1; j = i
        while j < len(s) and depth:
            if s[j] == "{": depth += 1
            elif s[j] == "}": depth -= 1
            j += 1
        if depth != 0: break
        inner = sqrt_convert(s[i:j-1])
        s = s[:m.start()] + "√(" + inner + ")" + s[j:]
        m = re.search(r"\\sqrt\{", s)
    return s

def frac_convert(s):
    m = re.search(r"\\(?:frac|dfrac|tfrac)\{", s)
    while m:
        i = m.end(); depth = 1; j = i
        while j < len(s) and depth:
            if s[j] == "{": depth += 1
            elif s[j] == "}": depth -= 1
            j += 1
        if depth != 0: break
        a = s[i:j-1]
        if j >= len(s) or s[j] != "{": break
        i2 = j + 1; depth = 1; j2 = i2
        while j2 < len(s) and depth:
            if s[j2] == "{": depth += 1
            elif s[j2] == "}": depth -= 1
            j2 += 1
        if depth != 0: break
        b = s[i2:j2-1]
        s = s[:m.start()] + "(" + frac_convert(a) + ")/(" + frac_convert(b) + ")" + s[j2:]
        m = re.search(r"\\(?:frac|dfrac|tfrac)\{", s)
    return s

def latex_to_text(s):
    s = re.sub(r"\\operatorname\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = frac_convert(s)
    s = re.sub(r"\^\{([^{}]*)\}", r"^(\1)", s)
    s = re.sub(r"\_\{([^{}]*)\}", r"_\1", s)
    s = sqrt_convert(s)
    for pat, rep in LATEX_REPL:
        s = s.replace(pat, rep)
    s = re.sub(r"\\([a-zA-Z]+)", r"\1", s)
    s = s.replace("\\_", "_").replace("\\%", "%")
    for _ in range(6):
        s2 = re.sub(r"\{([^{}]*)\}", r"\1", s)
        if s2 == s: break
        s = s2
    s = re.sub(r"\s+", " ", s).strip()
    return s

def clean_text(s):
    s = re.sub(r'"([^"\n]*)"', "\u201c\\1\u201d", s)
    s = re.sub(r"\*\*([^*\n]+)\*\*", r"\1", s)
    s = s.replace("`", "")
    return s

def set_font(run, cn="宋体", en="Times New Roman", size=None, bold=None, color=None):
    run.font.name = en
    r = run._element
    rPr = r.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts"); rPr.append(rFonts)
    rFonts.set(qn("w:ascii"), en); rFonts.set(qn("w:hAnsi"), en); rFonts.set(qn("w:eastAsia"), cn)
    if size is not None: run.font.size = Pt(size)
    if bold is not None: run.font.bold = bold
    if color is not None: run.font.color.rgb = color

def inline_formula(par, text, font_size=None):
    text = clean_text(text)
    parts = re.split(r"(\$[^$\n]*\$)", text)
    for p in parts:
        if not p: continue
        if p.startswith("$") and p.endswith("$") and len(p) > 1:
            run = par.add_run(latex_to_text(p[1:-1]))
        else:
            run = par.add_run(p)
        if font_size: run.font.size = Pt(font_size)
        set_font(run, "宋体", "Times New Roman")
    return par

def style_paragraph(par, align=WD_ALIGN_PARAGRAPH.JUSTIFY, line15=True, indent2=True, before=0, after=0):
    pf = par.paragraph_format
    pf.alignment = align
    if line15: pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    if indent2: pf.first_line_indent = Pt(24)
    pf.space_before = Pt(before); pf.space_after = Pt(after); pf.widow_control = True

def add_heading_cn(doc, text, level):
    text = clean_text(text)
    h = doc.add_heading("", level=level)
    run = h.add_run(text)
    set_font(run, "黑体", "Times New Roman", size={1:16,2:14,3:12}[level], bold=True, color=RGBColor(0,0,0))
    pf = h.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf.space_before = Pt({1:14,2:11,3:9}[level]); pf.space_after = Pt({1:6,2:5,3:4}[level])
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE; pf.keep_with_next = True
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts"); rPr.append(rFonts)
    rFonts.set(qn("w:eastAsia"), "黑体")
    return h

def add_table(doc, rows):
    ncol = max(len(r) for r in rows)
    table = doc.add_table(rows=len(rows), cols=ncol)
    table.style = "Table Grid"; table.alignment = WD_TABLE_ALIGNMENT.CENTER; table.autofit = True
    for i, row in enumerate(rows):
        for j in range(ncol):
            cell = table.cell(i, j)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            txt = row[j] if j < len(row) else ""
            par = cell.paragraphs[0]
            par.paragraph_format.space_before = Pt(0); par.paragraph_format.space_after = Pt(0)
            par.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            par.paragraph_format.first_line_indent = Pt(0)
            par.paragraph_format.left_indent = Pt(0); par.paragraph_format.right_indent = Pt(0)
            inline_formula(par, txt, font_size=10.5)
            if i == 0:
                par.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in par.runs: run.font.bold = True
            else:
                if j == 0: par.alignment = WD_ALIGN_PARAGRAPH.CENTER
                elif re.fullmatch(r"[0-9,.~≈≤≥\s%-]+", txt): par.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                else: par.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for i, row in enumerate(rows):
        tr = table.rows[i]._tr
        trPr = tr.get_or_add_trPr()
        trPr.append(OxmlElement("w:cantSplit"))
        if i == 0:
            trPr.append(OxmlElement("w:tblHeader"))
            for j in range(ncol):
                tc = table.cell(i, j)._tc
                tcPr = tc.get_or_add_tcPr()
                shd = OxmlElement("w:shd")
                shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), "D9D9D9")
                vAlign = tcPr.find(qn("w:vAlign"))
                if vAlign is not None: vAlign.addprevious(shd)
                else: tcPr.append(shd)
    return table

def add_picture_block(doc, img_path, caption):
    img = Image.open(img_path)
    w_px, h_px = img.size
    width_cm = 14.5; height_cm = width_cm * h_px / w_px
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.space_before = Pt(6); par.paragraph_format.space_after = Pt(0)
    par.paragraph_format.first_line_indent = Pt(0); par.paragraph_format.keep_with_next = True
    run = par.add_run()
    run.add_picture(img_path, width=Cm(width_cm), height=Cm(height_cm))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(4); cap.paragraph_format.space_after = Pt(6)
    cap.paragraph_format.first_line_indent = Pt(0)
    cr = cap.add_run(caption)
    set_font(cr, "宋体", "Times New Roman", size=10.5)

def add_code_block(doc, lines):
    for ln in lines:
        par = doc.add_paragraph()
        par.paragraph_format.left_indent = Pt(24)
        par.paragraph_format.space_before = Pt(0); par.paragraph_format.space_after = Pt(0)
        par.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        par.paragraph_format.first_line_indent = Pt(0)
        run = par.add_run(ln if ln else " ")
        set_font(run, "宋体", "Consolas", size=10.5)
    doc.paragraphs[-1].paragraph_format.space_after = Pt(6)

def parse_md(path):
    with open(path, encoding="utf-8") as f:
        raw = f.read().splitlines()
    if raw and raw[0].startswith("\ufeff"):
        raw[0] = raw[0].lstrip("\ufeff")
    blocks = []
    i = 0
    while i < len(raw):
        line = raw[i]; s = line.strip()
        if not s: i += 1; continue
        if s.startswith("```"):
            i += 1; code = []
            while i < len(raw) and not raw[i].strip().startswith("```"):
                code.append(raw[i]); i += 1
            i += 1; blocks.append(("code", code)); continue
        m = re.match(r"^(#{1,6})\s+(.*)$", s)
        if m:
            blocks.append(("h", len(m.group(1)), m.group(2).strip())); i += 1; continue
        if s.startswith("|"):
            rows = []
            while i < len(raw) and raw[i].strip().startswith("|"):
                row = [c.strip() for c in raw[i].strip().strip("|").split("|")]
                rows.append(row); i += 1
            rows = [r for r in rows if not all(re.fullmatch(r":?-{2,}:?", c or "-") for c in r)]
            blocks.append(("table", rows)); continue
        m = re.match(r"^!\[(.*?)\]\((.*?)\)\s*$", s)
        if m:
            blocks.append(("img", m.group(1), m.group(2))); i += 1; continue
        if s.startswith("$$"):
            i += 1; fml = []
            if s.endswith("$$") and len(s) > 2: fml.append(s[2:-2])
            else:
                while i < len(raw) and not raw[i].strip().startswith("$$"):
                    fml.append(raw[i]); i += 1
                i += 1
            blocks.append(("formula", "\n".join(fml))); continue
        if s.startswith(">"):
            blocks.append(("quote", s.lstrip(">").strip())); i += 1; continue
        if s.startswith("- ") or s.startswith("* "):
            blocks.append(("para", s[2:].strip(), True)); i += 1; continue
        blocks.append(("para", s, False)); i += 1
    return blocks

def build_docx(title, subtitle, md_path, out_path):
    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Cm(21.0); sec.page_height = Cm(29.7)
    sec.top_margin = Cm(2.5); sec.bottom_margin = Cm(2.5)
    sec.left_margin = Cm(2.5); sec.right_margin = Cm(2.5)
    footer = sec.footer
    fp = footer.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fldChar1 = OxmlElement("w:fldChar"); fldChar1.set(qn("w:fldCharType"), "begin")
    instrText = OxmlElement("w:instrText"); instrText.set(qn("xml:space"), "preserve"); instrText.text = " PAGE "
    fldChar2 = OxmlElement("w:fldChar"); fldChar2.set(qn("w:fldCharType"), "end")
    run = fp.add_run()
    run._element.append(fldChar1); run._element.append(instrText); run._element.append(fldChar2)
    set_font(run, "宋体", "Times New Roman", size=10.5)
    normal = doc.styles["Normal"]
    normal.font.size = Pt(12); normal.font.name = "Times New Roman"
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t.paragraph_format.space_after = Pt(18)
    tr = t.add_run(title)
    set_font(tr, "黑体", "Times New Roman", size=18, bold=True, color=RGBColor(0,0,0))
    st = doc.add_paragraph(); st.alignment = WD_ALIGN_PARAGRAPH.CENTER
    st.paragraph_format.space_after = Pt(12)
    sr = st.add_run(subtitle)
    set_font(sr, "楷体", "Times New Roman", size=12, color=RGBColor(0,0,0))
    blocks = parse_md(md_path)
    fig_no = 0
    src_par = doc.add_paragraph()
    style_paragraph(src_par, indent2=False, after=8)
    inline_formula(src_par, "依据：远端官方 config（hf-mirror / ModelScope 拉取留档）+ transformers 源码 "
                            "（E:\\transformers 可编辑安装）+ 官方论文 / model card。所有数值按 config 实际取值与"
                            "源码 forward 路径逐项复算并标注来源。")
    for b in blocks:
        kind = b[0]
        if kind == "h":
            level, text = b[1], b[2]
            if level == 1: continue
            add_heading_cn(doc, text, min(level - 1, 3))
        elif kind == "para":
            p = doc.add_paragraph()
            style_paragraph(p, indent2=not b[2] if len(b) > 2 else True)
            inline_formula(p, b[1])
        elif kind == "quote":
            p = doc.add_paragraph(); style_paragraph(p, indent2=False); inline_formula(p, b[1])
        elif kind == "table":
            add_table(doc, b[1]); doc.add_paragraph().paragraph_format.space_after = Pt(0)
        elif kind == "img":
            fig_no += 1
            img_path = os.path.normpath(os.path.join(os.path.dirname(md_path), b[2]))
            if os.path.exists(img_path):
                add_picture_block(doc, img_path, f"图 {fig_no}：{b[1].strip()}")
            else:
                p = doc.add_paragraph(); style_paragraph(p, indent2=False); p.add_run(f"[图片缺失: {b[2]}]")
        elif kind == "formula":
            text = latex_to_text(" ".join(b[1].splitlines()))
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(4); p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.first_line_indent = Pt(0)
            run = p.add_run(text); set_font(run, "宋体", "Times New Roman", size=12)
        elif kind == "code":
            add_code_block(doc, b[1])
    doc.save(out_path)
    return fig_no

if __name__ == "__main__":
    for model, title, subtitle in MODELS:
        d = os.path.join(BASE, model)
        md_path = os.path.join(d, "index.md")
        out = os.path.join(d, f"{OUT_NAME[model]}-家族架构详解.docx")
        try:
            figs = build_docx(title, subtitle, md_path, out)
            print(f"{model}: {out}  图片 {figs} 张")
        except Exception as e:
            print(f"{model}: FAILED  {e}")
