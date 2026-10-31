# -*- coding: utf-8 -*-
"""把公众号版 index.md 转换为 Word 文档（docx），风格参考 gemma-4-26B-A4B-it-param-breakdown.docx。
用法: python md2docx.py
输出: 各模型目录下 Gemma-X-家族架构详解.docx
"""
import os, re, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from PIL import Image
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

BASE = r"E:\transformers\bayon-note\MiniMax"
MODELS = [
    ("minimax_text_01", "MiniMax-Text-01 家族架构详解", "MiniMaxForCausalLM · 456B总参/45.9B激活 · 80层 · Lightning:Softmax=7:1 · 32专家top2"),
]

CJK = {"黑体": "SimHei", "宋体": "SimSun", "楷体": "KaiTi", "等线": "DengXian"}

# ---------- LaTeX -> 可读文本 ----------
# 注意：\top 必须排在 \to 之前（否则 \top 被 \to 误吞成 →p）
LATEX_REPL = [
    ("\\top", "ᵀ"),
    ("\\to", "→"),
    ("\\approx", "≈"), ("\\pm", "±"), ("\\mp", "∓"), ("\\infty", "∞"),
    ("\\partial", "∂"), ("\\nabla", "∇"), ("\\in", "∈"), ("\\notin", "∉"),
    ("\\subset", "⊂"), ("\\subseteq", "⊆"), ("\\supset", "⊃"), ("\\supseteq", "⊇"),
    ("\\cup", "∪"), ("\\cap", "∩"), ("\\emptyset", "∅"), ("\\varnothing", "∅"),
    ("\\rightarrow", "→"), ("\\Rightarrow", "⇒"), ("\\leftarrow", "←"), ("\\Leftarrow", "⇐"),
    ("\\leftrightarrow", "↔"), ("\\Leftrightarrow", "⇔"), ("\\mapsto", "↦"),
    ("\\equiv", "≡"), ("\\ne", "≠"), ("\\leq", "≤"), ("\\geq", "≥"),
    ("\\times", "×"), ("\\div", "÷"), ("\\cdot", "·"), ("\\odot", "⊙"),
    ("\\otimes", "⊗"), ("\\oplus", "⊕"), ("\\pmb", ""),
    ("\\ell", "ℓ"), ("\\prime", "′"), ("\\forall", "∀"), ("\\exists", "∃"),
    ("\\neg", "¬"), ("\\sim", "∼"), ("\\simeq", "≃"), ("\\cong", "≅"),
    ("\\propto", "∝"), ("\\dots", "…"), ("\\ldots", "…"), ("\\cdots", "⋯"),
    ("\\vdots", "⋮"), ("\\ddots", "⋱"), ("\\star", "⋆"), ("\\ast", "*"),
    ("\\circ", "∘"), ("\\dagger", "†"), ("\\ddagger", "‡"),
    ("\\theta", "θ"), ("\\gamma", "γ"), ("\\lambda", "λ"), ("\\alpha", "α"),
    ("\\beta", "β"), ("\\sigma", "σ"), ("\\Sigma", "Σ"), ("\\pi", "π"),
    ("\\Phi", "Φ"), ("\\phi", "ϕ"), ("\\varphi", "φ"), ("\\Delta", "Δ"),
    ("\\delta", "δ"), ("\\epsilon", "ε"), ("\\varepsilon", "ε"), ("\\mu", "μ"),
    ("\\nu", "ν"), ("\\omega", "ω"), ("\\Omega", "Ω"), ("\\eta", "η"),
    ("\\rho", "ρ"), ("\\tau", "τ"), ("\\kappa", "κ"), ("\\zeta", "ζ"),
    ("\\xi", "ξ"), ("\\chi", "χ"), ("\\psi", "ψ"), ("\\Psi", "Ψ"),
    ("\\Gamma", "Γ"), ("\\Lambda", "Λ"), ("\\Xi", "Ξ"), ("\\Upsilon", "Υ"),
    ("\\cos", "cos"), ("\\sin", "sin"), ("\\tan", "tan"), ("\\cot", "cot"),
    ("\\sec", "sec"), ("\\csc", "csc"), ("\\log", "log"), ("\\ln", "ln"),
    ("\\lg", "lg"), ("\\exp", "exp"), ("\\max", "max"), ("\\min", "min"),
    ("\\arg", "arg"), ("\\dim", "dim"), ("\\tanh", "tanh"),
    ("\\operatorname", ""), ("\\mathrm", ""), ("\\text", ""),
    ("\\left", ""), ("\\right", ""), ("\\!", ""),
    ("\\quad", " "), ("\\qquad", " "), ("\\;", " "), ("\\:", " "),
    ("\\,", " "), ("\\ ", " "), ("\\^", "^"),
]

def sqrt_convert(s):
    """把 \\sqrt{...} 深度转换为 √(...)，支持嵌套花括号。"""
    m = re.search(r"\\sqrt\{", s)
    while m:
        i = m.end()
        depth = 1
        j = i
        while j < len(s) and depth:
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                depth -= 1
            j += 1
        if depth != 0:
            break
        inner = sqrt_convert(s[i:j-1])
        s = s[:m.start()] + "√(" + inner + ")" + s[j:]
        m = re.search(r"\\sqrt\{", s)
    return s

def frac_convert(s):
    """把 \\frac{...}{...} 深度转换为 (a)/(b)，参数支持嵌套花括号。"""
    m = re.search(r"\\(?:frac|dfrac|tfrac)\{", s)
    while m:
        i = m.end()
        depth = 1
        j = i
        while j < len(s) and depth:
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                depth -= 1
            j += 1
        if depth != 0:
            break
        a = s[i:j-1]
        if j >= len(s) or s[j] != "{":
            break
        i2 = j + 1
        depth = 1
        j2 = i2
        while j2 < len(s) and depth:
            if s[j2] == "{":
                depth += 1
            elif s[j2] == "}":
                depth -= 1
            j2 += 1
        if depth != 0:
            break
        b = s[i2:j2-1]
        s = s[:m.start()] + "(" + frac_convert(a) + ")/(" + frac_convert(b) + ")" + s[j2:]
        m = re.search(r"\\(?:frac|dfrac|tfrac)\{", s)
    return s

def latex_to_text(s):
    # 1) 去壳：operatorname / mathrm / text，保留内容
    s = re.sub(r"\\operatorname\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    # 2) cases 分段函数 → cases(...)，行分隔用；，去掉 & 列对齐符
    s = re.sub(r"\\begin\{cases\}", "cases(", s)
    s = re.sub(r"\\end\{cases\}", ")", s)
    s = s.replace("\\\\", "；")
    s = s.replace("&", "")
    # 3) 分式（支持嵌套）+ 帽子/波浪/上横线/二项式
    s = frac_convert(s)
    s = re.sub(r"\\hat\{([^{}]*)\}", r"\1̂", s)
    s = re.sub(r"\\tilde\{([^{}]*)\}", r"\1̃", s)
    s = re.sub(r"\\bar\{([^{}]*)\}", r"¯\1", s)
    s = re.sub(r"\\binom\{([^{}]*)\}\{([^{}]*)\}", r"C(\1,\2)", s)
    # 4) 上标/下标：^{...} → ^(...)（标杆式 P^(s)），_{...} → _内容（标杆式 P_embed）
    s = re.sub(r"\^\{([^{}]*)\}", r"^(\1)", s)
    s = re.sub(r"\_\{([^{}]*)\}", r"_\1", s)
    # 5) 根号
    s = sqrt_convert(s)
    s = re.sub(r"\\sqrt(\w)", r"√\1", s)
    # 6) 括号尺寸命令 \big/\Big/\bigg/\Bigg 及左右变体
    s = re.sub(r"\\[bB]ig{1,2}[lr]?", "", s)
    # 7) 符号替换（\top 已在 \to 前）
    for pat, rep in LATEX_REPL:
        s = s.replace(pat, rep)
    # 8) 剩余裸命令兜底：\foo → foo
    s = re.sub(r"\\([a-zA-Z]+)", r"\1", s)
    # 9) 转义字符
    s = s.replace("\\_", "_").replace("\\%", "%").replace("\\&", "&").replace("\\#", "#")
    # 10) 迭代去除残余花括号
    for _ in range(6):
        s2 = re.sub(r"\{([^{}]*)\}", r"\1", s)
        if s2 == s:
            break
        s = s2
    s = re.sub(r"\s+", " ", s).strip()
    return s

def clean_text(s):
    """中文正文里成对 ASCII 双引号 → 中文引号；去 md 加粗/行内代码标记"""
    s = re.sub(r'"([^"\n]*)"', "\u201c\\1\u201d", s)
    s = re.sub(r"\*\*([^*\n]+)\*\*", r"\1", s)
    s = s.replace("`", "")
    return s

def inline_formula(par, text, font_size=None):
    """把包含 $...$ 的文本写入段落（去 $，LaTeX 转文本）。"""
    text = clean_text(text)
    parts = re.split(r"(\$[^$\n]*\$)", text)
    for p in parts:
        if not p:
            continue
        if p.startswith("$") and p.endswith("$") and len(p) > 1:
            run = par.add_run(latex_to_text(p[1:-1]))
        else:
            run = par.add_run(p)
        if font_size:
            run.font.size = Pt(font_size)
        set_font(run, "宋体", "Times New Roman")
    return par

# ---------- 字体 ----------
def set_font(run, cn="宋体", en="Times New Roman", size=None, bold=None, color=None):
    run.font.name = en
    r = run._element
    rPr = r.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:ascii"), en)
    rFonts.set(qn("w:hAnsi"), en)
    rFonts.set(qn("w:eastAsia"), cn)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if color is not None:
        run.font.color.rgb = color

def style_paragraph(par, align=WD_ALIGN_PARAGRAPH.JUSTIFY, line15=True, indent2=True, before=0, after=0):
    pf = par.paragraph_format
    pf.alignment = align
    if line15:
        pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    if indent2:
        pf.first_line_indent = Pt(24)  # 2 字符 @12pt
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.widow_control = True

def add_heading_cn(doc, text, level):
    text = clean_text(text)
    h = doc.add_heading("", level=level)
    run = h.add_run(text)
    set_font(run, "黑体", "Times New Roman",
             size={1: 16, 2: 14, 3: 12}[level], bold=True, color=RGBColor(0, 0, 0))
    pf = h.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf.space_before = Pt({1: 14, 2: 11, 3: 9}[level])
    pf.space_after = Pt({1: 6, 2: 5, 3: 4}[level])
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    pf.keep_with_next = True
    # 标题字体也需 eastAsia
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts"); rPr.append(rFonts)
    rFonts.set(qn("w:eastAsia"), "黑体")
    return h

def add_table(doc, rows):
    """rows: list[list[str]]，首行为表头。"""
    ncol = max(len(r) for r in rows)
    table = doc.add_table(rows=len(rows), cols=ncol)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    for i, row in enumerate(rows):
        for j in range(ncol):
            cell = table.cell(i, j)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            txt = row[j] if j < len(row) else ""
            par = cell.paragraphs[0]
            par.paragraph_format.space_before = Pt(0)
            par.paragraph_format.space_after = Pt(0)
            par.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            par.paragraph_format.first_line_indent = Pt(0)
            par.paragraph_format.left_indent = Pt(0)
            par.paragraph_format.right_indent = Pt(0)
            inline_formula(par, txt, font_size=10.5)
            if i == 0:
                par.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in par.runs:
                    run.font.bold = True
            else:
                # 首列居中，其他列：数字右对齐，文本左对齐
                if j == 0:
                    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
                elif re.fullmatch(r"[0-9,.~≈≤≥\s%-]+", txt):
                    par.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                else:
                    par.alignment = WD_ALIGN_PARAGRAPH.LEFT
            # 行不可跨页
            trPr = cell._tc.get_or_add_tcPr()
    # 表头底纹 + 跨页重复 + 行 cantSplit
    for i, row in enumerate(rows):
        tr = table.rows[i]._tr
        trPr = tr.get_or_add_trPr()
        cantSplit = OxmlElement("w:cantSplit")
        trPr.append(cantSplit)
        if i == 0:
            tblHeader = OxmlElement("w:tblHeader")
            trPr.append(tblHeader)
            for j in range(ncol):
                tc = table.cell(i, j)._tc
                tcPr = tc.get_or_add_tcPr()
                shd = OxmlElement("w:shd")
                shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto")
                shd.set(qn("w:fill"), "D9D9D9")
                vAlign = tcPr.find(qn("w:vAlign"))
                if vAlign is not None:
                    vAlign.addprevious(shd)   # OOXML 顺序: tcW < ... < shd < vAlign
                else:
                    tcPr.append(shd)
    return table

def add_picture_block(doc, img_path, caption):
    img = Image.open(img_path)
    w_px, h_px = img.size
    width_cm = 14.5
    height_cm = width_cm * h_px / w_px
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.space_before = Pt(6)
    par.paragraph_format.space_after = Pt(0)
    par.paragraph_format.first_line_indent = Pt(0)
    par.paragraph_format.keep_with_next = True   # 图与题注同页
    run = par.add_run()
    run.add_picture(img_path, width=Cm(width_cm), height=Cm(height_cm))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(4)
    cap.paragraph_format.space_after = Pt(6)
    cap.paragraph_format.first_line_indent = Pt(0)
    cr = cap.add_run(caption)
    set_font(cr, "宋体", "Times New Roman", size=10.5)
    return par

def add_code_block(doc, lines):
    for ln in lines:
        par = doc.add_paragraph()
        par.paragraph_format.left_indent = Pt(24)
        par.paragraph_format.space_before = Pt(0)
        par.paragraph_format.space_after = Pt(0)
        par.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        par.paragraph_format.first_line_indent = Pt(0)
        run = par.add_run(ln if ln else " ")
        set_font(run, "宋体", "Consolas", size=10.5)
    # 代码块前后留白
    doc.paragraphs[-1].paragraph_format.space_after = Pt(6)

# ---------- Markdown 解析 ----------
def parse_md(path):
    with open(path, encoding="utf-8") as f:
        raw = f.read().splitlines()
    # 去 BOM
    if raw and raw[0].startswith("\ufeff"):
        raw[0] = raw[0].lstrip("\ufeff")
    blocks = []
    i = 0
    while i < len(raw):
        line = raw[i]
        s = line.strip()
        if not s:
            i += 1; continue
        if s.startswith("```"):
            i += 1
            code = []
            while i < len(raw) and not raw[i].strip().startswith("```"):
                code.append(raw[i]); i += 1
            i += 1
            blocks.append(("code", code)); continue
        m = re.match(r"^(#{1,6})\s+(.*)$", s)
        if m:
            blocks.append(("h", len(m.group(1)), m.group(2).strip())); i += 1; continue
        if s.startswith("|"):
            rows = []
            while i < len(raw) and raw[i].strip().startswith("|"):
                row = [c.strip() for c in raw[i].strip().strip("|").split("|")]
                rows.append(row); i += 1
            # 去掉 md 分隔行
            rows = [r for r in rows if not all(re.fullmatch(r":?-{2,}:?", c or "-") for c in r)]
            blocks.append(("table", rows)); continue
        m = re.match(r"^!\[(.*?)\]\((.*?)\)\s*$", s)
        if m:
            blocks.append(("img", m.group(1), m.group(2))); i += 1; continue
        if s.startswith("$$"):
            i += 1
            fml = []
            if s.endswith("$$") and len(s) > 2:
                fml.append(s[2:-2])
            else:
                while i < len(raw) and not raw[i].strip().startswith("$$"):
                    fml.append(raw[i]); i += 1
                i += 1
            blocks.append(("formula", "\n".join(fml))); continue
        if s.startswith(">"):
            q = s.lstrip(">").strip()
            blocks.append(("quote", q)); i += 1; continue
        if s.startswith("- ") or s.startswith("* "):
            blocks.append(("para", s[2:].strip(), True)); i += 1; continue
        blocks.append(("para", s, False)); i += 1
    return blocks

def build_docx(model_dir, title, subtitle, md_path, img_dir, out_path):
    doc = Document()
    # 页面设置 A4 2.5cm
    sec = doc.sections[0]
    sec.page_width = Cm(21.0); sec.page_height = Cm(29.7)
    sec.top_margin = Cm(2.5); sec.bottom_margin = Cm(2.5)
    sec.left_margin = Cm(2.5); sec.right_margin = Cm(2.5)
    # 页脚页码（居中 PAGE 域）
    footer = sec.footer
    fp = footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fldChar1 = OxmlElement("w:fldChar"); fldChar1.set(qn("w:fldCharType"), "begin")
    instrText = OxmlElement("w:instrText"); instrText.set(qn("xml:space"), "preserve")
    instrText.text = " PAGE "
    fldChar2 = OxmlElement("w:fldChar"); fldChar2.set(qn("w:fldCharType"), "end")
    run = fp.add_run()
    run._element.append(fldChar1); run._element.append(instrText); run._element.append(fldChar2)
    set_font(run, "宋体", "Times New Roman", size=10.5)
    # 默认样式
    normal = doc.styles["Normal"]
    normal.font.size = Pt(12)
    normal.font.name = "Times New Roman"
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    # Title
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t.paragraph_format.space_after = Pt(18)
    tr = t.add_run(title)
    set_font(tr, "黑体", "Times New Roman", size=18, bold=True, color=RGBColor(0, 0, 0))
    # Subtitle
    st = doc.add_paragraph()
    st.alignment = WD_ALIGN_PARAGRAPH.CENTER
    st.paragraph_format.space_after = Pt(12)
    sr = st.add_run(subtitle)
    set_font(sr, "楷体", "Times New Roman", size=12, color=RGBColor(0, 0, 0))
    # 依据段：取自 md 首个段落之后（我们预先插入固定来源说明）
    blocks = parse_md(md_path)
    fig_no = 0
    # 依据段固定文案
    src_par = doc.add_paragraph()
    style_paragraph(src_par, indent2=False, after=8)
    inline_formula(src_par, "依据：官方 config（hf-mirror/ModelScope 远端拉取，留档 _tools\_cfgpull）+ transformers 源码 "
                            "（E:\\transformers）+ 官方论文/model card。所有数值按 config 实际取值与源码 forward 路径逐项复算并标注来源。")
    for b in blocks:
        kind = b[0]
        if kind == "h":
            level, text = b[1], b[2]
            if level == 1:
                continue  # md 的 # 标题已在 Title
            add_heading_cn(doc, text, min(level - 1, 3))
        elif kind == "para":
            text = b[1]
            is_list = b[2] if len(b) > 2 else False
            p = doc.add_paragraph()
            style_paragraph(p, indent2=not is_list)
            inline_formula(p, text)
        elif kind == "quote":
            p = doc.add_paragraph()
            style_paragraph(p, indent2=False)
            inline_formula(p, b[1])
        elif kind == "table":
            add_table(doc, b[1])
            doc.add_paragraph().paragraph_format.space_after = Pt(0)
        elif kind == "img":
            fig_no += 1
            alt = b[1].strip()
            rel = b[2]
            img_path = os.path.normpath(os.path.join(os.path.dirname(md_path), rel))
            if os.path.exists(img_path):
                add_picture_block(doc, img_path, f"图 {fig_no}：{alt}")
            else:
                p = doc.add_paragraph()
                style_paragraph(p, indent2=False)
                p.add_run(f"[图片缺失: {rel}]")
        elif kind == "formula":
            text = latex_to_text(" ".join(b[1].splitlines()))
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.first_line_indent = Pt(0)
            run = p.add_run(text)
            set_font(run, "宋体", "Times New Roman", size=12)
        elif kind == "code":
            add_code_block(doc, b[1])
    doc.save(out_path)
    return fig_no

if __name__ == "__main__":
    for model, title, subtitle in MODELS:
        d = os.path.join(BASE, model)
        md_path = os.path.join(d, "index.md")
        img_dir = os.path.join(d, "_arch")
        OUT_NAME = {"minimax_text_01": "MiniMax-Text-01"}
        out = os.path.join(d, f"{OUT_NAME[model]}-家族架构详解.docx")
        try:
            figs = build_docx(d, title, subtitle, md_path, img_dir, out)
            print(f"{model}: {out}  图片 {figs} 张")
        except Exception as e:
            print(f"{model}: FAILED  {e}")
