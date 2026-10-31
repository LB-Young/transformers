# -*- coding: utf-8 -*-
"""提取 5 份 md 中所有 $...$ 公式，用 md2docx.latex_to_text 转换后列出仍含反斜杠/未转换记号的片段"""
import os, re, sys, importlib.util

BASE = r"E:\transformers\bayon-note\Gemma"
MODELS = ["gemma-1", "gemma-2", "gemma-3", "gemma-3n", "gemma-4"]

spec = importlib.util.spec_from_file_location("m2d", os.path.join(BASE, "_tools", "md2docx.py"))
m2d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m2d)
lt = m2d.latex_to_text

out_lines = []
def pl(s):
    out_lines.append(s)

dollar_re = re.compile(r"\$[^$\n]+\$")

def collect(path):
    """收集 md 中所有 $...$ 公式与表格中 $...$ 片段"""
    out = []
    for ln in open(path, encoding="utf-8"):
        for m in dollar_re.finditer(ln):
            out.append(m.group(0))
    return out

for model in MODELS:
    md = os.path.join(BASE, model, "index.md")
    if not os.path.exists(md):
        continue
    frags = collect(md)
    bad = []
    for f in frags:
        core = f[1:-1]
        t = lt(core)
        if "\\" in t:
            bad.append((core, t))
    pl(f"== {model}: 共 {len(frags)} 个公式片段，转换后仍含反斜杠 {len(bad)} 个 ==")
    for core, t in bad:
        pl(f"  MD : {core}")
        pl(f"  DOCX: {t}")

open(os.path.join(BASE, "_tools", "_latex_scan.txt"), "w", encoding="utf-8").write("\n".join(out_lines))
print("written", len(out_lines))
