# -*- coding: utf-8 -*-
"""抽查：从 md 实际提取公式并显示转换前后对照（含表格单元格内公式）"""
import os, re, sys, importlib.util
BASE = r"E:\transformers\bayon-note\Gemma"
spec = importlib.util.spec_from_file_location("m2d", os.path.join(BASE, "_tools", "md2docx.py"))
m2d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m2d)
lt = m2d.latex_to_text

out = []
def pl(s): out.append(s)

# 找包含指定关键词的行
KEYS = ["inv_freq", "cases", "softcap", "hat{x}", "down_proj", "rotate_half", "approx0.26", "K^\\top", "pm50"]
seen = set()
for model in ["gemma-1","gemma-2","gemma-3","gemma-3n","gemma-4"]:
    md = os.path.join(BASE, model, "index.md")
    if not os.path.exists(md):
        continue
    for ln in open(md, encoding="utf-8"):
        for m in re.finditer(r"\$[^$\n]+\$", ln):
            core = m.group(0)[1:-1]
            if core in seen:
                continue
            if any(k in core for k in KEYS):
                seen.add(core)
                pl(f"MD   : {core}")
                pl(f"DOCX : {lt(core)}")
                pl("")

open(os.path.join(BASE, "_tools", "_latex_samples.txt"), "w", encoding="utf-8").write("\n".join(out))
print("ok", len(out))
