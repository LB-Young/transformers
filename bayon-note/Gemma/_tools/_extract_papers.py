# -*- coding: utf-8 -*-
"""提取 4 篇 Gemma 论文 PDF 全文文本到 txt，便于精读训练数据章节。"""
import os
import fitz  # PyMuPDF

base = r"E:\agents\agents\papers\Google\其他"
outdir = r"E:\transformers\bayon-note\Gemma\_tools\_gemma_papers_txt"
os.makedirs(outdir, exist_ok=True)

papers = {
    "gemma1": "Gemma.pdf",
    "gemma2": "Gemma-2.pdf",
    "gemma3": "Gemma-3.pdf",
    "gemma4": "Gemma-4.pdf",
}

for key, fname in papers.items():
    path = os.path.join(base, fname)
    doc = fitz.open(path)
    parts = []
    for i, page in enumerate(doc):
        parts.append(f"\n===== PAGE {i+1} =====\n")
        parts.append(page.get_text("text"))
    text = "".join(parts)
    out = os.path.join(outdir, f"{key}.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"{key}: {len(doc)} pages, {len(text)} chars -> {out}")
