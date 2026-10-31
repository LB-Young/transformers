# -*- coding: utf-8 -*-
import sys
from docx import Document
p = r"E:\transformers\bayon-note\InternVL\internvl-3\InternVL-3-家族架构详解.docx"
doc = Document(p)
hits = []
for ti, tbl in enumerate(doc.tables):
    for row in tbl.rows:
        cells = [c.text.strip() for c in row.cells]
        line = " | ".join(cells)
        if "InternVL3" in line or "8.9" in line:
            hits.append(f"[table{ti}] {line}")
print("\n".join(hits))
print("---- 8.9 occurrences in whole doc ----")
n = 0
for para in doc.paragraphs:
    if "8.9" in para.text:
        n += 1; print("PARA:", para.text)
for tbl in doc.tables:
    for row in tbl.rows:
        for c in row.cells:
            if "8.9" in c.text:
                n += 1; print("CELL:", c.text)
print("8.9 count =", n)
