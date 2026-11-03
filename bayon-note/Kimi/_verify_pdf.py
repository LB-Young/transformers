# -*- coding: utf-8 -*-
"""Verify downloaded K3 PDF header and page count."""
p = r"E:\transformers\bayon-note\Kimi\kimi_k3\_tools\_cfgpull\k3_tech_report.pdf"
with open(p, "rb") as f:
    head = f.read(16)
    f.seek(-1024, 2)
    tail = f.read()
import os
print("size:", os.path.getsize(p))
print("head:", head)
print("has %%EOF:", b"%%EOF" in tail)
print("page markers:", tail.count(b"/Type /Page") + tail.count(b"/Type/Page"))
