# -*- coding: utf-8 -*-
"""检查 docx 中图片 extent 与裁切属性"""
import zipfile, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from lxml import etree

p = r"E:\transformers\bayon-note\Gemma\gemma-3\Gemma-3-家族架构详解.docx"
z = zipfile.ZipFile(p)
xml = z.read("word/document.xml").decode("utf-8")
root = etree.fromstring(xml.encode("utf-8"))
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
for ext in root.iter("{%s}ext" % A):
    cx = int(ext.get("cx")) / 360000
    cy = int(ext.get("cy")) / 360000
    print("ext cx=%.2fcm cy=%.2fcm ratio=%.3f" % (cx, cy, cx / cy if cy else 0))
crops = root.iter("{%s}srcRect" % A)
for c in crops:
    print("crop:", dict(c.attrib) if len(c.attrib) else "(empty)")
# 每张图的描述/名字
for blip in root.iter("{%s}blip" % A):
    print("blip embed:", blip.get("{%s}embed" % "http://schemas.openxmlformats.org/officeDocument/2006/relationships"))
