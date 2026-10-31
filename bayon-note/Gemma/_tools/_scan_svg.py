# -*- coding: utf-8 -*-
"""扫描 5 个 Gemma HTML 中所有 .dg svg：viewBox vs 元素内容边界，找出越界项"""
import os, re, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE = r"E:\transformers\bayon-note\Gemma"
models = ["gemma-1", "gemma-2", "gemma-3", "gemma-3n", "gemma-4"]

def content_bounds(svg):
    """解析 svg 片段中的 rect/circle/text/path 元素，估算内容最大 y。"""
    maxy = 0
    maxx = 0
    # rect y + height
    for m in re.finditer(r'<rect[^>]*\by="([\d.]+)"[^>]*\bheight="([\d.]+)"', svg):
        y = float(m.group(1)) + float(m.group(2)); maxy = max(maxy, y)
    for m in re.finditer(r'<circle[^>]*\bcy="([\d.]+)"[^>]*\br="([\d.]+)"', svg):
        y = float(m.group(1)) + float(m.group(2)); maxy = max(maxy, y)
    # text y（基线）+ 12 字高
    for m in re.finditer(r'<text[^>]*\by="([\d.]+)"', svg):
        maxy = max(maxy, float(m.group(1)) + 12)
    # path 里可能含大坐标（粗略取数字峰值）
    for m in re.finditer(r'[Mm](\d+(?:\.\d+)?),(\d+(?:\.\d+)?)', svg):
        maxy = max(maxy, float(m.group(2))); maxx = max(maxx, float(m.group(1)))
    return maxx, maxy

for model in models:
    p = os.path.join(BASE, model, "index.html")
    if not os.path.exists(p):
        print(model, "NO HTML"); continue
    html = open(p, encoding="utf-8").read()
    # 逐个 svg（含 viewBox 的）
    for i, m in enumerate(re.finditer(r'<svg[^>]*viewBox="([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)"[^>]*>([\s\S]*?)</svg>', html)):
        vb_x, vb_y, vb_w, vb_h = float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))
        maxx, maxy = content_bounds(m.group(5))
        flag = ""
        if maxy > vb_h + 2 or maxx > vb_w + 2:
            flag = "  <<< 越界!"
        print(f"{model} svg#{i}: viewBox {vb_w:.0f}x{vb_h:.0f}  内容边界 x≤{maxx:.0f} y≤{maxy:.0f}{flag}")
