# -*- coding: utf-8 -*-
"""列出指定 svg 中超出 viewBox 高度的元素明细"""
import os, re, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

def details(html_path, svg_index):
    html = open(html_path, encoding="utf-8").read()
    svgs = re.findall(r'<svg[^>]*viewBox="([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)"[^>]*>([\s\S]*?)</svg>', html)
    if svg_index >= len(svgs):
        print("no such svg"); return
    vb = (float(svgs[svg_index][0]), float(svgs[svg_index][1]), float(svgs[svg_index][2]), float(svgs[svg_index][3]))
    body = svgs[svg_index][4]
    print(f"viewBox: {vb[0]:.0f},{vb[1]:.0f} {vb[2]:.0f}x{vb[3]:.0f}")
    print("--- rect 超出 ---")
    for m in re.finditer(r'<rect[^>]*\by="([\d.]+)"[^>]*\bheight="([\d.]+)"', body):
        y = float(m.group(1)) + float(m.group(2))
        if y > vb[3]:
            print(f"  rect y={m.group(1)} h={m.group(2)} -> bottom {y:.0f}")
    print("--- text 超出(基线+14) ---")
    for m in re.finditer(r'<text[^>]*\by="([\d.]+)"', body):
        y = float(m.group(1)) + 14
        if y > vb[3]:
            print(f"  text y={m.group(1)} -> 约{y:.0f}  ctx={body[max(0,m.start()-80):m.start()][-80:]!r}")
    print("--- circle 超出 ---")
    for m in re.finditer(r'<circle[^>]*\bcy="([\d.]+)"[^>]*\br="([\d.]+)"', body):
        y = float(m.group(1)) + float(m.group(2))
        if y > vb[3]:
            print(f"  circle cy={m.group(1)} r={m.group(2)} -> bottom {y:.0f}")

details(r"E:\transformers\bayon-note\Gemma\gemma-3\index.html", 2)
print()
details(r"E:\transformers\bayon-note\Gemma\gemma-4\index.html", 2)
print()
details(r"E:\transformers\bayon-note\Gemma\gemma-4\index.html", 3)
