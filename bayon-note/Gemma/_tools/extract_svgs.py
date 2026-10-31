# -*- coding: utf-8 -*-
"""从 5 份 gemma-X/index.html 提取核心架构图 SVG(class=dg) → 生成独立 HTML 页面，
供 Chrome headless 截图。同时输出每张图的命名与尺寸清单。

用法: python extract_svgs.py
输出: 每个模型目录下 _arch_html/<序号>_<图名>.html；stdout 打印 <目录>|<文件名>|<宽>|<高>|<md 插入章节标题关键词>
"""
import os, re, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE = r"E:\transformers\bayon-note\Gemma"
MODELS = ["gemma-1", "gemma-2", "gemma-3", "gemma-3n", "gemma-4"]

svg_re = re.compile(r'<svg\b[^>]*class="dg"[^>]*>[\s\S]*?</svg>', re.I)
style_re = re.compile(r'<style[^>]*>[\s\S]*?</style>', re.I)
h_re = re.compile(r'<h([23])[^>]*>(.*?)</h\1>', re.S)
section_re = re.compile(r'<section\b[^>]*>', re.I)

def text_of(html):
    t = re.sub(r'<[^>]+>', '', html)
    t = t.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
    return t.strip()

for m in MODELS:
    d = os.path.join(BASE, m)
    html_path = os.path.join(d, "index.html")
    if not os.path.exists(html_path):
        print(f"SKIP {m}: no index.html")
        continue
    with open(html_path, encoding='utf-8') as f:
        html = f.read()
    css = "\n".join(re.findall(r'<style[^>]*>([\s\S]*?)</style>', html, re.I))
    outdir = os.path.join(d, "_arch_html")
    os.makedirs(outdir, exist_ok=True)
    # 收集标题（h2/h3）位置，用于给 svg 命名
    heads = []  # (pos, level, text)
    for mm in h_re.finditer(html):
        heads.append((mm.start(), mm.group(1), text_of(mm.group(2))))
    idx = 0
    for mm in svg_re.finditer(html):
        idx += 1
        svg = mm.group(0)
        # 找到该 svg 之前最近的 h2/h3
        name = f"架构图{idx}"
        for pos, lv, tx in reversed(heads):
            if pos < mm.start():
                name = re.sub(r'[：:：\s/\\:*?"<>|]+', '_', tx)
                name = re.sub(r'^_+|_+$', '', name)
                name = name[:24]
                break
        # 尺寸
        vb = re.search(r'viewBox="([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"', svg)
        w = int(float(vb.group(3))) if vb else 1200
        h = int(float(vb.group(4))) if vb else 700
        fname = f"{idx:02d}_{name}.html"
        page = ('<!DOCTYPE html><html><head><meta charset="utf-8"><style>' + css +
                '</style></head><body style="margin:0;padding:0;background:#ffffff;">' +
                svg + '</body></html>')
        with open(os.path.join(outdir, fname), 'w', encoding='utf-8') as f:
            f.write(page)
        print(f"{m}|{fname}|{w}|{h}|{name}")
