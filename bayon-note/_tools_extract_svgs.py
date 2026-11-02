# -*- coding: utf-8 -*-
"""提取 bayon-note 单模型系列 index.html 中的内联 SVG 架构图，
生成独立渲染页（保留原样式），供 Chrome headless 截图转 PNG 后插入 docx。
"""
import os
import re
import json

BASE = r"E:\transformers\bayon-note"
MODELS = ["BLIP", "CLIP", "Siglip", "YOLOS"]

# 每个模型里各 svg 的中文命名（顺序对应 html 中出现的顺序）
SVG_NAMES = {
    "BLIP":  ["01_blip端到端架构总览", "02_blip_med三模式", "03_blip_capfilt数据引导"],
    "CLIP":  ["01_clip端到端双塔架构"],
    "Siglip":["01_siglip两代视觉塔对比"],
    "YOLOS": ["01_yolos端到端架构总览", "02_yolos匈牙利匹配"],
}

def extract_style(html):
    m = re.search(r'<style>(.*?)</style>', html, re.S)
    return m.group(1) if m else ""

def extract_svgs(html):
    return re.findall(r'<svg class="dg".*?</svg>', html, re.S)

def viewbox_of(svg):
    m = re.search(r'viewBox="([\d\.]+) ([\d\.]+) ([\d\.]+) ([\d\.]+)"', svg)
    if not m:
        return 1000, 500
    return float(m.group(3)), float(m.group(4))

def main():
    manifest = {}
    for model in MODELS:
        html_path = os.path.join(BASE, model, "index.html")
        with open(html_path, encoding="utf-8") as f:
            html = f.read()
        style = extract_style(html)
        svgs = extract_svgs(html)
        names = SVG_NAMES[model]
        assert len(svgs) == len(names), f"{model}: svg 数 {len(svgs)} != 命名数 {len(names)}"
        arch_dir = os.path.join(BASE, model, "_arch")
        os.makedirs(arch_dir, exist_ok=True)
        page_dir = os.path.join(arch_dir, "svg_pages")
        os.makedirs(page_dir, exist_ok=True)
        entries = []
        for i, (svg, name) in enumerate(zip(svgs, names)):
            w, h = viewbox_of(svg)
            page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<style>{style}
body{{margin:0; background:#ffffff;}}
.diagram-shell{{padding:14px; border:1px solid #e5e4df; border-radius:12px; width:{w}px;}}
.diagram-shell svg{{display:block; width:{w}px; height:{h}px; min-width:0;}}
</style>
</head>
<body><div class="diagram-shell">{svg}</div></body>
</html>
"""
            page_path = os.path.join(page_dir, f"{i+1:02d}_{name}.html")
            with open(page_path, "w", encoding="utf-8") as f:
                f.write(page)
            entries.append({"name": name, "page": page_path, "w": int(w), "h": int(h), "idx": i + 1})
        manifest[model] = entries
        print(f"[{model}] 提取 {len(svgs)} 个 svg -> {page_dir}")
    with open(os.path.join(BASE, "_arch_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("manifest:", os.path.join(BASE, "_arch_manifest.json"))

if __name__ == "__main__":
    main()
