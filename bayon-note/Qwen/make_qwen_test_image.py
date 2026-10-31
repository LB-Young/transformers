# -*- coding: utf-8 -*-
"""生成 Qwen 多模态调试用的默认测试图（彩色几何图形）。"""
from PIL import Image, ImageDraw, ImageFont

OUT = r"E:\transformers\bayon-note\Qwen\qwen_test_image.png"

img = Image.new("RGB", (800, 500), "#f2f5f9")
d = ImageDraw.Draw(img)

# 三个彩色几何图形：红圆 / 蓝方块 / 绿三角
d.ellipse([110, 100, 280, 270], fill="#e74c3c", outline="#c0392b", width=6)
d.rectangle([350, 100, 570, 270], fill="#3498db", outline="#2471a3", width=6)
d.polygon([(700, 270), (780, 110), (610, 110)], fill="#2ecc71", outline="#1e8449", width=6)

# 底部信息条
d.rectangle([0, 400, 800, 460], fill="#34495e")
try:
    title_font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 28)
    body_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
except OSError:
    title_font = ImageFont.load_default()
    body_font = ImageFont.load_default()
d.text((50, 45), "Qwen Vision Test Image", fill="#2c3e50", font=title_font)
d.text((40, 412), "red circle  |  blue square  |  green triangle", fill="#ecf0f1", font=body_font)

img.save(OUT)
print("saved:", OUT, img.size)
