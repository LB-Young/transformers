# -*- coding: utf-8 -*-
"""Generate GLM-4V-9B (2024 multimodal) architecture diagrams (PNG)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = r"E:\transformers\bayon-note\GLM\glm_4v\_arch"
os.makedirs(OUT, exist_ok=True)

INK = "#23272F"; SUB = "#66707B"; EDGE = "#C9C4BA"
ACCENT = "#1D4ED8"; ACCENT_SOFT = "#E2EAFB"; ACCENT_LINE = "#C0D0F4"
MM = "#0E7490"; MM_SOFT = "#E0F2F7"; MM_LINE = "#A5D8E6"
NODE = "#FFFDF9"; CORE = "#EEF3FD"

def box(ax, x, y, w, h, text, fc=NODE, ec=EDGE, fs=10.5, bold=False, tc=INK):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                       linewidth=1.3, edgecolor=ec, facecolor=fc)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", linespacing=1.45)

def arrow(ax, p1, p2, color=EDGE, lw=1.6):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13, linewidth=lw, color=color)
    ax.add_patch(a)

# ===== 图1：端到端（图像→视觉塔→投影→文本塔）=====
fig, ax = plt.subplots(figsize=(14, 6.6))
ax.set_xlim(0, 14); ax.set_ylim(0, 6.6); ax.axis("off")

# 视觉支路
box(ax, 0.2, 4.3, 1.7, 1.0, "图像\n(tile ≤1120px)\n动态分辨率", fc=MM_SOFT, ec=MM_LINE, fs=9.5)
box(ax, 2.2, 4.3, 1.9, 1.0, "Patch 切分\npatch=14\n1120/14=80×80", fc=MM_SOFT, ec=MM_LINE, fs=9.5)
box(ax, 4.4, 4.0, 2.6, 1.6,
    "视觉塔 ViT（63层）\nhidden 1792 · heads 16\ninter 15360 · GELU\nnum_positions=6401",
    fc="#d6eef5", ec=MM, fs=9.5, bold=True)
box(ax, 7.3, 4.3, 1.9, 1.0, "空间下采样\nscaling=8\n8×8 patch 合并", fc=MM_SOFT, ec=MM_LINE, fs=9.5)
box(ax, 9.5, 4.3, 1.8, 1.0, "投影\n1792→4096", fc=MM_SOFT, ec=MM_LINE, fs=9.5)

arrow(ax, (1.9, 4.8), (2.2, 4.8))
arrow(ax, (4.1, 4.8), (4.4, 4.8))
arrow(ax, (7.0, 4.8), (7.3, 4.8))
arrow(ax, (9.2, 4.8), (9.5, 4.8))

# 文本支路
box(ax, 0.2, 1.3, 1.7, 1.0, "文本 token\n+ <boi>...<eoi>\n(词表 151552)", fs=9.5)
box(ax, 2.2, 1.3, 1.6, 1.0, "词嵌入\n(4096)\ntie=False", fs=9.5)
box(ax, 4.4, 0.9, 3.4, 1.8,
    "N=40 × GLM-4DecoderLayer\n（文本塔 = GLM-4-9B）\n4×RMSNorm 双残差 · GQA 32Q/2KV\npartial rotary 0.5 · SiGLU",
    fc=CORE, ec=ACCENT, fs=9.5, bold=True)
box(ax, 8.2, 1.3, 1.4, 1.0, "最终\nRMSNorm", fs=9.5)
box(ax, 9.9, 1.3, 1.5, 1.0, "LM Head\n→ 下一个 token", fc=ACCENT_SOFT, ec=ACCENT, fs=9.5, bold=True)

arrow(ax, (1.9, 1.8), (2.2, 1.8))
arrow(ax, (3.8, 1.8), (4.4, 1.8))
arrow(ax, (7.8, 1.8), (8.2, 1.8))
arrow(ax, (9.6, 1.8), (9.9, 1.8))

# 视觉 token 注入
arrow(ax, (10.4, 4.3), (10.4, 2.3), color=MM, lw=2.0)
ax.text(10.55, 3.3, "图像 token\n(每 tile 100 个)\n插入文本序列", color=MM, fontsize=9, va="center")

ax.text(7, 6.2, "GLM-4V-9B（2024）端到端：图像经 63 层视觉塔编码为 4096 维 token，注入 <boi>…<eoi> 位置，与文本共享 40 层 GLM-4 解码器",
        ha="center", fontsize=11.0, fontweight="bold", color=INK)
ax.text(7, 0.15,
        "视觉塔：patch=14/image=1120 → 80×80=6400 patch（num_positions=6401）；scaling_factor=8 → 合并为 10×10=100 token/tile；投影 1792→4096。文本塔与 GLM-4-9B 完全一致。",
        ha="center", fontsize=8.8, color=SUB)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "01_glm4v端到端总览.png"), dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

# ===== 图2：视觉塔 token 流（动态分辨率/tile）=====
fig, ax = plt.subplots(figsize=(14, 5.6))
ax.set_xlim(0, 14); ax.set_ylim(0, 5.6); ax.axis("off")

box(ax, 0.2, 2.2, 1.8, 1.2, "原图\n(任意比例)", fc=MM_SOFT, ec=MM_LINE, fs=10)
box(ax, 2.3, 2.2, 1.9, 1.2, "等比缩放\n按 tile 网格\n切分为多块", fc=MM_SOFT, ec=MM_LINE, fs=9.5)
box(ax, 4.5, 2.2, 1.9, 1.2, "每块 resize\n到 1120×1120\n(image_size)", fc=MM_SOFT, ec=MM_LINE, fs=9.5)
box(ax, 6.7, 2.2, 2.0, 1.2, "Patchify\n14×14 →\n80×80=6400", fc="#d6eef5", ec=MM, fs=9.5, bold=True)
box(ax, 9.0, 2.2, 1.9, 1.2, "ViT 63层\n+ 位置编码\n(1792 维)", fc="#d6eef5", ec=MM, fs=9.5, bold=True)
box(ax, 11.2, 2.2, 2.5, 1.2, "8×8 空间合并\n→ 10×10=100\ntoken/tile", fc=MM_SOFT, ec=MM_LINE, fs=9.5)

for x1, x2 in [(2.0,2.3),(4.2,4.5),(6.4,6.7),(8.7,9.0),(10.9,11.2)]:
    arrow(ax, (x1, 2.8), (x2, 2.8), color=MM)

box(ax, 4.5, 0.5, 6.7, 0.9,
    "多 tile 的 100-token 序列首尾相接，外包 <boi>(151339) … <eoi>(151340)，拼接到文本 embedding 序列对应位置",
    fc=NODE, ec=EDGE, fs=9.5)
arrow(ax, (12.45, 2.2), (11.0, 1.4), color=MM, lw=1.6)

ax.text(7, 5.1, "GLM-4V-9B 视觉塔 token 流：动态分辨率 → 多 1120 tile → patchify → ViT → 8×8 合并（100 token/tile）",
        ha="center", fontsize=11.0, fontweight="bold", color=INK)
ax.text(7, 0.08,
        "推导：1120/14=80 patch/边；scaling_factor=8 表示 8×8 patch 合并为 1 个 token，故 (80/8)²=100。num_positions=6401=80×80+1 与单 tile patch 数自洽。",
        ha="center", fontsize=8.8, color=SUB)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "02_glm4v视觉塔token流.png"), dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

# ===== 图3：文本解码层（复用 GLM-4 4×RMSNorm，标注视觉 token 同走）=====
fig, ax = plt.subplots(figsize=(13.5, 5.4))
ax.set_xlim(0, 13.5); ax.set_ylim(0, 5.4); ax.axis("off")
box(ax, 0.3, 2.2, 1.5, 1.0, "hidden\n(文本+\n图像token)", fs=9.5)
box(ax, 2.2, 3.6, 1.9, 0.9, "① input_layernorm\n(RMSNorm)", fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=9)
box(ax, 2.2, 2.4, 1.9, 0.9, "② post_self_attn\n_layernorm", fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=9)
box(ax, 4.6, 2.9, 3.4, 1.5, "Self-Attention（GQA 32Q/2KV）\npartial rotary 0.5 · 40 层\n视觉 token 与文本 token 全连接",
    fc=CORE, ec=ACCENT, fs=9.2, bold=True)
box(ax, 8.5, 3.1, 1.0, 0.7, "⊕", fs=14, bold=True)
box(ax, 2.2, 0.9, 1.9, 0.9, "③④ post_attention/\npost_mlp RMSNorm", fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=9)
box(ax, 4.6, 0.6, 3.4, 1.2, "SiGLU FFN（融合 gate_up）\ninter=13696",
    fc=CORE, ec=ACCENT, fs=9.2, bold=True)
box(ax, 8.5, 0.9, 1.0, 0.7, "⊕", fs=14, bold=True)
box(ax, 10.2, 2.2, 1.6, 1.0, "下一层 /\nLM Head", fs=9.5)
arrow(ax, (1.8, 2.7), (2.2, 3.9))
arrow(ax, (4.1, 4.05), (4.6, 3.9))
arrow(ax, (8.0, 3.65), (8.5, 3.45))
arrow(ax, (9.5, 3.45), (10.2, 3.0))
arrow(ax, (1.05, 2.2), (1.05, 1.35))
arrow(ax, (1.05, 1.35), (2.2, 1.35))
arrow(ax, (4.1, 1.35), (4.6, 1.2))
arrow(ax, (8.0, 1.2), (8.5, 1.25))
arrow(ax, (9.5, 1.25), (10.2, 2.5))
ax.text(6.75, 5.0, "GLM-4V-9B 文本解码层 = GLM-4-9B（4×RMSNorm 双残差）；图像 token 经投影后与文本 token 拼接，在同一注意力中全连接",
        ha="center", fontsize=10.5, fontweight="bold", color=INK)
ax.text(6.75, 0.05, "视觉塔只负责把图像编码成 token；跨模态融合发生在共享文本解码器的自注意力中（无独立融合层）。",
        ha="center", fontsize=8.8, color=SUB)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "03_glm4v文本解码层.png"), dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

print("done:", os.listdir(OUT))
