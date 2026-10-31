# -*- coding: utf-8 -*-
"""Generate GLM-Z1 (reasoning, GLM-4 Dense backbone) architecture diagrams (PNG)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = r"E:\transformers\bayon-note\GLM\glm_z1\_arch"
os.makedirs(OUT, exist_ok=True)

INK = "#23272F"; SUB = "#66707B"; EDGE = "#C9C4BA"
ACCENT = "#B45309"; ACCENT_SOFT = "#FBEFD9"; ACCENT_LINE = "#EFD3A0"
NODE = "#FFFDF9"; CORE = "#FDF3E3"

def box(ax, x, y, w, h, text, fc=NODE, ec=EDGE, fs=10.5, bold=False, tc=INK):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                       linewidth=1.3, edgecolor=ec, facecolor=fc)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", linespacing=1.45)

def arrow(ax, p1, p2, color=EDGE, lw=1.6):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13, linewidth=lw, color=color))

# ===== 图1：Z1 家族总览（= GLM-4 Dense 0414 权重 + 长思维链训练）=====
fig, ax = plt.subplots(figsize=(14, 6.6))
ax.set_xlim(0, 14); ax.set_ylim(0, 6.6); ax.axis("off")
box(ax, 0.2, 3.0, 1.6, 1.0, "prompt token\n(词表 151552)", fs=9.5)
box(ax, 2.1, 3.0, 1.4, 1.0, "Embed\n(解绑)", fs=9.5)
box(ax, 3.9, 2.6, 3.0, 1.8,
    "GLM-4 Dense 解码器（同 glm4 原生模块）\nZ1-9B: 4096·40层·32Q/2KV\nZ1-32B: 6144·61层·48Q/2KV\n4×RMSNorm 双残差 · partial rotary 0.5",
    fc=CORE, ec=ACCENT, fs=9.2, bold=True)
box(ax, 7.3, 3.0, 1.4, 1.0, "最终\nRMSNorm", fs=9.5)
box(ax, 9.1, 3.0, 1.5, 1.0, "LM Head\n→ 自回归续写", fc=ACCENT_SOFT, ec=ACCENT, fs=9.5, bold=True)
arrow(ax, (1.8, 3.5), (2.1, 3.5))
arrow(ax, (3.5, 3.5), (3.9, 3.5))
arrow(ax, (6.9, 3.5), (7.3, 3.5))
arrow(ax, (8.7, 3.5), (9.1, 3.5))
ax.text(11.6, 4.2, "训练目标：\n长思维链（CoT）\n强化学习微调\n（不改架构）", ha="center", va="center",
        fontsize=9.5, color=ACCENT, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", fc=NODE, ec=ACCENT_LINE))
ax.text(7, 6.1, "GLM-Z1 家族（reasoning）端到端：架构 = GLM-4 Dense（Z1-9B=GLM-4-9B-0414；Z1-32B=GLM-4-32B-0414）",
        ha="center", fontsize=11.0, fontweight="bold", color=INK)
ax.text(7, 0.6,
        "config 逐项核验：Z1-9B 与 GLM-4-9B-0414 相同；Z1-32B 与 GLM-4-32B-0414 相同（hidden 6144/61层/48Q·2KV/inter 23040/max 32768/rope_theta 1e4/attention_bias=false）。差异仅在长思维链推理训练。",
        ha="center", fontsize=8.6, color=SUB)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "01_glmz1端到端总览.png"), dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

# ===== 图2：推理时 CoT 生成流（行为示意，非新结构）=====
fig, ax = plt.subplots(figsize=(14, 4.8))
ax.set_xlim(0, 14); ax.set_ylim(0, 4.8); ax.axis("off")
box(ax, 0.3, 1.9, 1.9, 1.1, "用户问题\n(prompt)", fs=10)
box(ax, 2.8, 1.9, 3.2, 1.1,
    "长思维链 token 序列\n(reasoning / CoT)\n数百~数千步自推理",
    fc=CORE, ec=ACCENT, fs=9.8, bold=True)
box(ax, 6.5, 1.9, 2.2, 1.1, "思维结束标记\n（待确认 tag）", fc=NODE, ec=EDGE, fs=9.5)
box(ax, 9.2, 1.9, 2.2, 1.1, "最终答案\n(answer)", fc=ACCENT_SOFT, ec=ACCENT, fs=10, bold=True)
arrow(ax, (2.2, 2.45), (2.8, 2.45))
arrow(ax, (6.0, 2.45), (6.5, 2.45))
arrow(ax, (8.7, 2.45), (9.2, 2.45))
ax.text(7, 4.2, "GLM-Z1 推理时行为（自回归；架构与 GLM-4 完全一致，不新增算子/层）",
        ha="center", fontsize=11.0, fontweight="bold", color=INK)
ax.text(7, 0.6,
        "Z1 是推理模型：用 GLM-4 Dense 权重做长思维链训练。具体 CoT 起止特殊 token id 未在 config 显式给出，标注「待确认」（以官方 chat template / model card 为准）。",
        ha="center", fontsize=8.8, color=SUB)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "02_glmz1推理CoT流.png"), dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

print("done:", os.listdir(OUT))
