# -*- coding: utf-8 -*-
"""Generate DeepSeek-V2 family architecture diagrams (PNG)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = r"E:\transformers\bayon-note\DeepSeek\deepseek_v2\_arch"
os.makedirs(OUT, exist_ok=True)

INK = "#23272F"; SUB = "#66707B"; EDGE = "#C9C4BA"
ACCENT = "#1D4ED8"; ACCENT_SOFT = "#E2EAFB"; ACCENT_LINE = "#C0D0F4"
NODE = "#FFFDF9"; CORE = "#EEF3FD"

def box(ax, x, y, w, h, text, fc=NODE, ec=EDGE, fs=10.5, bold=False, tc=INK):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                       linewidth=1.3, edgecolor=ec, facecolor=fc)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", linespacing=1.45)

def arrow(ax, p1, p2, color=EDGE, lw=1.6, style="-|>"):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=13,
                        linewidth=lw, color=color)
    ax.add_patch(a)

# ============ 图1：端到端总览 ============
fig, ax = plt.subplots(figsize=(13.5, 7.0))
ax.set_xlim(0, 13.5); ax.set_ylim(0, 7.0); ax.axis("off")

box(ax, 0.2, 3.1, 1.4, 1.0, "文本 token\n(词表 102400)", fc="#F4F4F2", ec=EDGE, fs=10)
box(ax, 1.9, 3.1, 1.4, 1.0, "词嵌入\n5120\ntie=false", fs=10)
box(ax, 3.6, 2.5, 2.9, 2.2,
    "60 × DecoderLayer\n──────────\nRMSNorm 前置残差块\nMLA 注意力（KV 压缩 576 维）\n第0层: dense SwiGLU(inter12288)\n第1-59层: DeepSeekMoE\n(160选6 + 共享专家2)",
    fc=CORE, ec=ACCENT, fs=9, bold=True)
box(ax, 6.9, 3.3, 1.3, 0.9, "最终\nRMSNorm", fs=10)
box(ax, 8.5, 3.3, 1.5, 0.9, "LM Head\n(tie=false)", fs=10)
box(ax, 10.4, 3.3, 1.5, 0.9, "下一 token\n(logits)", fc=ACCENT_SOFT, ec=ACCENT, fs=10, bold=True)

arrow(ax, (1.6, 3.6), (1.9, 3.6))
arrow(ax, (3.3, 3.6), (3.6, 3.6))
arrow(ax, (6.5, 3.6), (6.9, 3.75))
arrow(ax, (8.2, 3.75), (8.5, 3.75))
arrow(ax, (10.0, 3.75), (10.4, 3.75))

ax.text(6.75, 6.4, "DeepSeek-V2 端到端数据流：MLA + DeepSeekMoE（236B-A21B）",
        ha="center", fontsize=12.5, fontweight="bold", color=INK)
ax.text(6.75, 1.4, "hidden 5120 · 60 层 · 128 Q 头 · kv_lora 512 · q_lora 1536 · qk nope128+rope64 · v_head128",
        ha="center", fontsize=9.5, color=SUB)
ax.text(6.75, 0.85, "上下文 163840（YaRN factor=40）· routed_scaling_factor=16.0 · aux_loss_alpha=0.001 · V2.5 仅多 ep_size=1 与 mscale 差异",
        ha="center", fontsize=9, color=SUB)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "01_deepseek_v2端到端总览.png"), dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

# ============ 图2：解码层解剖 + MLA/MoE 核心机制 ============
fig, ax = plt.subplots(figsize=(13.5, 7.6))
ax.set_xlim(0, 13.5); ax.set_ylim(0, 7.6); ax.axis("off")

box(ax, 0.2, 3.4, 1.3, 1.0, "hidden\n(x)", fs=10)
box(ax, 1.9, 5.6, 1.7, 0.8, "① RMSNorm\n(eps1e-6)", fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=9)

box(ax, 4.0, 5.0, 4.2, 1.9,
    "MLA 多头潜在注意力\nQ: q_a(5120→1536) → RMSNorm → q_b(→128×192)\nKV: kv_a(→576=512+64) → RMSNorm(512) → 缓存\nq_pe/k_pe 64 维 RoPE → expand_kv → softmax(QK^T/√192)V",
    fc=CORE, ec=ACCENT, fs=9, bold=True)

box(ax, 8.7, 5.5, 1.0, 0.7, "⊕", fs=14, bold=True)
box(ax, 10.3, 3.4, 1.5, 1.0, "下一层\nhidden", fs=10)

box(ax, 1.9, 2.6, 1.7, 0.8, "② RMSNorm", fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=9)
box(ax, 4.0, 1.6, 4.2, 2.2,
    "FFN 子层（layer≥1：DeepSeekMoE）\nRouter: softmax → 8组选3组 → top6 → ×16.0\n路由专家: 160 选 6（细粒度 inter1536，加权合并）\n共享专家: 2 个全时并行（inter 3072）\nlayer 0: 普通 SwiGLU dense(inter12288)",
    fc=CORE, ec=ACCENT, fs=9, bold=True)
box(ax, 8.7, 2.7, 1.0, 0.7, "⊕", fs=14, bold=True)

arrow(ax, (1.5, 3.9), (1.9, 6.0))
arrow(ax, (3.6, 6.0), (4.0, 5.95))
arrow(ax, (8.2, 5.95), (8.7, 5.85))
arrow(ax, (9.7, 5.85), (10.3, 4.3))
arrow(ax, (0.85, 3.4), (0.85, 3.0))
arrow(ax, (0.85, 3.0), (1.9, 3.0))
arrow(ax, (3.6, 3.0), (4.0, 2.7))
arrow(ax, (8.2, 2.7), (8.7, 3.05))
arrow(ax, (9.7, 3.05), (10.3, 3.9))

ax.text(6.75, 7.15, "DeepSeek-V2 解码层（DeepseekV2DecoderLayer）：MLA 注意力 + MoE FFN",
        ha="center", fontsize=12.5, fontweight="bold", color=INK)
ax.text(6.75, 0.7, "KV cache 存压缩潜在向量 512+64=576 维/token/层（论文口径 -93.3%）· 辅助负载均衡 aux_loss_alpha=0.001 · 源码：models\\deepseek_v2\\modeling_deepseek_v2.py",
        ha="center", fontsize=8.8, color=SUB)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "02_deepseek_v2解码层与核心机制.png"), dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

print("done:", os.listdir(OUT))
