# -*- coding: utf-8 -*-
"""Generate DeepSeek-LLM (v1) family architecture diagrams (PNG)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = r"E:\transformers\bayon-note\DeepSeek\deepseek_v1\_arch"
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
fig, ax = plt.subplots(figsize=(13.5, 6.8))
ax.set_xlim(0, 13.5); ax.set_ylim(0, 6.8); ax.axis("off")

box(ax, 0.3, 3.0, 1.5, 1.0, "文本 token\ninput_ids\n(词表 102400)", fc="#F4F4F2", ec=EDGE, fs=10)
box(ax, 2.3, 3.0, 1.6, 1.0, "词嵌入\nEmbedding\n(hidden 4096/8192)", fs=10)
box(ax, 4.4, 2.6, 2.6, 1.8,
    "N × DecoderLayer\n──────────\nRMSNorm 前置残差块\n自注意力：7B MHA / 67B GQA\n(head_dim128 · RoPE θ=1e4)\n+ SwiGLU FFN",
    fc=CORE, ec=ACCENT, fs=9.5, bold=True)
box(ax, 7.5, 3.3, 1.5, 0.9, "最终\nRMSNorm", fs=10)
box(ax, 9.4, 3.3, 1.6, 0.9, "LM Head\n(tie=false 解绑)", fs=10)
box(ax, 11.4, 3.3, 1.6, 0.9, "下一 token\n(logits)", fc=ACCENT_SOFT, ec=ACCENT, fs=10, bold=True)

arrow(ax, (1.8, 3.5), (2.3, 3.5))
arrow(ax, (3.9, 3.5), (4.4, 3.5))
arrow(ax, (7.0, 3.5), (7.5, 3.75))
arrow(ax, (9.0, 3.75), (9.4, 3.75))
arrow(ax, (11.0, 3.75), (11.4, 3.75))

ax.text(6.75, 6.1, "DeepSeek-LLM（一代）端到端数据流：LLaMA 式 dense 主干",
        ha="center", fontsize=12.5, fontweight="bold", color=INK)
ax.text(6.75, 1.3, "7B：hidden 4096 · 30 层 · 32Q/32KV(MHA) · inter 11008   ｜   67B：hidden 8192 · 95 层 · 64Q/8KV(GQA) · inter 22016",
        ha="center", fontsize=9.5, color=SUB)
ax.text(6.75, 0.75, "上下文 4096 · rope_theta=10000 · rms_norm_eps=1e-6 · 加载路径：transformers 标准 llama 模块（无 deepseek_v1 原生模块）",
        ha="center", fontsize=9, color=SUB)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "01_deepseek_v1端到端总览.png"), dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

# ============ 图2：解码层解剖 + 核心机制 ============
fig, ax = plt.subplots(figsize=(13.5, 7.2))
ax.set_xlim(0, 13.5); ax.set_ylim(0, 7.2); ax.axis("off")

box(ax, 0.3, 3.2, 1.4, 1.0, "hidden\n(x)", fs=10)
box(ax, 2.2, 5.2, 2.0, 0.9, "① RMSNorm\n(input_layernorm, eps1e-6)", fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=9)

box(ax, 4.8, 4.9, 3.6, 1.6,
    "自注意力子层\n7B: MHA 32Q/32KV   67B: GQA 64Q/8KV(8:1)\nRoPE(θ=1e4, head_dim128) → softmax(QK^T/√128)V → o_proj",
    fc=CORE, ec=ACCENT, fs=9.5, bold=True)

box(ax, 8.9, 5.2, 1.1, 0.7, "⊕", fs=14, bold=True)
box(ax, 10.6, 3.2, 1.6, 1.0, "下一层\nhidden", fs=10)

box(ax, 2.2, 2.2, 2.0, 0.9, "② RMSNorm\n(post_attn_layernorm)", fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=9)
box(ax, 4.8, 1.9, 3.6, 1.6,
    "FFN 子层（SwiGLU）\ndown( silu(gate) ⊙ up )\n7B inter=11008   67B inter=22016",
    fc=CORE, ec=ACCENT, fs=9.5, bold=True)
box(ax, 8.9, 2.2, 1.1, 0.7, "⊕", fs=14, bold=True)

arrow(ax, (1.7, 3.7), (2.2, 5.65))
arrow(ax, (4.2, 5.65), (4.8, 5.7))
arrow(ax, (8.4, 5.7), (8.9, 5.55))
arrow(ax, (10.0, 5.55), (10.6, 4.1))
arrow(ax, (1.0, 3.2), (1.0, 2.65))
arrow(ax, (1.0, 2.65), (2.2, 2.65))
arrow(ax, (4.2, 2.65), (4.8, 2.7))
arrow(ax, (8.4, 2.7), (8.9, 2.55))
arrow(ax, (10.0, 2.55), (10.6, 3.6))

ax.text(6.75, 6.7, "DeepSeek-LLM 解码层：单 RMSNorm 前置残差块（对应 LlamaDecoderLayer）",
        ha="center", fontsize=12.5, fontweight="bold", color=INK)
ax.text(6.75, 0.8, "无注意力偏置 · RoPE 全维旋转 128 · 词嵌入与 LM Head 解绑(tie=false) · 中文词表 102400",
        ha="center", fontsize=9.5, color=SUB)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "02_deepseek_v1解码层与核心机制.png"), dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

print("done:", os.listdir(OUT))
