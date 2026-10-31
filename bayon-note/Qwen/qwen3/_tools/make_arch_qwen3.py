# -*- coding: utf-8 -*-
"""Generate Qwen3 family architecture diagrams (PNG)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = r"E:\transformers\bayon-note\Qwen\qwen3\_arch"
os.makedirs(OUT, exist_ok=True)

INK = "#23272F"; SUB = "#66707B"; EDGE = "#C9C4BA"
ACCENT = "#1D4ED8"; ACCENT_SOFT = "#E2EAFB"; ACCENT_LINE = "#C0D0F4"
MM = "#0E7490"; MM_SOFT = "#E0F2F7"
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
fig, ax = plt.subplots(figsize=(13.5, 7.2))
ax.set_xlim(0, 13.5); ax.set_ylim(0, 7.2); ax.axis("off")

box(ax, 0.3, 4.6, 1.5, 1.0, "文本 token\ninput_ids\n(词表 151936)", fc="#F4F4F2", ec=EDGE, fs=10)
box(ax, 0.3, 1.2, 1.5, 1.0, "图像/视频\npixel_values\n(仅 3-VL)", fc=MM_SOFT, ec=MM, fs=10, tc="#155E75")

box(ax, 2.4, 4.6, 1.7, 1.0, "词嵌入\nEmbed\n(tie≤4B)", fs=10)
box(ax, 2.4, 1.2, 1.7, 1.0, "视觉塔 ViT\npatch16 · depth24\nspatial_merge=2", fc=MM_SOFT, ec=MM, fs=9.5, tc="#155E75")

box(ax, 4.6, 2.6, 1.6, 1.6, "early fusion\n按 <image>\n占位位填入\n视觉 token", fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=9.5)

box(ax, 6.7, 2.4, 2.4, 2.0,
    "N × Qwen3DecoderLayer\n──────────\n前置 RMSNorm\nGQA 全注意力\n(head_dim128 · QK-Norm · RoPE θ1e6)\n+ SwiGLU FFN / 128选8 MoE",
    fc=CORE, ec=ACCENT, fs=9.5, bold=True)

box(ax, 9.7, 4.4, 1.6, 0.9, "最终\nRMSNorm", fs=10)
box(ax, 9.7, 2.7, 1.6, 0.9, "LM Head\n(≥8B 解绑)", fs=10)
box(ax, 11.7, 3.4, 1.5, 0.9, "下一 token\n(logits)", fc=ACCENT_SOFT, ec=ACCENT, fs=10, bold=True)

arrow(ax, (1.8, 5.1), (2.4, 5.1))
arrow(ax, (1.8, 1.7), (2.4, 1.7), color=MM)
arrow(ax, (4.1, 5.1), (5.4, 3.8))
arrow(ax, (4.1, 1.7), (5.4, 3.0), color=MM)
arrow(ax, (6.2, 3.4), (6.7, 3.4))
arrow(ax, (9.1, 4.0), (9.7, 4.85))
arrow(ax, (10.5, 4.4), (10.5, 3.6))
arrow(ax, (11.3, 3.15), (11.7, 3.85))

ax.text(6.75, 6.6, "Qwen3 端到端数据流：文本主干（上）+ 3-VL 视觉路径（青）early fusion",
        ha="center", fontsize=12.5, fontweight="bold", color=INK)
ax.text(6.75, 0.55, "Dense 0.6B–32B：28/28/36/36/40/64 层；MoE 30B-A3B(48层) / 235B-A22B(94层)，FFN 子层换 128 选 8 专家；无滑窗、全因果注意力",
        ha="center", fontsize=9.5, color=SUB)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "01_qwen3端到端总览.png"), dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

# ============ 图2：解码层解剖 + 核心机制 ============
fig, ax = plt.subplots(figsize=(13.5, 7.6))
ax.set_xlim(0, 13.5); ax.set_ylim(0, 7.6); ax.axis("off")

box(ax, 0.3, 3.4, 1.4, 1.0, "hidden\n(x)", fs=10)
box(ax, 2.2, 5.4, 2.0, 0.9, "① RMSNorm\n(input_layernorm)", fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=9.5)
box(ax, 2.2, 3.4, 2.0, 0.9, "② RMSNorm\n(post_attn_layernorm)", fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=9.5)

box(ax, 4.8, 5.1, 3.4, 1.5,
    "GQA 全注意力\nq/k 投影 → q_norm/k_norm(head_dim) → RoPE(θ1e6)\nrepeat_kv(KV=8 或 4) → softmax(QK^T/√128)V → o_proj",
    fc=CORE, ec=ACCENT, fs=9.5, bold=True)

box(ax, 4.8, 1.9, 3.4, 1.5,
    "FFN 子层（二选一）\nDense: SwiGLU( gate·up ) → down\nMoE: TopKRouter(128选8) → 专家桶 → 加权合并",
    fc=CORE, ec=ACCENT, fs=9.5, bold=True)

box(ax, 8.9, 5.3, 1.2, 0.8, "⊕", fs=14, bold=True)
box(ax, 8.9, 2.0, 1.2, 0.8, "⊕", fs=14, bold=True)
box(ax, 10.8, 3.4, 1.6, 1.0, "下一层\nhidden", fs=10)

arrow(ax, (1.7, 3.9), (2.2, 5.85))
arrow(ax, (4.2, 5.85), (4.8, 5.85))
arrow(ax, (8.2, 5.85), (8.9, 5.7))
arrow(ax, (10.1, 5.7), (10.8, 4.3))
arrow(ax, (1.0, 3.4), (1.0, 2.4))
arrow(ax, (1.0, 2.4), (2.2, 3.85))
arrow(ax, (4.2, 3.85), (4.8, 2.65))
arrow(ax, (8.2, 2.65), (8.9, 2.4))
arrow(ax, (10.1, 2.4), (10.8, 3.6))

ax.text(6.75, 7.2, "Qwen3 解码层（Qwen3DecoderLayer）：单 RMSNorm 前置残差块",
        ha="center", fontsize=12.5, fontweight="bold", color=INK)
ax.text(6.75, 0.9, "attention_bias=false · sliding_window=null（全因果）· 小尺寸 tie 嵌入；MoE 档 decoder_sparse_step=1，每层 FFN 都走专家",
        ha="center", fontsize=9.5, color=SUB)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "02_qwen3解码层与核心机制.png"), dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

# ============ 图3：3-VL 视觉塔 token 流 ============
fig, ax = plt.subplots(figsize=(13.5, 6.6))
ax.set_xlim(0, 13.5); ax.set_ylim(0, 6.6); ax.axis("off")

box(ax, 0.2, 2.6, 1.5, 1.1, "图像/视频\n像素\nH×W×3", fc=MM_SOFT, ec=MM, fs=10, tc="#155E75")
box(ax, 2.1, 2.6, 1.7, 1.1, "PatchEmbed\npatch 16×16\ntemporal 2", fc=MM_SOFT, ec=MM, fs=9.5, tc="#155E75")
box(ax, 4.2, 2.4, 2.6, 1.5, "24 层双向视觉 Transformer\nhidden 1024 · heads 16 · inter 4096\ngelu_pytorch_tanh\ndeepstack 取层 [5, 11, 17]", fc=MM_SOFT, ec=MM, fs=9.5, tc="#155E75", bold=False)
box(ax, 7.2, 2.6, 1.9, 1.1, "spatial_merge\n2×2 合并\n→ 视觉 token", fc=MM_SOFT, ec=MM, fs=9.5, tc="#155E75")
box(ax, 9.5, 2.6, 1.7, 1.1, "投影到\ntext hidden\n(=2048)", fc=MM_SOFT, ec=MM, fs=9.5, tc="#155E75")
box(ax, 11.6, 2.4, 1.7, 1.5, "early fusion\n按 image=151655\n位填入文本序列", fc=ACCENT_SOFT, ec=ACCENT, fs=9.5, bold=True)

for x1, x2 in [(1.7,2.1),(3.8,4.2),(6.8,7.2),(9.1,9.5),(11.2,11.6)]:
    arrow(ax, (x1, 3.15), (x2, 3.15), color=MM if x2 < 11.2 else ACCENT)

ax.text(6.75, 5.9, "Qwen3-VL 视觉塔 → 文本融合 token 流（本地 3-VL-2B config 逐字段）",
        ha="center", fontsize=12.5, fontweight="bold", color=INK)
ax.text(6.75, 1.1, "文本塔：28 层 · 16Q/8KV · head_dim128 · rope_theta 5e6 · M-RoPE[24,20,20] · 上下文 262144；特殊 token：vs=151652 / ve=151653 / image=151655 / video=151656",
        ha="center", fontsize=9.5, color=SUB)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "03_qwen3VL视觉塔.png"), dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

print("done:", os.listdir(OUT))
