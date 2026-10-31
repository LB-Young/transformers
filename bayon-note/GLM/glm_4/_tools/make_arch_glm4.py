# -*- coding: utf-8 -*-
"""Generate GLM-4 family (dense) architecture diagrams (PNG)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = r"E:\transformers\bayon-note\GLM\glm_4\_arch"
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
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=13, linewidth=lw, color=color)
    ax.add_patch(a)

# ============ 图1：端到端总览 ============
fig, ax = plt.subplots(figsize=(13.5, 6.8))
ax.set_xlim(0, 13.5); ax.set_ylim(0, 6.8); ax.axis("off")

box(ax, 0.3, 2.9, 1.5, 1.0, "文本 token\ninput_ids\n(词表 151552)", fc="#F4F4F2", ec=EDGE, fs=10)
box(ax, 2.3, 2.9, 1.5, 1.0, "词嵌入\nEmbed\n(4096)\ntie=False", fs=10)
box(ax, 4.3, 2.2, 3.0, 2.4,
    "N × GLM-4DecoderLayer\n────────────\n4×RMSNorm 双残差块\nGQA 32Q / 2KV · head_dim 128\npartial rotary 0.5（仅前64维）\nQKV bias（9B）· SiGLU",
    fc=CORE, ec=ACCENT, fs=9.5, bold=True)
box(ax, 7.8, 3.1, 1.5, 0.9, "最终\nRMSNorm", fs=10)
box(ax, 7.8, 1.9, 1.5, 0.9, "LM Head\n(解绑)", fs=10)
box(ax, 9.9, 2.5, 1.6, 0.9, "下一 token\n(logits)", fc=ACCENT_SOFT, ec=ACCENT, fs=10, bold=True)

arrow(ax, (1.8, 3.4), (2.3, 3.4))
arrow(ax, (3.8, 3.4), (4.3, 3.4))
arrow(ax, (7.3, 3.4), (7.8, 3.55))
arrow(ax, (8.55, 3.1), (8.55, 2.8))
arrow(ax, (9.3, 2.35), (9.9, 2.85))

ax.text(6.75, 6.2, "GLM-4 家族（Dense）端到端数据流：Embed → N×DecoderLayer → RMSNorm → LM Head",
        ha="center", fontsize=12.5, fontweight="bold", color=INK)
ax.text(6.75, 0.6,
        "GLM-4-9B（2024-01）：hidden 4096 · 40 层 · 32Q/2KV · inter 13696 · add_qkv_bias · 基座 seq 8192（推理扩 128K）；"
        "GLM-4-32B-0414（2025-04）：hidden 6144 · 61 层 · 48Q/2KV · inter 23040 · attention_bias=false · rope_theta=1e4",
        ha="center", fontsize=9.0, color=SUB)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "01_glm4端到端总览.png"), dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

# ============ 图2：解码层解剖（4×RMSNorm 双残差）+ partial rotary ============
fig, ax = plt.subplots(figsize=(13.5, 7.4))
ax.set_xlim(0, 13.5); ax.set_ylim(0, 7.4); ax.axis("off")

box(ax, 0.3, 3.3, 1.3, 0.9, "hidden\n(x)", fs=10)
# 注意力子层
box(ax, 2.1, 5.4, 1.9, 0.8, "① input_layernorm\n(RMSNorm)", fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=9)
box(ax, 2.1, 4.2, 1.9, 0.8, "② post_self_attn\n_layernorm", fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=9)
box(ax, 4.6, 4.7, 3.6, 1.5,
    "Self-Attention（GQA 32Q / 2KV）\nq/k/v_proj（9B 带 bias）→ partial rotary 0.5\n→ repeat_kv → softmax(QK^T/√128)V → o_proj",
    fc=CORE, ec=ACCENT, fs=9.2, bold=True)
box(ax, 8.9, 5.1, 1.1, 0.7, "⊕", fs=14, bold=True)
# FFN 子层
box(ax, 2.1, 2.6, 1.9, 0.8, "③ post_attention\n_layernorm", fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=9)
box(ax, 2.1, 1.3, 1.9, 0.8, "④ post_mlp\n_layernorm", fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=9)
box(ax, 4.6, 1.7, 3.6, 1.5,
    "SiGLU FFN（融合 gate_up_proj）\nup_states = chunk(gate_up(x))\nffn = down( up ⊙ silu(gate) )",
    fc=CORE, ec=ACCENT, fs=9.2, bold=True)
box(ax, 8.9, 1.9, 1.1, 0.7, "⊕", fs=14, bold=True)
box(ax, 10.7, 3.3, 1.6, 0.9, "下一层\nhidden", fs=10)

arrow(ax, (1.6, 3.75), (2.1, 5.75))
arrow(ax, (4.0, 5.8), (4.6, 5.6))
arrow(ax, (8.2, 5.6), (8.9, 5.45))
arrow(ax, (10.0, 5.45), (10.7, 4.0))
arrow(ax, (1.0, 3.3), (1.0, 3.0))
arrow(ax, (1.0, 3.0), (2.1, 3.0))
arrow(ax, (4.0, 3.0), (4.6, 2.45))
arrow(ax, (8.2, 2.45), (8.9, 2.25))
arrow(ax, (10.0, 2.25), (10.7, 3.5))

ax.text(6.75, 6.9, "GLM-4 解码层（Glm4DecoderLayer）：4×RMSNorm「pre+post」双残差块（ChatGLM 血统）",
        ha="center", fontsize=12.0, fontweight="bold", color=INK)
ax.text(6.75, 0.45,
        "partial rotary：head_dim=128，partial_rotary_factor=0.5 → 仅前 64 维施加 RoPE（interleave 方式），后 64 维原样透传；"
        "SiGLU：gate_up_proj 融合，chunk(2) 拆分 gate/up；9B QKV 带 bias、32B-0414 去 bias",
        ha="center", fontsize=9.0, color=SUB)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "02_glm4解码层与核心机制.png"), dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

print("done:", os.listdir(OUT))
