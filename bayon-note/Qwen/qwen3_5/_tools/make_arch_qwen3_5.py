# -*- coding: utf-8 -*-
"""Generate Qwen3.5 family architecture diagrams (PNG)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = r"E:\transformers\bayon-note\Qwen\qwen3_5\_arch"
os.makedirs(OUT, exist_ok=True)

INK = "#23272F"; SUB = "#66707B"; EDGE = "#C9C4BA"
ACCENT = "#1D4ED8"; ACCENT_SOFT = "#E2EAFB"; ACCENT_LINE = "#C0D0F4"
MM = "#0E7490"; MM_SOFT = "#E0F2F7"
NODE = "#FFFDF9"; CORE = "#EEF3FD"
LIN = "#FDEBD0"; LIN_EDGE = "#E59866"   # linear attention orange

def box(ax, x, y, w, h, text, fc=NODE, ec=EDGE, fs=10.5, bold=False, tc=INK):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                       linewidth=1.3, edgecolor=ec, facecolor=fc)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", linespacing=1.45)

def arrow(ax, p1, p2, color=EDGE, lw=1.6):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13,
                        linewidth=lw, color=color)
    ax.add_patch(a)

# ============ 图1：端到端 + 3:1 层排程 ============
fig, ax = plt.subplots(figsize=(13.5, 7.4))
ax.set_xlim(0, 13.5); ax.set_ylim(0, 7.4); ax.axis("off")

box(ax, 0.2, 4.7, 1.5, 1.0, "文本 token\n(input_ids)\n词表 248320", fs=10)
box(ax, 0.2, 1.4, 1.5, 1.0, "图像/视频\npixel_values\n(原生视觉)", fc=MM_SOFT, ec=MM, fs=9.5, tc="#155E75")
box(ax, 2.1, 4.7, 1.6, 1.0, "词嵌入\nEmbed", fs=10)
box(ax, 2.1, 1.4, 1.6, 1.0, "视觉塔 ViT\npatch16 · depth12/27\nspatial_merge=2", fc=MM_SOFT, ec=MM, fs=9, tc="#155E75")
box(ax, 4.1, 2.9, 1.5, 1.2, "early fusion\n按 image=248056\n占位位填入", fc=ACCENT_SOFT, ec=ACCENT, fs=9)

# layer schedule strip
box(ax, 6.1, 4.9, 6.9, 1.4, "", fc="#ffffff", ec=EDGE)
ax.text(9.55, 6.0, "N 层 = 每 4 层一组：3×线性 + 1×全注意力（layer_types）", ha="center", fontsize=10.5, fontweight="bold", color=INK)
xs = [6.35, 7.75, 9.15, 10.55]
labs = ["L: Gated\nDeltaNet", "L: Gated\nDeltaNet", "L: Gated\nDeltaNet", "F: Gated\nAttention"]
fcs = [LIN, LIN, LIN, CORE]
ecs = [LIN_EDGE, LIN_EDGE, LIN_EDGE, ACCENT]
for i, (xx, lb, fc, ec) in enumerate(zip(xs, labs, fcs, ecs)):
    box(ax, xx, 5.05, 1.15, 0.8, lb, fc=fc, ec=ec, fs=8.5)

box(ax, 6.1, 2.6, 6.9, 1.7,
    "每层后接：RMSNorm → SwiGLU FFN（Dense） / 共享专家+256选8路由专家（MoE）\n"
    "L 层：递归状态 O(n)；F 层：GQA + 输出门 sigmoid(gate) + RoPE 64 维",
    fc="#ffffff", ec=EDGE, fs=9.5)

box(ax, 12.05, 3.0, 1.35, 0.9, "最终\nRMSNorm", fs=9.5)
box(ax, 12.05, 1.7, 1.35, 0.9, "LM Head\n→ 下一 token", fs=9.5)

arrow(ax, (1.7, 5.2), (2.1, 5.2))
arrow(ax, (1.7, 1.9), (2.1, 1.9), color=MM)
arrow(ax, (3.7, 5.2), (4.85, 3.9))
arrow(ax, (3.7, 1.9), (4.85, 3.2), color=MM)
arrow(ax, (5.6, 3.5), (6.1, 3.5))
arrow(ax, (13.0, 4.3), (12.72, 3.9))
arrow(ax, (12.72, 3.0), (12.72, 2.6))

ax.text(6.75, 7.05, "Qwen3.5 端到端：文本主干 + 原生视觉 early fusion，3:1 混合注意力层排程",
        ha="center", fontsize=12.5, fontweight="bold", color=INK)
ax.text(6.75, 0.55, "0.8B: 24层/1024；9B: 32层/4096；27B: 64层/5120/inter17408；35B-A3B: 40层/2048/256选8+共享专家；122B-A10B、397B-A17B: MoE 旗舰",
        ha="center", fontsize=9, color=SUB)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "01_qwen3_5端到端总览.png"), dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

# ============ 图2：两种注意力单元解剖 ============
fig, ax = plt.subplots(figsize=(13.5, 7.6))
ax.set_xlim(0, 13.5); ax.set_ylim(0, 7.6); ax.axis("off")

# left: Gated DeltaNet
box(ax, 0.3, 0.6, 6.2, 6.2, "", fc="#FDF6EE", ec=LIN_EDGE)
ax.text(3.4, 6.4, "Gated DeltaNet（线性注意力，3/4 的层）", ha="center", fontsize=11.5, fontweight="bold", color="#9C5410")
steps_l = [
    ("x → in_proj_qkv 融合 QKV", 5.6),
    ("depthwise Conv1d kernel=4 + silu", 5.0),
    ("q,k,v 分头（K 头=16, V 头=16/32, head_dim=128）", 4.4),
    ("beta=sigmoid(b) 输入门；g=-exp(A_log)*softplus(a+dt) 衰减门", 3.7),
    ("递归状态 S = g*S + beta*(v ⊗ k)；out = (q⊗S)*RMSNormGated(z)", 2.9),
    ("out_proj → hidden（训练 chunk / 解码 recurrent 双实现）", 2.1),
]
for t, yy in steps_l:
    ax.text(3.4, yy, t, ha="center", fontsize=9.3, color=INK)

# right: Gated Attention
box(ax, 7.0, 0.6, 6.2, 6.2, "", fc="#F0F4FD", ec=ACCENT)
ax.text(10.1, 6.4, "Gated Attention（全注意力，1/4 的层）", ha="center", fontsize=11.5, fontweight="bold", color=ACCENT)
steps_r = [
    ("x → q_proj（输出 hidden*head_dim*2）", 5.6),
    ("chunk 成 (query, gate) —— 输出门", 5.0),
    ("q_norm/k_norm 逐头 RMSNorm(head_dim=256)", 4.4),
    ("RoPE（partial_rotary 0.25 = 64 维）", 3.7),
    ("GQA repeat_kv（0.8B 8Q/2KV；35B 16Q/2KV；27B 24Q/4KV）", 2.9),
    ("softmax(QK^T/√256)V → * sigmoid(gate) → o_proj", 2.1),
]
for t, yy in steps_r:
    ax.text(10.1, yy, t, ha="center", fontsize=9.3, color=INK)

ax.text(6.75, 7.25, "Qwen3.5 两种 Token Mixer 解剖（modeling_qwen3_5.py:490-808）",
        ha="center", fontsize=12.5, fontweight="bold", color=INK)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "02_qwen3_5混合注意力解剖.png"), dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

print("done:", os.listdir(OUT))
