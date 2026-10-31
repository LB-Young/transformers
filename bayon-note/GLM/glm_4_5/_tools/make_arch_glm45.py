# -*- coding: utf-8 -*-
"""Generate GLM-4.5 (MoE) family architecture diagrams (PNG)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = r"E:\transformers\bayon-note\GLM\glm_4_5\_arch"
os.makedirs(OUT, exist_ok=True)

INK = "#23272F"; SUB = "#66707B"; EDGE = "#C9C4BA"
ACCENT = "#7C3AED"; ACCENT_SOFT = "#EDE7FB"; ACCENT_LINE = "#D3C4F5"
NODE = "#FFFDF9"; CORE = "#F1ECFD"

def box(ax, x, y, w, h, text, fc=NODE, ec=EDGE, fs=10.5, bold=False, tc=INK):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                       linewidth=1.3, edgecolor=ec, facecolor=fc)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", linespacing=1.45)

def arrow(ax, p1, p2, color=EDGE, lw=1.6):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13, linewidth=lw, color=color))

# ===== 图1：端到端总览 =====
fig, ax = plt.subplots(figsize=(14, 6.4))
ax.set_xlim(0, 14); ax.set_ylim(0, 6.4); ax.axis("off")
box(ax, 0.2, 2.7, 1.5, 1.0, "token\n(词表 151552)", fs=9.5)
box(ax, 2.0, 2.7, 1.4, 1.0, "Embed\n(5120)\ntie=False", fs=9.5)
box(ax, 3.8, 3.9, 2.6, 1.0, "前 K=3 层\nDense FFN\n(first_k_dense_replace)", fc=NODE, ec=EDGE, fs=9)
box(ax, 3.8, 2.2, 2.6, 1.3,
    "其余 89 层 MoE\nRouter → top-8 / 160 专家\n+ 1 共享专家",
    fc=CORE, ec=ACCENT, fs=9.5, bold=True)
box(ax, 6.8, 2.7, 1.4, 1.0, "最终\nRMSNorm", fs=9.5)
box(ax, 8.6, 3.3, 1.5, 0.9, "LM Head\n(解绑)", fs=9.5)
box(ax, 8.6, 2.0, 1.5, 0.9, "MTP 头\nnum_nextn=1", fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=9)
box(ax, 10.6, 2.7, 1.5, 0.9, "下一 token\n(logits)", fc=ACCENT_SOFT, ec=ACCENT, fs=9.5, bold=True)
arrow(ax, (1.7, 3.2), (2.0, 3.2))
arrow(ax, (3.4, 3.2), (3.8, 4.2))
arrow(ax, (3.4, 3.2), (3.8, 2.85))
arrow(ax, (6.4, 4.4), (6.8, 3.4))
arrow(ax, (6.4, 2.85), (6.8, 3.2))
arrow(ax, (8.2, 3.2), (8.6, 3.75))
arrow(ax, (9.35, 3.3), (9.35, 2.9))
arrow(ax, (10.1, 2.45), (10.6, 2.95))
ax.text(7, 5.9, "GLM-4.5（glm4_moe）端到端：Embed → 前 K 层 Dense + 其余 MoE（Router+160专家+共享专家）→ RMSNorm → LM Head / MTP 头",
        ha="center", fontsize=11.0, fontweight="bold", color=INK)
ax.text(7, 0.5,
        "GLM-4.5：hidden 5120 · 92 层 · 96Q/8KV · ≈355B-A32B（160 路由专家 + 1 共享，top-8）；GLM-4.5-Air：4096/46层/128专家 ≈106B-A12B；GLM-4.6：同 4.5，max_position=202752",
        ha="center", fontsize=8.8, color=SUB)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "01_glm45端到端总览.png"), dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

# ===== 图2：MoE 解码层解剖 =====
fig, ax = plt.subplots(figsize=(14, 7.2))
ax.set_xlim(0, 14); ax.set_ylim(0, 7.2); ax.axis("off")
box(ax, 0.3, 3.2, 1.3, 1.0, "hidden\n(x)", fs=10)
# 注意力子层（前置 RMSNorm + QK-Norm）
box(ax, 2.0, 5.5, 1.9, 0.9, "① input_layernorm\n(RMSNorm 前置)", fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=9)
box(ax, 4.3, 5.4, 4.0, 1.3,
    "Self-Attention（GQA 96Q/8KV）\nQ/K 先过 Glm4MoeRMSNorm(head_dim)  ← QK-Norm\npartial rotary 0.5 · attention_bias",
    fc=CORE, ec=ACCENT, fs=9.2, bold=True)
box(ax, 8.8, 5.6, 1.0, 0.7, "⊕", fs=14, bold=True)
# MoE FFN 子层
box(ax, 2.0, 2.0, 1.9, 0.9, "② post_attention_\nlayernorm（前置）", fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=9)
box(ax, 4.3, 3.5, 1.7, 0.9, "Router\n(TopkRouter)\nsigmoid scores", fc=NODE, ec=EDGE, fs=9)
box(ax, 6.4, 3.4, 3.4, 1.3,
    "路由专家 160 选 top-8（num_experts_per_tok=8）\nweights = sigmoid(router(x))\nnorm_topk_prob · × routed_scaling_factor(2.5)\nexperts: Glm4MoeMLP(moe_inter=1536)",
    fc=CORE, ec=ACCENT, fs=9.0, bold=True)
box(ax, 6.4, 1.9, 3.4, 0.9,
    "共享专家 = Glm4MoeMLP(moe_inter×n_shared)\n输出 = Σ(top8 w_i·E_i(x)) + Shared(x)",
    fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=9.0)
box(ax, 10.2, 2.6, 1.0, 0.7, "⊕", fs=14, bold=True)
box(ax, 11.6, 3.0, 1.6, 1.0, "下一层\nhidden", fs=10)

arrow(ax, (1.6, 3.7), (2.0, 5.8))
arrow(ax, (3.9, 5.95), (4.3, 6.0))
arrow(ax, (8.3, 6.05), (8.8, 5.95))
arrow(ax, (9.8, 5.95), (11.6, 3.8))
arrow(ax, (1.0, 3.2), (1.0, 2.45))
arrow(ax, (1.0, 2.45), (2.0, 2.45))
arrow(ax, (3.9, 2.45), (4.3, 3.7))
arrow(ax, (6.0, 3.9), (6.4, 4.0))
arrow(ax, (6.0, 3.7), (6.4, 2.6))
arrow(ax, (9.8, 3.0), (10.2, 2.9))
arrow(ax, (11.2, 2.95), (11.6, 3.4))
ax.text(7, 6.85, "GLM-4.5 MoE 解码层（Glm4MoeDecoderLayer）：仅 2×RMSNorm 前置（简化自 GLM-4 的 4×）；注意力带 QK-Norm；FFN = top-8 路由专家 + 1 共享专家",
        ha="center", fontsize=10.5, fontweight="bold", color=INK)
ax.text(7, 0.9,
        "Router 用 sigmoid（非 softmax）打分 → top-8 选择 → norm_topk_prob 归一 → 乘 routed_scaling_factor；共享专家以残差方式与路由专家输出相加。"
        "前 first_k_dense_replace=3 层仍用 dense MLP。",
        ha="center", fontsize=8.8, color=SUB)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "02_glm45MoE解码层.png"), dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

print("done:", os.listdir(OUT))
