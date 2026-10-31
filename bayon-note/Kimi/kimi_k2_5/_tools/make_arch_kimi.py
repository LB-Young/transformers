# -*- coding: utf-8 -*-
"""Generate Kimi K2.5 / K2-Thinking / K2-VL architecture diagrams (PNG)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = r"E:\transformers\bayon-note\Kimi"

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

def new_canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax

def save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, name), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

# ============ A. K2.5 端到端总览 ============
d = os.path.join(ROOT, "kimi_k2_5", "_arch")
fig, ax = new_canvas(13.5, 7.2)
box(ax, 0.3, 4.7, 1.5, 1.0, "文本 token\ninput_ids\n(词表 163840)", fc="#F4F4F2", ec=EDGE, fs=10)
box(ax, 0.3, 1.3, 1.5, 1.0, "图像/视频\npixel_values\n(原生分辨率)", fc=MM_SOFT, ec=MM, fs=10, tc="#155E75")
box(ax, 2.4, 4.7, 1.6, 1.0, "词嵌入\nembed_tokens", fs=10)
box(ax, 2.4, 1.3, 1.6, 1.0, "MoonViT-3D\npatch14·27层·1152\n轴向2D-RoPE", fc=MM_SOFT, ec=MM, fs=9.5, tc="#155E75")
box(ax, 4.5, 1.3, 1.7, 1.0, "PatchMerger\n(2,2)时空池化\n→投影7168", fc=MM_SOFT, ec=MM, fs=9.5, tc="#155E75")
box(ax, 6.6, 3.0, 1.6, 1.5, "early fusion\n按 image=163605\nvideo=163840\n位 masked_scatter", fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=9.5)
box(ax, 8.7, 2.8, 2.6, 2.0,
    "61 × K2 MoE 解码层\n──────────\n前置 RMSNorm\nMLA 注意力(512/1536 低秩)\n第0层Dense / 1-60层 MoE\n384 routed+1 shared · top8",
    fc=CORE, ec=ACCENT, fs=9.5, bold=True)
box(ax, 11.7, 4.5, 1.4, 0.9, "最终\nRMSNorm", fs=10)
box(ax, 11.7, 2.9, 1.4, 0.9, "LM Head\n(INT4发布)", fs=10)
box(ax, 11.7, 1.3, 1.4, 0.9, "下一 token\n(logits)", fc=ACCENT_SOFT, ec=ACCENT, fs=10, bold=True)
arrow(ax, (1.8, 5.2), (2.4, 5.2))
arrow(ax, (1.8, 1.8), (2.4, 1.8), color=MM)
arrow(ax, (4.0, 1.8), (4.5, 1.8), color=MM)
arrow(ax, (6.2, 1.8), (7.0, 3.0), color=MM)
arrow(ax, (4.0, 5.2), (6.6, 3.8))
arrow(ax, (8.2, 3.75), (8.7, 3.75))
arrow(ax, (11.3, 4.2), (11.7, 4.95))
arrow(ax, (12.4, 4.5), (12.4, 3.8))
arrow(ax, (12.4, 2.9), (12.4, 2.2))
ax.text(6.75, 6.6, "Kimi K2.5 端到端数据流：K2 MoE 文本塔（上）+ MoonViT-3D 视觉路径（青）early fusion",
        ha="center", fontsize=12.5, fontweight="bold", color=INK)
ax.text(6.75, 0.55, "文本塔 hidden7168·61层·64Q/64KV·MLA·384+1专家·top8·YaRN64·256K；开源权重 compressed-tensors int4 group32",
        ha="center", fontsize=9.5, color=SUB)
save(fig, d, "01_k25端到端总览.png")

# ============ B. K2.5 解码层 + MoE ============
fig, ax = new_canvas(13.5, 7.6)
box(ax, 0.3, 3.4, 1.3, 1.0, "hidden\n(x)\n7168", fs=10)
box(ax, 2.0, 5.5, 1.9, 0.9, "① RMSNorm\ninput_layernorm", fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=9.5)
box(ax, 2.0, 3.3, 1.9, 0.9, "② RMSNorm\npost_attn_norm", fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=9.5)
box(ax, 4.4, 5.1, 3.6, 1.6,
    "MLA 自注意力（全因果）\nq_a(7168→1536)→RMSNorm→q_b(→64×192)\nkv_a(→512+64)→缓存潜向量→kv_b展开\nRoPE 仅加在 64 维 rope 部分",
    fc=CORE, ec=ACCENT, fs=9.5, bold=True)
box(ax, 4.4, 1.7, 3.6, 1.6,
    "FFN 子层（按层号二选一）\n第0层: Dense SwiGLU (inter 18432)\n1-60层: TopKRouter sigmoid→8/384专家\n+ 常驻共享专家(inter 2048)",
    fc=CORE, ec=ACCENT, fs=9.5, bold=True)
box(ax, 8.6, 5.4, 1.0, 0.8, "⊕", fs=14, bold=True)
box(ax, 8.6, 1.9, 1.0, 0.8, "⊕", fs=14, bold=True)
box(ax, 10.4, 3.4, 1.5, 1.0, "下一层\nhidden\n7168", fs=10)
arrow(ax, (1.6, 3.9), (2.0, 5.95))
arrow(ax, (3.9, 5.95), (4.4, 5.95))
arrow(ax, (8.0, 5.9), (8.6, 5.8))
arrow(ax, (9.6, 5.8), (10.4, 4.2))
arrow(ax, (0.95, 3.4), (0.95, 2.6))
arrow(ax, (0.95, 2.6), (2.0, 3.75))
arrow(ax, (3.9, 3.75), (4.4, 2.5))
arrow(ax, (8.0, 2.5), (8.6, 2.3))
arrow(ax, (9.6, 2.3), (10.4, 3.6))
ax.text(6.75, 7.2, "Kimi K2.5 文本塔解码层（DeepseekV3DecoderLayer）：pre-norm 双残差",
        ha="center", fontsize=12.5, fontweight="bold", color=INK)
ax.text(6.75, 0.8, "路由 routed_scaling_factor=2.827 · noaux_tc 无辅助损失 · aux_loss_alpha=0.0 · 第0层 dense 其余 MoE",
        ha="center", fontsize=9.5, color=SUB)
save(fig, d, "02_k25解码层与MoE.png")

# ============ C. K2-Thinking 端到端 ============
d = os.path.join(ROOT, "kimi_k2_thinking", "_arch")
fig, ax = new_canvas(13.5, 6.4)
box(ax, 0.3, 2.6, 1.6, 1.1, "文本 token\ninput_ids\n(词表 163840)", fc="#F4F4F2", ec=EDGE, fs=10)
box(ax, 2.4, 2.6, 1.6, 1.1, "词嵌入\nembed_tokens\n(tie=false)", fs=10)
box(ax, 4.5, 2.4, 2.6, 1.5,
    "61 × MoE 解码层\n──────────\nRMSNorm · MLA 注意力\n第0层Dense / 1-60层 MoE\n384 routed+1 shared · top8",
    fc=CORE, ec=ACCENT, fs=9.5, bold=True)
box(ax, 7.6, 3.2, 1.5, 0.9, "最终\nRMSNorm", fs=10)
box(ax, 7.6, 1.9, 1.5, 0.9, "LM Head\n(与嵌入解绑)", fs=10)
box(ax, 9.6, 2.5, 1.6, 1.0, "下一 token\n(logits)", fc=ACCENT_SOFT, ec=ACCENT, fs=10, bold=True)
arrow(ax, (1.9, 3.15), (2.4, 3.15))
arrow(ax, (4.0, 3.15), (4.5, 3.15))
arrow(ax, (7.1, 3.4), (7.6, 3.65))
arrow(ax, (8.35, 3.2), (8.35, 2.8))
arrow(ax, (9.1, 2.35), (9.6, 2.9))
ax.text(6.75, 5.8, "Kimi K2 Thinking 端到端：纯文本 MoE（架构=K2-0905，差异在推理后训练）",
        ha="center", fontsize=12.5, fontweight="bold", color=INK)
ax.text(6.75, 1.0, "1.04T 总参 / 32B 激活 · hidden7168 · MLA(512/1536) · 384专家top8 · YaRN64 · 256K · INT4开源 · eos=163586",
        ha="center", fontsize=9.5, color=SUB)
save(fig, d, "01_k2thinking端到端总览.png")

# ============ D. K2-Thinking 解码层 ============
fig, ax = new_canvas(13.5, 7.2)
box(ax, 0.3, 3.4, 1.3, 1.0, "hidden\n(x)\n7168", fs=10)
box(ax, 2.0, 5.5, 1.9, 0.9, "① RMSNorm\ninput_layernorm", fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=9.5)
box(ax, 2.0, 3.3, 1.9, 0.9, "② RMSNorm\npost_attn_norm", fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=9.5)
box(ax, 4.4, 5.1, 3.6, 1.6,
    "MLA 自注意力（全因果）\nq_a(7168→1536)→RMSNorm→q_b(→64×192)\nkv_a(→512+64)→缓存潜向量→kv_b展开\nRoPE 仅加在 64 维 rope 部分",
    fc=CORE, ec=ACCENT, fs=9.5, bold=True)
box(ax, 4.4, 1.7, 3.6, 1.6,
    "FFN 子层（按层号二选一）\n第0层: Dense SwiGLU (inter 18432)\n1-60层: sigmoid路由→8/384专家\n+ 常驻共享专家(inter 2048)",
    fc=CORE, ec=ACCENT, fs=9.5, bold=True)
box(ax, 8.6, 5.4, 1.0, 0.8, "⊕", fs=14, bold=True)
box(ax, 8.6, 1.9, 1.0, 0.8, "⊕", fs=14, bold=True)
box(ax, 10.4, 3.4, 1.5, 1.0, "下一层\nhidden\n7168", fs=10)
arrow(ax, (1.6, 3.9), (2.0, 5.95))
arrow(ax, (3.9, 5.95), (4.4, 5.95))
arrow(ax, (8.0, 5.9), (8.6, 5.8))
arrow(ax, (9.6, 5.8), (10.4, 4.2))
arrow(ax, (0.95, 3.4), (0.95, 2.6))
arrow(ax, (0.95, 2.6), (2.0, 3.75))
arrow(ax, (3.9, 3.75), (4.4, 2.5))
arrow(ax, (8.0, 2.5), (8.6, 2.3))
arrow(ax, (9.6, 2.3), (10.4, 3.6))
ax.text(6.75, 6.9, "Kimi K2 Thinking 解码层（DeepseekV3DecoderLayer）：与 K2-0905/K2.5 文本塔同构",
        ha="center", fontsize=12.5, fontweight="bold", color=INK)
ax.text(6.75, 0.8, "无视觉塔 · 无线性注意力 · routed_scaling_factor=2.827 · aux_loss_alpha=0.0 · beta_fast=1.0",
        ha="center", fontsize=9.5, color=SUB)
save(fig, d, "02_k2thinking解码层与MoE.png")

# ============ E. K2.5-VL 端到端 ============
d = os.path.join(ROOT, "kimi_k2_vl", "_arch")
fig, ax = new_canvas(13.5, 6.6)
box(ax, 0.2, 2.6, 1.5, 1.1, "文本 token\n词表 163840", fc="#F4F4F2", ec=EDGE, fs=10)
box(ax, 2.0, 2.6, 1.5, 1.1, "embed\n→7168", fs=10)
box(ax, 0.2, 4.6, 1.5, 1.1, "图像/视频\n原生分辨率\n(不切tile)", fc=MM_SOFT, ec=MM, fs=9.5, tc="#155E75")
box(ax, 2.0, 4.6, 1.7, 1.1, "MoonViT-3D\npatch14·27层\n1152·轴向2D-RoPE", fc=MM_SOFT, ec=MM, fs=9.5, tc="#155E75")
box(ax, 4.0, 4.6, 1.6, 1.1, "(2,2)时空\n均值池化\ntoken÷4", fc=MM_SOFT, ec=MM, fs=9.5, tc="#155E75")
box(ax, 5.9, 4.6, 1.7, 1.1, "PatchMerger\nLN→Linear→GELU\nLinear→7168", fc=MM_SOFT, ec=MM, fs=9.5, tc="#155E75")
box(ax, 8.0, 3.4, 1.7, 1.4, "early fusion\nmasked_scatter\n按 image=163605\n位填入", fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=9.5)
box(ax, 10.1, 3.2, 2.0, 1.8,
    "61 × K2 MoE 文本塔\nMLA + 384专家 top8\nhidden 7168 · 256K",
    fc=CORE, ec=ACCENT, fs=9.5, bold=True)
arrow(ax, (1.7, 3.15), (2.0, 3.15))
arrow(ax, (1.7, 5.15), (2.0, 5.15), color=MM)
arrow(ax, (3.7, 5.15), (4.0, 5.15), color=MM)
arrow(ax, (5.6, 5.15), (5.9, 5.15), color=MM)
arrow(ax, (7.6, 5.15), (8.4, 4.8), color=MM)
arrow(ax, (3.5, 3.15), (8.0, 3.9))
arrow(ax, (9.7, 4.1), (10.1, 4.1))
ax.text(6.75, 6.1, "Kimi K2.5-VL 端到端：MoonViT-3D → PatchMerger → K2 MoE 文本塔",
        ha="center", fontsize=12.5, fontweight="bold", color=INK)
ax.text(6.75, 1.4, "视觉塔 patch14·27层·16头·4304·gelu_tanh·双向非因果 · 视频4帧共享权重+4×时间压缩",
        ha="center", fontsize=9.5, color=SUB)
save(fig, d, "01_k25vl端到端总览.png")

# ============ F. K2.5-VL 核心机制 ============
fig, ax = new_canvas(13.5, 6.6)
box(ax, 0.2, 3.6, 1.7, 1.2, "PatchEmbed\nConv2d patch14\nstride14 无偏置", fc=MM_SOFT, ec=MM, fs=9.5, tc="#155E75")
box(ax, 2.3, 3.6, 1.9, 1.2, "位置编码\n64×64 可学习网格\n超网格双三次插值\n+正弦时间表", fc=MM_SOFT, ec=MM, fs=9, tc="#155E75")
box(ax, 4.6, 3.4, 3.2, 1.6,
    "27 × 视觉 Transformer 层（双向非因果）\nQ/K/V/O 带偏置 · LayerNorm(eps1e-5)\n轴向2D-RoPE(空间维36) · gelu_pytorch_tanh FFN4304\ncu_seqlens 变长图掩码",
    fc=MM_SOFT, ec=MM, fs=9.5, tc="#155E75", bold=True)
box(ax, 8.2, 3.6, 1.6, 1.2, "merge (2,2)\n时空均值池化\ntoken÷4", fc=MM_SOFT, ec=MM, fs=9.5, tc="#155E75")
box(ax, 10.2, 3.6, 2.0, 1.2, "PatchMerger\nLN1152→Linear4608\nGELU→Linear7168", fc=ACCENT_SOFT, ec=ACCENT, fs=9.5, bold=True)
arrow(ax, (1.9, 4.2), (2.3, 4.2), color=MM)
arrow(ax, (4.2, 4.2), (4.6, 4.2), color=MM)
arrow(ax, (7.8, 4.2), (8.2, 4.2), color=MM)
arrow(ax, (9.8, 4.2), (10.2, 4.2), color=MM)
ax.text(6.75, 6.0, "Kimi K2.5-VL 核心机制：MoonViT-3D 层内结构（SigLIP-SO-400M 初始化）",
        ha="center", fontsize=12.5, fontweight="bold", color=INK)
ax.text(6.75, 2.4, "图像与视频权重完全共享 · 最多4帧为一时空体积 · 长视频经4×时间压缩进入256K上下文",
        ha="center", fontsize=9.5, color=SUB)
save(fig, d, "02_k25vl核心机制.png")

# ============ G. K2.5-VL 视觉塔 token 流 ============
fig, ax = new_canvas(13.5, 6.2)
box(ax, 0.2, 2.5, 1.5, 1.1, "图像 H×W\n原生分辨率\n(NaViT 打平)", fc=MM_SOFT, ec=MM, fs=9.5, tc="#155E75")
box(ax, 2.0, 2.5, 1.6, 1.1, "patch14\n→ grid_h×grid_w\n个 patch", fc=MM_SOFT, ec=MM, fs=9.5, tc="#155E75")
box(ax, 3.9, 2.4, 2.2, 1.3, "27层 MoonViT-3D\nbidirectional\n轴向2D-RoPE", fc=MM_SOFT, ec=MM, fs=9.5, tc="#155E75")
box(ax, 6.4, 2.5, 1.8, 1.1, "2×2 空间池化\n→ (gh/2)(gw/2)\n视觉 token", fc=MM_SOFT, ec=MM, fs=9.5, tc="#155E75")
box(ax, 8.5, 2.5, 1.7, 1.1, "投影器\n→7168维\n每token", fc=MM_SOFT, ec=MM, fs=9.5, tc="#155E75")
box(ax, 10.5, 2.4, 2.2, 1.3, "填入 <|image_pad|>\n=163605 占位位\n进 K2 MoE 塔", fc=ACCENT_SOFT, ec=ACCENT, fs=9.5, bold=True)
for x1, x2, c in [(1.7,2.0,MM),(3.6,3.9,MM),(6.1,6.4,MM),(8.2,8.5,MM),(10.2,10.5,ACCENT)]:
    arrow(ax, (x1, 3.05), (x2, 3.05), color=c)
ax.text(6.75, 5.6, "Kimi K2.5-VL 视觉塔 → 文本融合 token 流（按官方 vision_config 逐字段）",
        ha="center", fontsize=12.5, fontweight="bold", color=INK)
ax.text(6.75, 1.4, "例：512×512 → 36×36 patch → 池化后 18×18=324 视觉token；上限 max_patches=16384 · video=163840 · vs=163602 · ve=163604",
        ha="center", fontsize=9.5, color=SUB)
save(fig, d, "03_k25vl视觉塔token流.png")

print("done")
