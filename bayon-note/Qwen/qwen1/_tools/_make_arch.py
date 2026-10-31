# -*- coding: utf-8 -*-
"""生成 qwen1 / qwen1_5 笔记所需架构图 PNG，写入各自 _arch/。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

BG = "#faf9f6"; PANEL = "#ffffff"; INK = "#23272f"; SUB = "#66707b"; LINE = "#e5e4df"
EDGE = "#c9c4ba"; ACCENT = "#1d4ed8"; ACCENT_INK = "#1e40af"
ACCENT_SOFT = "#e2eafb"; ACCENT_LINE = "#c0d0f4"; HL_SOFT = "#eef3fd"

ROOT = r"E:\transformers\bayon-note\Qwen"

def box(ax, x, y, w, h, text, fc=PANEL, ec=EDGE, lw=1.3, fs=10.5, bold=False, tc=INK, rounded=0.025):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rounded}",
                       fc=fc, ec=ec, lw=lw, mutation_aspect=1)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", linespacing=1.5)

def arrow(ax, p1, p2, color=EDGE, lw=1.6, style="-|>"):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=13,
                        color=color, lw=lw, shrinkA=0, shrinkB=0)
    ax.add_patch(a)

# ============ Qwen1 图1：端到端总览 ============
fig, ax = plt.subplots(figsize=(11.2, 4.6), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(0, 41); ax.axis("off")
fig.patch.set_facecolor(BG)

box(ax, 1, 16, 11, 9, "输入 token\ninput_ids\n(B,T)", fs=10)
box(ax, 16, 16, 14, 9, "词嵌入 wte\nEmbedding\n151936 × d_model\n不与 LM Head 共享", fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=9.5)
arrow(ax, (12, 20.5), (16, 20.5))

# 解码层堆
box(ax, 34, 5, 30, 31, "", fc=HL_SOFT, ec=ACCENT, lw=1.6, rounded=0.03)
ax.text(49, 33.2, "QWenBlock × N（24/32/40/80 层）", ha="center", fontsize=10, fontweight="bold", color=ACCENT_INK)
box(ax, 36, 23.5, 26, 7.5, "ln_1: RMSNorm → QWenAttention\n融合 c_attn: d→3·(kv_channels·H)\nRoPE(kv_channels=128, base 1e4)\n推理超长时 query × logn 系数", fs=8.8)
box(ax, 36, 13.5, 26, 7.5, "ln_2: RMSNorm → QWenMLP\nw1·silu(w2)，内维 = intermediate/2\nc_proj 无 bias（no_bias=true）", fs=8.8)
ax.text(49, 11.6, "残差连接：x + sublayer(LN(x))", ha="center", fontsize=8.5, color=SUB)
arrow(ax, (60, 27.2), (63.2, 27.2), color=ACCENT)
arrow(ax, (63.2, 27.2), (63.2, 17.2), color=ACCENT)
arrow(ax, (63.2, 17.2), (60, 17.2), color=ACCENT)
arrow(ax, (30, 20.5), (34, 20.5))

box(ax, 68, 16, 11, 9, "最终 RMSNorm\nln_f\neps 1e-6", fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=9.5)
arrow(ax, (64, 20.5), (68, 20.5))
box(ax, 82, 16, 16, 9, "LM Head\nd → 词表\n独立权重\ntie=False", fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=9.5)
arrow(ax, (79, 20.5), (82, 20.5))
ax.text(90, 13.6, "下一 token 分布", ha="center", fontsize=9, color=SUB)
ax.text(49, 2.2, "纯文本主干（1_8B / 7B / 14B / 72B 四尺寸共用）：tiktoken BPE 分词 → wte → N×Block → ln_f → 独立 LM Head",
        ha="center", fontsize=9, color=SUB)
fig.savefig(os.path.join(ROOT, "qwen1", "_arch", "01_端到端总览.png"), facecolor=BG, bbox_inches="tight")
plt.close(fig)

# ============ Qwen1 图2：核心机制（logn-attn + 动态 NTK RoPE） ============
fig, ax = plt.subplots(figsize=(11.2, 5.2), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")
fig.patch.set_facecolor(BG)

ax.text(25, 44.5, "机制① LogN 注意力缩放（use_logn_attn）", ha="center", fontsize=11, fontweight="bold", color=ACCENT_INK)
box(ax, 4, 22.5, 42, 18,
    "训练长度 seq_length 内：query 不缩放（系数=1）\n推理 KV 长度超过 seq_length 时：\n  logn_i = log( i , seq_length )，i>seq_length\n  query ← query × logn_i（缓冲表预生成至 32767）\n含义：序列越长、位置索引越大，对 query 做\n对数级幅度放大，补偿长上下文 softmax 分数\n稀释，稳定超长生成。",
    fs=9.2, rounded=0.03)
ax.text(25, 20.3, "代码：modeling_qwen.py:289-294 建表；482-490 应用", ha="center", fontsize=8.3, color=SUB)

ax.text(75, 44.5, "机制② 动态 NTK 扩展 RoPE（use_dynamic_ntk）", ha="center", fontsize=11, fontweight="bold", color=ACCENT_INK)
box(ax, 54, 22.5, 42, 18,
    "按真实序列长度即时调整旋转基频：\n  c = log2( true_len / seq_length ) + 1\n  ntk_alpha = 2^ceil(c) − 1，下限取 1\n  base′ = rotary_emb_base · ntk_alpha^(d/(d−2))\ninv_freq 随 base′ 重算并缓存。\n8B/7B/14B 开启；72B 关闭（base 直接取 1e6）。",
    fs=9.2, rounded=0.03)
ax.text(75, 20.3, "代码：modeling_qwen.py:735-739；1287-1298", ha="center", fontsize=8.3, color=SUB)

box(ax, 4, 4, 92, 11.5,
    "层内注意力流：c_attn 一次线性层切出 Q/K/V（d → 3·kv_channels·H）→ 分头 → RoPE 仅作用于 kv_channels=128 维（rotary_pct=1.0）\n→ 长度超限先乘 logn 系数 → 因果掩码 → softmax/√d → c_proj 输出。全程 MHA：Q 与 KV 头数相同（无 GQA 字段）。",
    fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=9.5)
fig.savefig(os.path.join(ROOT, "qwen1", "_arch", "02_核心机制_logn_attn与动态NTK.png"), facecolor=BG, bbox_inches="tight")
plt.close(fig)

# ============ Qwen1.5 图1：端到端总览 ============
fig, ax = plt.subplots(figsize=(11.2, 4.6), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(0, 41); ax.axis("off")
fig.patch.set_facecolor(BG)

box(ax, 1, 16, 11, 9, "输入 token\ninput_ids\n(B,T)", fs=10)
box(ax, 16, 16, 14, 9, "词嵌入 embed_tokens\nEmbedding\n151936/152064 × d\n小尺寸 tied", fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=9.5)
arrow(ax, (12, 20.5), (16, 20.5))

box(ax, 34, 5, 30, 31, "", fc=HL_SOFT, ec=ACCENT, lw=1.6, rounded=0.03)
ax.text(49, 33.2, "Qwen2DecoderLayer × N（24…80 层）", ha="center", fontsize=10, fontweight="bold", color=ACCENT_INK)
box(ax, 36, 23.5, 26, 7.5, "input_layernorm: RMSNorm\n→ Qwen2Attention\nq/k/v_proj（bias=True）· RoPE θ=1e6\nGQA：32B 为 40Q/8KV，其余 MHA", fs=8.6)
box(ax, 36, 13.5, 26, 7.5, "post_attention_layernorm: RMSNorm\n→ Qwen2MLP（SwiGLU）\ndown( silu(gate) · up )，无 bias", fs=8.6)
ax.text(49, 11.6, "残差连接：residual + sublayer(norm(x))", ha="center", fontsize=8.5, color=SUB)
arrow(ax, (60, 27.2), (63.2, 27.2), color=ACCENT)
arrow(ax, (63.2, 27.2), (63.2, 17.2), color=ACCENT)
arrow(ax, (63.2, 17.2), (60, 17.2), color=ACCENT)
arrow(ax, (30, 20.5), (34, 20.5))

box(ax, 68, 16, 11, 9, "最终 RMSNorm\nnorm\neps 1e-6", fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=9.5)
arrow(ax, (64, 20.5), (68, 20.5))
box(ax, 82, 16, 16, 9, "LM Head\nd → 词表\n0.5B tied\n其余独立", fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=9.5)
arrow(ax, (79, 20.5), (82, 20.5))
ax.text(90, 13.6, "下一 token 分布", ha="center", fontsize=9, color=SUB)
ax.text(49, 2.2, "纯文本主干（入口类 Qwen2ForCausalLM，model_type=qwen2；0.5B / 1.8B / 7B / 14B / 32B / 72B 六尺寸共用）",
        ha="center", fontsize=9, color=SUB)
fig.savefig(os.path.join(ROOT, "qwen1_5", "_arch", "01_端到端总览.png"), facecolor=BG, bbox_inches="tight")
plt.close(fig)

# ============ Qwen1.5 图2：注意力头分配 + 滑窗配置 ============
fig, ax = plt.subplots(figsize=(11.2, 5.4), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(0, 48); ax.axis("off")
fig.patch.set_facecolor(BG)

ax.text(25, 45.5, "① KV 头分配（Qwen1.5 家族）", ha="center", fontsize=11, fontweight="bold", color=ACCENT_INK)
rows = [
    ("0.5B", "16 Q / 16 KV", "MHA 1:1"),
    ("1.8B", "16 Q / 16 KV", "MHA 1:1"),
    ("7B",   "32 Q / 32 KV", "MHA 1:1"),
    ("14B",  "40 Q / 40 KV", "MHA 1:1"),
    ("32B",  "40 Q / 8 KV",  "GQA 5:1（唯一）"),
    ("72B",  "64 Q / 64 KV", "MHA 1:1"),
]
y = 42.5
for name, heads, kind in rows:
    hl = "GQA" in kind
    box(ax, 5, y-2.6, 10, 3.4, name, fs=9, fc=PANEL)
    box(ax, 17, y-2.6, 16, 3.4, heads, fs=9, fc=(ACCENT_SOFT if hl else PANEL), ec=(ACCENT_LINE if hl else EDGE))
    box(ax, 35, y-2.6, 14, 3.4, kind, fs=9, fc=(ACCENT_SOFT if hl else PANEL), ec=(ACCENT_LINE if hl else EDGE))
    y -= 5.0

ax.text(75, 45.5, "② 滑窗配置（sliding_window / max_window_layers）", ha="center", fontsize=11, fontweight="bold", color=ACCENT_INK)
box(ax, 56, 13.5, 40, 29,
    "config 显式字段：\n  sliding_window = 32768\n  max_window_layers = 21 / 28 / 35 / 70\n  use_sliding_window = false\n\nv5 加载时 __post_init__：\n  use_sliding_window=false → sliding_window 置 None\n  → layer_types 全部 full_attention\n  → 六尺寸实际均为全因果注意力\n\n（该字段模板自 Qwen2，官方博客注明\n 可自行调大 max_position 与 sliding_window）",
    fs=8.8, rounded=0.03)

box(ax, 4, 1.5, 92, 8,
    "attention 结构：q_proj(kv 含 bias) → 分头 → RoPE(θ=1e6，全 head_dim) → （GQA 层 repeat_kv 复制 KV）→ softmax(QKᵀ/√d)V → o_proj(无 bias)。\n缩放 = head_dim^(−0.5)；head_dim = hidden/heads（0.5B=64，其余=128）。",
    fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=9.2)
fig.savefig(os.path.join(ROOT, "qwen1_5", "_arch", "02_注意力头分配与滑窗配置.png"), facecolor=BG, bbox_inches="tight")
plt.close(fig)

print("OK: 4 figures generated")
