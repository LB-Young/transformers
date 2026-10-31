# -*- coding: utf-8 -*-
"""Generate Kimi family architecture diagrams (PNG)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = r"E:\transformers\bayon-note\Kimi"
INK = "#2327F"[:7] if False else "#23272F"; SUB = "#66707B"; EDGE = "#C9C4BA"
ACCENT = "#1D4ED8"; ACCENT_SOFT = "#E2EAFB"; ACCENT_LINE = "#C0D0F4"
NODE = "#FFFDF9"; CORE = "#EEF3FD"; WARN = "#FBEEE6"; WARN_EDGE = "#E0A878"

def box(ax, x, y, w, h, text, fc=NODE, ec=EDGE, fs=10.5, bold=False, tc=INK):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                       linewidth=1.3, edgecolor=ec, facecolor=fc)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", linespacing=1.45)

def arrow(ax, p1, p2, color=EDGE, lw=1.6, style="-|>"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=13, linewidth=lw, color=color))

# ===================== Kimi-K2-0905 图1：端到端 =====================
out = os.path.join(ROOT, "kimi_k2_0905", "_arch"); os.makedirs(out, exist_ok=True)
fig, ax = plt.subplots(figsize=(13.5, 7.4))
ax.set_xlim(0, 13.5); ax.set_ylim(0, 7.4); ax.axis("off")

box(ax, 0.3, 4.9, 1.6, 1.0, "文本 token\ninput_ids\n词表 163840", fc="#F4F4F2", ec=EDGE, fs=10)
box(ax, 2.3, 4.9, 1.7, 1.0, "词嵌入 Embedding\n(untied 解绑)", fs=10)
box(ax, 4.5, 5.0, 1.5, 0.8, "第 0 层\nDense FFN\n(first_k_dense=1)", fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=9.5)
box(ax, 6.5, 3.0, 3.0, 2.6,
    "层 1–60（共 60 个 MoE 层）\nmoe_layer_freq=1，每层都走专家\n──────────\n① RMSNorm → MLA 注意力\n  kv_lora_rank 512 · q_lora_rank 1536\n  qk_nope 128 / qk_rope 64 / v_head 128\n② RMSNorm → MoE FFN\n  384 路由专家 top-8 + 1 共享专家\n  moe_intermediate 2048",
    fc=CORE, ec=ACCENT, fs=9.3, bold=True)
box(ax, 10.0, 5.0, 1.5, 0.8, "最终\nRMSNorm", fs=10)
box(ax, 10.0, 3.6, 1.5, 0.8, "LM Head\n7168→163840\nuntied", fs=10)
box(ax, 11.9, 4.2, 1.3, 0.8, "下一 token\nlogits", fc=ACCENT_SOFT, ec=ACCENT, fs=10, bold=True)

arrow(ax, (1.9, 5.4), (2.3, 5.4))
arrow(ax, (4.0, 5.4), (4.5, 5.4))
arrow(ax, (6.0, 5.4), (6.5, 5.0))
arrow(ax, (9.5, 5.0), (10.0, 5.4))
arrow(ax, (10.75, 5.0), (10.75, 4.4))
arrow(ax, (11.5, 4.0), (11.9, 4.5))

ax.text(6.75, 6.95, "Kimi-K2-0905 端到端数据流：DeepseekV3ForCausalLM（model_type=kimi_k2）· hidden 7168 · 61 层",
        ha="center", fontsize=12.5, fontweight="bold", color=INK)
ax.text(6.75, 0.6, "上下文 262144（rope_theta 50000 + YaRN factor 64，original_max_position 4096）· 词表 163840 · tie=false · fp8 e4m3 官方权重",
        ha="center", fontsize=9.5, color=SUB)
plt.tight_layout()
plt.savefig(os.path.join(out, "01_k2_0905端到端总览.png"), dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

# ===================== Kimi-K2-0905 图2：MLA + MoE 核心机制 =====================
fig, ax = plt.subplots(figsize=(13.5, 7.6))
ax.set_xlim(0, 13.5); ax.set_ylim(0, 7.6); ax.axis("off")

box(ax, 0.3, 3.4, 1.3, 0.9, "hidden x\n[B,T,7168]", fs=10)
# MLA branch
box(ax, 2.1, 5.0, 3.0, 1.7,
    "MLA 注意力（低秩联合压缩）\nq 下投影 → q_lora_rank 1536\nkv 下投影 → kv_lora_rank 512\nqk_nope_head_dim 128（无位置）\nqk_rope_head_dim 64（RoPE 在压缩 K）\nv_head_dim 128 · 64 头",
    fc=CORE, ec=ACCENT, fs=9.2, bold=True)
# MoE branch
box(ax, 2.1, 1.6, 3.0, 1.9,
    "MoE FFN（细粒度专家）\n路由：sigmoid 打分 · noaux_tc\ntop-8 · routed_scaling 2.827\nn_routed_experts = 384\nn_shared_experts = 1（共享）\nmoe_intermediate_size = 2048\naux_loss_alpha = 0.001",
    fc=CORE, ec=ACCENT, fs=9.2, bold=True)

box(ax, 5.7, 5.4, 1.0, 0.7, "⊕", fs=13, bold=True)
box(ax, 5.7, 2.3, 1.0, 0.7, "⊕", fs=13, bold=True)
box(ax, 7.3, 3.4, 1.5, 0.9, "下一层\nhidden", fs=10)

arrow(ax, (1.6, 3.85), (2.1, 5.6))
arrow(ax, (1.0, 3.4), (1.0, 2.5))
arrow(ax, (1.0, 2.5), (2.1, 2.5))
arrow(ax, (5.1, 5.85), (5.7, 5.75))
arrow(ax, (5.1, 2.55), (5.7, 2.6))
arrow(ax, (6.7, 5.75), (7.3, 4.2))
arrow(ax, (6.7, 2.6), (7.3, 3.6))

ax.text(6.75, 7.2, "Kimi-K2-0905 核心机制：DeepSeek-V3 式 MLA（上）+ 细粒度 MoE 路由（下）",
        ha="center", fontsize=12.5, fontweight="bold", color=INK)
ax.text(6.75, 0.7, "与 Kimi-K2-Base 同架构；0905 更新把 max_position 从 131072 扩到 262144（YaRN）并附带 fp8 e4m3 权重配置",
        ha="center", fontsize=9.5, color=SUB)
plt.tight_layout()
plt.savefig(os.path.join(out, "02_k2_0905_MLA与MoE机制.png"), dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

# ===================== moonshot_8b 图：通用 dense 示意（待确认） =====================
out = os.path.join(ROOT, "moonshot_8b", "_arch"); os.makedirs(out, exist_ok=True)
fig, ax = plt.subplots(figsize=(13.5, 6.6))
ax.set_xlim(0, 13.5); ax.set_ylim(0, 6.6); ax.axis("off")
box(ax, 0.3, 2.8, 1.5, 1.0, "文本 token\ninput_ids", fc="#F4F4F2", ec=EDGE, fs=10)
box(ax, 2.2, 2.8, 1.6, 1.0, "词嵌入 Embedding\n(参数待确认)", fc=WARN, ec=WARN_EDGE, fs=9.5)
box(ax, 4.3, 2.6, 3.2, 1.4,
    "N × DecoderLayer（dense）\n前置 RMSNorm → 自注意力 → ⊕\n前置 RMSNorm → SwiGLU FFN → ⊕\n层数 / hidden / 头数 / 词表：均待确认",
    fc=WARN, ec=WARN_EDGE, fs=9.5, bold=True)
box(ax, 8.0, 2.8, 1.5, 1.0, "最终\nRMSNorm", fc=WARN, ec=WARN_EDGE, fs=10)
box(ax, 9.9, 2.8, 1.5, 1.0, "LM Head\n→ 词表", fc=WARN, ec=WARN_EDGE, fs=10)
box(ax, 11.8, 2.9, 1.3, 0.8, "下一 token", fc=ACCENT_SOFT, ec=ACCENT, fs=10, bold=True)
for p1,p2 in [(1.8,2.2),(3.8,4.3),(7.5,8.0),(9.5,9.9),(11.4,11.8)]:
    arrow(ax, (p1,3.3),(p2,3.3))
ax.text(6.75, 6.0, "Moonshot-8B 架构示意（官方 repo 已 404，本图为 dense Transformer 通用骨架，非实测）",
        ha="center", fontsize=12, fontweight="bold", color=INK)
ax.text(6.75, 0.7, "一切超参待确认：dense / 全注意力 / 长上下文（128K?）均未获官方 config 复核 · 详见正文「待确认」清单",
        ha="center", fontsize=9.5, color=SUB)
plt.tight_layout()
plt.savefig(os.path.join(out, "01_moonshot8b架构示意.png"), dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

# ===================== moonshot_32b_a32b 图：MoE 通用示意（待确认） =====================
out = os.path.join(ROOT, "moonshot_32b_a32b", "_arch"); os.makedirs(out, exist_ok=True)
fig, ax = plt.subplots(figsize=(13.5, 6.8))
ax.set_xlim(0, 13.5); ax.set_ylim(0, 6.8); ax.axis("off")
box(ax, 0.3, 3.0, 1.5, 1.0, "文本 token\ninput_ids", fc="#F4F4F2", ec=EDGE, fs=10)
box(ax, 2.2, 3.0, 1.6, 1.0, "词嵌入\n(参数待确认)", fc=WARN, ec=WARN_EDGE, fs=9.5)
box(ax, 4.3, 2.6, 3.4, 1.8,
    "N × DecoderLayer（MoE）\n注意力子层：类型待确认（GQA/MLA?）\nFFN 子层：路由 MoE\n总参数 / 激活 A32B 口径待确认\n专家数 / top-k / 共享专家待确认",
    fc=WARN, ec=WARN_EDGE, fs=9.3, bold=True)
box(ax, 8.2, 3.0, 1.5, 1.0, "最终\nRMSNorm", fc=WARN, ec=WARN_EDGE, fs=10)
box(ax, 10.1, 3.0, 1.5, 1.0, "LM Head\n→ 词表", fc=WARN, ec=WARN_EDGE, fs=10)
box(ax, 12.0, 3.1, 1.2, 0.8, "下一 token", fc=ACCENT_SOFT, ec=ACCENT, fs=10, bold=True)
for p1,p2 in [(1.8,2.2),(3.8,4.3),(7.7,8.2),(9.7,10.1),(11.6,12.0)]:
    arrow(ax, (p1,3.5),(p2,3.5))
ax.text(6.75, 6.2, "Moonshot-32B-A32B 架构示意（官方 repo 已 404，本图为 MoE Transformer 通用骨架，非实测）",
        ha="center", fontsize=12, fontweight="bold", color=INK)
ax.text(6.75, 0.7, "「A32B」= 每 token 激活约 32B 参数；总参数与专家配置均待确认 · 早期真实开源模型 Moonlight-16B-A3B 见演进章对照",
        ha="center", fontsize=9.5, color=SUB)
plt.tight_layout()
plt.savefig(os.path.join(out, "01_moonshot32bMoE架构示意.png"), dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

print("done")
for r,_,fs in os.walk(ROOT):
    for f in fs:
        if f.endswith(".png"): print("  ", os.path.join(r,f))
