# -*- coding: utf-8 -*-
"""生成 GLM 早期代架构图（端到端总览 + 核心机制），中文 PNG。"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = r"E:\transformers\bayon-note\GLM"

BLUE = "#1d4ed8"; SOFT = "#e2eafb"; LINE = "#c0d0f4"
INK = "#23272f"; SUB = "#66707b"; EDGE = "#c9c4ba"
PANEL = "#ffffff"; COREBG = "#eef3fd"


def box(ax, x, y, w, h, text, sub=None, fc=PANEL, ec=EDGE, tc=INK, fs=11, bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                       fc=fc, ec=ec, lw=1.4)
    ax.add_patch(p)
    weight = "bold" if bold else "normal"
    ax.text(x + w / 2, y + h / 2 + (0.18 if sub else 0), text, ha="center", va="center",
            fontsize=fs, color=tc, fontweight=weight)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.22, sub, ha="center", va="center",
                fontsize=fs - 2, color=SUB)


def arrow(ax, x1, y1, x2, y2, color=EDGE, lw=1.6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                mutation_scale=13, color=color, lw=lw))


def save(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", path)


def mech_box(ax, x, y, w, h, title, body, fc=COREBG, ec=BLUE):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                       fc=fc, ec=ec, lw=1.5)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="center",
            fontsize=11.5, color=INK, fontweight="bold")
    lines = body.split("\n")
    n = len(lines)
    start = y + h - 0.85
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, start - i * 0.34, ln, ha="center", va="center",
                fontsize=9.5, color=SUB)


# ============ GLM-130B ============
def glm130b_overview():
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    box(ax, 0.3, 2.4, 1.5, 1.2, "输入序列", "[gMASK]/[MASK]\nblank infilling", fs=10)
    arrow(ax, 1.8, 3.0, 2.4, 3.0)
    box(ax, 2.4, 2.4, 1.7, 1.2, "词嵌入", "icetk 双语\n150,000", fs=10)
    arrow(ax, 4.1, 3.0, 4.7, 3.0)
    box(ax, 4.7, 1.6, 3.0, 2.8, "70 × GLM Decoder Layer",
        "DeepNorm Post-LN · GeGLU FFN(32768)\n96 头 · head_dim 128 · RoPE\n双向注意力(unmasked context)",
        fc=COREBG, ec=BLUE, bold=True, fs=11)
    arrow(ax, 7.7, 3.0, 8.3, 3.0)
    box(ax, 8.3, 2.4, 1.5, 1.2, "最终 LN", "DeepNorm", fs=10)
    arrow(ax, 9.8, 3.0, 10.3, 3.0)
    box(ax, 10.3, 2.4, 1.5, 1.2, "LM Head", "softmax", fs=10)
    ax.text(6.2, 5.3, "GLM-130B 端到端主干（hidden=12288 · 70 层 · 最大序列 2048）",
            ha="center", fontsize=13, fontweight="bold", color=INK)
    ax.text(6.2, 0.7, "目标：自回归 blank infilling——未掩码上下文双向注意力，被掩码 span 自回归生成",
            ha="center", fontsize=10, color=SUB)
    save(fig, os.path.join(ROOT, "glm_130b", "_arch", "01_end_to_end.png"))


def glm130b_mech():
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7.2); ax.axis("off")
    ax.text(6, 6.9, "GLM-130B 核心机制：DeepNorm · 双向 prefix 注意力 · GeGLU · RoPE",
            ha="center", fontsize=13, fontweight="bold", color=INK)
    mech_box(ax, 0.4, 4.5, 5.4, 1.9, "DeepNorm（Post-LN 残差）",
             "y = LayerNorm( α·x + SubLayer(x) )，α=(2N)^(1/4)\nFFN/v_proj/out_proj 初始化缩放 (2N)^(-1/4)\n所有 bias 初始化为 0，N=70",
             fc=SOFT, ec=LINE)
    mech_box(ax, 6.2, 4.5, 5.4, 1.9, "GeGLU 前馈网络",
             "FFN = ( GeLU(x·W1) ⊙ x·V )·W2\n为保持参数量，d_ffn = 8/3·d_h\n8/3 × 12288 = 32768",
             fc=SOFT, ec=LINE)
    mech_box(ax, 0.4, 0.7, 5.4, 3.3, "双向 prefix 注意力掩码",
             "未掩码上下文：双向注意力\n被掩码 span：内部自回归（三角）\n两目标由 [MASK]/[gMASK] 区分\n9×8−2=70 层（流水线均衡）")
    mech_box(ax, 6.2, 0.7, 5.4, 3.3, "RoPE 与 INT4 特性",
             "对 Q/K 施加旋转位置编码\n头数 96 → head_dim=128\nFP16 · Embedding Gradient Shrink\nINT4 训练后量化几乎无损")
    save(fig, os.path.join(ROOT, "glm_130b", "_arch", "02_core_mech.png"))


# ============ ChatGLM-6B ============
def chatglm6b_overview():
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    box(ax, 0.3, 2.4, 1.5, 1.2, "输入 token", "bos=130004\neos=130005", fs=10)
    arrow(ax, 1.8, 3.0, 2.4, 3.0)
    box(ax, 2.4, 2.4, 1.7, 1.2, "词嵌入", "vocab 130528\n(ice tokenizer)", fs=10)
    arrow(ax, 4.1, 3.0, 4.7, 3.0)
    box(ax, 4.7, 1.6, 3.0, 2.8, "28 × GLM Block",
        "LayerNorm(eps=1e-5) · GeGLU FFN(16384)\n32 头 · head_dim 128 · 2D-RoPE\nprefix-LM 双向/自回归混合",
        fc=COREBG, ec=BLUE, bold=True, fs=11)
    arrow(ax, 7.7, 3.0, 8.3, 3.0)
    box(ax, 8.3, 2.4, 1.5, 1.2, "最终 LN", "LayerNorm", fs=10)
    arrow(ax, 9.8, 3.0, 10.3, 3.0)
    box(ax, 10.3, 2.4, 1.5, 1.2, "LM Head", "softmax", fs=10)
    ax.text(6.2, 5.3, "ChatGLM-6B 端到端主干（hidden=4096 · 28 层 · max_seq=2048）",
            ha="center", fontsize=13, fontweight="bold", color=INK)
    ax.text(6.2, 0.7, "config: model_type=chatglm · ChatGLMModel · trust_remote_code 加载（transformers v5 无原生 chatglm 模块）",
            ha="center", fontsize=10, color=SUB)
    save(fig, os.path.join(ROOT, "chatglm_6b", "_arch", "01_end_to_end.png"))


def chatglm6b_mech():
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7.2); ax.axis("off")
    ax.text(6, 6.9, "ChatGLM-6B 核心机制：2D 旋转位置编码 · prefix-LM 双支路掩码",
            ha="center", fontsize=13, fontweight="bold", color=INK)
    mech_box(ax, 0.4, 4.5, 5.4, 1.9, "2D RoPE（position_encoding_2d=true）",
             "第 1 组 = 序列位置 id\n第 2 组 = span 内相对位置 id\n两组旋转分别作用于 Q/K 两半",
             fc=SOFT, ec=LINE)
    mech_box(ax, 6.2, 4.5, 5.4, 1.9, "prefix-LM 注意力掩码",
             "prefix（上下文）双向注意力\ngeneration（生成）支路因果三角\n由 attention_mask 统一表达",
             fc=SOFT, ec=LINE)
    mech_box(ax, 0.4, 0.7, 5.4, 3.3, "FFN 与归一化",
             "inner_hidden_size=16384 (=4×4096)\n门控前馈（激活沿 GLM 传统）\nlayernorm_epsilon=1e-5\n标准 LayerNorm（非 RMSNorm）")
    mech_box(ax, 6.2, 0.7, 5.4, 3.3, "特殊标记（config 字段）",
             "mask=130000 · gmask=130001\nbos=130004 · eos=130005 · pad=3\nMHA 32 头（无 KV 分组字段）\nmax_sequence_length=2048")
    save(fig, os.path.join(ROOT, "chatglm_6b", "_arch", "02_core_mech.png"))


# ============ ChatGLM2/3 ============
def chatglm23_overview():
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    box(ax, 0.3, 2.4, 1.5, 1.2, "输入 token", "eos=2\npad=0", fs=10)
    arrow(ax, 1.8, 3.0, 2.4, 3.0)
    box(ax, 2.4, 2.4, 1.7, 1.2, "词嵌入", "padded_vocab\n65024", fs=10)
    arrow(ax, 4.1, 3.0, 4.7, 3.0)
    box(ax, 4.7, 1.6, 3.0, 2.8, "28 × GLM Block",
        "RMSNorm · SwiGLU FFN(13696)\n32 Q 头 / 2 KV 组(MQA) · kv_channels=128\n1D-RoPE(original_rope) · QKV bias",
        fc=COREBG, ec=BLUE, bold=True, fs=10.5)
    arrow(ax, 7.7, 3.0, 8.3, 3.0)
    box(ax, 8.3, 2.4, 1.5, 1.2, "最终 RMSNorm", "post_layer_norm", fs=10)
    arrow(ax, 9.8, 3.0, 10.3, 3.0)
    box(ax, 10.3, 2.4, 1.5, 1.2, "LM Head", "tie=false", fs=10)
    ax.text(6.2, 5.3, "ChatGLM2/3-6B 端到端主干（hidden=4096 · 28 层 · 架构数字两版完全一致）",
            ha="center", fontsize=12.5, fontweight="bold", color=INK)
    ax.text(6.2, 0.7, "差异仅在 seq_length：ChatGLM2=32768（FlashAttention 32K）· ChatGLM3=8192（config）",
            ha="center", fontsize=10, color=SUB)
    save(fig, os.path.join(ROOT, "chatglm2_3", "_arch", "01_end_to_end.png"))


def chatglm23_mech():
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7.2); ax.axis("off")
    ax.text(6, 6.9, "ChatGLM2/3 核心机制：MQA(GQA) · RMSNorm+SwiGLU · 标准 1D RoPE",
            ha="center", fontsize=13, fontweight="bold", color=INK)
    mech_box(ax, 0.4, 4.5, 5.4, 1.9, "Multi-Query Attention",
             "multi_query_attention=true · group=2\n32 个 Q 头共享 2 组 KV\n官方：推理提速约 42%",
             fc=SOFT, ec=LINE)
    mech_box(ax, 6.2, 4.5, 5.4, 1.9, "QKV bias / 去偏置",
             "add_qkv_bias=true（仅 QKV 带 bias）\nadd_bias_linear=false（其余去偏置）\nattention_softmax_in_fp32=true",
             fc=SOFT, ec=LINE)
    mech_box(ax, 0.4, 0.7, 5.4, 3.3, "归一化与前馈换代",
             "rmsnorm=true 取代 LayerNorm\npost_layer_norm=true\nffn_hidden_size=13696 (~10/3·4096)\nSwiGLU；去 2D-RoPE→1D")
    mech_box(ax, 6.2, 0.7, 5.4, 3.3, "上下文与权重解绑",
             "original_rope=true（标准旋转）\nChatGLM2: seq_length=32768 (32K)\nChatGLM3: seq_length=8192\ntie_word_embeddings=false")
    save(fig, os.path.join(ROOT, "chatglm2_3", "_arch", "02_core_mech.png"))


if __name__ == "__main__":
    for d in ["glm_130b", "chatglm_6b", "chatglm2_3"]:
        os.makedirs(os.path.join(ROOT, d, "_arch"), exist_ok=True)
    glm130b_overview(); glm130b_mech()
    chatglm6b_overview(); chatglm6b_mech()
    chatglm23_overview(); chatglm23_mech()
    print("ALL DONE")
