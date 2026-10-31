# -*- coding: utf-8 -*-
"""Generate MiniMax family architecture diagrams (PNG). Uses common312 python."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import os

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = r"E:\transformers\bayon-note\MiniMax"

INK = "#23272F"; SUB = "#66707B"; EDGE = "#C9C4BA"
ACCENT = "#1D4ED8"; ACCENT_SOFT = "#E2EAFB"; ACCENT_LINE = "#C0D0F4"
MM = "#0E7490"; MM_SOFT = "#E0F2F7"
NODE = "#FFFDF9"; CORE = "#EEF3FD"
WARN = "#B45309"; WARN_SOFT = "#FDF0DC"

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

def new_ax(w=13.5, h=7.2):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax

def save(fig, d, name):
    outdir = os.path.join(ROOT, d, "_arch")
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, name), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", d, name)

# ============================================================
# minimax_text_01
# ============================================================
def text01_overview():
    fig, ax = new_ax(13.5, 7.4)
    box(ax, 0.3, 4.6, 1.7, 1.1, "文本 token\ninput_ids\n词表 200064", fc="#F4F4F2", ec=EDGE, fs=10)
    box(ax, 2.5, 4.6, 1.7, 1.1, "词嵌入\nEmbed\n(不共享)", fs=10)
    # layer stack
    box(ax, 4.7, 1.6, 4.6, 4.6, "", fc=CORE, ec=ACCENT)
    ax.text(7.0, 5.85, "80 × MiniMaxDecoderLayer", ha="center", fontsize=11.5, fontweight="bold", color=ACCENT)
    # 7 lightning layers
    box(ax, 5.0, 4.35, 4.0, 1.15, "70 层 · Lightning 线性注意力\n块递归状态 (I/O 感知)\n+ 门控残差 α·h+β·s", fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=9.5)
    box(ax, 5.0, 2.95, 4.0, 1.15, "每 8 层插 1 层 · 全 Softmax 注意力\nGQA 64Q/8KV · 部分 RoPE(64/128)\n共 10 层 (第 8/16/…/80)", fc=WARN_SOFT, ec=WARN, fs=9.5, tc=WARN)
    box(ax, 5.0, 1.75, 4.0, 1.0, "每层 FFN：32 专家 MoE\ntop-2 路由 · SiGLU · 无共享专家", fc=NODE, ec=EDGE, fs=9.5)
    box(ax, 9.7, 4.6, 1.5, 1.0, "最终\nRMSNorm", fs=10)
    box(ax, 9.7, 2.9, 1.5, 1.0, "LM Head\n到词表", fs=10)
    box(ax, 11.6, 3.6, 1.6, 0.9, "下一 token", fc=ACCENT_SOFT, ec=ACCENT, fs=10, bold=True)
    arrow(ax, (2.0, 5.15), (2.5, 5.15))
    arrow(ax, (4.2, 5.15), (4.7, 5.15))
    arrow(ax, (9.3, 5.15), (9.7, 5.15))
    arrow(ax, (10.45, 4.6), (10.45, 3.9))
    arrow(ax, (11.2, 3.4), (11.6, 4.0))
    ax.text(7.0, 0.8, "456B 总参 / 每 token 激活约 45.9B（≈A46B）· 训练上下文 1M，推理外推至 4M",
            ha="center", fontsize=10.5, color=SUB)
    save(fig, "minimax_text_01", "arch_overview.png")

def text01_mech():
    fig, ax = new_ax(13.5, 7.4)
    ax.text(6.75, 7.05, "Lightning Attention（块线性注意力）核心机制  block_size = 256", ha="center", fontsize=12.5, fontweight="bold", color=ACCENT)
    # intra block
    box(ax, 0.5, 4.6, 3.6, 1.9, "块内 (intra-block)\nQK^T 衰减掩码 → V\n还原局部精确依赖\n（类因果局部注意力）", fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=10)
    # inter block recurrent
    box(ax, 4.7, 4.6, 4.1, 1.9, "块间 (inter-block) 递归状态\nS ∈ R^{B×H×d×d}\nS ← S·γ_block + (K·γ_k)^T·V\n下一块：Q·γ_q @ S", fc=CORE, ec=ACCENT, fs=9.8, bold=True)
    # decay
    box(ax, 9.3, 4.6, 3.7, 1.9, "逐头指数衰减\ng = 1-层号/层数 (层数自适应)\nquery_decay/key_decay = exp(-g·pos)\n遗忘远块、保留近块", fc=WARN_SOFT, ec=WARN, fs=9.6, tc=WARN)
    # gated residual
    box(ax, 2.0, 1.6, 9.5, 1.8,
        "门控残差（post-norm）：  hidden = x·α + Sublayer(LN(x)，  α = 3.5566，β = 1.0\n"
        "注意力层与 MoE 层均套同一门控残差；Linear Attention 不挂 RoPE，RoPE 只在 10 个全注意力层",
        fc=NODE, ec=EDGE, fs=10.5, bold=True)
    arrow(ax, (4.1, 5.55), (4.7, 5.55))
    arrow(ax, (8.8, 5.55), (9.3, 5.55))
    save(fig, "minimax_text_01", "arch_mech.png")

# ============================================================
# minimax_visual_01  (= MiniMax-VL-01)
# ============================================================
def vl01_overview():
    fig, ax = new_ax(13.5, 7.4)
    box(ax, 0.3, 4.7, 1.6, 1.0, "文本 token\n词表 200064", fc="#F4F4F2", ec=EDGE, fs=10)
    box(ax, 0.3, 2.2, 1.6, 1.0, "图像\npixel_values\n336 基准", fc=MM_SOFT, ec=MM, fs=10, tc="#155E75")
    box(ax, 2.3, 4.7, 1.5, 1.0, "词嵌入", fs=10)
    box(ax, 2.3, 2.2, 1.5, 1.0, "CLIP ViT\npatch14·24层\nhidden1024", fc=MM_SOFT, ec=MM, fs=9.5, tc="#155E75")
    box(ax, 4.2, 3.4, 1.6, 1.1, "线性投影\n→6144\n(gelu)", fc=MM_SOFT, ec=MM, fs=9.5, tc="#155E75")
    box(ax, 6.2, 3.0, 1.7, 1.6, "early fusion\n按 <image>\n(=200025)\n填入视觉 token", fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=9.5)
    box(ax, 8.3, 3.0, 2.6, 1.6, "MiniMax 文本塔\n= Text-01 架构\n80层·7:1 闪电注意力\n上下文 8192/块", fc=CORE, ec=ACCENT, fs=9.5, bold=True)
    box(ax, 11.3, 3.4, 1.7, 0.9, "文本/logits", fc=ACCENT_SOFT, ec=ACCENT, fs=10, bold=True)
    arrow(ax, (1.9, 5.2), (2.3, 5.2))
    arrow(ax, (1.9, 2.7), (2.3, 2.7), color=MM)
    arrow(ax, (3.8, 2.7), (4.2, 3.5), color=MM)
    arrow(ax, (3.8, 5.2), (6.2, 4.0))
    arrow(ax, (5.8, 3.95), (6.2, 3.95), color=MM)
    arrow(ax, (7.9, 3.8), (8.3, 3.8))
    arrow(ax, (10.9, 3.8), (11.3, 3.8))
    ax.text(6.75, 1.7, "512B 视觉-语言 token 续训 · grid pinpoints 动态分辨率切块", ha="center", fontsize=10, color=SUB)
    save(fig, "minimax_visual_01", "arch_overview.png")

def vl01_vision():
    fig, ax = new_ax(13.5, 6.6)
    ax.text(6.75, 6.2, "视觉塔 token 流：CLIP ViT + 动态分辨率切块", ha="center", fontsize=12.5, fontweight="bold", color=MM)
    box(ax, 0.4, 3.2, 2.0, 1.3, "原始图像\n按 pinpoints\n切 336 瓦片\n(最多 6×6)", fc=MM_SOFT, ec=MM, fs=10, tc="#155E75")
    box(ax, 2.9, 3.2, 2.2, 1.3, "patch 14×14\n每瓦 24×24\n=576 patch", fc=MM_SOFT, ec=MM, fs=10, tc="#155E75")
    box(ax, 5.6, 3.2, 2.2, 1.3, "CLIP ViT-L 编码\n24层·hidden1024\n取最后一层", fc=MM_SOFT, ec=MM, fs=10, tc="#155E75")
    box(ax, 8.3, 3.2, 2.2, 1.3, "投影层\nhidden1024→6144\n(projector gelu)", fc=MM_SOFT, ec=MM, fs=9.6, tc="#155E75")
    box(ax, 11.0, 3.2, 2.1, 1.3, "拼入序列\n替换 <image>\n占位(200025)", fc=ACCENT_SOFT, ec=ACCENT, fs=9.8)
    for x in [(2.4, 2.9), (5.1, 5.6), (7.8, 8.3), (10.5, 11.0)]:
        arrow(ax, (x[0], 3.85), (x[1], 3.85), color=MM)
    ax.text(6.75, 1.9, "grid pinpoints: [(336),(672),(1008),(1344),(1680),(2016)] 的长宽组合；每瓦独立编码后展平拼到文本序列",
            ha="center", fontsize=10, color=SUB)
    save(fig, "minimax_visual_01", "arch_vision.png")

def vl01_mech():
    fig, ax = new_ax(13.5, 6.0)
    ax.text(6.75, 5.6, "文本塔核心：与 MiniMax-Text-01 相同的 7:1 闪电注意力 + 32 专家 MoE", ha="center", fontsize=12, fontweight="bold", color=ACCENT)
    box(ax, 1.0, 2.6, 3.4, 1.6, "70 层 Lightning\n块递归线性注意力\nblock 256 · 衰减", fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=10)
    box(ax, 5.0, 2.6, 3.4, 1.6, "10 层全 Softmax\nGQA 64/8 · RoPE\n每 8 层 1 个", fc=WARN_SOFT, ec=WARN, fs=10, tc=WARN)
    box(ax, 9.0, 2.6, 3.4, 1.6, "32 专家 MoE\ntop-2 · 无共享专家\nSiGLU", fc=NODE, ec=EDGE, fs=10)
    save(fig, "minimax_visual_01", "arch_mech.png")

# ============================================================
# minimax_m1
# ============================================================
def m1_overview():
    fig, ax = new_ax(13.5, 7.0)
    box(ax, 0.3, 4.4, 1.7, 1.0, "长输入 token\n最长 1M", fc="#F4F4F2", ec=EDGE, fs=10)
    box(ax, 2.4, 4.4, 1.6, 1.0, "词嵌入\n200064", fs=10)
    box(ax, 4.4, 2.4, 4.4, 3.6, "", fc=CORE, ec=ACCENT)
    ax.text(6.6, 5.6, "80 × DecoderLayer（同 Text-01）", ha="center", fontsize=11, fontweight="bold", color=ACCENT)
    box(ax, 4.7, 4.25, 3.8, 1.0, "70 Lightning + 10 全注意力\n(每 8 层 1 个)", fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=9.5)
    box(ax, 4.7, 2.95, 3.8, 1.0, "32 专家 MoE · top-2\n门控残差 α=3.5566", fc=NODE, ec=EDGE, fs=9.5)
    box(ax, 9.2, 4.4, 1.5, 0.9, "RMSNorm", fs=10)
    box(ax, 9.2, 3.1, 1.5, 0.9, "LM Head", fs=10)
    box(ax, 11.1, 3.6, 1.9, 0.9, "长思维链\n(40K/80K)", fc=WARN_SOFT, ec=WARN, fs=10, bold=True, tc=WARN)
    arrow(ax, (2.0, 4.9), (2.4, 4.9)); arrow(ax, (4.0, 4.9), (4.4, 4.9))
    arrow(ax, (8.8, 4.9), (9.2, 4.85)); arrow(ax, (9.95, 4.4), (9.95, 4.0)); arrow(ax, (10.7, 3.55), (11.1, 3.9))
    ax.text(6.6, 1.7, "架构 = Text-01（456B/45.9B）；差异在 RL 后训练（CISPO）与长思维推理，不在网络结构",
            ha="center", fontsize=10, color=SUB)
    save(fig, "minimax_m1", "arch_overview.png")

def m1_mech():
    fig, ax = new_ax(13.5, 6.2)
    ax.text(6.75, 5.8, "M1 核心：混合注意力让「测试时算力」近线性扩展", ha="center", fontsize=12.5, fontweight="bold", color=ACCENT)
    box(ax, 0.8, 3.0, 3.5, 1.6, "长输入\n由 70 层 Lightning\n近线性吸收\n(状态固定)", fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=10)
    box(ax, 5.0, 3.0, 3.5, 1.6, "长思维链生成\n每步只读写小块\n递归状态，省算力", fc=CORE, ec=ACCENT, fs=10, bold=True)
    box(ax, 9.2, 3.0, 3.5, 1.6, "CISPO 算法\n裁剪重要性采样权重\n而非 token，提 RL 效率", fc=WARN_SOFT, ec=WARN, fs=10, tc=WARN)
    ax.text(6.75, 2.0, "100K 生成长度下 FLOPs 约为 DeepSeek-R1 的 25%；512×H800 三周完成全量 RL",
            ha="center", fontsize=10, color=SUB)
    save(fig, "minimax_m1", "arch_mech.png")

# ============================================================
# minimax_m2
# ============================================================
def m2_overview():
    fig, ax = new_ax(13.5, 7.2)
    box(ax, 0.3, 4.6, 1.6, 1.0, "文本 token\n词表 200064", fc="#F4F4F2", ec=EDGE, fs=10)
    box(ax, 2.2, 4.6, 1.5, 1.0, "词嵌入", fs=10)
    box(ax, 4.1, 2.0, 4.4, 4.6, "", fc=CORE, ec=ACCENT)
    ax.text(6.3, 6.1, "62 × MiniMaxM2DecoderLayer", ha="center", fontsize=11.5, fontweight="bold", color=ACCENT)
    box(ax, 4.4, 4.55, 3.8, 1.2, "全部标准 GQA Softmax 注意力\n48Q/8KV·head128\nQK-Norm + 部分RoPE(64/128)\n（取消 Lightning 线性层）", fc=WARN_SOFT, ec=WARN, fs=9.3, tc=WARN)
    box(ax, 4.4, 2.95, 3.8, 1.2, "256 专家 MoE · top-8\nsigmoid 打分 + 校正偏置\nexpert inter=1536", fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=9.5)
    box(ax, 4.4, 2.15, 3.8, 0.6, "普通残差（取消 α 门控）", fc=NODE, ec=EDGE, fs=9.3)
    box(ax, 8.9, 4.6, 1.4, 0.9, "RMSNorm", fs=10)
    box(ax, 8.9, 3.2, 1.4, 0.9, "LM Head", fs=10)
    box(ax, 10.7, 4.6, 1.4, 0.9, "MTP 头 ×3\n(预测未来)", fc=MM_SOFT, ec=MM, fs=9.3, tc="#155E75")
    box(ax, 12.3, 3.8, 1.0, 0.9, "token", fc=ACCENT_SOFT, ec=ACCENT, fs=10, bold=True)
    arrow(ax, (1.9, 5.1), (2.2, 5.1)); arrow(ax, (3.7, 5.1), (4.1, 5.1))
    arrow(ax, (8.5, 5.1), (8.9, 5.05)); arrow(ax, (9.6, 4.6), (9.6, 4.1)); arrow(ax, (10.3, 3.65), (10.7, 4.9))
    ax.text(6.3, 1.2, "约 229.9B 总参 / 每 token 激活 9.8B · 上下文 196608 · 发布权重 FP8(e4m3fn)", ha="center", fontsize=10, color=SUB)
    save(fig, "minimax_m2", "arch_overview.png")

def m2_mech():
    fig, ax = new_ax(13.5, 6.4)
    ax.text(6.75, 6.0, "M2 核心机制：sigmoid 路由 + QK-Norm + MTP", ha="center", fontsize=12.5, fontweight="bold", color=ACCENT)
    box(ax, 0.6, 3.0, 3.6, 1.8, "Router（门）\nhidden→256\nsigmoid 打分\n+ e_score 校正偏置\ntop-8 后归一化", fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=9.8)
    box(ax, 4.9, 3.0, 3.6, 1.8, "QK-Norm\n对 q/k 投影后\n各做 RMSNorm\n(per-layer 拼接头)\n稳定长上下文", fc=WARN_SOFT, ec=WARN, fs=9.8, tc=WARN)
    box(ax, 9.2, 3.0, 3.6, 1.8, "MTP 多 token 预测\n3 个 MTP 模块\n额外预测未来 token\n加速训练", fc=MM_SOFT, ec=MM, fs=9.8, tc="#155E75")
    ax.text(6.75, 2.0, "路由从 softmax 改 sigmoid + 可学习偏置，缓解专家负载；取消 Lightning、回到全软注意力",
            ha="center", fontsize=10, color=SUB)
    save(fig, "minimax_m2", "arch_mech.png")

# ============================================================
# minimax_m3_vl  (= MiniMax-M3)
# ============================================================
def m3_overview():
    fig, ax = new_ax(13.5, 7.4)
    box(ax, 0.2, 5.0, 1.5, 0.9, "文本 token", fc="#F4F4F2", ec=EDGE, fs=9.5)
    box(ax, 0.2, 3.3, 1.5, 0.9, "图像/视频", fc=MM_SOFT, ec=MM, fs=9.5, tc="#155E75")
    box(ax, 2.0, 3.3, 1.7, 0.9, "CLIP 视觉塔\npatch14·32层\nhidden1280", fc=MM_SOFT, ec=MM, fs=9.0, tc="#155E75")
    box(ax, 4.1, 4.1, 1.5, 0.9, "投影→6144\npatch merge", fc=MM_SOFT, ec=MM, fs=9.0, tc="#155E75")
    box(ax, 6.0, 4.1, 1.4, 1.2, "early fusion\n<image>=200025\n<video>=200026", fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=8.8)
    box(ax, 7.8, 4.6, 1.7, 1.0, "前 3 层 Dense\n全注意力\ninter 12288", fc=NODE, ec=EDGE, fs=9.0)
    box(ax, 9.9, 3.6, 1.9, 2.0, "57 层 MoE + MSA\n128专家 top4\n+1 共享专家\n块稀疏注意力", fc=CORE, ec=ACCENT, fs=9.2, bold=True)
    box(ax, 12.0, 4.3, 1.3, 0.9, "LM Head\n+7 MTP", fc=ACCENT_SOFT, ec=ACCENT, fs=9.0)
    arrow(ax, (1.7, 5.45), (6.0, 4.9))
    arrow(ax, (1.7, 3.75), (2.0, 3.75), color=MM); arrow(ax, (3.7, 3.75), (4.1, 4.4), color=MM)
    arrow(ax, (5.6, 4.55), (6.0, 4.55), color=MM)
    arrow(ax, (7.4, 4.7), (7.8, 4.9)); arrow(ax, (9.5, 5.0), (9.9, 4.9))
    arrow(ax, (11.8, 4.6), (12.0, 4.75))
    ax.text(6.7, 2.6, "原生多模态 · 1M 上下文 · MSA(MiniMax Sparse Attention) · SwiGLU-OAI(α1.702)",
            ha="center", fontsize=10, color=SUB)
    save(fig, "minimax_m3_vl", "arch_overview.png")

def m3_msa():
    fig, ax = new_ax(13.5, 6.2)
    ax.text(6.75, 5.8, "MSA 块稀疏注意力（MiniMax Sparse Attention）", ha="center", fontsize=12.5, fontweight="bold", color=ACCENT)
    box(ax, 0.6, 3.0, 3.6, 1.8, "Index 分支\n4 个 index 头\n对 128-token KV 块\n打分", fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=9.8)
    box(ax, 4.9, 3.0, 3.6, 1.8, "选块\n每查询每 GQA 组\ntop-16 块\n+ 1 个局部块", fc=CORE, ec=ACCENT, fs=9.8, bold=True)
    box(ax, 9.2, 3.0, 3.6, 1.8, "Main 分支\n只在选中块上\n做 GQA 注意力\ndense 精确计算", fc=WARN_SOFT, ec=WARN, fs=9.8, tc=WARN)
    arrow(ax, (4.2, 3.9), (4.9, 3.9)); arrow(ax, (8.5, 3.9), (9.2, 3.9))
    ax.text(6.75, 2.0, "块大小 128 · index_head_dim128 · 1M 上下文下单 token 算力约为 M2 的 1/20",
            ha="center", fontsize=10, color=SUB)
    save(fig, "minimax_m3_vl", "arch_msa.png")

def m3_vision():
    fig, ax = new_ax(13.5, 6.2)
    ax.text(6.75, 5.8, "M3 视觉塔 token 流（3D RoPE + patch 合并压缩）", ha="center", fontsize=12.5, fontweight="bold", color=MM)
    box(ax, 0.5, 3.0, 2.0, 1.5, "图像/视频\n最大 2016\n分段≤4帧", fc=MM_SOFT, ec=MM, fs=9.8, tc="#155E75")
    box(ax, 2.9, 3.0, 2.2, 1.5, "CLIP ViT\npatch14·32层\nhidden1280\n3D 轴向 RoPE", fc=MM_SOFT, ec=MM, fs=9.3, tc="#155E75")
    box(ax, 5.5, 3.0, 2.2, 1.5, "patch merge\n空间×2 时间×2\ntoken 压缩", fc=MM_SOFT, ec=MM, fs=9.3, tc="#155E75")
    box(ax, 8.1, 3.0, 2.2, 1.5, "投影→6144\nprojection", fc=MM_SOFT, ec=MM, fs=9.3, tc="#155E75")
    box(ax, 10.7, 3.0, 2.3, 1.5, "拼入文本序列\n576/img\n(200025/26)", fc=ACCENT_SOFT, ec=ACCENT, fs=9.2)
    for x in [(2.5, 2.9), (5.1, 5.5), (7.7, 8.1), (10.3, 10.7)]:
        arrow(ax, (x[0], 3.75), (x[1], 3.75), color=MM)
    ax.text(6.75, 2.0, "video 帧间用 temporal patch embedding 建模时序；支持图片与视频统一编码",
            ha="center", fontsize=10, color=SUB)
    save(fig, "minimax_m3_vl", "arch_vision.png")

for f in [text01_overview, text01_mech, vl01_overview, vl01_vision, vl01_mech,
          m1_overview, m1_mech, m2_overview, m2_mech, m3_overview, m3_msa, m3_vision]:
    f()
print("ALL DONE")
