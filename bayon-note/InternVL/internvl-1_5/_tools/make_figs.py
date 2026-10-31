# -*- coding: utf-8 -*-
"""Generate architecture diagrams for InternVL 1.5 family doc."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.font_manager as fm
import os

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_arch")
os.makedirs(OUT, exist_ok=True)

# frozen palette
INK = "#24292f"
SUB = "#6b7280"
LINE = "#e3e6ea"
TEAL = "#0e7490"
TEAL_INK = "#155e75"
TEAL_SOFT = "#e0f2f5"
TEAL_LINE = "#b7dbe4"
AMBER = "#b45309"
AMBER_SOFT = "#f7ead6"
AMBER_LINE = "#e8cfae"
WHITE = "#ffffff"


def box(ax, x, y, w, h, text, fc=WHITE, ec=SUB, fs=10, fw="normal", tc=INK, r=0.02, lw=1.3):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, mutation_aspect=1)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, fontweight=fw, linespacing=1.45)


def arrow(ax, x1, y1, x2, y2, color=SUB, lw=1.6, style="-|>", ls="-"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13,
                        color=color, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0)
    ax.add_patch(a)


# ============================================================
# Fig 1: end-to-end overall architecture
# ============================================================
fig, ax = plt.subplots(figsize=(12.2, 5.4), dpi=200)
ax.set_xlim(0, 122); ax.set_ylim(0, 54); ax.axis("off")

# input image tiles
box(ax, 2, 30, 16, 18, "输入图像\n动态分块\n1~12 块 ×448²\n(测试可到 40 块)", fc=TEAL_SOFT, ec=TEAL_LINE, fs=9.5)
box(ax, 2, 8, 16, 14, "缩略图 thumbnail\n整图缩到 448²\n提供全局上下文", fc=TEAL_SOFT, ec=TEAL_LINE, fs=9)

# ViT
box(ax, 26, 22, 22, 24, "InternViT 视觉编码器\n300M：1024维·24层·16头\n6B：3200维·45层·25头\npatch=14 · 448px",
    fc=WHITE, ec=TEAL, fs=9.5, fw="bold", tc=TEAL_INK, lw=1.8)
arrow(ax, 18, 39, 26, 36, color=TEAL)
arrow(ax, 18, 15, 26, 30, color=TEAL, ls="--")

# pixel shuffle
box(ax, 54, 28, 18, 12, "像素重排 pixel shuffle\ndownsample=0.5\ntoken 数 ÷ 4", fc=AMBER_SOFT, ec=AMBER_LINE, fs=9.5)
arrow(ax, 48, 34, 54, 34, color=AMBER, lw=1.8)

# MLP projector
box(ax, 78, 28, 18, 12, "MLP 投影器\nLinear·GELU·Linear\n(随机初始化)", fc=AMBER_SOFT, ec=AMBER, fs=9.5, fw="bold", tc=AMBER, lw=1.6)
arrow(ax, 72, 34, 78, 34, color=AMBER, lw=1.8)

# LLM
box(ax, 100, 20, 19, 26, "InternLM2 语言模型\n1.8B：2048维·24层\n20B：6144维·48层\n文本自回归解码",
    fc=WHITE, ec=AMBER, fs=9.5, fw="bold", tc=AMBER, lw=1.8)
arrow(ax, 96, 34, 100, 38, color=AMBER, lw=1.8)

# text branch
box(ax, 100, 4, 19, 10, "文本 token 嵌入\n<image> 占位符位置\n替换为视觉 token", fc=TEAL_SOFT, ec=TEAL_LINE, fs=9)
arrow(ax, 96, 9, 100, 9, color=TEAL)
arrow(ax, 109.5, 14, 109.5, 20, color=TEAL)

ax.text(61, 48.5, "图 1  InternVL 1.5 端到端整体架构（ViT–MLP–LLM，无 QLLaMA 桥接）",
        ha="center", fontsize=11.5, fontweight="bold", color=INK)
ax.text(61, 2, "每个 448² tile 经 ViT 得 32×32=1024 个 patch token，像素重排后压缩为 16×16=256 个视觉 token，再经 MLP 投影到 LLM 词嵌入维度",
        ha="center", fontsize=8.6, color=SUB)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_overall.png"), bbox_inches="tight", facecolor="white")
plt.close(fig)

# ============================================================
# Fig 2: InternViT block detail
# ============================================================
fig, ax = plt.subplots(figsize=(11.5, 6.6), dpi=200)
ax.set_xlim(0, 115); ax.set_ylim(0, 66); ax.axis("off")

box(ax, 38, 58, 39, 6, "图像 3×448×448", fc=TEAL_SOFT, ec=TEAL_LINE, fs=10)
arrow(ax, 57.5, 58, 57.5, 54.5)
box(ax, 33, 47, 49, 7, "Patch Embedding：卷积核/步长 14 → 32×32 patch", fc=WHITE, ec=TEAL, fs=9.5)
arrow(ax, 57.5, 47, 57.5, 43.5)
box(ax, 33, 36, 49, 7, "+ 可学习位置编码（无 class token）", fc=WHITE, ec=TEAL, fs=9.5)
arrow(ax, 57.5, 36, 57.5, 32.5)

# repeated block
box(ax, 20, 6, 75, 26, "", fc="#fbfdfe", ec=TEAL, lw=1.8)
ax.text(57.5, 29.5, "重复 N 次（300M：24 层 / 6B：45 层）", ha="center", fontsize=9.5, color=TEAL_INK, fontweight="bold")
box(ax, 24, 20, 20, 6, "LayerNorm\n6B 用 RMSNorm", fc=TEAL_SOFT, ec=TEAL_LINE, fs=8.6)
box(ax, 50, 20, 22, 6, "自注意力（多头）\n6B：QK Norm", fc=WHITE, ec=TEAL, fs=8.6)
box(ax, 76, 20, 16, 6, "残差相加", fc=AMBER_SOFT, ec=AMBER_LINE, fs=8.6)
box(ax, 24, 10, 20, 6, "LayerNorm", fc=TEAL_SOFT, ec=TEAL_LINE, fs=8.6)
box(ax, 50, 10, 22, 6, "MLP（GELU）\nintermediate=4×hidden", fc=WHITE, ec=TEAL, fs=8.6)
box(ax, 76, 10, 16, 6, "残差相加", fc=AMBER_SOFT, ec=AMBER_LINE, fs=8.6)
arrow(ax, 44, 23, 50, 23, color=TEAL)
arrow(ax, 72, 23, 76, 23, color=AMBER)
arrow(ax, 44, 13, 50, 13, color=TEAL)
arrow(ax, 72, 13, 76, 13, color=AMBER)
arrow(ax, 57.5, 32.5, 57.5, 26, color=SUB)

box(ax, 33, 0.5, 49, 5, "取指定层特征（select_layer=-1）→ reshape 为 32×32 特征图", fc=AMBER_SOFT, ec=AMBER, fs=9)
arrow(ax, 57.5, 6, 57.5, 5.5, color=AMBER)

# side annotation
ax.text(112, 22,
        "规格对照\n"
        "300M: h=1024\n  heads=16\n  inter=4096\n  qkv_bias=T\n  qk_norm=F\n"
        "6B: h=3200\n  heads=25\n  inter=12800\n  qkv_bias=F\n  qk_norm=T",
        ha="right", va="center", fontsize=8.4, color=INK,
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=LINE))

ax.text(57.5, 64, "图 2  InternViT 视觉编码器层结构（pre-norm ViT 块，两种规格同构）",
        ha="center", fontsize=11.5, fontweight="bold", color=INK)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_vit.png"), bbox_inches="tight", facecolor="white")
plt.close(fig)

# ============================================================
# Fig 3: dynamic tiling mechanism
# ============================================================
fig, ax = plt.subplots(figsize=(12.2, 5.6), dpi=200)
ax.set_xlim(0, 122); ax.set_ylim(0, 56); ax.axis("off")

ax.text(61, 53, "图 3  动态高分辨率分块机制（按长宽比匹配 35 种预设网格）",
        ha="center", fontsize=11.5, fontweight="bold", color=INK)

# original image
box(ax, 3, 26, 18, 22, "输入图像\n800×1300\n（任意比例）", fc="#fafbfc", ec=SUB, fs=9.5)
arrow(ax, 21, 37, 27, 37, color=TEAL, lw=1.8)
ax.text(24, 40, "匹配比例", fontsize=8.5, color=TEAL_INK, ha="center")

# grid 2x3 tiles
gx, gy, tw, th = 28, 22, 8, 10
for r in range(2):
    for c in range(3):
        ax.add_patch(Rectangle((gx + c * tw, gy + (1 - r) * th), tw - 0.4, th - 0.4,
                               fc=TEAL_SOFT, ec=TEAL_LINE, lw=1.2))
ax.text(gx + 1.5 * tw, gy + 2 * th + 1.5, "缩放为 896×1344（2:3 网格）",
        ha="center", fontsize=8.6, color=SUB)
ax.text(gx + 1.5 * tw, gy - 2.5, "6 个 tile，每个 448×448", ha="center", fontsize=8.8, color=TEAL_INK)

arrow(ax, gx + 3 * tw, 37, 56, 37, color=TEAL, lw=1.8)

# per tile tokens
box(ax, 57, 30, 22, 14, "每个 tile\nViT: 1024 token\npixel shuffle: 256 token", fc=WHITE, ec=TEAL, fs=9)
arrow(ax, 79, 37, 84, 37, color=AMBER, lw=1.8)

# thumbnail
box(ax, 28, 6, 24, 9, "缩略图：整图压到 448×448\n→ 256 个 token（全局视图）", fc=AMBER_SOFT, ec=AMBER_LINE, fs=8.8)
arrow(ax, 52, 10.5, 60, 30, color=AMBER, ls="--")

# token count
box(ax, 85, 26, 34, 18,
    "视觉 token 总数\n训练：1~12 tile → 256~3328 token\n测试：最多 40 tile → 10496 token\n（零样本外推到 4K 分辨率）",
    fc=AMBER_SOFT, ec=AMBER, fs=9, fw="bold", tc=AMBER, lw=1.6)

ax.text(61, 1, "规则：优先选面积不超过原图 2 倍的最接近预设比例；use_thumbnail=true 时永远额外拼一个缩略图 token 序列",
        ha="center", fontsize=8.6, color=SUB)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_tiles.png"), bbox_inches="tight", facecolor="white")
plt.close(fig)

# ============================================================
# Fig 4: token stream
# ============================================================
fig, ax = plt.subplots(figsize=(12.2, 5.0), dpi=200)
ax.set_xlim(0, 122); ax.set_ylim(0, 50); ax.axis("off")
ax.text(61, 47, "图 4  Token 流：视觉特征如何拼接进 LLM 的 token 序列",
        ha="center", fontsize=11.5, fontweight="bold", color=INK)

# input sequence row
seqs = ["<bos>", "<img>", "<img>", "...", "<img>", "<human> 问题 <eoh>", "回答...", "<eos>"]
cols = [TEAL_SOFT, "#e6f4ea", "#e6f4ea", "#e6f4ea", "#e6f4ea", AMBER_SOFT, AMBER_SOFT, "#f3f4f6"]
x = 3
for s, c in zip(seqs, cols):
    w = 15 if len(s) > 6 else 10
    box(ax, x, 34, w, 8, s, fc=c, ec=LINE, fs=8.6)
    x += w + 0.6
ax.text(61, 44.5, "输入 input_ids（文本模板 internlm2-chat，<image> 占位符共 N×256 个）", ha="center", fontsize=9, color=SUB)

# replace
arrow(ax, 61, 34, 61, 28, color=TEAL, lw=2)
ax.text(64, 31, "把 <image> 位置的词嵌入\n替换为 MLP 输出的视觉嵌入", fontsize=9, color=TEAL_INK, va="center")

# embedded row
box(ax, 3, 16, 12, 8, "文本\n嵌入", fc=TEAL_SOFT, ec=TEAL_LINE, fs=8.6)
box(ax, 16, 16, 52, 8, "视觉嵌入序列（经 ViT + pixel shuffle + MLP）：每 tile 256 维向量，沿序列维拼接",
    fc=AMBER_SOFT, ec=AMBER, fs=9, fw="bold", tc=AMBER)
box(ax, 69, 16, 22, 8, "文本嵌入\n（问题）", fc=TEAL_SOFT, ec=TEAL_LINE, fs=8.6)
box(ax, 92, 16, 27, 8, "文本嵌入（回答，自回归生成）", fc="#f3f4f6", ec=LINE, fs=8.6)

arrow(ax, 61, 16, 61, 10, color=SUB, lw=2)
box(ax, 20, 2, 82, 8, "InternLM2 因果 Transformer：视觉 token 与文本 token 共享同一套注意力，无跨注意力桥接模块",
    fc=WHITE, ec=TEAL, fs=9.5, fw="bold", tc=TEAL_INK)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_tokenflow.png"), bbox_inches="tight", facecolor="white")
plt.close(fig)

# ============================================================
# Fig 5: evolution 1.0 -> 1.5
# ============================================================
fig, ax = plt.subplots(figsize=(12.2, 5.8), dpi=200)
ax.set_xlim(0, 122); ax.set_ylim(0, 58); ax.axis("off")
ax.text(61, 55, "图 5  InternVL 1.0 → 1.5 架构演进", ha="center", fontsize=11.5, fontweight="bold", color=INK)

# 1.0 row
ax.text(8, 46, "1.0\n(2312.14238)", fontsize=9.5, color=SUB, ha="center", fontweight="bold")
box(ax, 18, 40, 20, 11, "InternViT-6B\n224px 固定分辨率", fc=TEAL_SOFT, ec=TEAL_LINE, fs=9)
box(ax, 42, 40, 24, 11, "QLLaMA 桥接模块\n8B 中间语言模型\n（对比学习桥接）", fc="#fdecec", ec="#c0392b", fs=9, tc="#8a3b3b")
box(ax, 70, 40, 22, 11, "LLaMA / QLLaMA\n解码器", fc="#f3f4f6", ec=LINE, fs=9)
ax.text(99, 45.5, "参数大、训练复杂\n分辨率低", fontsize=8.8, color="#8a3b3b", va="center")

arrow(ax, 38, 45.5, 42, 45.5, color=SUB)
arrow(ax, 66, 45.5, 70, 45.5, color=SUB)

# arrow down
arrow(ax, 61, 38, 61, 32, color=AMBER, lw=2.2, style="-|>")
ax.text(64, 35, "三点改进", fontsize=9.5, color=AMBER, fontweight="bold")

# 1.5 row
ax.text(8, 22, "1.5\n(2404.16821)", fontsize=9.5, color=TEAL_INK, ha="center", fontweight="bold")
box(ax, 18, 16, 22, 12, "InternViT-6B/300M\n448px 连续学习\n动态分辨率", fc=TEAL_SOFT, ec=TEAL, fs=9, fw="bold", tc=TEAL_INK)
box(ax, 44, 16, 20, 12, "像素重排\n+ MLP 投影\n（直接对接）", fc=AMBER_SOFT, ec=AMBER, fs=9, fw="bold", tc=AMBER)
box(ax, 68, 16, 22, 12, "InternLM2\n1.8B/20B 直接当\nLLM 解码器", fc="#f3f4f6", ec=LINE, fs=9)
ax.text(99, 22, "ViT-MLP-LLM\n轻量、可迁移", fontsize=8.8, color=TEAL_INK, va="center")

arrow(ax, 40, 22, 44, 22, color=AMBER, lw=1.8)
arrow(ax, 64, 22, 68, 22, color=AMBER, lw=1.8)

ax.text(61, 8,
        "① 强视觉编码器：InternViT-6B 连续学习（48 层裁到 45 层、224→448px、动态分辨率）\n"
        "② 去掉 8B QLLaMA 桥接，改为像素重排（token ÷4）+ 随机初始化 MLP 直接拼接\n"
        "③ 高质量中英双语数据 + 动态分辨率 tile（1~12 训练 / 40 测试）+ 缩略图全局视图",
        ha="center", fontsize=8.8, color=INK, linespacing=1.7)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig5_evo.png"), bbox_inches="tight", facecolor="white")
plt.close(fig)

print("DONE", os.listdir(OUT))
