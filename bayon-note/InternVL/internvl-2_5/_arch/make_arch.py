# -*- coding: utf-8 -*-
"""InternVL 2.5 家族架构详解 — 5 类架构图（论文配图级 PNG）。
配色严格遵循 pattern-spec §3 冻结值。输出到本目录。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib import font_manager as fm

# 中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11

# 冻结配色
INK   = "#24292f"
SUB   = "#6b7280"
LINE  = "#e3e6ea"
BG    = "#f6f7f9"
PANEL = "#ffffff"
ACC   = "#0e7490"   # accent
ACCIN = "#155e75"   # accent-ink
ACCSF = "#e0f2f5"   # accent-soft
ACCLN = "#b7dbe4"   # accent-line
HL    = "#b45309"   # hl
HLSF  = "#f7ead6"   # hl-soft
HLLN  = "#e8cfae"   # hl-line

OUT = r"E:\transformers\bayon-note\InternVL\internvl-2_5\_arch"

def box(ax, x, y, w, h, text, fc=PANEL, ec=LINE, lw=1.4, fs=10.5, tc=INK,
        sub=None, sub_fs=8.8, sub_c=SUB, rounded=0.02, bold=False):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.012,rounding_size={rounded}",
                       linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p)
    weight = "bold" if bold else "normal"
    if sub is None:
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fs, color=tc, weight=weight, zorder=3)
    else:
        ax.text(x + w/2, y + h*0.62, text, ha="center", va="center",
                fontsize=fs, color=tc, weight="bold", zorder=3)
        ax.text(x + w/2, y + h*0.26, sub, ha="center", va="center",
                fontsize=sub_fs, color=sub_c, zorder=3)

def arrow(ax, x1, y1, x2, y2, color=SUB, lw=1.6, style="-|>", ls="-", ms=14):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                        mutation_scale=ms, linewidth=lw, color=color,
                        linestyle=ls, zorder=1, shrinkA=0, shrinkB=0)
    ax.add_patch(a)

def new_ax(w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.patch.set_facecolor(PANEL)
    return fig, ax

def save(fig, name):
    fig.savefig(f"{OUT}\\{name}", dpi=200, bbox_inches="tight",
                facecolor=PANEL, edgecolor="none", pad_inches=0.12)
    plt.close(fig)
    print("saved", name)

# ============ 图 1：端到端整体架构 ============
fig, ax = new_ax(13.5, 6.2)
# 输入图像
box(ax, 2, 40, 12, 22, "输入图像", fc=ACCSF, ec=ACC, lw=1.8, fs=11, bold=True,
    sub="任意宽高比 / 分辨率\n单图 · 多图 · 视频帧")
arrow(ax, 14, 51, 19, 51)
# 动态切图
box(ax, 19, 38, 15, 26, "动态分辨率切图", fc=HLSF, ec=HL, lw=1.8, fs=10.5, bold=True,
    sub="按宽高比切 448×448 tile\n≤12 tile + 1 缩略图\n(force_image_size=448)")
arrow(ax, 34, 51, 39, 51)
# InternViT
box(ax, 39, 38, 16, 26, "InternViT 视觉塔", fc=ACCSF, ec=ACC, lw=1.8, fs=10.5, bold=True,
    sub="300M: h1024/24层/16头\n6B: h3200/45层/25头\n本地1B用 300M-448px-V2.5")
arrow(ax, 55, 51, 60, 51)
# pixel shuffle
box(ax, 60, 44, 13, 14, "Pixel Unshuffle", fc=PANEL, ec=ACCIN, lw=1.6, fs=10, bold=True,
    sub="1024→256 token\ndownsample=0.5")
arrow(ax, 73, 51, 78, 51)
# MLP projector
box(ax, 78, 44, 12, 14, "2层 MLP 投影", fc=PANEL, ec=ACCIN, lw=1.6, fs=10, bold=True,
    sub="4096→896\n随机初始化")
arrow(ax, 84, 44, 84, 30, color=ACCIN)
# LLM
box(ax, 72, 12, 22, 16, "Qwen2.5-0.5B LLM", fc=HLSF, ec=HL, lw=1.8, fs=10.5, bold=True,
    sub="h896 / 24层 / 14Q头 / 2KV头\n图像token与文本embedding拼接")
# text input branch
box(ax, 40, 14, 18, 12, "文本 token", fc=PANEL, ec=LINE, fs=10, bold=True,
    sub="ChatML 模板 · internvl2_5")
arrow(ax, 58, 20, 72, 20)
arrow(ax, 31, 40, 31, 26, color=SUB, ls="--")
ax.text(31, 33, "缩略图 tile\n同样过 ViT", ha="center", va="center", fontsize=8, color=SUB)
# output
arrow(ax, 94, 20, 98, 20, color=HL)
ax.text(97, 24, "下一\ntoken", ha="center", va="center", fontsize=9, color=HL, weight="bold")
ax.text(50, 90, "图 1  InternVL 2.5 端到端整体架构（ViT–MLP–LLM 范式，以本地 1B 为主尺寸）",
        ha="center", fontsize=12, color=INK, weight="bold")
ax.text(50, 6, "视觉特征经 MLP 投影后，在 embedding 层通过 masked_scatter 替换图像占位 token，再送入因果 LLM 自回归生成。",
        ha="center", fontsize=9, color=SUB)
save(fig, "fig1_overall.png")

# ============ 图 2：InternViT 层细节 ============
fig, ax = new_ax(12.5, 7.2)
ax.text(50, 98, "图 2  InternViT 视觉编码器层结构（本地 1B = InternViT-300M-448px-V2.5）",
        ha="center", fontsize=12, color=INK, weight="bold")
box(ax, 36, 88, 28, 7, "输入图像 tile  448×448×3", fc=ACCSF, ec=ACC, fs=10, bold=True)
arrow(ax, 50, 88, 50, 84)
box(ax, 32, 76, 36, 8, "Patch Embedding  Conv2d 14×14, stride 14", fc=PANEL, ec=ACCIN, fs=9.3, bold=True,
    sub="448/14 = 32 → 32×32 = 1024 patch token（+1 CLS）")
arrow(ax, 50, 76, 50, 72)
# 堆叠 block 容器（标题置顶，不压内框）
ax.add_patch(FancyBboxPatch((20, 38), 60, 33,
             boxstyle="round,pad=0.012,rounding_size=0.02",
             linewidth=1.8, edgecolor=ACC, facecolor=ACCSF, zorder=2))
ax.text(50, 67, "InternViT Block × 24（300M 规格）", ha="center", va="center",
        fontsize=11, color=ACCIN, weight="bold", zorder=3)
box(ax, 24, 55, 24, 8, "Multi-Head Self-Attention", fc=PANEL, ec=ACCIN, fs=9.0, bold=True,
    sub="16 头 · head_dim=64 · qkv_bias=True")
box(ax, 52, 55, 24, 8, "LayerNorm（pre-norm）", fc=PANEL, ec=ACCIN, fs=9.0, bold=True,
    sub="300M 用标准 LN · 无 QK-Norm")
box(ax, 24, 43, 24, 8, "MLP / FFN", fc=PANEL, ec=ACCIN, fs=9.0, bold=True,
    sub="GELU · 1024→4096→1024")
box(ax, 52, 43, 24, 8, "残差连接", fc=PANEL, ec=ACCIN, fs=9.0, bold=True,
    sub="x + Attn(LN(x))；x + FFN(LN(x))")
arrow(ax, 50, 38, 50, 34)
box(ax, 32, 26, 36, 8, "取 last_hidden_state，丢弃 CLS", fc=HLSF, ec=HL, lw=1.6, fs=9.3, bold=True,
    sub="[:, 1:, :] → 1024 个 1024 维 token")
arrow(ax, 50, 26, 50, 22)
box(ax, 30, 13, 40, 8, "Pixel Unshuffle（见投影层）", fc=PANEL, ec=HL, fs=9.3, bold=True,
    sub="1024 → 256 token · 通道 1024→4096")
# 侧注：6B 变体差异
box(ax, 3, 42, 14, 26, "6B 变体差异", fc=HLSF, ec=HL, lw=1.5, fs=9.5, bold=True,
    sub="hidden 3200\n45 层\n25 头\nQK-Norm: 有\nRMSNorm\n(本1B不用)")
save(fig, "fig2_vit.png")

# ============ 图 3：动态分辨率 tile 机制 ============
fig, ax = new_ax(13.0, 6.6)
# 原图
ax.add_patch(Rectangle((4, 55), 20, 26, facecolor=ACCSF, edgecolor=ACC, lw=1.6))
ax.text(14, 68, "原图\n(任意比例)", ha="center", va="center", fontsize=10, color=ACCIN, weight="bold")
arrow(ax, 24, 68, 30, 68)
ax.text(27, 71, "按比例缩放", ha="center", fontsize=8.5, color=SUB)
# tile 网格 3x4 = 12
gx, gy = 32, 50
cols, rows = 4, 3
tw, th = 5.5, 6.5
for r in range(rows):
    for c in range(cols):
        ax.add_patch(Rectangle((gx + c*tw, gy + (rows-1-r)*th), tw-0.3, th-0.3,
                               facecolor=PANEL, edgecolor=ACC, lw=1.1))
ax.text(gx + cols*tw/2, gy + rows*th + 3, "动态 tile 网格（最多 12 个 448×448）",
        ha="center", fontsize=9.5, color=ACCIN, weight="bold")
arrow(ax, gx + cols*tw + 1, 62, gx + cols*tw + 6, 62)
# thumbnail
ax.add_patch(Rectangle((gx + cols*tw + 6, 56), 9, 12, facecolor=HLSF, edgecolor=HL, lw=1.6))
ax.text(gx + cols*tw + 10.5, 62, "缩略图\n1 tile\n(use_thumbnail)", ha="center", va="center",
        fontsize=8.3, color=HL, weight="bold")
# 每 tile token
arrow(ax, 60, 45, 60, 38, color=SUB)
box(ax, 38, 20, 46, 16, "每个 448 tile → 256 个视觉 token", fc=HLSF, ec=HL, lw=1.8, fs=10.5, bold=True,
    sub="32×32=1024 patch → 去CLS → pixel shuffle → 16×16=256\n最大 tile 数 12+1=13 → 最多 13×256 = 3328 视觉 token",
    sub_fs=9)
ax.text(65, 92, "图 3  动态分辨率与像素 tile 机制（dynamic_image_size / max_dynamic_patch=12 / use_thumbnail）",
        ha="center", fontsize=11.5, color=INK, weight="bold")
save(fig, "fig3_tile.png")

# ============ 图 4：token 流 / 占位 token 注入 ============
fig, ax = new_ax(13.0, 5.8)
chain = [
    (3, "图像像素", "(b,3,448,448)", ACCSF, ACC),
    (20, "InternViT", "1024×1024", ACCSF, ACC),
    (37, "去CLS+PixelShuffle", "256×4096", PANEL, ACCIN),
    (56, "MLP 投影", "256×896", PANEL, ACCIN),
    (76, "图像占位 token", "替换<image>", HLSF, HL),
]
for x, t, s, fc, ec in chain:
    box(ax, x, 55, 15, 16, t, fc=fc, ec=ec, lw=1.6, fs=9.8, bold=True, sub=s, sub_fs=8.5)
for i in range(len(chain)-1):
    arrow(ax, chain[i][0]+15, 63, chain[i+1][0], 63)
# embedding 拼接示意
box(ax, 6, 22, 88, 18, "", fc=BG, ec=LINE, lw=1.2)
ax.text(50, 37, "Embedding 层拼接（masked_scatter 替换）", ha="center", fontsize=10, color=INK, weight="bold")
# 段
segs = [(10, "文本 BOS", 12), (24, "[图像占位 256×896]", 26, HLSF, HL),
        (52, "文本指令…", 20), (74, "文本回答…", 20)]
ax.add_patch(Rectangle((10,25), 12, 7, facecolor=PANEL, edgecolor=SUB))
ax.text(16, 28.5, "BOS", ha="center", va="center", fontsize=8.5)
ax.add_patch(Rectangle((23,25), 28, 7, facecolor=HLSF, edgecolor=HL, lw=1.5))
ax.text(37, 28.5, "图像占位（被视觉特征替换）", ha="center", va="center", fontsize=8.5, color=HL, weight="bold")
ax.add_patch(Rectangle((52,25), 18, 7, facecolor=PANEL, edgecolor=SUB))
ax.text(61, 28.5, "文本指令", ha="center", va="center", fontsize=8.5)
ax.add_patch(Rectangle((71,25), 18, 7, facecolor=PANEL, edgecolor=SUB))
ax.text(80, 28.5, "文本回答", ha="center", va="center", fontsize=8.5)
ax.text(50, 14, "→ 因果 LLM 前向：视觉 token 与文本 token 在同一序列内联合注意力，自回归输出下一 token",
        ha="center", fontsize=9.2, color=ACCIN)
ax.text(50, 90, "图 4  视觉 token 流与图像占位 token 的拼接位置（本地 1B：LLM hidden=896）",
        ha="center", fontsize=11.5, color=INK, weight="bold")
save(fig, "fig4_tokenflow.png")

# ============ 图 5：2.0 → 2.5 演进对比 ============
fig, ax = new_ax(13.0, 6.8)
ax.text(50, 90, "图 5  InternVL 2.0 → 2.5 演进对比（架构同构：ViT–MLP–LLM；升级在模型、数据、测试时扩展）",
        ha="center", fontsize=11.5, color=INK, weight="bold")
# 左：2.0
ax.add_patch(FancyBboxPatch((4, 22), 40, 58, boxstyle="round,pad=0.012,rounding_size=0.02",
             linewidth=2, edgecolor=ACC, facecolor=ACCSF, zorder=2))
ax.text(24, 74, "InternVL 2.0（2024-07）", ha="center", fontsize=12, color=ACCIN, weight="bold", zorder=3)
items20 = ["视觉塔：InternViT-300M / 6B（V1.5）",
           "语言底座：Qwen2 / InternLM2 / Phi-3",
           "1B 底座：Qwen2-0.5B（vocab 151655）",
           "预训练样本：7.3M",
           "推理：直接回答为主",
           "家族：开源 1B–76B + Pro 108B"]
for i, t in enumerate(items20):
    ax.text(7, 66 - i*6.4, "• " + t, ha="left", va="center", fontsize=9.3, color=INK)
# 右：2.5
ax.add_patch(FancyBboxPatch((56, 22), 40, 58, boxstyle="round,pad=0.012,rounding_size=0.02",
             linewidth=2, edgecolor=HL, facecolor=HLSF, zorder=2))
ax.text(76, 74, "InternVL 2.5（2024-12）", ha="center", fontsize=12, color=HL, weight="bold", zorder=3)
items25 = ["视觉塔升级：300M/6B 增量 NTP → V2.5",
           "底座升级：Qwen2.5 / InternLM2.5",
           "1B 底座：Qwen2.5-0.5B（vocab 151674）",
           "样本翻倍：16.3M + 严格去重过滤",
           "推理：CoT 思维链 test-time scaling",
           "家族：1B – 78B（新增 38B/78B）"]
for i, t in enumerate(items25):
    ax.text(59, 66 - i*6.4, "• " + t, ha="left", va="center", fontsize=9.3, color=INK)
# 中间箭头
arrow(ax, 44, 51, 56, 51, color=HL, lw=2.6, ms=22)
ax.text(50, 55, "升级", ha="center", fontsize=10, color=HL, weight="bold")
ax.text(50, 14, "保持不变：动态分辨率 448 tile + pixel unshuffle(1024→256) + 2层MLP + 因果 LLM；3 阶段管线（MLP warmup → ViT 增量 → 全量 SFT）。",
        ha="center", fontsize=8.8, color=SUB)
save(fig, "fig5_evo.png")

print("ALL DONE")
