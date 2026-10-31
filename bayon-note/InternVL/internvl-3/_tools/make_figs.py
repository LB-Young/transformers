# -*- coding: utf-8 -*-
"""InternVL-3 家族架构详解 — 论文配图级架构图（5 张 PNG）"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = r"E:\transformers\bayon-note\InternVL\internvl-3\_arch"
os.makedirs(OUT, exist_ok=True)

# 冻结配色（与 HTML/docx 统一）
INK   = "#24292f"
SUB   = "#6b7280"
LINE  = "#c9cfd6"
ACC   = "#0e7490"
ACC_S = "#e0f2f5"
ACC_L = "#b7dbe4"
HL    = "#b45309"
HL_S  = "#f7ead6"
HL_L  = "#e8cfae"
PANEL = "#ffffff"

def box(ax, x, y, w, h, text, fc=PANEL, ec=INK, lw=1.2, fs=10.5, tc=INK, bold=False, round=0.02):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.012,rounding_size={round}",
                       fc=fc, ec=ec, lw=lw, mutation_aspect=1.0)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", linespacing=1.45)

def arrow(ax, x1, y1, x2, y2, color=INK, lw=1.6, style="-|>", ls="-"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                        color=color, lw=lw, linestyle=ls, shrinkA=2, shrinkB=2)
    ax.add_patch(a)

def tag(ax, x, y, text, fs=9, color=SUB, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color)

# ============================================================
# 图 1：端到端整体架构（图像 → ViT → 投影 → LLM）
# ============================================================
fig, ax = plt.subplots(figsize=(13.2, 5.6), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(0, 42); ax.axis("off")

# 输入图像
box(ax, 1.5, 15, 11, 12, "输入图像\nH×W×3\n(动态分辨率)\n最多 12 个\n448×448 切片", fc=HL_S, ec=HL, fs=9.5, bold=True)
# tile 预处理
box(ax, 15, 15, 11, 12, "动态分辨率\n切片预处理\nresize→切 448\n+ 缩略图\n(thumbnail)", fc=PANEL, ec=LINE, fs=9.5)
# ViT
box(ax, 29, 14, 15, 14, "InternViT-300M\n视觉编码器\n24 层 · hidden 1024\n16 头 · patch 14×14\n输出 1024 维特征", fc=ACC_S, ec=ACC, fs=10, bold=True)
# pixel shuffle
box(ax, 47, 15, 12, 12, "Pixel Unshuffle\n(像素重组 0.5×)\n空间 32×32→16×16\n通道 1024→4096\n每切片 256 token", fc=PANEL, ec=LINE, fs=9.5)
# projector
box(ax, 62, 15, 12, 12, "MLP 投影器\nLayerNorm(4096)\nLinear 4096→896\nGELU\nLinear 896→896", fc=HL_S, ec=HL, fs=9.5, bold=True)
# LLM
box(ax, 77, 14, 15, 14, "Qwen2.5-0.5B\n语言模型 (Qwen2 架构)\n24 层 · hidden 896\n14 Q 头 / 2 KV 头\nvocab 151674", fc=ACC_S, ec=ACC, fs=10, bold=True)

arrow(ax, 12.5, 21, 15, 21)
arrow(ax, 26, 21, 29, 21)
arrow(ax, 44, 21, 47, 21)
arrow(ax, 59, 21, 62, 21, color=HL)
arrow(ax, 74, 21, 77, 21, color=ACC)

# 底部数据流标注
tag(ax, 7.0, 11.5, "pixel_values", fs=8.5)
tag(ax, 20.5, 11.5, "N×(3,448,448)", fs=8.5)
tag(ax, 36.5, 11.5, "(B,1025,1024) 含 CLS", fs=8.5)
tag(ax, 53.0, 11.5, "(B,256,4096)", fs=8.5)
tag(ax, 68.0, 11.5, "(B,256,896)", fs=8.5)
tag(ax, 84.5, 11.5, "自回归文本生成", fs=8.5, color=ACC)

# 顶部：文本支路
box(ax, 62, 31, 30, 6.5, "文本 token 序列：… <image>×(256×切片数) …\n图像占位 embedding 在 forward 中被视觉特征 masked_scatter 替换",
    fc=PANEL, ec=ACC_L, fs=9.5)
arrow(ax, 68, 27, 68, 31, color=ACC, ls="--")
arrow(ax, 84, 27, 84, 31, color=ACC, ls="--")

ax.text(50, 39.5, "InternVL-3 端到端架构：ViT – MLP – LLM（以本地 InternVL3-1B-hf 为例，38B/78B 视觉塔换为 InternViT-6B-V2.5）",
        ha="center", fontsize=12.5, fontweight="bold", color=INK)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig1_overview.png"), bbox_inches="tight", facecolor="white")
plt.close()

# ============================================================
# 图 2：InternViT-300M 视觉编码器层细节
# ============================================================
fig, ax = plt.subplots(figsize=(10.5, 9.2), dpi=200)
ax.set_xlim(0, 60); ax.set_ylim(0, 77); ax.axis("off")
ax.text(30, 74.5, "InternViT-300M-448px-V2.5 视觉编码器结构（本地 vision_config 逐字段核对）",
        ha="center", fontsize=12, fontweight="bold", color=INK)

# patch embedding
box(ax, 12, 66, 36, 5.5, "Patch Embedding：448×448×3 → patch 14×14 → 32×32=1024 个 patch token + 1 个 CLS token",
    fc=ACC_S, ec=ACC, fs=9.5, bold=True)
arrow(ax, 30, 66, 30, 63.5)
box(ax, 12, 58, 36, 5, "可学习绝对位置编码（use_absolute_position_embeddings=true）\n+ V2PE 可变视觉位置编码（长上下文，δ=1）",
    fc=PANEL, ec=LINE, fs=9)
arrow(ax, 30, 58, 30, 55.5)

# 单层 block ×24
box(ax, 8, 30, 44, 24, "", fc="#fbfcfd", ec=ACC, lw=1.4)
ax.text(30, 51.5, "× 24 个 ViT Block（hidden=1024, intermediate=4096, GELU）",
        ha="center", fontsize=10, fontweight="bold", color=ACC)
box(ax, 13, 44, 34, 4.5, "LayerNorm → 多头自注意力（16 头，qkv_bias=true）", fc=PANEL, ec=INK, fs=9.5)
box(ax, 13, 38.5, 34, 4, "× LayerScale（init=0.1）→ 残差相加", fc=HL_S, ec=HL, fs=9.5)
box(ax, 13, 33, 34, 4.5, "LayerNorm → MLP：1024→4096(GELU)→1024", fc=PANEL, ec=INK, fs=9.5)
box(ax, 13, 30.2, 34, 3.2, "× LayerScale（init=0.1）→ 残差相加", fc=HL_S, ec=HL, fs=9)

arrow(ax, 30, 55.5, 30, 53.2)
arrow(ax, 30, 44, 30, 42.5, color=HL)
arrow(ax, 30, 38.5, 30, 37.5, color=HL)

# 输出处理
arrow(ax, 30, 30, 30, 27.5)
box(ax, 8, 20.5, 44, 7, "取最后一层隐藏态（vision_feature_layer=-1）\n默认策略：丢弃 CLS，保留 1024 个 patch 特征\nreshape 为 32×32 空间网格 → 进入 Pixel Unshuffle",
    fc=ACC_S, ec=ACC, fs=9.5, bold=True)
arrow(ax, 30, 20.5, 30, 18)
box(ax, 8, 11.5, 44, 5.5, "use_mean_pooling=true（分类头侧使用）；本地 config：layer_norm_eps=1e-6,\nnorm_type=layer_norm, attention_bias=true, use_qk_norm=false",
    fc=PANEL, ec=LINE, fs=8.8)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig2_vit.png"), bbox_inches="tight", facecolor="white")
plt.close()

# ============================================================
# 图 3：动态分辨率 / tile 机制
# ============================================================
fig, ax = plt.subplots(figsize=(12.5, 6.4), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")

ax.text(50, 38, "动态分辨率切图机制（max_patches=12 / min_patches=1，每切片 448×448）",
        ha="center", fontsize=12.5, fontweight="bold", color=INK)

# 原图
box(ax, 3, 16, 16, 14, "原始图像\nH×W\n（任意宽高比）", fc=HL_S, ec=HL, fs=10, bold=True)
arrow(ax, 19, 23, 24, 23)
# 尺寸对齐
box(ax, 24, 16, 17, 14, "按短边缩放到\n448 的倍数\n计算切片网格\n(行数×列数)", fc=PANEL, ec=LINE, fs=9.5)
arrow(ax, 41, 23, 46, 23)
# tile 网格
box(ax, 46, 14, 24, 18, "切分为 N 个\n448×448 切片\n（N≤12）\n逐切片独立过 ViT", fc=ACC_S, ec=ACC, fs=9.5, bold=True)
arrow(ax, 70, 23, 75, 23)
# thumbnail
box(ax, 75, 22, 22, 10, "缩略图 thumbnail\n整图缩到 448×448\n补全全局视野", fc=HL_S, ec=HL, fs=9.5, bold=True)
box(ax, 75, 10, 22, 8, "每切片 → 256 视觉 token\n整图 token 数 = 256×N (+1 缩略图切片)", fc=PANEL, ec=LINE, fs=9)
arrow(ax, 86, 22, 86, 18.2, color=HL)

# 示例
ax.text(50, 7.5, "示例：一张 1344×448 的宽图 → 切成 3 个 448×448 切片 + 1 个缩略图 = 4 个 ViT 输入，共 4×256 = 1024 个视觉 token",
        ha="center", fontsize=9.5, color=SUB)
ax.text(50, 4, "来源：preprocessor_config.json（size=448×448, max_patches=12, min_patches=1）+ processing_internvl.py 切片占位逻辑",
        ha="center", fontsize=8.5, color=SUB)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig3_tile.png"), bbox_inches="tight", facecolor="white")
plt.close()

# ============================================================
# 图 4：token 流（图像占位 → embedding 替换 → LLM）
# ============================================================
fig, ax = plt.subplots(figsize=(12.8, 5.8), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(0, 38); ax.axis("off")

ax.text(50, 36, "视觉 token 流：占位符替换与拼接（modeling_internvl.py forward 路径）",
        ha="center", fontsize=12.5, fontweight="bold", color=INK)

# 文本序列
box(ax, 3, 22, 30, 9, "input_ids 文本序列\n… user: <image> … assistant:\n<image> 出现 image_token_id=151667\n每切片展开为 256 个占位 token",
    fc=PANEL, ec=LINE, fs=9)
arrow(ax, 33, 26.5, 38, 26.5)
# embedding
box(ax, 38, 22, 18, 9, "embed_tokens\n查表得 inputs_embeds\n(B, L, 896)", fc=ACC_S, ec=ACC, fs=9.5, bold=True)
arrow(ax, 56, 26.5, 61, 26.5)
# masked scatter
box(ax, 61, 22, 20, 9, "masked_scatter\n按 image_token_id 掩码\n把 (B,256N,896)\n视觉特征写回占位位置",
    fc=HL_S, ec=HL, fs=9, bold=True)
arrow(ax, 81, 26.5, 86, 26.5)
# LLM
box(ax, 86, 22, 12, 9, "Qwen2\nDecoder\n×24", fc=ACC_S, ec=ACC, fs=9.5, bold=True)

# 上方视觉特征支路
box(ax, 38, 8, 43, 8, "ViT + Pixel Unshuffle + MLP 输出：(B, 256×N, 896)\nBOI/EOI 特殊 token 包裹图像段；视觉 token 仅作条件，不计算 loss（text-only loss）",
    fc=PANEL, ec=ACC_L, fs=9)
arrow(ax, 68, 16, 70, 22, color=ACC, ls="--")
ax.text(50, 3, "约束校验：n_image_tokens × 896 == image_features.numel()，不等即报错（get_placeholder_mask）",
        ha="center", fontsize=9, color=SUB)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig4_tokenflow.png"), bbox_inches="tight", facecolor="white")
plt.close()

# ============================================================
# 图 5：2.5 → 3.0 演进对比
# ============================================================
fig, ax = plt.subplots(figsize=(12.5, 8.6), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")

ax.text(50, 60, "InternVL 2.5 → InternVL 3.0 架构与训练范式演进",
        ha="center", fontsize=13, fontweight="bold", color=INK)

rows = [
    ("预训练范式", "MLP warmup 对齐 + 多阶段微调\n（语言预训练与多模态对齐分离）", "原生多模态预训练：语言+多模态语料\n单阶段联合训练，全部参数共同优化\n约 200B token（文本 50B : 多模态 150B = 1:3）"),
    ("视觉编码器", "InternViT-300M / 6B-448px\n（qkv_bias，基础 ViT 结构）", "InternViT-V2.5：LayerScale(0.1)、mean pooling、\n绝对位置编码 + V2PE 可变视觉位置编码"),
    ("语言底座", "Qwen2.5 / InternLM2.5 系列", "Qwen2.5 全系列 + InternLM3-8B（9B 型号）\n全部用 base 模型初始化，不用 instruct 版"),
    ("后训练", "SFT（16.3M 样本）", "SFT 扩到 21.7M 样本 + MPO 混合偏好优化\n（DPO + BCO + LM 损失，约 30 万偏好对）"),
    ("测试时策略", "常规推理", "Best-of-N + VisualPRM-8B 分步奖励评分\n思维链推理可叠加奖励模型选优"),
    ("上下文", "4K–32K 常规", "V2PE 支持 32K 序列（头并行+序列并行）"),
]
y = 53
for i, (k, v25, v3) in enumerate(rows):
    fc = "#fbfcfd" if i % 2 == 0 else "white"
    box(ax, 2, y-6.2, 16, 6.2, k, fc=ACC_S, ec=ACC, fs=10, bold=True)
    box(ax, 18, y-6.2, 38, 6.2, v25, fc=fc, ec=LINE, fs=8.8)
    box(ax, 56, y-6.2, 42, 6.2, v3, fc=HL_S, ec=HL, fs=8.8, bold=False)
    y -= 7.6

ax.text(10, 55.2, "维度", fontsize=9.5, color=SUB, fontweight="bold")
ax.text(37, 55.2, "InternVL 2.5", ha="center", fontsize=10.5, color=SUB, fontweight="bold")
ax.text(77, 55.2, "InternVL 3.0", ha="center", fontsize=10.5, color=HL, fontweight="bold")
ax.text(50, 1.5, "依据：arXiv:2504.10479 §2 与 arXiv:2412.05271；本地 config 为 3.0 原生 internvl 格式",
        ha="center", fontsize=8.5, color=SUB)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig5_evolution.png"), bbox_inches="tight", facecolor="white")
plt.close()

print("done:", os.listdir(OUT))
