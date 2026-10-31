# -*- coding: utf-8 -*-
"""InternVL 3.5 家族架构详解 — 论文配图级架构图生成。配色严格对齐规格冻结值。"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_arch")
os.makedirs(OUT, exist_ok=True)

# ---- 冻结配色 ----
INK   = "#24292f"
SUB   = "#6b7280"
LINE  = "#e3e6ea"
PANEL = "#ffffff"
ACC   = "#0e7490"   # accent 青
ACC_I = "#155e75"
ACC_S = "#e0f2f5"   # accent-soft
ACC_L = "#b7dbe4"
HL    = "#b45309"   # 琥珀强调
HL_S  = "#f7ead6"
HL_L  = "#e8cfae"
MONO  = "#6d5a2e"

# 中文字体
for fp in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyh.ttf", r"C:\Windows\Fonts\simhei.ttf"]:
    if os.path.exists(fp):
        fm.fontManager.addfont(fp)
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "none"

def box(ax, x, y, w, h, text, fc=PANEL, ec=LINE, lw=1.4, fs=11, tc=INK,
         bold=False, sub=None, rounded=0.02, zorder=2):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={rounded}",
                       fc=fc, ec=ec, lw=lw, zorder=zorder)
    ax.add_patch(p)
    weight = "bold" if bold else "normal"
    ax.text(x + w/2, y + h/2 + (0.12 if sub else 0), text,
            ha="center", va="center", fontsize=fs, color=tc, fontweight=weight, zorder=zorder+1)
    if sub:
        ax.text(x + w/2, y + h/2 - h*0.28, sub, ha="center", va="center",
                fontsize=fs-2.2, color=SUB, zorder=zorder+1)
    return (x + w/2, y + h/2)

def arrow(ax, p1, p2, color=SUB, lw=1.6, style="-|>", ls="-", zorder=1, ms=14):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=ms,
                        color=color, lw=lw, ls=ls, zorder=zorder,
                        shrinkA=2, shrinkB=2)
    ax.add_patch(a)

def new_ax(w_in, h_in):
    fig, ax = plt.subplots(figsize=(w_in, h_in), dpi=200)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax

def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.12)
    plt.close(fig)
    print("saved", name)

# ============================================================
# 图1 整体端到端架构
# ============================================================
fig, ax = new_ax(12.5, 5.6)
ax.text(50, 96, "InternVL 3.5 端到端架构：图像 → InternViT → 像素洗牌 → MLP 投影 → Qwen3 语言模型",
        ha="center", fontsize=13.5, fontweight="bold", color=INK)

# 输入图像
box(ax, 2, 55, 12, 26, "输入图像", fc=ACC_S, ec=ACC_L, bold=True, fs=11, sub="动态分辨率\ntile 化")
# 切 tile
box(ax, 18, 55, 13, 26, "图像预处理", fc=PANEL, ec=LINE, fs=10.5,
    sub="缩放到 448 倍数\n切 448×448 tile\n+ 缩略图")
arrow(ax, (14, 68), (18, 68))

# ViT
box(ax, 35, 55, 16, 26, "InternViT-300M", fc=HL_S, ec=HL, lw=1.8, bold=True, fs=11.5,
    sub="24 层 · hidden 1024\n16 头 · patch 14\n输出 32×32=1024")
arrow(ax, (31, 68), (35, 68))

# pixel shuffle
box(ax, 55, 55, 14, 26, "Pixel Shuffle", fc=PANEL, ec=ACC, lw=1.6, bold=True, fs=11,
    sub="scale=0.5\n32×32×1024\n→16×16×4096")
arrow(ax, (51, 68), (55, 68))

# projector
box(ax, 73, 55, 13, 26, "MLP 投影器", fc=ACC_S, ec=ACC, lw=1.8, bold=True, fs=11,
    sub="LN(4096)\nLinear→1024\nGELU→Linear")
arrow(ax, (69, 68), (73, 68))

# LLM
box(ax, 89, 55, 9, 26, "Qwen3\n-0.6B", fc=PANEL, ec=LINE, bold=True, fs=10.5,
    sub="28 层\nhidden 1024")
arrow(ax, (86, 68), (89, 68))

# 底部 token 流说明条
box(ax, 2, 18, 96, 26, "", fc="#fbfcfd", ec=LINE, lw=1.0)
ax.text(4, 40, "单 tile 视觉 token 流（本地 1B 实测）", fontsize=11, fontweight="bold", color=ACC_I)
steps = [
    ("patch 切分", "448/14=32\n32×32=1024"),
    ("去 CLS", "default 策略\n1024 token"),
    ("像素洗牌", "16×16=256\n通道 4096"),
    ("投影到 LLM", "256×1024"),
    ("替换占位符", "image_token\n151671"),
]
x = 5
for i, (t, s) in enumerate(steps):
    fc = HL_S if i in (2, 3) else PANEL
    ec = HL if i in (2, 3) else LINE
    box(ax, x, 22, 16, 13, t, fc=fc, ec=ec, fs=9.8, bold=(i in (2,3)), sub=s)
    if i < len(steps)-1:
        arrow(ax, (x+16, 28.5), (x+18.5, 28.5), color=ACC, ms=11)
    x += 19
ax.text(50, 12, "每个 448×448 tile 恰好产出 image_seq_length = 256 个视觉 token（config 顶层 image_seq_length=256）",
        ha="center", fontsize=10, color=MONO, style="italic")
ax.text(50, 6, "图像特征经 masked_scatter 写回输入嵌入，再送入 Qwen3 自回归解码（InternVLForConditionalGeneration）",
        ha="center", fontsize=9.5, color=SUB)
save(fig, "fig1_overall.png")

# ============================================================
# 图2 InternViT-300M 层细节
# ============================================================
fig, ax = new_ax(9.5, 7.2)
ax.text(50, 97, "InternViT-300M 视觉编码器结构（transformers internvl_vision）",
        ha="center", fontsize=13, fontweight="bold", color=INK)

# embeddings
box(ax, 30, 82, 40, 11, "Patch Embeddings", fc=ACC_S, ec=ACC_L, bold=True, fs=11,
    sub="Conv2d: 3→1024, kernel=stride=14   ·   + CLS token   ·   绝对位置编码 (use_absolute_position_embeddings)")
arrow(ax, (50, 82), (50, 77))

# 24 layers container
box(ax, 18, 30, 64, 47, "", fc="#fbfcfd", ec=ACC_L, lw=1.4, rounded=0.015)
ax.text(50, 73, "24 × Transformer Encoder Layer", ha="center", fontsize=11,
        fontweight="bold", color=ACC_I)
# sub layer 1
box(ax, 24, 57, 52, 11, "LayerNorm → Multi-Head Self-Attention", fc=PANEL, ec=LINE, fs=10.5,
    sub="16 头 · head_dim=64 · attention_bias=True · 无 QK-Norm")
arrow(ax, (50, 57), (50, 53.5), ms=11)
# layer scale 1
box(ax, 30, 49, 40, 4.2, "LayerScale (init 0.1) + 残差", fc=HL_S, ec=HL_L, fs=9.3)
arrow(ax, (50, 49), (50, 45.5), ms=11)
# sub layer 2
box(ax, 24, 34, 52, 11, "LayerNorm → MLP (GELU)", fc=PANEL, ec=LINE, fs=10.5,
    sub="hidden 1024 → intermediate 4096 → 1024")
arrow(ax, (50, 34), (50, 30.5), ms=11)
box(ax, 30, 26.5, 40, 4.2, "LayerScale (init 0.1) + 残差", fc=HL_S, ec=HL_L, fs=9.3)

# output
arrow(ax, (50, 30), (50, 26))
box(ax, 24, 9, 52, 14, "输出处理", fc=HL_S, ec=HL, lw=1.6, bold=True, fs=11,
    sub="use_mean_pooling=True：对 patch token 均值池化（跳过 CLS）\nLayerNorm eps=1e-6")
ax.text(50, 3.5, "规格：hidden_size=1024 · num_hidden_layers=24 · num_attention_heads=16 · image_size=[448,448] · patch_size=[14,14]",
        ha="center", fontsize=9, color=MONO)
save(fig, "fig2_vit.png")

# ============================================================
# 图3 动态分辨率 tile 机制
# ============================================================
fig, ax = new_ax(12.5, 5.4)
ax.text(50, 96, "动态高分辨率（Dynamic High Resolution）tile 机制",
        ha="center", fontsize=13.5, fontweight="bold", color=INK)

# 原图
box(ax, 3, 50, 16, 34, "任意比例\n输入图像", fc=ACC_S, ec=ACC_L, bold=True, fs=11)
arrow(ax, (19, 67), (25, 67))

# resize
box(ax, 25, 50, 18, 34, "长边缩放", fc=PANEL, ec=LINE, fs=10.5, bold=True,
    sub="缩放到边长为\n448 的倍数\nmin/max patch\n约束")
arrow(ax, (43, 67), (49, 67))

# grid tiles
box(ax, 49, 50, 30, 34, "切分 N 个 448×448 tile", fc=PANEL, ec=LINE, fs=10.5, bold=True,
    sub="每个 tile 独立过 ViT\n各自产出 256 token")
# draw mini grid
for r in range(2):
    for c in range(3):
        ax.add_patch(Rectangle((53+c*7.5, 55+r*8), 6.5, 6.8, fc="white", ec=ACC_L, lw=1.0))
arrow(ax, (79, 67), (85, 67))

# thumbnail
box(ax, 84, 68, 13, 18, "缩略图", fc=HL_S, ec=HL, lw=1.6, bold=True, fs=10.5,
    sub="整图压到448²")
ax.text(90.5, 63, "= 1 个 tile · 256 token", ha="center", fontsize=8.6, color=SUB)

# token count bar
box(ax, 3, 14, 94, 26, "", fc="#fbfcfd", ec=LINE, lw=1.0)
ax.text(5, 36, "视觉 token 总量计算", fontsize=11, fontweight="bold", color=ACC_I)
ax.text(5, 30, "T = (tile 数) × 256  +  缩略图 256   （外加 BOI / EOI 两个特殊 token）",
        fontsize=10.5, color=INK)
examples = [("1 tile 图像", "1×256+256=512"), ("2×2=4 tile", "4×256+256=1280"),
            ("3×3=9 tile", "9×256+256=2560"), ("本地 1B 常用", "受 max_dynamic_patch 约束")]
x = 6
for t, s in examples:
    box(ax, x, 17, 21, 9, t, fc=PANEL, ec=LINE, fs=9.8, sub=s)
    x += 24
ax.text(50, 8, "3.5 在 tile 之上新增 ViR（Visual Resolution Router）：按语义为每个 tile 动态选择 1/4 或 1/16 压缩率，进一步削减视觉 token。",
        ha="center", fontsize=9.6, color=HL)
save(fig, "fig3_tile.png")

# ============================================================
# 图4 token 流 / 嵌入替换
# ============================================================
fig, ax = new_ax(12.5, 5.0)
ax.text(50, 96, "视觉 token 如何进入语言模型：占位符替换（masked_scatter）",
        ha="center", fontsize=13.5, fontweight="bold", color=INK)

# text with placeholder
box(ax, 4, 66, 92, 16, "", fc="#fbfcfd", ec=LINE, lw=1.0)
ax.text(6, 77, "文本序列 input_ids：", fontsize=10.5, fontweight="bold", color=ACC_I)
segs = [("<|user|>", PANEL, LINE), ("图：", PANEL, LINE), ("<IMG_CONTEXT>", HL_S, HL),
        ("问题？", PANEL, LINE), ("<|assistant|>", PANEL, LINE)]
x = 20
for t, fc, ec in segs:
    w = 16 if t == "<IMG_CONTEXT>" else 14
    box(ax, x, 71, w, 8, t, fc=fc, ec=ec, fs=9.3, bold=(t=="<IMG_CONTEXT>"))
    x += w + 1.5
ax.text(50, 68, "image_token_id = 151671（config 顶层），prompt 中由 processor 展开为 BOI + 256×N 个占位符 + EOI",
        ha="center", fontsize=9.3, color=SUB)

# arrow down
arrow(ax, (50, 66), (50, 58), color=ACC, lw=2)

# embeddings row
box(ax, 4, 36, 92, 20, "", fc="#fbfcfd", ec=LINE, lw=1.0)
ax.text(6, 51, "inputs_embeds（embed_tokens 查表后）：", fontsize=10.5, fontweight="bold", color=ACC_I)
emb = [("文本嵌入\n1024", PANEL, LINE), ("视觉特征\n256×1024", HL_S, HL),
       ("文本嵌入\n1024", PANEL, LINE), ("文本嵌入\n1024", PANEL, LINE)]
x = 14
for t, fc, ec in emb:
    w = 26 if "视觉" in t else 19
    box(ax, x, 39, w, 9, t, fc=fc, ec=ec, fs=9.3, bold=("视觉" in t))
    x += w + 2
ax.text(50, 37.5, "special_image_mask 命中占位符位置 → inputs_embeds.masked_scatter(mask, image_features)",
        ha="center", fontsize=9.3, color=MONO)

arrow(ax, (50, 36), (50, 29), color=ACC, lw=2)
box(ax, 18, 14, 64, 13, "Qwen3-0.6B 28 层因果自注意力 → lm_head → 下一 token 概率",
    fc=ACC_S, ec=ACC, lw=1.8, bold=True, fs=11)
ax.text(50, 8, "校验：占位符 token 数必须等于视觉特征数（n_image_tokens == features），否则报错（get_placeholder_mask）",
        ha="center", fontsize=9.3, color=SUB)
save(fig, "fig4_tokenflow.png")

# ============================================================
# 图5 Cascade RL + ViR/DvD 效率（3.0→3.5 训练/推理）
# ============================================================
fig, ax = new_ax(12.5, 6.0)
ax.text(50, 96, "InternVL 3.5 后训练与推理效率：Cascade RL、ViR、DvD",
        ha="center", fontsize=13.5, fontweight="bold", color=INK)

# Cascade RL
ax.text(8, 86, "① Cascade RL（级联强化学习）", fontsize=11.5, fontweight="bold", color=ACC_I)
box(ax, 4, 68, 26, 14, "SFT 起点", fc=PANEL, ec=LINE, fs=10.5, bold=True)
arrow(ax, (30, 75), (35, 75), color=ACC, lw=1.8)
box(ax, 35, 66, 28, 18, "离线 RL（热身）", fc=ACC_S, ec=ACC, lw=1.6, bold=True, fs=10.5,
    sub="MPO = 偏好损失(DPO)\n+ 质量损失(BCO)\n+ 生成损失(LM)")
arrow(ax, (63, 75), (68, 75), color=ACC, lw=1.8)
box(ax, 68, 66, 28, 18, "在线 RL（精修）", fc=HL_S, ec=HL, lw=1.8, bold=True, fs=10.5,
    sub="GSPO（无参考模型）\n同 query 组内归一化优势\nefficient rollout")
ax.text(83, 63, "→ 推理 +16.0%", ha="center", fontsize=10, color=HL, fontweight="bold")

# ViR
ax.text(8, 55, "② ViR 视觉分辨率路由（InternVL3.5-Flash）", fontsize=11.5, fontweight="bold", color=ACC_I)
box(ax, 4, 38, 30, 12, "每个 tile", fc=PANEL, ec=LINE, fs=10, bold=True)
arrow(ax, (34, 44), (39, 44), color=SUB)
box(ax, 39, 38, 26, 12, "语义判断", fc=PANEL, ec=LINE, fs=10, bold=True, sub="ViCO 轻量训练")
arrow(ax, (65, 44), (70, 44), color=SUB)
box(ax, 70, 44, 26, 6, "压缩 1/4", fc=ACC_S, ec=ACC_L, fs=9.5)
box(ax, 70, 36, 26, 6, "压缩 1/16", fc=HL_S, ec=HL_L, fs=9.5)

# DvD
ax.text(8, 29, "③ DvD 解耦部署", fontsize=11.5, fontweight="bold", color=ACC_I)
box(ax, 10, 12, 36, 11, "GPU-A：ViT 编码", fc=ACC_S, ec=ACC, lw=1.5, fs=10.5, bold=True,
    sub="图像编码并行")
box(ax, 58, 12, 36, 11, "GPU-B：LLM 自回归", fc=HL_S, ec=HL, lw=1.5, fs=10.5, bold=True,
    sub="文本解码")
arrow(ax, (46, 17.5), (58, 17.5), color=ACC, lw=1.8, style="<|-|>")
ax.text(52, 24, "视觉与语言计算负载均衡 → 最高 4.05× 推理加速", ha="center",
        fontsize=10.5, color=HL, fontweight="bold")
save(fig, "fig5_cascade_rl.png")

# ============================================================
# 图6 家族参数量分布（柱状，真实 Table 1 数据）
# ============================================================
fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=200)
models = ["1B", "2B", "4B", "8B", "14B", "20B-A4B", "30B-A3B", "38B", "241B-A28B"]
vision = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 5.5, 5.5]
lang   = [0.8, 2.0, 4.4, 8.2, 14.8, 20.9, 30.5, 32.8, 235.1]
import numpy as np
x = np.arange(len(models))
w = 0.62
b1 = ax.bar(x, vision, w, label="视觉编码器 (InternViT-300M/6B)", color=ACC_L, edgecolor=ACC)
b2 = ax.bar(x, lang, w, bottom=vision, label="语言模型 (Qwen3 / GPT-OSS)", color=HL_S, edgecolor=HL)
ax.set_yscale("log")
ax.set_ylabel("参数量（B，对数刻度）", fontsize=11)
ax.set_title("InternVL 3.5 家族参数量构成（论文 Table 1）", fontsize=13, fontweight="bold", color=INK)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9.5)
for i, (v, l) in enumerate(zip(vision, lang)):
    total = v + l
    ax.text(i, total*1.12, f"{total:.1f}B", ha="center", fontsize=8.6, color=INK)
ax.legend(fontsize=9.5, loc="upper left", frameon=False)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="y", ls=":", color=LINE)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig6_family_params.png"), dpi=200, bbox_inches="tight", facecolor="white")
print("saved fig6_family_params.png")
print("ALL DONE")
