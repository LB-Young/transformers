# -*- coding: utf-8 -*-
"""Generate InternVL 2.0 architecture diagrams (paper-figure level) into ../_arch/.
Palette frozen per pattern-spec v1."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib import font_manager

# ---------- palette (frozen) ----------
INK = "#24292F"
SUB = "#6B7280"
LINE = "#E3E6EA"
ACCENT = "#0E7490"
ACCENT_INK = "#155E75"
ACCENT_SOFT = "#E0F2F5"
ACCENT_LINE = "#B7DBE4"
HL = "#B45309"
HL_SOFT = "#F7EAD6"
HL_LINE = "#E8CFAE"
PANEL = "#FFFFFF"
BG = "#F6F7F9"

# ---------- fonts ----------
for f in ["Microsoft YaHei", "SimHei"]:
    try:
        font_manager.findfont(f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [f, "DejaVu Sans"]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_arch")
os.makedirs(OUT, exist_ok=True)


def new_ax(w=13.2, h=6.2):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def box(ax, x, y, w, h, text, fc=PANEL, ec=LINE, lw=1.4, fs=10, weight="normal",
        tc=INK, sub=None, sub_fs=8.4, sub_color=SUB, rounded=0.025):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rounded*100}",
                       fc=fc, ec=ec, lw=lw, mutation_aspect=1)
    ax.add_patch(p)
    if sub:
        ax.text(x + w/2, y + h*0.62, text, ha="center", va="center",
                fontsize=fs, color=tc, fontweight=weight)
        ax.text(x + w/2, y + h*0.28, sub, ha="center", va="center",
                fontsize=sub_fs, color=sub_color)
    else:
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fs, color=tc, fontweight=weight)


def arrow(ax, x1, y1, x2, y2, color=SUB, lw=1.6, style="-|>", ls="-"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13,
                        color=color, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0)
    ax.add_patch(a)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.18)
    plt.close(fig)
    print("saved", path)


# =====================================================================
# 图1 端到端整体架构图
# =====================================================================
fig, ax = new_ax(13.6, 5.6)
ax.set_ylim(0, 100)
# 主流程 6 个节点
nodes = [
    (2, 52, 13, 26, "输入图像", "800×1300 示例", PANEL, LINE, INK),
    (18, 52, 15, 26, "动态切 tile", "N 个 448×448\n+ 1 个缩略图", ACCENT_SOFT, ACCENT_LINE, ACCENT_INK),
    (36, 52, 16, 26, "InternViT 视觉塔", "300M：hidden1024\n24 层 · 16 头 · patch14", ACCENT_SOFT, ACCENT_LINE, ACCENT_INK),
    (55, 52, 14, 26, "Pixel Shuffle", "scale=0.5\n32×32→16×16\n通道 1024→4096", PANEL, LINE, INK),
    (72, 52, 14, 26, "MLP 投影层", "LN→Linear(4096→896)\n→GELU→Linear(896)", HL_SOFT, HL_LINE, HL),
    (89, 52, 9.5, 26, "Qwen2\nLLM", "24 层 GQA\nhidden896", HL_SOFT, HL_LINE, HL),
]
for (x, y, w, h, t, s, fc, ec, tc) in nodes:
    box(ax, x, y, w, h, t, fc=fc, ec=ec, fs=10.5, weight="bold", sub=s, sub_fs=8.2, sub_color=tc)
for x1, x2 in [(15, 18), (33, 36), (52, 55), (69, 72), (86, 89)]:
    arrow(ax, x1, 65, x2, 65, color=ACCENT, lw=2.0)

# 下方 token 条带
ax.text(2, 30, "拼接入 LLM 的 token 序列（masked_scatter 替换 <image> 占位符嵌入）",
        fontsize=9.5, color=SUB)
segs = [
    (2, 22, "User: <image>", PANEL, LINE),
    (25, 26, "tile1 特征\n256×896", ACCENT_SOFT, ACCENT_LINE),
    (52, 26, "... tileN 特征\nN<=12", ACCENT_SOFT, ACCENT_LINE),
    (79, 12, "缩略图\n256×896", ACCENT_SOFT, ACCENT_LINE),
    (92, 6.5, "问题/\n回答文本", PANEL, LINE),
]
for (x, w, t, fc, ec) in segs:
    box(ax, x, 8, w, 14, t, fc=fc, ec=ec, fs=8.6, sub=None)
arrow(ax, 14.5, 52, 14.5, 22, color=LINE, lw=1.2, ls="--")
arrow(ax, 79, 52, 79, 22, color=LINE, lw=1.2, ls="--")
ax.text(50, 44, "单张 448×448 tile 输出 32×32=1024 个 patch token，pixel shuffle 后为 16×16=256 个视觉 token",
        ha="center", fontsize=9, color=ACCENT_INK)
save(fig, "fig1_overall.png")

# =====================================================================
# 图2 视觉编码器 InternViT 细节
# =====================================================================
fig, ax = new_ax(12.6, 7.4)
ax.set_ylim(0, 100)
# 左列：输入与 patch embed
box(ax, 3, 80, 18, 14, "448×448  RGB 图像", sub="3 通道", fs=10, weight="bold")
arrow(ax, 12, 80, 12, 74)
box(ax, 3, 60, 18, 14, "Patch Embed", sub="卷积 14×14, stride 14\n32×32=1024 patch", fc=ACCENT_SOFT, ec=ACCENT_LINE)
arrow(ax, 12, 60, 12, 54)
box(ax, 3, 40, 18, 14, "+ 位置编码", sub="+ 1 个 CLS token\n→ 1025×1024", fc=ACCENT_SOFT, ec=ACCENT_LINE)
arrow(ax, 12, 40, 12, 34)
box(ax, 3, 16, 18, 14, "select_layer = -1", sub="取最后一层输出\n[:, 1:, :] 去 CLS", fc=HL_SOFT, ec=HL_LINE, tc=HL)

# 中央：24 层 block 展开
box(ax, 30, 14, 42, 80, "", fc=PANEL, ec=ACCENT, lw=1.8, rounded=0.02)
ax.text(51, 90, "InternViT Block × 24（hidden 1024 · intermediate 4096 · gelu）",
        ha="center", fontsize=10.5, fontweight="bold", color=ACCENT_INK)
box(ax, 34, 74, 34, 11, "LayerNorm（eps 1e-6）", fs=9.5, sub=None)
arrow(ax, 51, 74, 51, 69)
box(ax, 34, 56, 34, 13, "多头自注意力（16 头 · head_dim 64）",
    sub="qkv_bias=true · qk_normalization=false\n无 flash-attn 以外的特殊结构", fs=9.5)
# 残差弧（上）
arrow(ax, 33, 79.5, 30.5, 79.5, color=HL, lw=1.6)
arrow(ax, 30.5, 79.5, 30.5, 62.5, color=HL, lw=1.6)
arrow(ax, 30.5, 62.5, 33.5, 62.5, color=HL, lw=1.6)
ax.text(24.5, 71, "残差", fontsize=8.5, color=HL, rotation=90, ha="center", va="center")
arrow(ax, 51, 56, 51, 51)
box(ax, 34, 39, 34, 11, "LayerNorm", fs=9.5, sub=None)
arrow(ax, 51, 39, 51, 34)
box(ax, 34, 18, 34, 13, "MLP（gelu）", sub="1024 → 4096 → 1024", fs=9.5)
arrow(ax, 33, 44.5, 30.5, 44.5, color=HL, lw=1.6)
arrow(ax, 30.5, 44.5, 30.5, 24.5, color=HL, lw=1.6)
arrow(ax, 30.5, 24.5, 33.5, 24.5, color=HL, lw=1.6)
# 右侧输出
arrow(ax, 72, 24.5, 78, 24.5, color=ACCENT, lw=1.8)
box(ax, 78, 16, 19, 22, "视觉特征", sub="(B, 1024, 32, 32)\n送 Pixel Shuffle", fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=10, weight="bold")
ax.text(51, 6, "注：本地 config 的 vision_config.model_type 字符串为 intern_vit_6b，但数值（1024/24层/16头/inter4096）即 InternViT-300M-448px 规格",
        ha="center", fontsize=8.6, color=HL)
save(fig, "fig2_vit.png")

# =====================================================================
# 图3 动态分辨率 tile 机制
# =====================================================================
fig, ax = new_ax(13.4, 6.8)
ax.set_ylim(0, 100)
# 左：原图 + 预定义比例
box(ax, 3, 58, 16, 30, "输入图像", sub="800×1300", fs=10.5, weight="bold")
arrow(ax, 19, 73, 25, 73, color=ACCENT, lw=1.8)
box(ax, 25, 56, 24, 34, "预定义宽高比匹配",
    sub="1:1  1:2  1:3  1:4\n1:5  1:6  2:3  3:2 …\n→ 命中 2:3", fs=9.5)
arrow(ax, 49, 73, 55, 73, color=ACCENT, lw=1.8)
box(ax, 55, 56, 24, 34, "按比例缩放后切 tile",
    sub="2:3 → 896×1344\n= 2列×3行 = 6 块\n每块 448×448", fs=9.5)
arrow(ax, 79, 73, 85, 73, color=ACCENT, lw=1.8)
# tile 网格示意
gx, gy = 85, 60
for i in range(2):
    for j in range(3):
        ax.add_patch(Rectangle((gx + i*6.5, gy + j*8.5), 6, 7.5, fc=ACCENT_SOFT, ec=ACCENT_LINE, lw=1.2))
ax.text(91.5, 55, "6 个 448×448 tile", ha="center", fontsize=8.6, color=ACCENT_INK)
# thumbnail 行
box(ax, 55, 30, 24, 14, "+ 全局缩略图 thumbnail", sub="整图缩到 448×448（use_thumbnail=true）", fs=9.5)
arrow(ax, 79, 37, 85, 37, color=ACCENT, lw=1.8)
ax.add_patch(Rectangle((86, 32), 11, 10, fc=HL_SOFT, ec=HL_LINE, lw=1.4))
ax.text(91.5, 28, "thumbnail 448×448", ha="center", fontsize=8.6, color=HL)
# 底部 token 核算
box(ax, 3, 8, 60, 16, "token 核算：每 tile = 1024 patch → pixel_shuffle → 256 token；本图 (6+1)×256 = 1792 视觉 token",
    sub="训练 max_dynamic_patch=12（config 实参），测试零样本可放大到 40 tile（约 4K）", fs=9.8)
ax.text(70, 16, "配置实参：dynamic_image_size=true\nforce_image_size=448 · downsample_ratio=0.5\nmin_dynamic_patch=1 · ps_version=v2",
        fontsize=8.8, color=SUB, va="center")
save(fig, "fig3_tile.png")

# =====================================================================
# 图4 token 流
# =====================================================================
fig, ax = new_ax(13.4, 5.4)
ax.set_ylim(0, 100)
ax.text(50, 94, "文本 token 嵌入与视觉特征的拼接（masked_scatter 替换）", ha="center",
        fontsize=11, fontweight="bold", color=INK)
# 输入流
box(ax, 3, 62, 20, 20, "input_ids", sub="含 <image> 占位符\n(b, s)", fs=10)
arrow(ax, 23, 72, 29, 72, color=SUB, lw=1.6)
box(ax, 29, 62, 22, 20, "embed_tokens 查表", sub="Qwen2 词表 151655\n→ (b, s, 896)", fc=ACCENT_SOFT, ec=ACCENT_LINE)
arrow(ax, 51, 72, 57, 72, color=SUB, lw=1.6)
box(ax, 57, 62, 22, 20, "special_image_mask", sub="input_ids == image_token\n布尔掩码", fc=HL_SOFT, ec=HL_LINE)
arrow(ax, 79, 72, 85, 72, color=SUB, lw=1.6)
box(ax, 85, 62, 13, 20, "masked\n_scatter", sub="占位位置\n写入视觉特征", fc=HL_SOFT, ec=HL_LINE)
# 视觉支路
box(ax, 15, 26, 26, 18, "pixel_values（全部 tile）", sub="(N, 3, 448, 448)", fs=10)
arrow(ax, 28, 26, 28, 20, color=ACCENT, lw=1.8)
box(ax, 15, 6, 26, 14, "视觉塔 + Pixel Shuffle + MLP", sub="→ (N_tile, 256, 896)", fc=ACCENT_SOFT, ec=ACCENT_LINE)
arrow(ax, 41, 13, 60, 13, color=ACCENT, lw=1.8)
arrow(ax, 60, 13, 60, 33, color=ACCENT, lw=1.8)
arrow(ax, 60, 33, 66, 33, color=ACCENT, lw=1.8)
arrow(ax, 66, 33, 66, 62, color=ACCENT, lw=1.8)
ax.text(63, 48, "按图像顺序摊平\n(N_tile×256, 896)", fontsize=8.6, color=ACCENT_INK, ha="center")
arrow(ax, 91.5, 62, 91.5, 30, color=SUB, lw=1.6)
box(ax, 78, 14, 19, 16, "Qwen2 解码器", sub="24 层 GQA\n因果注意力", fc=HL_SOFT, ec=HL_LINE, fs=10, weight="bold")
save(fig, "fig4_tokenflow.png")

# =====================================================================
# 图5 版本演进对比（1.5 → 2.0）
# =====================================================================
fig, ax = new_ax(13.4, 7.6)
ax.set_ylim(0, 100)
# 表头
box(ax, 2, 88, 22, 9, "维度", fs=10, weight="bold")
box(ax, 25, 88, 36, 9, "InternVL 1.5（2024-04）", fc=PANEL, ec=LINE, fs=10, weight="bold")
box(ax, 62, 88, 36, 9, "InternVL 2.0（2024-07）", fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=10, weight="bold")
rows = [
    ("视觉塔", "主模型 6B-448px；Mini-2B/4B\n已用蒸馏 300M（24层/1024d/16头）",
     "300M-448px（小模型）+ 6B-448px-V1.5（大模型）\n两档覆盖全家族，连续学习升级"),
    ("语言底座", "已有三档：InternLM2-20B（V1.5）\n+1.8B（Mini-2B）+Phi-3-mini（Mini-4B）",
     "在 InternLM2 系+Phi-3 基础上扩展为 7 种：\nQwen2-0.5B / InternLM2.5-7B / Yi-34B / Llama3-70B 等"),
    ("动态分辨率", "tile 1–40×448 + thumbnail\n4K 测试",
     "保留：训练 ≤12 tile、测试 ≤40 tile\npixel shuffle 0.5 → 256 token/tile 不变"),
    ("输入模态", "图 + 文本",
     "图+视频+医疗影像+多图\n（8k 上下文沿用 1.5 Mini，非新增）"),
    ("输出能力", "仅文本",
     "VisionLLMv2 解码器：文本 + 框 + 掩码 + 图像生成\n（外接下游任务解码器）"),
    ("家族规模", "~1B / 2B / 26B",
     "1B → 108B（Progressive Scaling：小到大、粗到精）"),
]
y = 78
for i, (k, c15, c20) in enumerate(rows):
    fc = PANEL if i % 2 == 0 else "#FAFBFC"
    box(ax, 2, y-10, 22, 10, k, fc=fc, fs=9.6, weight="bold", tc=INK)
    box(ax, 25, y-10, 36, 10, c15, fc=fc, fs=8.8)
    box(ax, 62, y-10, 36, 10, c20, fc=ACCENT_SOFT if i % 2 == 0 else "#EAF6F8", fs=8.8, tc=ACCENT_INK)
    y -= 11
save(fig, "fig5_evolution.png")

# =====================================================================
# 图6 训练管线（两阶段 + 渐进扩展）
# =====================================================================
fig, ax = new_ax(13.4, 6.4)
ax.set_ylim(0, 100)
# Stage 1
box(ax, 3, 58, 44, 32, "Stage 1 · 多模态对齐预训练",
    sub="可训练：仅 MLP 投影层（ViT / LLM 冻结）\n数据：扩展 1.5 预训练集\ncaption · VQA · detection · grounding\nOCR（PaddleOCR）· 考题 · OmniCorpus 交织",
    fc=ACCENT_SOFT, ec=ACCENT_LINE, fs=11, weight="bold", sub_fs=9)
arrow(ax, 47, 74, 55, 74, color=HL, lw=2.2)
box(ax, 55, 58, 42, 32, "Stage 2 · 监督微调 SFT",
    sub="可训练：ViT + MLP + LLM 全参数\n数据：5M 高质量双语集\n+ 视频（EgoTaskQA/Mementos…）\n+ 医疗（PMC-VQA/Slake/VQA-RAD…）\n手写 SROIE/FUNSD · ShareGPT-4o",
    fc=HL_SOFT, ec=HL_LINE, fs=11, weight="bold", sub_fs=9, tc=HL)
# 渐进扩展带
box(ax, 3, 18, 94, 22, "Progressive Scaling Strategy（渐进扩展）",
    sub="小尺寸模型（1B/2B/4B…）先行训好 → 其权重作为更大模型的初始化参考，语言底座逐级增大，数据由粗到精清洗\n以较低成本完成 1B → 108B 全家族训练；视觉塔连续学习、可跨底座复用",
    fc=PANEL, ec=ACCENT, lw=1.8, fs=11, weight="bold", sub_fs=9)
ax.text(50, 8, "注：1.5→2.0 保留两阶段训练范式；2.0 的新增量在数据模态（视频/医疗）与底座多样性",
        ha="center", fontsize=9, color=SUB)
save(fig, "fig6_pipeline.png")

print("ALL DONE")
