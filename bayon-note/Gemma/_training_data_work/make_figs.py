# -*- coding: utf-8 -*-
"""生成 Gemma 训练数据演进 docx 用图：图1 token 预算对比，图2 数据处理管道演进。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 中文字体
for f in ["Microsoft YaHei", "SimHei", "SimSun"]:
    try:
        font_manager.findfont(f, fallback_to_default=False)
        break
    except Exception:
        continue
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

OUT = r"E:\transformers\bayon-note\Gemma\_training_data_work"

# ---- 图1：预训练 token 预算对比 ----
generations = ["Gemma 1", "Gemma 2", "Gemma 3", "Gemma 4"]
data = [
    (["2B", "7B"], [3, 6]),
    (["2B", "9B", "27B"], [2, 8, 13]),
    (["1B", "4B", "12B", "27B"], [2, 4, 12, 14]),
    (["E2B~31B"], [None]),  # 未披露
]
fig, ax = plt.subplots(figsize=(11.2, 6.4), dpi=150)
x_positions = []
labels = []
colors = ["#1A2B4A", "#52647D", "#8D9AAB", "#C9D0D9"]
idx = 0
for gi, (sizes, toks) in enumerate(data):
    for si, s in enumerate(sizes):
        x_positions.append(idx)
        labels.append(s)
        if toks[si] is None:
            ax.bar(idx, 14, color=colors[gi], alpha=0.35, edgecolor=colors[gi], hatch="//")
            ax.text(idx, 14.6, "未披露", ha="center", fontsize=9, color="#767676")
        else:
            ax.bar(idx, toks[si], color=colors[gi], width=0.7)
            ax.text(idx, toks[si] + 0.3, f"{toks[si]}T", ha="center", fontsize=9, color="#252525")
        idx += 1
# 代际分隔线
boundaries = [3, 6, 10]
for b in boundaries:
    ax.axvline(b - 0.5, color="#D8D8D8", linewidth=1.2)
ax.set_xticks(x_positions)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("预训练 token 预算（万亿 tokens）", fontsize=11, color="#252525")
ax.set_ylim(0, 17)
ax.set_title("各代 Gemma 模型预训练 token 预算对比", fontsize=13, color="#252525", pad=14)
ax.grid(axis="y", color="#D8D8D8", linewidth=0.8)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
# 图例（代际）
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
leg = ax.legend(handles, generations, loc="upper right", frameon=False, fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}\\fig1_token_budget.png", dpi=150)
plt.close(fig)
print("fig1 saved")

# ---- 图2：数据处理管道演进（预训练/SFT/RL 三阶段 × 四代） ----
fig2, ax2 = plt.subplots(figsize=(11.2, 6.4), dpi=150)
ax2.axis("off")
stages = ["预训练数据", "SFT 数据", "RL / 对齐数据"]
gen_cols = ["#1A2B4A", "#52647D", "#8D9AAB", "#C9D0D9"]
g_texts = {
    "Gemma 1": [
        "英文 web/数学/代码\n3T(2B) / 6T(7B)\n启发式+模型分类器过滤\n阶段式训练提升后期数据质量",
        "纯文本纯英文\n人工+合成 prompt-response\nLM 逐对评估选混合\n多阶段过滤合成数据",
        "英文偏好对\nBradley-Terry 奖励模型\n新颖 RL 算法\n高容量模型自动评分",
    ],
    "Gemma 2": [
        "13T(27B)/8T(9B)/2T(2B)\n含科学文章\n+ 预训练知识蒸馏\n过滤管道同 Gemma 1",
        "扩展 Gemma 1.1 数据\n+ LMSYS-chat-1M prompts\n行为克隆（teacher 合成响应）\n+ 学生分布蒸馏",
        "更大对话向 reward model\n（比 policy 大一个数量级）\n阶段后模型平均",
    ],
    "Gemma 3": [
        "14T(27B)/12T/4T/2T\n图文混合 + 多语言(单语+平行)\n质量重加权(Sachdeva 2024)\n262k tokenizer",
        "大 IT teacher 知识蒸馏\n（改进版，同 RL 阶段衔接）\n数据过滤同前",
        "BOND/WARM/WARP 改进版\n多奖励：人类偏好(权重平均)\n+ 代码执行 + 数学 ground-truth",
    ],
    "Gemma 4": [
        "类似 Gemma 3 配方\nweb/代码/图像/音频\ncutoff 2025-01\ntoken 预算未披露",
        "类似 Gemma 3\n新增 thinking mode 数据\nIT 输出 <turn|>",
        "类似 Gemma 3 的 RL\n+ thinking mode 推理轨迹",
    ],
}
# 布局：行=4 代，列=3 阶段
rows = list(g_texts.keys())
nrow, ncol = 4, 3
cell_w, cell_h = 1.0, 0.72
x0, y0 = 0.5, 3.6
# 阶段列标题
for c, st in enumerate(stages):
    ax2.text(x0 + c * (cell_w + 0.18) + cell_w / 2, y0 + 0.28, st, ha="center",
             fontsize=12, fontweight="bold", color="#252525")
for r, gen in enumerate(rows):
    y = y0 - (r + 1) * (cell_h + 0.28)
    # 代标签
    ax2.text(x0 - 0.28, y + cell_h / 2, gen, ha="right", va="center", fontsize=11,
             fontweight="bold", color=gen_cols[r])
    for c in range(ncol):
        x = x0 + c * (cell_w + 0.18)
        rect = plt.Rectangle((x, y), cell_w, cell_h, facecolor=gen_cols[r] + "1A",
                             edgecolor=gen_cols[r], linewidth=1.0, zorder=1)
        ax2.add_patch(rect)
        ax2.text(x + cell_w / 2, y + cell_h / 2, g_texts[gen][c], ha="center", va="center",
                 fontsize=9.0, color="#252525", zorder=2, linespacing=1.45)
# 阶段间箭头
for c in range(ncol - 1):
    ax2.annotate("", xy=(x0 + (c + 1) * (cell_w + 0.18) - 0.02, y0 - 1.55),
                 xytext=(x0 + (c + 1) * (cell_w + 0.18) - cell_w - 0.16, y0 - 1.55),
                 arrowprops=dict(arrowstyle="-|>", color="#767676", lw=1.4))
ax2.set_xlim(x0 - 0.8, x0 + ncol * (cell_w + 0.18) + 0.1)
ax2.set_ylim(y0 - nrow * (cell_h + 0.28) - 0.55, y0 + 0.55)
ax2.set_title("Gemma 1→4 三阶段训练数据管道演进", fontsize=13, color="#252525", pad=16)
fig2.tight_layout()
fig2.savefig(f"{OUT}\\fig2_pipeline_evolution.png", dpi=150)
plt.close(fig2)
print("fig2 saved")
