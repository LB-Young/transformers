# -*- coding: utf-8 -*-
"""Gemma 训练算法演进 · 图素材生成（matplotlib，common312 环境执行）"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager

# 中文字体
for fp in [r'C:\Windows\Fonts\msyh.ttc', r'C:\Windows\Fonts\simhei.ttf']:
    try:
        font_manager.fontManager.addfont(fp)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
        break
    except Exception:
        continue
plt.rcParams['axes.unicode_minus'] = False

NAVY = '#1A2B4A'; MID = '#52647D'; LIGHT = '#8D9AAB'; PALE = '#C9D0D9'
AMBER = '#C8861A'; GREY = '#767676'; GRID = '#D8D8D8'

OUT = r'E:\transformers\bayon-note\Gemma\_training_algo_work'

# ---------- 图1：三阶段 x 四代 算法矩阵 ----------
fig, ax = plt.subplots(figsize=(14, 8), dpi=120)
fig.patch.set_facecolor('white')
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')

phases = ['预训练算法', '指令微调算法', '强化学习算法']
gens = ['Gemma 1\n2024-02', 'Gemma 2\n2024-06', 'Gemma 3\n2025-03', 'Gemma 4\n2026-07']
# (行, 列) -> (主文案, 副文案)
cells = {
    (0, 0): ('标准 next-token 预测', '分阶段训练提升后期数据质量'),
    (0, 1): ('全词汇概率蒸馏', 'min Σx −P_T·log P_S'),
    (0, 2): ('每 token 256-logit 采样蒸馏', '加权采样 + 重归一化'),
    (0, 3): ('继承 Gemma 3 蒸馏配方', '—'),
    (1, 0): ('LM 逐对评估选数据', '基线大模型判优（Gemini 思路）'),
    (1, 1): ('行为克隆 + on-policy 蒸馏', 'Agarwal 2024 / Gu 2024'),
    (1, 2): ('大型 IT 教师改进版蒸馏', '与 RL 阶段衔接'),
    (1, 3): ('继承 Gemma 3 + thinking 数据', '先输出推理轨迹再作答'),
    (2, 0): ('经典 RLHF（新颖 RL 算法）', 'Bradley-Terry + 自动评分'),
    (2, 1): ('换更大对话向奖励模型', '阶段模型平均（WARP 思想）'),
    (2, 2): ('BOND / WARM / WARP 改进版', '多奖励：人类反馈 + 代码 + 数学'),
    (2, 3): ('继承 Gemma 3 + 推理轨迹', '—'),
}

left = 0.6; top = 8.0; w = 2.18; h = 2.4
gapx = 0.14; gapy = 0.4

# 阶段行标签
for i, p in enumerate(phases):
    y = top - i * (h + gapy)
    ax.text(0.05, y + h / 2, p, ha='left', va='center', fontsize=15, color=NAVY, fontweight='bold')

for j, g in enumerate(gens):
    x = left + j * (w + gapx)
    ax.text(x + w / 2, top + 0.35, g, ha='center', va='center', fontsize=13, color=NAVY, fontweight='bold')
    for i in range(3):
        y = top - i * (h + gapy)
        main, sub = cells[(i, j)]
        rect = mpatches.FancyBboxPatch((x, y - h), w, h,
                                       boxstyle='round,pad=0.02,rounding_size=0.12',
                                       linewidth=1.2, edgecolor=MID, facecolor='white')
        ax.add_patch(rect)
        ax.text(x + w / 2, y - h * 0.35, main, ha='center', va='center', fontsize=11.5, color='#252525', fontweight='bold')
        if sub and sub != '—':
            ax.text(x + w / 2, y - h * 0.65, sub, ha='center', va='center', fontsize=9, color=GREY)

plt.tight_layout()
plt.savefig(OUT + r'\fig1_algo_matrix.png', dpi=200, facecolor='white')
plt.close()

# ---------- 图2：蒸馏目标函数演进（Gemma 2 vs Gemma 3） ----------
fig, ax = plt.subplots(figsize=(14, 6.4), dpi=120)
fig.patch.set_facecolor('white')
ax.set_xlim(0, 14); ax.set_ylim(0, 7.6); ax.axis('off')
ax.text(7, 7.2, '预训练知识蒸馏目标演进', ha='center', fontsize=17, color=NAVY, fontweight='bold')

def box(x, y, w, h, text, fc, ec, fs=11.5, weight='normal'):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.14',
                                   linewidth=1.4, edgecolor=ec, facecolor=fc)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color='#252525', fontweight=weight)
    return rect

def arrow(x1, y1, x2, y2, color=MID, lw=1.6):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, mutation_scale=16))

# ---- Gemma 2：全词汇分布蒸馏 ----
ax.text(3.5, 6.45, 'Gemma 2 · 全词汇概率蒸馏', ha='center', fontsize=13, color=NAVY, fontweight='bold')
box(0.4, 5.0, 2.6, 1.2, 'Teacher\n（大模型）', '#F4F4F4', MID)
box(3.6, 5.0, 2.6, 1.2, '完整词汇表概率\nP_T(x | x_c)', '#FFFFFF', AMBER)
box(7.2, 5.0, 2.6, 1.2, '交叉熵\n−Σx P_T · log P_S', '#FFFFFF', MID)
box(10.0, 5.0, 2.6, 1.2, 'Student\n（Gemma 2）', '#F4F4F4', MID)
arrow(3.0, 5.6, 3.6, 5.6); arrow(6.2, 5.6, 7.2, 5.6); arrow(9.8, 5.6, 10.0, 5.6)
ax.text(0.4, 4.35, '目标：让学生分布逼近教师全分布\n（Gemini 1.5 同法；高内存与带宽开销）',
        ha='left', va='top', fontsize=9.5, color=GREY, wrap=True)

# ---- Gemma 3：每 token 256-logit 采样蒸馏 ----
ax.text(3.5, 3.55, 'Gemma 3 · 每 token 256-logit 采样蒸馏', ha='center', fontsize=13, color=NAVY, fontweight='bold')
box(0.4, 2.2, 2.6, 1.2, 'Teacher\n（大模型）', '#F4F4F4', MID)
box(3.6, 2.2, 2.6, 1.2, '按概率加权采样\n256 logits / token', '#FFFFFF', AMBER)
box(7.2, 2.2, 2.6, 1.2, '仅采样集交叉熵\n未采样位置置零重归一化', '#FFFFFF', MID)
box(10.0, 2.2, 2.6, 1.2, 'Student\n（Gemma 3）', '#F4F4F4', MID)
arrow(3.0, 2.8, 3.6, 2.8); arrow(6.2, 2.8, 7.2, 2.8); arrow(9.8, 2.8, 10.0, 2.8)
ax.text(0.4, 1.55, '大幅降低内存与通信开销，蒸馏训练可规模化\n消融：短训练视窗小 teacher 更优，长训练大 teacher 更优',
        ha='left', va='top', fontsize=9.5, color=GREY)

# 底部连接线
arrow(1.7, 4.2, 1.7, 3.5, color=PALE, lw=1.2)
ax.text(2.05, 3.9, '细化', ha='left', fontsize=9.5, color=GREY)

plt.tight_layout()
plt.savefig(OUT + r'\fig2_distill_evolution.png', dpi=200, facecolor='white')
plt.close()
print('figs done')
