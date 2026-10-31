# Gemma 训练算法演进详解 · outline

## 任务
基于 Gemma 1/2/3/4 四篇技术报告（不含 3n），系统整理训练算法的演进：预训练算法（蒸馏、量化、分阶段训练）、指令微调算法（行为克隆、蒸馏、数据选择）、强化学习对齐算法（RLHF、BOND/WARM/WARP、多奖励）与配套技术（推测解码、训练基础设施），产出 docx + html。

## 核心判断（主线）
训练算法沿五条主线演进：
1. **蒸馏逐步深化**：Gemma 1 无蒸馏（标准 next-token + 分阶段数据加权）→ Gemma 2 预训练引入全词汇概率蒸馏 → Gemma 3 细化为每 token 256-logit 加权采样蒸馏，并给出小/大 teacher 消融结论 → Gemma 4 继承 Gemma 3 配方。
2. **后训练从"数据混合"转向"教师蒸馏 + 强化学习"**：SFT 从 LM 逐对评估选数 → 行为克隆 + on-policy 蒸馏 → 大型 IT 教师蒸馏；RL 从经典 RLHF（Bradley-Terry + 未命名新颖算法）→ 换更大对话向奖励模型 + 模型平均（WARP 思想）→ BOND/WARM/WARP 改进版 + 多奖励信号（人类反馈、代码执行、数学 ground-truth）。
3. **对齐从单一偏好对走向多信号融合**，并在 Gemma 4 加入思维链（thinking）推理轨迹训练。
4. **工程配套升级**：量化感知训练（Gemma 3 QAT 5,000 步 → Gemma 4 扩展至编码器与更低 bit 位宽 + fp16 稳定标量）、推测解码（Gemma 4 MTP drafter：4 层 Transformer、交叉注意力到主模型 KV、top-k token cluster 投影）、训练基础设施（TPUv4/v5e/v5p → v6e、Slice-Granularity Elasticity 容错续训）。
5. **事实边界**：Gemma 1 的 RL 算法未披露名称（论文称 novel RL algorithm）；Gemma 2/3/4 的优化器、学习率等超参论文未披露；写作时如实标注。

## 关键事实速查（论文出处）
- Gemma 1：分阶段训练（stage training，借鉴 Gemini，后期提高高质量数据权重）；SFT 数据用 LM 逐对评估选择（Zheng 2023 思路）；RLHF=人类偏好对 + Bradley-Terry 奖励模型 + 新颖强化学习算法（未命名）+ 高容量模型自动评分缓解 reward hacking；训练基础设施 TPUv4/v5e/v5p、256-way data replication、ZeRO-3、GSPMD、Pathways、MegaScale XLA。
- Gemma 2：预训练知识蒸馏 min Σₓ −P_T(x|x_c)·log P_S(x|x_c)（Gemini 1.5 亦用）；SFT=行为克隆（teacher 合成响应）+ on-policy 蒸馏（Agarwal 2024；Gu et al. 2024）；RLHF 类似 Gemma 1.1 但奖励模型大一个数量级、更面向多轮对话；模型平均（Ramé et al. 2024，WARP 系）。
- Gemma 3：预训练蒸馏=按教师概率加权采样每 token 256 logits，学生仅在此采样集上交叉熵，未采样概率置零并重归一化；质量重加权（数据）；QAT 约 5,000 步、以非量化 checkpoint 概率为目标、适配预训练与后训练分布、支持 int4（per-channel/per-block）与 SFP8；消融：短训练视窗小 teacher 更优、长训练大 teacher 更优；post-training=大 IT 教师改进版蒸馏 + RL 微调（BOND、WARM、WARP 改进版）；多奖励=权重平均奖励模型（人类反馈数据）、代码执行反馈（Gehring et al. 2024）、数学 ground-truth 奖励（DeepSeek-AI 2025；Lambert et al. 2024）。
- Gemma 4：预训练类似 Gemma 3；QAT 双格式（mobile: per-channel int2/int4 权重 + int8 激活；Q4_0: blockwise），block 内标量缩放保 fp16 稳定；QAT 扩展至图像/音频编码器（W8A8 图像编码器 2× 内存缩减、44% 延迟下降；音频编码器 78% on-disk 缩减）；MTP drafter（4 层 Transformer、256/1024 维、3 local+1 global、交叉注意力到主模型 KV、E2B/E4B 用 top-k token cluster 将投影从 262k 降到 4096 维）；thinking mode（推理轨迹）；基础设施 TPUv4/v6e、Slice-Granularity Elasticity（局部故障少 slice 续训，中断从数分钟降到数秒）。

## 结构规划（docx，阿拉伯层级）
1. 概述与演进主线
2. 预训练算法演进（2.1-2.4 逐代、2.5 小结表1 + 图1 算法矩阵）
3. 指令微调（SFT）算法演进（3.1-3.4、3.5 小结表2）
4. 强化学习对齐算法演进（4.1-4.4、4.5 小结表3）
5. 蒸馏、量化与工程配套（5.1 蒸馏目标演进 图2、5.2 QAT、5.3 推测解码与训练基础设施 表4）
6. 总结与论文写作建议
7. 参考文献

## 引用核查表
| 引用名称 | 编号 | 发布方 | 年份 | 状态 | 来源 |
|---|---|---|---|---|---|
| Gemma 技术报告（1） | arXiv:2403.08295 | Google DeepMind | 2024 | 已核实（本地 PDF） | 本地 Gemma.pdf |
| Gemma 2 技术报告 | arXiv:2408.00118 | Google DeepMind | 2024 | 已核实（本地 PDF） | 本地 Gemma-2.pdf |
| Gemma 3 技术报告 | arXiv:2503.19786 | Google DeepMind | 2025 | 已核实（本地 PDF） | 本地 Gemma-3.pdf |
| Gemma 4 技术报告 | arXiv:2607.02770 | Google DeepMind | 2026 | 已核实（本地 PDF） | 本地 Gemma-4.pdf |
