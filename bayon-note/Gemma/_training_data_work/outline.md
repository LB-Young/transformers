# Gemma 训练数据演进详解 · outline

## 任务
基于 4 篇 Gemma 技术报告（Gemma 1/2/3/4，无 3n），系统整理预训练、SFT、强化学习等全部阶段的训练数据改进，产出 docx + html 双载体文档，供用户论文引用。

## 核心判断（主线）
Gemma 系列训练数据的演进沿四条主线展开：
1. **规模扩张**：预训练 token 预算从 Gemma 1（2B=3T / 7B=6T）扩到 Gemma 2（2B=2T / 9B=8T / 27B=13T）再到 Gemma 3（1B=2T / 4B=4T / 12B=12T / 27B=14T）；Gemma 4 论文未披露 token 预算（事实边界，标注"未披露"），数据范围扩展至图像与音频。
2. **多语言/多模态扩展**：Gemma 1 纯英文文本 → Gemma 2 仍英文主导（含科学文章）→ Gemma 3 增加多语言（单语+平行语料、语言不平衡处理）与图文混合 → Gemma 4 含图像、音频（E2B/E4B/12B），cutoff 2025-01。
3. **合成数据与蒸馏深化**：Gemma 1 人工+合成 SFT 对 → Gemma 2 预训练阶段知识蒸馏 + post-training teacher 合成响应 + LMSYS-chat-1M prompts → Gemma 3/4 post-training 全面转向"大 IT teacher 蒸馏 + RL"。
4. **对齐数据演进**：Gemma 1 经典 RLHF（Bradley-Terry + 英文偏好对）→ Gemma 2 更大的对话向 reward model + 模型平均 → Gemma 3 BOND/WARM/WARP 改进 + 多奖励信号（人类偏好、代码执行反馈、数学 ground-truth）→ Gemma 4 继承并新增 thinking mode。
5. **过滤/治理贯穿始终**：每代都执行——去除个人信息与敏感数据、评估集去污染、降低 recitation、in-context attribution/hedging/refusal 事实性子集；从启发式+模型分类器逐步演进到质量重加权（Gemma 3，Sachdeva 2024）。

## 结构规划（docx，阿拉伯小数层级）
1. 概述与演进主线（总-分-总；摘要式开篇）
2. 预训练数据演进（2.1-2.4 逐代，2.5 对比表+token 预算条形图）
3. 指令微调（SFT）数据演进（3.1-3.4 逐代，3.5 对比表）
4. 强化学习对齐数据演进（4.1-4.4 逐代，4.5 对比表）
5. 贯穿各代的共同数据处理主线（过滤管道演进，流程示意）
6. 总结与论文写作建议
参考文献（4 篇论文 arXiv 编号）

html 同内容，时间轴+对比表+ECharts 条形图，信息图/研报风格。

## 关键事实速查（均来自论文原文，将写入正文）
- Gemma 1：2B=3T tokens、7B=6T tokens；英文 web/数学/代码；256k tokenizer（Gemini 子集）；阶段式训练提高后期高质量数据权重；过滤=启发式+模型分类器；SFT=纯文本纯英文合成+人工对；LM-based 评估选混合；RLHF=Bradley-Terry 奖励模型+新颖 RL 算法+高容量自动评分者。
- Gemma 2：27B=13T、9B=8T、2B=2T；web/代码/科学文章；预训练知识蒸馏（teacher 概率）；post-training 扩展自 Gemma 1.1（内部+外部公开数据；LMSYS-chat-1M 的 prompts 不含 answers）；SFT=行为克隆（teacher 合成响应）+学生分布蒸馏；RLHF=更大 reward model（比 policy 大一个数量级、面向多轮对话）+ 模型平均。
- Gemma 3：27B=14T、12B=12T、4B=4T、1B=2T；增量含图文混合；多语言单语+平行；262k tokenizer（Gemini 2.0 同款）；质量重加权（Sachdeva 2024）；蒸馏=每 token 采样 256 logits；post-training=改进版知识蒸馏（大 IT teacher）+RL（BOND/WARM/WARP 改进）；多奖励=权重平均奖励模型（人类反馈）、代码执行反馈、数学 ground-truth。
- Gemma 4：预训练类似 Gemma 3；大规模多样化数据（web/代码/图像/音频，cutoff 2025-01）；token 预算未披露；262k tokenizer；post-training 类似 Gemma 3 + thinking mode；IT 输出 <turn|>。
- 事实边界：Gemma 4 token 预算、各代 SFT/RL 数据的具体体量论文均未披露，正文以"论文未披露"标注，不编造。

## 引用核查表
| 引用名称 | 编号 | 发布方 | 年份 | 状态 | 来源 |
|---|---|---|---|---|---|
| Gemma 技术报告（1） | arXiv:2403.08295 | Google DeepMind | 2024 | 已核实（本地 PDF） | 本地 Gemma.pdf |
| Gemma 2 技术报告 | arXiv:2408.00118 | Google DeepMind | 2024 | 已核实（本地 PDF） | 本地 Gemma-2.pdf |
| Gemma 3 技术报告 | arXiv:2503.19786 | Google DeepMind | 2025 | 已核实（本地 PDF） | 本地 Gemma-3.pdf |
| Gemma 4 技术报告 | arXiv:2607.02770 | Google DeepMind | 2026 | 已核实（本地 PDF） | 本地 Gemma-4.pdf |
