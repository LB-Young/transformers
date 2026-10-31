# GLM-130B 官方 config 留档说明

GLM-130B **无 Hugging Face 公开权重仓库**，亦无 HF 格式 config.json。
本目录不放置远端 config，配置数字以以下官方来源为准（已在 index.md / index.html 逐字段标注）：

- 官方论文：GLM-130B: An Open Bilingual Pre-Trained Model, arXiv:2210.02414
  - §2.1 架构（DeepNorm、α=(2N)^(1/4)、双向 GLM）
  - §2.3 训练配置（70 层、hidden 12288、序列 2048）
  - 附录 Table 11 完整训练配置（ffn_hidden_size=32768、deepnorm=True 等）
  - §B.1 icetk 词表 150000；§B.3 GeGLU（d_ffn=8/3·d_h）
- 官方 GitHub：https://github.com/THUDM/GLM-130B
- 清华 KEG 博客：https://keg.cs.tsinghua.edu.cn/glm-130b/zh/posts/glm-130b/

核验日期：2026-10-31。num_attention_heads=96 为官方公开口径，论文附录 config 未逐字段列出，标「待确认」。
