# MiniMax-M2.5 架构详解（M2 系列迭代版）

> **一句话**：2026 年 2 月发布的 M2 系列迭代模型，官方口径「在编程、工具调用和搜索、办公等生产力场景都达到或刷新了行业 SOTA」——官方 config 与 M2（及 M2.1）**逐字段 diff 后网络结构完全一致**（62 层 / 3072 / 256 专家 top-8 / 全 Softmax GQA / QK-Norm / MTP×3 / FP8），差异在训练配方与能力定位，不在网络。

> 命名说明：官方开源仓为 **`MiniMaxAI/MiniMax-M2.5`**，`model_type=minimax_m2`、架构 `MiniMaxM2ForCausalLM`，与 M2 共用本地原生模块 `src/transformers/models/minimax_m2/`。API 系模型，平台侧同时提供 M2.5 与 M2.5-highspeed 两个变体；本笔记以 config 字段 + 官方 release notes 双口径记录。

## 一、家族全景

| 维度 | 官方取值 | 来源 |
|---|---|---|
| 发布日期 | 2026-02 | 官方 release notes（platform.minimaxi.com） |
| 定位 | 编程、工具调用、搜索、办公等生产力场景 SOTA | 官方 release notes |
| 架构类别 | decoder-only · MoE · 全 Softmax GQA（与 M2 同构） | 远端官方 config |
| 层数 / hidden | 62 层 / 3072 | 远端官方 config |
| 注意力头 / KV 头 | 48 / 8（GQA） | 远端官方 config |
| 头维 / RoPE 旋转维 | 128 / 64（部分旋转） | 远端官方 config |
| RoPE 基频 `rope_theta` | 5,000,000（5e6） | 远端官方 config |
| 专家中间维 `intermediate_size` | 1536（每个路由专家） | 远端官方 config |
| 路由专家数 / top-k | 256 / 8 | 远端官方 config |
| 共享专家 `shared_intermediate_size` | 0（无共享专家） | 远端官方 config |
| 路由打分 `scoring_func` | sigmoid（+ `use_routing_bias=true`） | 远端官方 config |
| QK-Norm | `use_qk_norm=true`，`qk_norm_type=per_layer` | 远端官方 config |
| 激活 | SiLU（`hidden_act=silu`） | 远端官方 config |
| MTP | `use_mtp=true` / `num_mtp_modules=3`（`mtp_transformer_layers=1`） | 远端官方 config |
| 最大位置 `max_position_embeddings` | 196608（192K，与 M2 相同） | 远端官方 config |
| 词表 | 200064（bos/eos 未在 config 单列，**待确认**） | 远端官方 config |
| 发布权重 | FP8（`quantization_config: float8_e4m3fn`，dynamic，weight_block 128×128） | 远端官方 config |
| 服务变体 | M2.5 与 M2.5-highspeed（highspeed 为部署侧加速变体） | 官方 release notes（config 无法核验） |

## 二、与 M2 的 config diff（逐字段）

以 `minimax_m2/_tools/_cfgpull/config.json` 为基准：

**网络字段完全一致**：层数 62、hidden 3072、Q/KV 头 48/8、head_dim 128、专家 256/top-8/inter 1536、共享专家 0、sigmoid 路由、per-layer QK-Norm、RoPE 64/5e6、MTP×3、上下文 196608、词表 200064、FP8 e4m3fn（modules_to_not_convert 同为 gate / e_score_correction_bias / lm_head）。

**仅有的差异（均非网络结构）**：
- `transformers_version`：4.57.1（M2）→ 4.46.1（M2.5）；
- M2.5 config 省略了 M2 中的一批可选默认字段（`attention_dropout`、`bos/eos_token_id`、`initializer_range`、`layernorm_*_beta_factor`、`mlp_intermediate_size=8192`、`output_router_logits`、`router_aux_loss_coef`、`router_jitter_noise`、`shared_moe_mode`、`sliding_window`）——均为默认值字段，不构成架构差异；
- 与 `minimax_m2_1`（M2.1）config **互相 diff 亦无网络字段差异**（连 transformers_version 都是 4.46.1）。

结论：**M2.5 是 M2 底座的第二次同架构训练迭代**——SOTA 口径来自后训练/强化学习与工具调用数据，网络未改。

## 三、端到端结构

词嵌入 → 62 层「GQA 全注意力 + 256 专家 MoE」堆叠（q/k 后接 per-layer RMSNorm）→ RMSNorm → LM Head；旁挂 3 个 MTP 头。与 M2/M2.1 逐字段同构，详见 `minimax_m2/index.md` 第三~六节。

## 四、注意力机制

| 层类型 | 层数 | 实现 |
|---|---|---|
| 全 Softmax GQA | 62（全部，`attn_type_list` 全 1） | 48 Q 头 / 8 KV 头，head_dim 128，q/k 后 RMSNorm，部分 RoPE 64/128 |

与 M2/M2.1 完全一致，无稀疏分支。

## 五、Token 与上下文

- 词表 200064（bos/eos 本 config 未单列，**待确认**）。
- `max_position_embeddings=196608`，纯 KV cache。

## 六、演进坐标

- **M2（2025-10-27）**：230B-A10B，全 Softmax，256 专家 top-8，QK-Norm + MTP + FP8。
- **M2.1（2025-12-22）**：同架构迭代，多语言编程专家。
- **M2.5（本模型，2026-02）**：同架构再迭代，编程/工具调用/搜索/办公 SOTA 口径，M2.5 / M2.5-highspeed 双变体。
- **M2.7（2026-03-18）**：同底座 + 上下文扩到 204800（见 `minimax_m2_7`）。
- **M3（2026-06-01）**：换 MSA 块稀疏注意力 + 原生多模态（见 `minimax_m3_vl`）。

## 七、参考文献与依据

- 远端官方 config：https://huggingface.co/MiniMaxAI/MiniMax-M2.5 （config.json，拉取日期 2026-11-03，hf-mirror）
- 官方 release notes：https://platform.minimaxi.com/docs/release-notes/models （2026 年 2 月条：MiniMax-M2.5 / M2.5-highspeed，编程/工具调用/搜索/办公 SOTA）
- transformers 源码：`src/transformers/models/minimax_m2/configuration_minimax_m2.py`、`modeling_minimax_m2.py`
- 对照基准：`E:\transformers\bayon-note\MiniMax\minimax_m2\index.md`、`minimax_m2_1\index.md`
- 依据口径：远端官方 config（hf-mirror）+ 官方 release notes；highspeed 变体与 SOTA 口径为 release notes 表述，config 无法承载处已标注
