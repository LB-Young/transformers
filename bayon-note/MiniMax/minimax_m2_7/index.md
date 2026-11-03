# MiniMax-M2.7 架构详解（M2 系列收官迭代版）

> **一句话**：2026 年 3 月 18 日发布的 M2 系列收官迭代，官方定位「开启模型的自我迭代」——官方 config 与 M2 底座**几乎逐字段相同**，两处真实差异是 `max_position_embeddings` 由 196608 扩到 **204800**（204K）、新增 `dtype=bfloat16` 字段；256 专家 top-8 / 全 Softmax GQA / per-layer QK-Norm / MTP×3 / FP8 全部保留，提供 M2.7 与 M2.7-highspeed 双变体。

> 命名说明：官方开源仓为 **`MiniMaxAI/MiniMax-M2.7`**，`model_type=minimax_m2`、架构 `MiniMaxM2ForCausalLM`，与 M2 共用本地原生模块 `src/transformers/models/minimax_m2/`。API 系模型，本笔记以 config 字段 + 官方 release notes 双口径记录。

## 一、家族全景

| 维度 | 官方取值 | 来源 |
|---|---|---|
| 发布日期 | 2026-03-18 | 官方 release notes（platform.minimaxi.com） |
| 定位 | 「开启模型的自我迭代」；M2.7 / M2.7-highspeed 双变体 | 官方 release notes |
| 架构类别 | decoder-only · MoE · 全 Softmax GQA（M2 底座） | 远端官方 config |
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
| 最大位置 `max_position_embeddings` | **204800（204K）** ← M2 为 196608 | 远端官方 config |
| 计算 dtype | 新增字段 `dtype=bfloat16`（M2 config 无此字段） | 远端官方 config |
| 词表 | 200064（bos/eos 未在 config 单列，**待确认**） | 远端官方 config |
| 发布权重 | FP8（`quantization_config: float8_e4m3fn`，dynamic，weight_block 128×128） | 远端官方 config |

## 二、与 M2 的 config diff（逐字段）

以 `minimax_m2/_tools/_cfgpull/config.json` 为基准：

**真实差异（2 处）**：
1. **`max_position_embeddings`：196608 → 204800**——上下文窗口在 config 层扩到 200K（呼应 M2 笔记里「公开口径 204K」的提法，M2.7 把它落成了 config 显式值）；
2. **新增 `dtype: "bfloat16"` 字段**——纯标注性字段，不改变网络结构。

**其余网络字段完全一致**：层数 62、hidden 3072、Q/KV 头 48/8、head_dim 128、专家 256/top-8/inter 1536、共享专家 0、sigmoid 路由、per-layer QK-Norm、RoPE 64/5e6、MTP×3、词表 200064、FP8 e4m3fn（modules_to_not_convert 同为 gate / e_score_correction_bias / lm_head）。

**省略的默认字段**：`transformers_version` 4.57.1→4.46.1；与 M2.1/M2.5 一样省略了 `attention_dropout`、`bos/eos_token_id`、`initializer_range`、`layernorm_*_beta_factor`、`mlp_intermediate_size=8192`、`output_router_logits`、`router_aux_loss_coef`、`router_jitter_noise`、`shared_moe_mode`、`sliding_window`（均为默认值字段，不构成架构差异）。

结论：**M2.7 是 M2 底座的收官微调版**——网络几乎没动，唯一结构性变化是上下文扩到 204K；「自我迭代」能力来自训练方法（config 无法承载，属 release notes 口径）。

## 三、端到端结构

词嵌入 → 62 层「GQA 全注意力 + 256 专家 MoE」堆叠（q/k 后接 per-layer RMSNorm）→ RMSNorm → LM Head；旁挂 3 个 MTP 头。与 M2/M2.1/M2.5 同构，详见 `minimax_m2/index.md` 第三~六节。

## 四、注意力机制

| 层类型 | 层数 | 实现 |
|---|---|---|
| 全 Softmax GQA | 62（全部，`attn_type_list` 全 1） | 48 Q 头 / 8 KV 头，head_dim 128，q/k 后 RMSNorm，部分 RoPE 64/128 |

与 M2 系完全一致，无稀疏分支。

## 五、Token 与上下文

- 词表 200064（bos/eos 本 config 未单列，**待确认**）。
- `max_position_embeddings=204800`（204K）——M2 系中唯一在 config 层显式扩窗的版本。

## 六、演进坐标

- **M2（2025-10-27）**：230B-A10B，全 Softmax，256 专家 top-8，192K 上下文。
- **M2.1（2025-12-22）** / **M2.5（2026-02）**：同架构同上下文迭代（见 `minimax_m2_1` / `minimax_m2_5`）。
- **M2.7（本模型，2026-03-18）**：同底座，上下文扩至 204800，主打「自我迭代」，M2.7 / M2.7-highspeed 双变体。
- **M3（2026-06-01）**：换 MSA 块稀疏注意力 + 原生多模态（见 `minimax_m3_vl`）。

## 七、参考文献与依据

- 远端官方 config：https://huggingface.co/MiniMaxAI/MiniMax-M2.7 （config.json，拉取日期 2026-11-03，hf-mirror）
- 官方 release notes：https://platform.minimaxi.com/docs/release-notes/models （2026-03-18 条：MiniMax-M2.7 / M2.7-highspeed，开启模型的自我迭代）
- transformers 源码：`src/transformers/models/minimax_m2/configuration_minimax_m2.py`、`modeling_minimax_m2.py`
- 对照基准：`E:\transformers\bayon-note\MiniMax\minimax_m2\index.md`
- 依据口径：远端官方 config（hf-mirror）+ 官方 release notes；「自我迭代」与 highspeed 变体为 release notes 表述，config 无法承载处已标注
