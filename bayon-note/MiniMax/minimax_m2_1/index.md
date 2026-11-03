# MiniMax-M2.1 架构详解（M2 系列迭代版）

> **一句话**：2025 年 12 月 22 日发布的 M2 系列迭代模型，官方定位「多语言编程专家」——与 MiniMax-M2 的官方 config **逐字段 diff 后网络结构完全一致**（62 层 / 3072 / 256 专家 top-8 / 全 Softmax GQA / QK-Norm / MTP×3 / FP8），差异主要在训练目标与发布口径（编程能力升级 + highspeed 变体），而非网络改动。

> 命名说明：官方开源仓为 **`MiniMaxAI/MiniMax-M2.1`**，`model_type=minimax_m2`、架构 `MiniMaxM2ForCausalLM`，与 M2 共用同一套本地原生模块 `src/transformers/models/minimax_m2/`。属 API 系模型（平台侧同时提供 M2.1 与 M2.1-highspeed 两个变体），官方 HF 仓库已开放 config 与权重，本笔记以 config 字段 + 官方 release notes 双口径记录。

## 一、家族全景

| 维度 | 官方取值 | 来源 |
|---|---|---|
| 发布日期 | 2025-12-22 | 官方 release notes（platform.minimaxi.com） |
| 定位 | 多语言编程专家，全面升级复杂编程体验 | 官方 release notes |
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
| 服务变体 | M2.1 与 M2.1-highspeed（highspeed 为部署侧加速变体） | 官方 release notes（config 无法核验） |

## 二、与 M2 的 config diff（逐字段）

以 `minimax_m2/_tools/_cfgpull/config.json` 为基准：

**网络字段完全一致**：层数 62、hidden 3072、Q/KV 头 48/8、head_dim 128、专家 256/top-8/inter 1536、共享专家 0、sigmoid 路由、per-layer QK-Norm、RoPE 64/5e6、MTP×3、上下文 196608、词表 200064、FP8 e4m3fn（modules_to_not_convert 同为 gate / e_score_correction_bias / lm_head）。

**仅有的差异（均非网络结构）**：
- `transformers_version`：4.57.1（M2）→ 4.46.1（M2.1）；
- M2.1 config 省略了 M2 中的一批可选默认字段（`attention_dropout`、`bos/eos_token_id`、`initializer_range`、`layernorm_*_beta_factor`、`mlp_intermediate_size=8192`、`output_router_logits`、`router_aux_loss_coef`、`router_jitter_noise`、`shared_moe_mode`、`sliding_window`）——均为默认值字段，不构成架构差异；
- M2 中 `mlp_intermediate_size=8192` 在 M2.1 config 中同样不出现（继承 M2 的「原生模块不消费该字段」口径）。

结论：**M2.1 是 M2 的「同架构训练迭代版」**——能力提升（多语言编程专家）来自后训练/数据与训练配方，网络一个字段都没改。

## 三、端到端结构

词嵌入 → 62 层「GQA 全注意力 + 256 专家 MoE」堆叠（q/k 后接 per-layer RMSNorm）→ RMSNorm → LM Head；旁挂 3 个 MTP 头。与 M2 逐字段同构，详见 `minimax_m2/index.md` 第三~六节，本目录不重复展开。

## 四、注意力机制

| 层类型 | 层数 | 实现 |
|---|---|---|
| 全 Softmax GQA | 62（全部，`attn_type_list` 全 1） | 48 Q 头 / 8 KV 头，head_dim 128，q/k 后 RMSNorm，部分 RoPE 64/128 |

无 Lightning、无稀疏分支，与 M2 完全一致。

## 五、Token 与上下文

- 词表 200064（与 M2 同族；bos/eos 本 config 未单列，**待确认**）。
- `max_position_embeddings=196608`，纯 KV cache，无 Lightning 状态。

## 六、演进坐标

- **M2（2025-10-27）**：230B-A10B，全 Softmax，256 专家 top-8，QK-Norm + MTP + FP8。
- **M2.1（本模型，2025-12-22）**：架构与 M2 逐字段相同，定位收窄为「多语言编程专家」，新增 M2.1-highspeed 服务变体。
- **M2.5（2026-02）/ M2.7（2026-03-18）**：同底座继续迭代（见 `minimax_m2_5` / `minimax_m2_7`）。
- **M3（2026-06-01）**：换 MSA 块稀疏注意力 + 原生多模态（见 `minimax_m3_vl`）。

## 七、参考文献与依据

- 远端官方 config：https://huggingface.co/MiniMaxAI/MiniMax-M2.1 （config.json，拉取日期 2026-11-03，hf-mirror）
- 官方 release notes：https://platform.minimaxi.com/docs/release-notes/models （2025-12-22 条：MiniMax-M2.1 / M2.1-highspeed，多语言编程专家）
- transformers 源码：`src/transformers/models/minimax_m2/configuration_minimax_m2.py`、`modeling_minimax_m2.py`
- 对照基准：`E:\transformers\bayon-note\MiniMax\minimax_m2\index.md`
- 依据口径：远端官方 config（hf-mirror）+ 官方 release notes；highspeed 变体与能力口径为 release notes 表述，config 无法承载处已标注
