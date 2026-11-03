# MiniMax-M3.1 Flash 架构详解（M3 底座效率版）

> **一句话**：2026 年 10 月 19 日发布的最新开源多模态模型，**与 M3 共用同一 `minimax_m3_vl` 底座**（文本塔 60 层 / 6144 / 128 专家 top-4 + 1 共享 / MSA 块稀疏 / SwiGLU-OAI / per-head QK-Norm / GemmaNorm / 1M 上下文 / CLIP 视觉塔），定位效率：官方口径约 3× 推理加速、约 70% agentic 成本下降，config 侧真实差异为 **全部 60 层统一走 MSA 稀疏（取消前 3 层全注意力）、去掉 MTP、rope_theta 5e6→1e7、视觉最大分辨率 2016→672、权重改 MxFP8 + MoE NVFP4 量化**。

> 命名说明：官方开源仓为 **`MiniMaxAI/MiniMax-M3.1-Flash`**，`model_type=minimax_m3_vl`、顶层架构 `MiniMaxM3SparseForConditionalGeneration`（与 M3 同一入口类），本地原生模块同为 `src/transformers/models/minimax_m3_vl/`。本笔记以远端官方 config 逐字段 diff + 官方博客双口径记录。

## 一、家族全景

| 维度 | 官方取值 | 来源 |
|---|---|---|
| 发布日期 | 2026-10-19 | 官方博客（minimaxai.co，文章「MiniMax M3.1 Flash: Redefining Agent Performance and Cost」） |
| 定位 | 效率导向：约 3× 推理加速、约 70% agentic 成本下降；保留 1M 上下文与多模态输入；开放权重 | 官方博客 |
| API 定价 | 约 $0.15 输入 / $0.60 输出（每 1M token） | 官方博客口径 |
| 文本塔层数 / hidden | 60 层 / 6144（与 M3 相同） | 远端 config `text_config` |
| 注意力头 / KV 头 | 64 / 4（GQA 16:1，与 M3 相同） | 远端 config |
| 头维 / RoPE 旋转维 | 128 / 64（partial 0.5，与 M3 相同） | 远端 config |
| RoPE 基频 `rope_theta` | **10,000,000（1e7）** ← M3 为 5e6 | 远端 config |
| Dense 层中间维 `dense_intermediate_size` | 12288（保留） | 远端 config |
| 路由专家中间维 `intermediate_size` | 3072（保留） | 远端 config |
| 共享专家中间维 `shared_intermediate_size` | 3072（1 个常驻共享专家，保留） | 远端 config |
| 路由专家数 / top-k | 128 / 4（保留） | 远端 config |
| 路由缩放 `routed_scaling_factor` | 2.0（保留） | 远端 config |
| 层类型 `moe_layer_freq` | 前 3 层 `0`（dense），后 57 层 `1`（sparse）——**与 M3 相同** | 远端 config |
| 注意力 `sparse_attention_freq` | **60 层全部为 1（全层 MSA）** ← M3 为前 3 层 0 + 后 57 层 1 | 远端 config `sparse_attention_config` |
| 稀疏块大小 / top-k 块 / local | block 128 / topk_blocks 16 / local 1（与 M3 相同） | 远端 config |
| 索引头 `sparse_num_index_heads` / `sparse_index_dim` | 4 / 128（与 M3 相同） | 远端 config |
| `sparse_disable_index_value` | **60 层全部为 1** ← M3 为前 3 层 0 + 后 57 层 1 | 远端 config |
| 激活 | SwiGLU-OAI（α=1.702，limit=7.0，保留） | 远端 config |
| Norm | GemmaNorm + per_head QK-Norm（保留） | 远端 config |
| MTP | **无（`num_mtp_modules` / `num_nextn_predict_layers` 字段均不出现）** ← M3 为 ×7 | 远端 config |
| 最大位置 | 1,048,576（1M，保留） | 远端 config |
| 视觉塔 | CLIP：hidden 1280 / 32 层 / patch14 / 投影 6144（与 M3 相同） | 远端 config `vision_config` |
| 视觉最大分辨率 `image_size` | **672** ← M3 为 2016 | 远端 config `vision_config` |
| 视觉 token 压缩 | patch_merge：空间×2、时间×2，3D 轴向 RoPE（保留） | 远端 config |
| 图像/视频 token id | 200025 / 200026（保留）；`image_seq_length=576` | 远端 config |
| 权重格式 | **MxFP8**（`quant_method=mxfp8`，weight_block [1,32]，group_size 16）+ MoE **NVFP4**（`moe_quant_algo=NVFP4`，`nvfp4-pack-quantized`）；忽略层含 gate/lm_head/embed_tokens/projector/vision_tower/patch_merge_mlp | 远端 config `quantization_config`（M3 为 bf16 开放权重、无量化字段） |

## 二、与 M3 的 config diff（逐字段）

以 `minimax_m3_vl/_tools/_cfgpull/config.json` 为基准：

**架构继承点（与 M3 完全一致）**：text_config 的 hidden 6144 / 60 层 / 64 Q 头 / 4 KV 头 / head_dim 128 / partial_rotary 0.5 / swigluoai（α=1.702, limit 7.0）/ use_gemma_norm / per-head QK-Norm / dense_intermediate 12288 / shared 3072 / 128 专家 top-4 / n_shared_experts 1 / routed_scaling 2.0 / sigmoid 路由 / moe_layer_freq 前 3 层 dense；稀疏块 128 / topk 16 / local 1 / index 4 头 128 维；1M 上下文；CLIP 视觉塔 1280/32 层/patch14/proj 6144/3D rope/patch_merge 时空×2/segment ≤4 帧；image 200025 / video 200026。

**真实差异（5 处，均为效率取向）**：
1. **MTP 取消**：`num_mtp_modules=7`、`num_nextn_predict_layers=1` 两个字段均不出现——Flash 版不挂 MTP 头，省推理分支；
2. **全层 MSA**：`sparse_attention_freq` 由「前 3 层 0 + 后 57 层 1」改为 **60 层全 1**，`sparse_disable_index_value` 同步改为全 1——前 3 个原本全注意力的 dense 层也并入稀疏注意力路径（dense 层类型本身仍由 `moe_layer_freq` 保持前 3 层 dense FFN）；
3. **RoPE 基频 5e6 → 1e7**（text_config.rope_theta）；
4. **视觉最大分辨率 2016 → 672**（vision_config.image_size；顶层 `image_grid_pinpoints` 字符串仍保留到 2016 的档位表，与 image_size 的关系**待确认**）；
5. **量化发布**：新增顶层 `quantization_config`——MxFP8（weight_block [1,32]、group_size 16）+ MoE NVFP4 pack 量化，gate / lm_head / embed_tokens / 多模态投影与视觉塔在忽略列表中。

结论：**M3.1-Flash = M3 底座 + 推理侧瘦身**（去 MTP、全层稀疏、视觉降分辨率、权重量化），换来官方口径约 3× 加速与约 70% agentic 成本下降；1M 上下文与多模态输入能力保留。

## 三、端到端结构

图像/视频 token（CLIP 塔 + patch_merge 压缩）与文本 token early-fusion 进 60 层堆叠：前 3 层 dense FFN + 全层 MSA 块稀疏注意力，后 57 层 128 专家 top-4 MoE（+1 共享专家），最后 LM Head；**无 MTP 头**（与 M3 的 ×7 不同）。

## 四、注意力机制

| 层 | 注意力 |
|---|---|
| 全部 60 层 | MSA 块稀疏（block 128，top-16 块 + 1 局部，index 4 头） |

与 M3 的差异：M3 是「前 3 层全 Softmax + 后 57 层 MSA」，Flash 把前 3 层也并入 MSA 路径——这是 config 里能看到的唯一注意力排布差异，索引分支参数（4 头 / 128 维 / topk 16 / block 128）不变。

## 五、Token 与上下文

- 上下文 1,048,576（1M）——官方口径明确保留。
- 词表继承 200064 体系，视觉/视频占位 200025 / 200026。

## 六、演进坐标

- **M2 系（M2/M2.1/M2.5/M2.7）**：全 Softmax GQA + 256 专家，API 系迭代。
- **M3（2026-06-01，06-12 开源权重）**：MSA 块稀疏 + 原生多模态 + 前 3 层 dense + MTP×7，bf16 开放权重。
- **M3.1 Flash（本模型，2026-10-19）**：同一底座效率版——去 MTP、全层 MSA、视觉分辨率 672、MxFP8 + NVFP4 量化发布，主打 agentic 场景的成本/速度。

## 七、参考文献与依据

- 远端官方 config：https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash （config.json，拉取日期 2026-11-03，hf-mirror）
- 官方博客：*MiniMax M3.1 Flash: Redefining Agent Performance and Cost*，https://minimaxai.co （2026-10-19；效率/成本/定价为博客口径）
- transformers 源码：`src/transformers/models/minimax_m3_vl/configuration_minimax_m3_vl.py`
- 对照基准：`E:\transformers\bayon-note\MiniMax\minimax_m3_vl\index.md`
- 依据口径：远端官方 config（hf-mirror）+ transformers 源码 + 官方博客；效率数字与定价为博客口径，config 无法承载处已标注
