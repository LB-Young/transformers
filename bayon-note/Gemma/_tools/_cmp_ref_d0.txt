Gemma-4-26B-A4B-it 参数构成拆解

为什么总参数是 26B、激活参数是 A4B

依据：Hugging Face config.json + transformers/models/gemma4/modular_gemma4.py。所有数值按 config 实际取值与源码 forward 路径逐项复算。

1. 关键配置速查（text_config）

| 符号 | 含义 | 取值 |
| --- | --- | --- |
| H | hidden_size | 2816 |
| L | num_hidden_layers | 30 |
| V | vocab_size | 262,144 |
| n_head | num_attention_heads | 16 |
| n_kv (sliding) | num_key_value_heads（sliding 层） | 8 |
| n_kv (full) | num_global_key_value_heads（full 层） | 2 |
| d_head (sliding) | head_dim（sliding 层） | 256 |
| d_head (full) | global_head_dim（full 层） | 512 |
| I_d | intermediate_size（shared/dense MLP） | 2112 |
| I_e | moe_intermediate_size（每个 routed expert） | 704 |
| N_e | num_experts | 128 |
| K | top_k_experts | 8 |
| L_s / L_f | sliding / full 层数 | 25 / 5 |
| — | tie_word_embeddings | true |
| — | hidden_size_per_layer_input（PLE） | 0（关闭） |
| — | num_kv_shared_layers | 0 |
| — | attention_k_eq_v | true（full 层 K=V） |

layer_types 在 30 层中按 "sliding×5 + full×1" 重复，最后一层强制 full，故 L_s=25、L_f=5。

2. 整体架构图

图 1  Gemma-4-26B-A4B-it 整体数据流：每一层 FFN 由 Shared Dense MLP 与 Routed MoE 并联组成

关键设计：源码 modular_gemma4.py 第 1218–1234 行显示，每一层的 FFN 子层是「Shared Dense MLP（恒激活）」+「Router 路由到 Top-K Experts（稀疏激活）」并联，二者输出相加后再做残差：

# modular_gemma4.py L1218-L1234
residual = hidden_states
hidden_states = self.pre_feedforward_layernorm(hidden_states)
hidden_states = self.mlp(hidden_states)                 # dense MLP, 每 token 都过
if self.enable_moe_block:
    hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)
    _, top_k_w, top_k_idx = self.router(hidden_states_flat)   # 选 8/128
    hidden_states_2 = self.pre_feedforward_layernorm_2(hidden_states_flat)
    hidden_states_2 = self.experts(hidden_states_2, top_k_idx, top_k_w)
    hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)
    hidden_states = hidden_states_1 + hidden_states_2         # 并联相加

3. 单层 FFN 内部 MoE 路由示意

图 2  128 个专家池，每 token 由 Router 选出 Top-8，加权求和后与 Shared Dense MLP 输出相加

N_e = 128 个专家，每 token 只触发 K = 8 个  →  专家部分激活比例 K/N_e = 8/128 = 1/16

4. 逐项参数计算

4.1 Token Embedding（与 lm_head 共享）

P_embed = V · H = 262,144 × 2,816 = 738,197,504 ≈ 0.738 B

因 tie_word_embeddings=true，输出侧 lm_head 不重复占用参数。

4.2 Attention（分两种层）

(a) Sliding 层（25 层，d_h=256, n_kv=8）

| 算子 | 形状 | 参数量 |
| --- | --- | --- |
| q_proj | 2816 × (16·256) = 2816 × 4096 | 11,534,336 |
| k_proj | 2816 × (8·256) = 2816 × 2048 | 5,767,168 |
| v_proj | 2816 × (8·256) = 2816 × 2048 | 5,767,168 |
| o_proj | 4096 × 2816 | 11,534,336 |
| q_norm / k_norm | 各 d_h=256 | 512 |
| v_norm | with_scale=False | 0 |

P_attn^(s) = 2816·(4096+2048+2048+4096) + 512 = 34,603,520

25 层合计 = 25 × 34,603,520 = 865,088,000

(b) Full 层（5 层，d_h=512, n_kv=2，K=V 共享 → v_proj 不存在）

| 算子 | 形状 | 参数量 |
| --- | --- | --- |
| q_proj | 2816 × (16·512) = 2816 × 8192 | 23,068,672 |
| k_proj | 2816 × (2·512) = 2816 × 1024 | 2,883,584 |
| v_proj | 无（v = k） | 0 |
| o_proj | 8192 × 2816 | 23,068,672 |
| q_norm / k_norm | 各 512 | 1,024 |

P_attn^(f) = 2816·(8192+1024+8192) + 1024 = 49,021,952

5 层合计 = 5 × 49,021,952 = 245,109,760

P_attn 总 = 865,088,000 + 245,109,760 = 1,110,197,760 ≈ 1.11 B

4.3 Shared Dense MLP（每层恒激活，SwiGLU 三门）

P_dense,layer = 3 · H · I_d = 3 × 2816 × 2112 = 17,842,176

P_dense = 30 × 17,842,176 = 535,265,280 ≈ 0.535 B

4.4 Router（每层一个）

Gemma4TextRouter 由 norm（with_scale=False，0 参数）、proj（H→N_e）、scale（H）、per_expert_scale（N_e）组成：

P_router,layer = 2816×128 + 2816 + 128 = 363,392

P_router = 30 × 363,392 = 10,901,760 ≈ 0.011 B

4.5 Routed Experts（参数大头）

MixtralExperts 将 gate/up 融合为一个参数张量：gate_up_proj [N_e, 2·I_e, H]，down_proj [N_e, H, I_e]。

P_exp,layer = N_e · (2·I_e·H + H·I_e) = N_e · 3·H·I_e

= 128 × 3 × 2816 × 704 = 761,266,176 ≈ 0.761 B / 层

P_exp,total = 30 × 761,266,176 = 22,837,985,280 ≈ 22.84 B

这是整个模型的参数主体，也是"总参数 26B"的主要来源。

4.6 其它 RMSNorm

每层 7 个 RMSNorm（input / post-attn / pre-ffn / post-ffn + MoE 的 3 个），每个维度 H=2816，合计约 0.59 M，相对其它项 < 0.01%，可忽略。

4.7 Vision Encoder（多模态视觉塔）

vision_config：hidden=1152, intermediate=4304, 27 层, 16 heads, head_dim=72。粗算约 0.558 B，与 NVIDIA NGC 标注的 "vision encoder ~550M" 一致。纯文本推理时它不参与激活，但计入总参数。

5. 总参数汇总

| 模块 | 参数量 | 占比 |
| --- | --- | --- |
| Token Embedding（共享 lm_head） | 0.738 B | 2.9% |
| Attention（25 sliding + 5 full） | 1.110 B | 4.3% |
| Shared Dense MLP（30 层） | 0.535 B | 2.1% |
| Router（30 层） | 0.011 B | 0.04% |
| Routed Experts（30 × 128） | 22.838 B | 88.6% |
| 各种 RMSNorm | ~0.001 B | ~0% |
| Text 小计 | ≈ 25.23 B | 97.9% |
| Vision Encoder | ≈ 0.56 B | 2.1% |
| 总计 | ≈ 25.79 B ≈ 26 B | 100% |

这就是名字里 26B 的来历：22.84B 全部分布在 30 层 × 128 个专家的庞大专家池里，其余所有"通用"参数加起来才 ~2.4B。

6. 激活参数（为什么是 A4B）

每来一个 token：Embedding 查表、Attention、Shared Dense MLP、Router 全部参数参与；只有 Experts 只走 Top-8，其余 120 个专家权重这一步不加载。

激活的专家参数：

P_exp,active = P_exp,total · (K / N_e) = 22.838 B × (8/128) = 22.838 / 16 = 1.427 B

固定激活部分（与是否稀疏无关）：

P_fixed = P_embed + P_attn + P_dense + P_router

= 0.738 + 1.110 + 0.535 + 0.011 = 2.394 B

总激活参数：

P_active = P_fixed + P_exp,active = 2.394 + 1.427 ≈ 3.82 B ≈ 4 B

| 项 | 参数量 |
| --- | --- |
| Embedding + Attention + Dense MLP + Router（恒激活） | 2.39 B |
| 8 / 128 个 Experts 被激活 | 1.43 B |
| 每 token 实际激活 | ≈ 3.82 B ≈ 4 B |

注：激活参数口径通常不含 vision encoder（纯文本解码时视觉塔不参与）；即使把 0.56B 视觉塔算上也只是 4.4B，命名口径仍取文本侧 ~4B。

7. 一句话总结

Gemma-4-26B-A4B-it 的 "26B" 几乎全部来自 30 层 × 128 个 routed experts（每个专家 3 × 2816 × 704），合计 22.84B；其余 embedding、attention、shared dense MLP、router 加起来才 ~2.4B。前向时每个 token 由 Router 从 128 个专家里只挑 8 个（即 1/16），专家部分只激活 22.84/16 ≈ 1.43B，加上恒激活的 2.39B，每个 token 真正参与矩阵乘法的参数约 3.82B，四舍五入即 A4B。
