# Kimi K3 家族架构详解

> 官方 HF 仓库：moonshotai/Kimi-K3（远端官方 config：hf-mirror.com 镜像拉取，2026-11-03）；技术报告：arXiv:2607.24653《Kimi K3: Open Frontier Intelligence》（47 页技术报告 PDF 已下载留档：`_tools/_cfgpull/k3_tech_report.pdf`，来源 GitHub 仓库 MoonshotAI/Kimi-K3 分支 commit 3cb39dfd）；本地 transformers 模块核验（Glob 于 E:\transformers\src\transformers\models\）：**无 `models/kimi_k3/` 模块**（顶层 `KimiK3ForConditionalGeneration` 走仓库自带 custom code，auto_map 指向 configuration_kimi_k3 / modeling_kimi_k3）；text_config 运行时实例化为 `KimiLinearForCausalLM`（auto_map 指向 modeling_kimi_linear.KimiLinearModel，本地 `models/kimi_linear/` 存在 configuration_kimi_linear.py / modeling_kimi_linear.py / modular_kimi_linear.py）。本机 E:\models 无该模型权重（官方仅发布 MXFP4 量化权重），全部 config 数值以远端官方 config.json 为准（留档 `_tools/_cfgpull/config.json`，拉取日期 2026-11-03）。

## 1. 家族全景

Kimi K3（2026-07-16 API 上线、2026-07-27 开源权重 + 47 页技术报告）是 Moonshot 当前旗舰：**2.8T 总参 / 104B 激活**的原生多模态 MoE，1M 上下文。架构沿三个维度扩展信息流——序列长度（Hybrid Attention：3×KDA + 1×Gated MLA，3:1 混合）、网络深度（AttnRes 注意力残差）、模型宽度（Stable LatentMoE，896 路由专家每 token 激活 16，稀疏度 56）；配合 SiTU-GLU 激活、Quantile Balancing 负载均衡、Per-Head Muon 优化器，整体 scaling efficiency 较 Kimi K2 提升约 **2.5×**（技术报告 §1/§3.2）。

一句话：**入口类 `KimiK3ForConditionalGeneration`（model_type `kimi_k3`）= 视觉塔 MoonViT-V2（patch14 / 27 层 / 1024 hidden / 401M，从零训练）+ PatchMergerV2 投影器 + 93 层混合文本塔（69 层 KDA 混合线性注意力 + 24 层 Gated MLA + 第 0 层 dense），每注意力层配对一个 Stable LatentMoE 前馈（16/896 路由 + 2 共享），Block AttnRes 跨层注意力残差，MXFP4 原生量化（QAT）发布，NoPE 直接外推 1M 上下文。**

### 1.1 官方 config 关键实参（来源：moonshotai/Kimi-K3/config.json，2026-11-03 拉取）

| 字段 | 取值 | 来源 |
|---|---|---|
| architectures | `KimiK3ForConditionalGeneration` | 官方 config |
| model_type | `kimi_k3` | 官方 config |
| text_config.model_type / architectures | `kimi_linear` / `KimiLinearForCausalLM`（auto_map → modeling_kimi_linear.KimiLinearModel） | 官方 config + 本地 models\kimi_linear\ 核验 |
| num_hidden_layers | 93 | 官方 config text_config |
| hidden_size | 7168 | 官方 config text_config |
| num_attention_heads / num_key_value_heads | 96 / 96 | 官方 config text_config |
| kv_lora_rank / q_lora_rank | 512 / 1536 | 官方 config text_config |
| qk_nope_head_dim / qk_rope_head_dim / v_head_dim | 128 / 64 / 128 | 官方 config text_config |
| mla_use_nope / mla_use_output_gate | true / true（MLA 层 NoPE + 通道级全秩输出门） | 官方 config text_config |
| linear_attn_config.num_heads / head_dim | 96 / 128（KDA 头） | 官方 config text_config |
| linear_attn_config.short_conv_kernel_size | 4 | 官方 config text_config |
| linear_attn_config.use_full_rank_gate / gate_lower_bound | true / -5.0 | 官方 config text_config |
| linear_attn_config.full_attn_layers | 24 层：4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,93 | 官方 config text_config |
| linear_attn_config.kda_layers | 69 层（每 3 个 KDA 后接 1 个 full-attn，3:1 混合） | 官方 config text_config |
| attn_res_block_size | 12（Block AttnRes：93 层分 8 块，报告另计 embedding 层共 9 块） | 官方 config text_config + 技术报告 §2.2 |
| first_k_dense_replace | 1（第 0 层为 dense FFN） | 官方 config text_config |
| num_experts / num_experts_per_token / num_shared_experts | 896 / 16 / 2 | 官方 config text_config |
| moe_intermediate_size | 3072（latent 空间） | 官方 config text_config |
| routed_expert_hidden_size | 3584（LatentMoE 潜维 ℓ，= hidden 7168 的 0.5×） | 官方 config text_config + 技术报告 Table 1 |
| latent_moe_use_norm / moe_renormalize | true / true | 官方 config text_config |
| moe_router_activation_func / topk_method / use_grouped_topk | sigmoid / noaux_tc / true | 官方 config text_config |
| routed_scaling_factor / num_expert_group | 1.0 / 1 | 官方 config text_config |
| intermediate_size（dense） | 33792 | 官方 config text_config |
| hidden_act | **situ**（SiTU-GLU） | 官方 config text_config |
| activation_situ_beta / activation_situ_linear_beta | 4.0 / 25.0（与技术报告 §2.3.2 β₁=4、β₂=25 一致） | 官方 config text_config + 技术报告 |
| max_position_embeddings | 1048576（1M） | 官方 config text_config |
| rope_theta / rope_scaling | **无**（NoPE：位置信息由 KDA 循环门控隐式编码，无 RoPE 外推） | 官方 config text_config + 技术报告 §3.4 |
| vocab_size | 163840 | 官方 config text_config |
| tie_word_embeddings（text_config） | false | 官方 config text_config |
| bos / eos / pad token id | 163584 / 163586 / 163839 | 官方 config text_config |
| media_placeholder_token_id | 163605（顶层；图像统一占位） | 官方 config 顶层 |
| image_placeholder | `<\|kimi_image_placeholder\|>` | 官方 config 顶层 |
| vision_config（是否存在） | 有（见 §6） | 官方 config 顶层 |
| quantization | **mxfp4-pack-quantized**（compressed-tensors；weights: float num_bits=4 group_size=32 symmetric，scale_dtype=torch.uint8）；忽略 self_attn / shared_experts / mlp 门投影 / lm_head / vision_tower / mm_projector | 官方 config quantization_config |

### 1.2 与 K2.5/K2.6 的代际差异（来源：技术报告 Table 1 + 本目录 config）

| 维度 | Kimi K2 / K2.5 / K2.6 | Kimi K3 |
|---|---|---|
| 总参 / 激活 | 1.04T / 32.6B | **2.78T / 104.2B** |
| 层数 | 61（第 0 层 dense） | **93**（+52%） |
| hidden / 头数 | 7168 / 64（Q=KV） | 7168 / **96**（+50%） |
| 注意力 | 全 MLA（RoPE q 侧低秩） | **混合：69 KDA + 24 Gated MLA**（NoPE） |
| FFN 激活 | SwiGLU | **SiTU-GLU**（β₁=4、β₂=25，软上限，输出有界 100） |
| 路由专家 | 384 routed + 1 共享，top-8（稀疏度 48） | **896 routed + 2 共享，top-16**（稀疏度 56，latent 3584） |
| 负载均衡 | sigmoid 路由 + 固定步长 bias | **Quantile Balancing**（分位数 + 直方图估计） |
| 跨层信息流 | 标准残差累加 | **AttnRes**（block 12，伪查询跨层检索） |
| 视觉塔 | MoonViT-3D（初始化自 SigLIP-400M，hidden 1152/27 层/16 头/FFN 4304） | **MoonViT-V2**（从零训练，hidden 1024/27 层/12 头/FFN 4096，约 401M，RMSNorm 无 bias，像素重排 2×2） |
| 上下文 | 262144（YaRN ×64） | **1048576（NoPE 直接外推）** |
| 权重格式 | INT4（K2.5/2.6 发布版）/ FP8（K2-0905） | **MXFP4 权重 / MXFP8 激活**（QAT 自 SFT 阶段起，无 BF16 发布） |
| 优化器 | MuonClip | **Per-Head Muon**（按头 Newton-Schulz 正交化）+ K2 权重裁剪 |

### 1.3 端到端总览

```
图像/视频 ──► MoonViT-V2 (patch14, 27层, hidden1024, 12头, 从零训练)
                 │ 像素重排 2×2 降采样 (token 数 ÷4) + 时间池化
                 ▼
文本 token ──► embed_tokens ──► PatchMergerV2 (投影器 → 7168)
                 │  media_placeholder_token_id=163605 占位位 masked_scatter
                 ▼
        93 层文本塔（第0层 dense）：
        每 block = 3×(KDA → Stable LatentMoE) + 1×(Gated MLA → Stable LatentMoE)
        （block 末端额外一层 Gated MLA，保证末层为全局注意力）
        每层输出经 AttnRes：伪查询 w 对 embedding + 各 block 输出做加权检索
                 ▼
        输出头（eos=163586；thinking 常开，reasoning_effort low/high/max）
```

## 2. 总体设计

K3 的架构由四段组成，全部可在官方 config 与技术报告中逐字段锚定：

1. **MoonViT-V2 视觉塔**（27 层 ViT，约 401M，RMSNorm、全部线性/注意力投影去 bias，**从零训练**而非 SigLIP 初始化；技术报告 §2.4：从零训练梯度范数更低更稳）；
2. **PatchMergerV2 投影器**（config `mm_projector_type=patchmergerv2`，lightweight MLP 投影到 7168 维）；
3. **混合注意力文本塔**：KDA 混合线性注意力负责长序列高效 token 混合，周期性 Gated MLA 层保留全局交互（3:1）；
4. **Stable LatentMoE**：每注意力层配对一个，在 latent 空间（ℓ=3584）做稀疏通道混合。

关键训练取舍（技术报告 §3）：原生多模态联合训练（从训练起步即 interleaved next-token prediction，无后期对齐阶段）；预训练上下文 8K→64K，四阶段课程扩到 1M；cosine LR（1% linear warmup），weight decay 0.1；RL 覆盖长程 coding / 通用 agent / 推理与知识，多 reasoning-effort 级别，多教师 on-policy 蒸馏合并为统一模型。

## 3. 解码层解剖

一个 block 内 4 个注意力层（3 KDA + 1 Gated MLA），每个注意力层后接一个 Stable LatentMoE 前馈子层（技术报告 Fig.2）：

- `RMSNorm → KDA → 残差；RMSNorm → Stable LatentMoE → 残差`（×3）
- `RMSNorm → Gated MLA → 残差；RMSNorm → Stable LatentMoE → 残差`（×1）
- 跨层：AttnRes 让每个模块选择性检索 embedding 与历史 block 输出（伪查询 w 加权，非均匀残差累加）。

第 0 层为 dense SwiGLU（intermediate 33792；first_k_dense_replace=1，官方 config）。

## 4. 核心机制

### 4.1 KDA（Kimi Delta Attention，混合线性注意力）

KDA 在 delta-rule 递推上叠加通道级遗忘门（技术报告 §2.1.1）：

- 状态递推 `S_t = (I − β_t k_t k_tᵀ)·Diag(α_t)·S_{t−1} + β_t k_t v_tᵀ`，输出 `õ_t = S_tᵀ q_t`；α_t 为逐通道一维留存因子，β_t 控制 delta-rule 写入强度；
- q/k/v 投影：ShortConv → Swish，q/k 再 L2Norm；β = Sigmoid(W_β x)；衰减 logit z 由低秩投影 W_α↑W_α↓ + 头级偏置 b_α^h 生成（细粒度通道级衰减）；
- 分块并行：chunk 间递推、chunk 内并行（UT 变换），相对衰减在 log 空间计算防溢出；
- 输出门：相较 Kimi-Linear 的低秩门，K3 改为**输入相关全秩门** `y_t = W_o[Sigmoid(W_g x_t) ⊙ RMSNorm(õ_t)]`（config use_full_rank_gate=true，gate_lower_bound=-5.0）。

### 4.2 Gated MLA（周期性全局注意力层）

- 沿用 DeepSeek-V2 系 MLA：KV 压缩到低维潜向量 c_t=W_c x_t，缓存潜向量、解码时上投影重建；
- **NoPE**：与 K2/K2.5 不同，K3 全部 MLA 层不加任何位置编码（config mla_use_nope=true）——位置敏感的序列混合由中间 KDA 层承担，MLA 只做无限制全局内容交互；上下文外推时无需重调 RoPE 频率或 YaRN；
- **通道级全秩输出门**：`y_t = W_o[Sigmoid(W_g x_t) ⊙ õ_t]`，W_g 为全秩（与 KDA 门一致），逐 token 调制全局注意力读出的通道；
- 训练时注意力输出保持 FP32（修正 flash attention 舍入偏差，技术报告 §2.1.2）。

### 4.3 AttnRes（Attention Residuals，注意力残差）

- 标准残差把所有历史信息压缩进单一状态 h_l（类似 RNN 的时间瓶颈）；AttnRes 把「注意力」搬到深度维：每层用可学习伪查询 q_l=w_l ∈ R^d 对 embedding（i=0）与历史各层输出做 softmax 加权检索（kernel φ(q,k)=exp(qᵀ·RMSNorm(k))，RMSNorm 防止大范数层主导权重）；
- 全量形式 O(L²d)，L<100 可承受；工程上用 **Block AttnRes**：93 层分 8 块（block size=12，config attn_res_block_size=12），块内标准残差、块间全注意力（含 embedding 作为固定源；技术报告 §2.2）。

### 4.4 Stable LatentMoE

- **LatentMoE**：共享专家（N_s=2，每 token 全参激活）走全宽路径；路由专家在紧凑 latent 空间 ℓ=3584 内运算（z=W↓x，专家作用于 z，聚合后 W↑ 投回 7168），把路由专家扩到 896 而通信/权重流量不随路由数线性增长；
- **稳定化三件套**（技术报告 §2.3）：① routed 聚合后上投影前加 RMSNorm（config latent_moe_use_norm=true）；② **SiTU-GLU** 替代 SwiGLU——门分支 β₁tanh(W_g x/β₁)·Sigmoid(W_g x)、up 分支 β₂tanh(W_u x/β₂)，β₁=4、β₂=25，输出有界 |f|≤100，防极端稀疏下激活爆炸；③ **Quantile Balancing（QB）**：无辅助损失路由（sigmoid 打分 + bias 修正），bias 按「目标负载 q=mk/n 对应的分位数」由单次前推更新（实际用逐专家直方图 + 一次 all-reduce 估计分位数），替代固定步长 bias 更新——896 专家规模下负载均衡的关键；
- 激活规格：每 token 激活 16/896 路由专家 + 2 共享，稀疏度 56（config num_experts_per_token=16，use_grouped_topk=true，topk_method=noaux_tc，routed_scaling_factor=1.0）。

## 5. 注意力布局与混合比例

- 93 层 = 第 0 层 dense + 69 层 KDA + 24 层 Gated MLA（config 列表：kda_layers 69 项、full_attn_layers 24 项；技术报告 §2.1「每 block 3 KDA + 1 Gated MLA」，末端额外一层 Gated MLA 保证末层全局注意力）；
- KDA 头：96 头、head_dim 128（linear_attn_config）；Gated MLA：kv_lora 512 / q_lora 1536、qk_nope 128 / qk_rope 64 / v 128、Q/KV 96 头；
- 无 RoPE、无 YaRN（config 无 rope_scaling 字段；NoPE 位置由 KDA 循环门控隐式编码，技术报告 §3.4）。

## 6. 多模态：MoonViT-V2 视觉塔

### 6.1 视觉塔实参（来源：moonshotai/Kimi-K3/config.json 的 vision_config）

| 字段 | 取值 | 说明 |
|---|---|---|
| patch_size | 14 | 官方 config |
| vt_hidden_size / vt_num_hidden_layers / vt_num_attention_heads | 1024 / 27 / 12 | 官方 config |
| vt_intermediate_size | 4096 | 官方 config |
| mm_hidden_size / text_hidden_size | 1024 / 7168 | 官方 config |
| mm_projector_type | **patchmergerv2** | 官方 config |
| merge_kernel_size | (2, 2) | 官方 config |
| qkv_hidden_size | 1536 | 官方 config |
| activation_func / mlp_type / norm_type | gelu_pytorch_tanh / mlp2 / rmsnorm | 官方 config |
| pos_emb_interpolation_mode | bilinear | 官方 config |
| init_pos_emb_height / width / time | 64 / 64 / 4 | 官方 config |

### 6.2 视觉机制（来源：技术报告 §2.4）

- **从零训练**：不初始化自 SigLIP（对比 MoonViT-3D 的 SigLIP 初始化），直接用 next-token prediction 训练——梯度范数更稳（报告 Fig.6），视觉评测与 SigLIP 基线持平；
- **图像/视频完全共享参数**：注意力分解为帧内空间 + 帧间时间两遍，时间池化沿时间维压缩 token；
- **像素重排（pixel-shuffle）2×2 降采样**：投影前把视觉 token 数再压 4×，使输入 ≤3584×3584 像素在 1M 上下文内可负担；
- 视觉编码器参数约 401M（官方模型卡 Model Summary）。

## 7. Token、权重格式与训练发布

- **词表 163840**（官方 config）；bos=163584、eos=163586、pad=163839、media_placeholder=163605；
- **权重格式**：仅发布 **MXFP4 量化权重 / MXFP8 激活**——量化感知训练（QAT）自 SFT 阶段起（config quantization_config: mxfp4-pack-quantized）；**无 BF16 全精度发布**（官方模型卡 §4）；
- **训练**：Per-Head Muon（按头 Newton-Schulz 正交化）+ K2 权重裁剪 + QB 负载均衡；cosine LR（1% warmup），weight decay 0.1；预训练上下文 8K→64K，四阶段课程扩到 1M；
- **推理/产品**：thinking 常开（reasoning_effort low/high/max，默认 max），preserved thinking history 模式（多轮回传 assistant 完整消息含 reasoning_content）；API 名 `kimi-k3`（输入 $3 / 输出 $15 / 缓存命中 $0.30 每 1M token——来源：官方平台定价，2026-11-03 核验）；
- **License**：Kimi K3 License（非 MIT；MaaS 年收入 >$20M 需单独协议——来源：官方模型卡 §7 + 官方 LICENSE）。

## 8. 演进

- **Kimi K2 / K2-0905 / K2.5 / K2.6 / K2.7-Code（2025-07 ~ 2026-06）**：DeepSeek-V3 系 MLA + 384 专家 MoE 的 1T 级复用与微调（见各独立目录；K2 论文 arXiv:2507.20534）。
- **Kimi K3（2026-07，本目录）**：架构换代——MLA → KDA+Gated MLA 混合（NoPE）、SwiGLU → SiTU-GLU、384+1 专家 → 896+2 专家（latent 3584，稀疏度 56）、固定步长 bias → Quantile Balancing、标准残差 → AttnRes、SigLIP 视觉塔 → 从零训练 MoonViT-V2、256K YaRN → 1M NoPE、INT4/FP8 → MXFP4 QAT；~2.5× scaling efficiency vs K2。
- **同代兄弟**：Kimi-Linear-48B-A3B（KDA 混合线性注意力的前身小模型，`models/kimi_linear/`，KDA 的直接来源）、Kimi-K3.1-Pro（见 `kimi_k3_1_pro` 目录，2026-10-24 官网展示的最新旗舰，公开信息尚少）、K2.8 Preview（2026-09-11 接管 kimi-for-coding；HF 无官方权重仓库，搜索 0 命中）、Kimi-VL-A3B（2025 上代小 VL）。

## 9. 参考文献

1. 远端官方 config：https://hf-mirror.com/moonshotai/Kimi-K3/resolve/main/config.json （留档：本目录 `_tools/_cfgpull/config.json`，拉取 2026-11-03）
2. 官方模型卡/仓库：https://huggingface.co/moonshotai/Kimi-K3 （Model Summary、评测、MXFP4 QAT、Kimi K3 License、preserved thinking、API 入口）
3. 技术报告：arXiv:2607.24653《Kimi K3: Open Frontier Intelligence》（47 页 PDF 留档：`_tools/_cfgpull/k3_tech_report.pdf`；章节锚点：§1 摘要/§2.1 KDA 与 Gated MLA/§2.2 AttnRes/§2.3 Stable LatentMoE/§2.4 MoonViT-V2/§2.5 Per-Head Muon/§3 Pre-Training/Table 1 架构对比）
4. 技术博客：https://www.kimi.com/blog/kimi-k3
5. transformers 源码（本机 E:\transformers 可编辑安装）：
   - 本地无 `models/kimi_k3/`（2026-11-03 Glob 核验）；顶层走仓库 custom code
   - `models/kimi_linear/configuration_kimi_linear.py`、`modeling_kimi_linear.py`、`modular_kimi_linear.py`（text_config 运行时实例化目标）
6. 前代对照：`kimi_k2_6/index.md`、`kimi_k2_5/index.md`（K2.5 技术报告 arXiv:2602.02276）
