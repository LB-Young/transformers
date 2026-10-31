# Gemma 3n 家族架构详解 — E2B / E4B 全尺寸 · MatFormer + 原生音频

- 两个尺寸：E2B（有效≈2B）· E4B（有效≈4B）
- 原生三路：文本 + 图像（MobileNetV5-300M）+ 音频（Conformer×12）
- MatFormer 嵌套 · AltUp · LAuReL(rank64) · PLE · 激活稀疏

本页以**本地检查点 `gemma-3n-E2B-it/config.json`**（权威基准）、ModelScope 下载的 **google/gemma-3n-E4B-it/config.json**，以及本地 **transformers 源码 `modeling_gemma3n.py / configuration_gemma3n.py`** 为唯一依据。Gemma 3n 是 Google 面向端侧（手机/笔记本）的开放权重多模态模型：文本骨干是 **MatFormer 嵌套 Transformer**（AltUp 四输入 + LAuReL 低秩增强残差 + 逐层 PLE 门控 + 前 10 层高斯 top-k 激活稀疏），混合注意力为「4 层滑窗 512 + 1 层全局」交错、末段共享 KV；视觉塔输出 256 个 soft token、音频塔输出至多 188 个 soft token。下文所有数字逐字段标注 config 来源，所有机制标注源码文件:行号。

## 1 家族全景：E2B / E4B 真实 config 对照

> 两尺寸均为 `Gemma3nForConditionalGeneration`。下列全部为 config 原文（`text_config` 子对象）：E2B 来自本地 `E:\models\model\gemma-3n-E2B-it\config.json`，E4B 来自 ModelScope `google/gemma-3n-E4B-it/config.json`（2026-10-22 下载核验）。

### 1.1 文本塔核心超参数（text_config）

| 字段（text_config） | E2B（本地检查点） | E4B（ModelScope config） |
|---|---|---|
| **architectures 入口类** | `Gemma3nForConditionalGeneration` ||
| **hidden_size** | **`2048`** | **`2048`** |
| **num_hidden_layers** | **`30`** | **`35`** |
| num_attention_heads（Q） | `8` | `8` |
| num_key_value_heads（GQA） | `2`（组比 8:2=4） | `2`（组比 8:2=4） |
| head_dim | `256`（8×256=2048） | `256`（8×256=2048） |
| **intermediate_size（逐层数组）** *MatFormer：每层一个 FFN 中间维* | **`[8192]`×30** *逐层数组，本检查点 30 层同值* | **`[16384]`×35** *逐层数组，本检查点 35 层同值* |
| layer_types（full 层位置，1-based） | 第 5,10,15,20,25,30 层 `full_attention`（6 层），余 24 层滑窗 | 第 5,10,15,20,25,30,35 层 `full_attention`（7 层），余 28 层滑窗 |
| sliding_window | `512` | `512` |
| **num_kv_shared_layers** *末段共享 KV 层数* | **`10`** *即第 21–30 层（1-based）共享* | **`15`** *即第 21–35 层（1-based）共享* |
| **activation_sparsity_pattern** | **`[0.95]`×10 + `[0.0]`×20** | **`[0.95]`×10 + `[0.0]`×25** |
| altup_num_inputs（AltUp 路数） | `4` *altup_active_idx=0 · coef_clip=120* | `4` *同上* |
| laurel_rank（LAuReL 瓶颈维） | `64` | `64` |
| hidden_size_per_layer_input（PLE 逐层小维） | `256` *vocab_size_per_layer_input=262144* | `256` *vocab_size_per_layer_input=262144* |
| rope_theta（全局层 base） | `1,000,000` | `1,000,000` |
| rope_local_base_freq（滑窗层 base） | `10,000` | `10,000` |
| max_position_embeddings | `32,768`（32K） | `32,768`（32K） |
| final_logit_softcapping | `30.0` | `30.0` |
| hidden_activation / rms_norm_eps | `gelu_pytorch_tanh`（GeGLU） · `1e-6` ||
| vocab_size | `262,400`（tie_word_embeddings=true，输入/输出共享） ||

> 要点：E2B 与 E4B **文本骨干维度完全相同**（hidden 2048 / 8Q-2KV / head_dim 256 / PLE 256 / laurel 64 / AltUp 4 / 滑窗 512 / 32K）。两尺寸的差异只有三处：**层数 30→35、FFN 中间维 8192→16384、KV 共享 10→15**。「E2B / E4B」是**有效参数**口径（≈2B / ≈4B），原始权重约 5B / 8B（Technical Report）。

### 1.2 视觉塔 / 音频塔（vision_config / audio_config，两尺寸完全一致）

| 字段 | 视觉塔 vision_config | 音频塔 audio_config |
|---|---|---|
| architecture | `mobilenetv5_300m_enc` *timm MobileNetV5-300M 编码器* | `gemma3n_audio` *SSCP 下采样 + Conformer×12* |
| hidden_size | `2048` | `1536` |
| 层数 / 头数 | —（CNN 主干，非 Transformer） | `12 层 / 8 头` |
| 下采样 | do_pooling=false；输出 reshape 16×16 | SSCP 卷积 [128,32] 通道，3×3 stride (2,2)×2；conf_reduction_factor=4 |
| 输入 | pixel_values（归一化图像） | input_features：128 维 mel 频谱 |
| soft token 数 | `256 / 图` *vision_soft_tokens_per_image=256* | `≤188 / 段` *audio_soft_tokens_per_image=188，不足补齐* |
| 内部 attention cap | — | `50.0` *conf_attention_logit_cap（仅 Conformer 内部）* |
| vocab_offset / vocab_size | `262144 / 128` | `262272 / 128` |

## 2 端到端数据流总览（文本 + 图像 + 音频三路）

> 三路输入：文本 `input_ids`、图像 `pixel_values`、音频 `input_features`。视觉/音频特征经各自 embedder（`Gemma3nMultimodalEmbedder`）投影进 2048 维文本空间，再以 `masked_scatter` 填进序列中 `image_token_id=262145 / audio_token_id=262273` 的占位位。主干：嵌入 $\times\sqrt{d}$ → AltUp 四路线堆叠 → N 个解码层 → 四分支求均值 → 最终 RMSNorm → 共享 LM Head → softcap 30。

### 2.1 端到端架构图

![架构图：端到端数据流总览（文本 + 图像 + 音频三路）](./_arch/01_2端到端数据流总览（文本_+_图像_+_音频三路.png)

> 全家族统一主干：嵌入（$\times\sqrt{d}$）→ AltUp 4 路 → 解码层 $\times N$ → 四分支均值 → RMSNorm → 共享 LM Head → softcap 30；E2B/E4B 差异仅在层数、FFN 中间维、KV 共享层数。

## 3 解码层解剖：Gemma3nTextDecoderLayer

> 源码 `modeling_gemma3n.py:1273-1346`。这是 MatFormer 的核心单元：输入是 `[4,B,T,2048]` 的 AltUp 四路线；残差路径叠加 LAuReL 低秩分支，FFN 门控带高斯 top-k 稀疏，末尾用 PLE（逐层小嵌入）做门控注入。

### 解码层展开架构图

![架构图：Gemma3nTextDecoderLayer 展开（LAuReL / PLE / 共享 KV）](./_arch/02_3解码层解剖_Gemma3nTextDecode.png)

### 3.1 残差与门控前向（源码 :1323-1344 直录）

```
predictions        = altup.predict(h)                 # :1306
active_prediction  = predictions[altup_active_idx]        # :1307
z                  = input_layernorm(active_prediction)    # :1309
laurel_output      = z + post_laurel_norm(W2·W1·z)       # :1310  (LAuReL, W1:2048→64)
attn               = post_attn_norm( self_attn(z, ...) ) # :1312-1321
attn_gated         = active_prediction + attn             # :1323
attn_laurel        = (attn_gated + laurel_output) / √2    # :1324
ffn                = post_ffn_norm( mlp(pre_ffn_norm(attn_laurel)) )  # :1326-1328
resid              = attn_laurel + ffn                   # :1329
corrected          = altup.correct(predictions, resid)    # :1330
first              = corrected[active_idx]·correct_scale# :1332-1334
first              = per_layer_proj( act(gate(first)) ⊙ per_layer_input ) # :1337-1342
corrected[1:]     += post_per_layer_norm(first)          # :1344  PLE 注入
return corrected
```

## 4 核心机制专节（本代独有）

### MatFormer 嵌套：intermediate_size 逐层数组

`config.intermediate_size` 是**逐层数组**：每层 MLP 取 `intermediate_size[layer_idx]`（`modeling:955`）。配置类允许标量（自动广播成 N 层，`configuration:138-139`）或显式 N 元列表（长度校验，`configuration:131-137`）。这是 MatFormer「嵌套子模型」的载体：理论上可对不同层配不同 FFN 中间维以抽取嵌套子网络；但发货的 E2B/E4B 检查点内部各层同值——E2B 全 30 层 `8192`，E4B 全 35 层 `16384`。

```
mlp 中间维 = config.intermediate_size[layer_idx]
E2B: [8192]×30   E4B: [16384]×35
```

*元信息：config text_config.intermediate_size；modeling:955；configuration:131-139*

### AltUp：Alternating Updates（altup_num_inputs=4）

每个解码层同时维护 **4 路隐藏态**（`[4,B,T,2048]`）：第 0 路是词嵌入，第 1–3 路由 3 个 `altup_projections`（2048→2048 Linear）生成并做幅值归一化（对齐第 0 路 RMS）。层内 `predict` 用 modality_router(tanh) 预测 4×4 系数矩阵做可学习混合；`correct` 再用真实残差 innovation=activated−prediction 修正 4 路；层末 4 路经 unembed 投影后**求均值**出主干。论文出处见 `modeling:989-997` 注释（NeurIPS 2023）。

```
stack=[h0, P1h0, P2h0, P3h0]  # 幅值对齐, :1711-1721
predictions = altup.predict(stack)      # :1015-1041
corrected   = altup.correct(predict, resid)  # :1043-1072
out = mean_i( unembed_i(corrected_i) )  # :1747-1759
```

*元信息：config altup_num_inputs=4 / active_idx=0 / coef_clip=120；modeling:988-1084,1631-1637,1711-1759*

### PLE：Per-Layer Embeddings（逐层嵌入）

一张大查表 `embed_tokens_per_layer = Embedding(262144, N×256)`（$\times\sqrt{256}$），reshape 成每层一条 256 维向量；另有一条从主嵌入投影而来的分支，二者相加后乘 $1/\sqrt{2}$。层内把主隐态经 gate 压到 256 维、与该层查表向量逐元素相乘，再投回 2048 维注入。可整体预计算并卸载到 CPU。

```
ple_lookup = Emb(262144, N·256) → [B,T,N,256]   # :1616-1621,1767-1772
ple = (proj_norm(proj(h0)) + ple_lookup)·(1/√2)  # :1774-1799
z = proj256→2048( act(gate2048→256 h) ⊙ ple_l )  # :1337-1342
```

*元信息：config hidden_size_per_layer_input=256 / vocab_size_per_layer_input=262144；modeling:1616-1621,1767-1799,1336-1344*

### 激活稀疏：高斯 top-k（前 10 层 cutoff 0.95）

仅作用于 GeGLU 的 `gate_proj` 输出：按正态分布分位点 $\Phi^{-1}(0.95)$ 求阈值，低于阈值的门控值经 $\mathrm{ReLU}(x-\mathrm{cutoff})$ 直接置零——即中间维约 95% 维度在每 token 上被关掉。E2B/E4B 均为**前 10 层 sparsity=0.95、其后稠密（0.0）**（配置默认规则 `configuration:146-150`）。注意 up_proj 不稀疏，稀疏的是门控支路。

$$
\mathrm{cutoff}=\mathrm{mean}(g)+\mathrm{std}(g)\cdot\Phi^{-1}(0.95),\qquad g\leftarrow\mathrm{ReLU}(g-\mathrm{cutoff})
$$

```
cutoff = mean(g) + std(g)·Normal(0,1).icdf(0.95)   # :982-984
g ← ReLU(g − cutoff)                                # :985
# E2B: [0.95]×10+[0.0]×20 ; E4B: [0.95]×10+[0.0]×25
```

*元信息：config activation_sparsity_pattern；modeling:962-985；configuration:146-156*

### LAuReL：Learned Augmented Residual Layer

在注意力子层前，对归一化输入额外走一条低秩瓶颈增强残差：`laurel_rank=64`，即 2048→64→2048，再过 post RMSNorm，最后与主残差按 $1/\sqrt{2}$ 缩放合并。每层仅 $2\times2048\times64\approx0.26\mathrm{M}$ 参数，却显著改变信息流。

```
laurel_output = z + post_norm( W2·W1·z )   # :943-947
attn_laurel = (active+attn + laurel_output)/√2  # :1324
```

*元信息：config laurel_rank=64；modeling:932-947,1310,1324*

### 共享 KV（末段层）

末段层不再拥有 `k_proj/v_proj/k_norm/v_norm`（权重直接不建，`_keys_to_ignore_on_load_unexpected` 跳过加载），而是复用进入共享区前最后一个**同类型**层（滑窗↔滑窗、全局↔全局）算好的 K/V 状态。E2B 共享后 10 层（第 21–30 层），E4B 共享后 15 层（第 21–35 层）。

```
first_shared = num_layers − num_kv_shared_layers  # :1177
E2B: 30−10=20(0-based) → 1-based 21..30
E4B: 35−15=20(0-based) → 1-based 21..35
k,v = shared_kv_states[same_type_last_non_shared]  # :1232-1236
```

*元信息：config num_kv_shared_layers=10/15；modeling:1177-1189,1232-1250,1642-1648*

### 双 RoPE 基频（按层类型）

位置编码按层类型分别缓存逆频率：全局层用 `rope_theta=1e6`，滑窗层用 `rope_local_base_freq=1e4`。`configuration:182-200` 把它们映射进 `rope_parameters[layer_type]["rope_theta"]`。

```
inv_freq(full)     = 1/(1e6^(2i/256))   # θ=1e6
inv_freq(sliding)  = 1/(1e4^(2i/256))   # θ=1e4
# modeling:1529-1588；configuration:193-200
```

*元信息：config rope_theta=1e6 / rope_local_base_freq=1e4*

### 输出端 logit soft-capping（=30）

LM Head 与主词嵌入共享权重（tied）。输出 logits 先除以 30、过 tanh、再乘回 30，压到 $\pm30$ 抑制极端 logit。**注意**：文本注意力内部**没有** softcap（仅音频 Conformer 内部有 logit cap=50）。

$$
\mathrm{softcap}(x)=c\cdot\tanh(x/c),\qquad c=30
$$

```
logits ← 30 · tanh( logits / 30 )   # :2297-2300
```

*元信息：config final_logit_softcapping=30.0；modeling:2296-2300；tied :2195*

## 5 注意力机制：GQA + 每 5 层 1 full + 共享 KV

> Gemma 3n 注意力（`modeling:1162-1270`）：Q/K/V 各经 head-wise RMSNorm（q_norm/k_norm/v_norm），`scaling=1.0`（取代 Gemma 2/3 的显式 query 缩放）；Q 8 头、K/V 2 头（GQA 组比 4:1），head_dim=256。

| 层模式 | E2B（30 层） | E4B（35 层） |
|---|---|---|
| sliding_attention 层 | 24 层，窗口 512，RoPE θ=1e4 | 28 层，窗口 512，RoPE θ=1e4 |
| **full_attention 层（1-based）** | **第 5,10,15,20,25,30（6 层），全局因果，θ=1e6** | **第 5,10,15,20,25,30,35（7 层），全局因果，θ=1e6** |
| KV 共享层 | 第 21–30 层（10 层），无 k/v 权重 | 第 21–35 层（15 层），无 k/v 权重 |
| 投影维度 | q: 2048→8×256=2048；k/v: 2048→2×256=512；o: 2048→2048（共享层仅 q/o 有） ||

> 层类型由 config `layer_types` 显式给出；缺省时配置类规则为「每 5 层第 5 层 full」（`configuration:141-144`）。滑窗层用 `create_sliding_window_causal_mask`、全局层用普通因果掩码（`modeling:1699-1702`）。

## 6 多模态：MobileNetV5 视觉塔 + Conformer 音频塔

### 6.1 视觉塔（两尺寸一致）

> 与 Gemma 3 使用的 ViT 系 SigLIP/SoViT 不同，Gemma 3n 面向端侧改用**轻量 CNN 主干 timm MobileNetV5-300M**（`vision_config.architecture=mobilenetv5_300m_enc`，hidden=2048）。输出特征图 `[B,2048,16,16]` reshape 为 `[B,256,2048]`，乘 $\sqrt{2048}$ 后经 embed_vision（RMSNorm→Linear 2048→2048→RMSNorm）注入 256 个 image_token 占位位（`modeling:1956-1974`）。

| 字段（vision_config） | Gemma 3n（E2B/E4B） | Gemma 3（对照口径） |
|---|---|---|
| 视觉编码器 | MobileNetV5-300M（CNN，约 300M） | SigLIP/SoViT（ViT 系） |
| hidden_size | `2048` | — |
| soft token / 图 | `256` | 每图 256–1024 patch（分辨率相关） |
| do_pooling | `false`（16×16 网格展平） | — |

### 6.2 音频塔（两尺寸一致，端侧原生）

> Gemma 3n 相比 Gemma 3 新增**端侧原生音频**：128 维 mel 频谱先经 2 层 SSCP 卷积（通道 [128,32]，3×3 stride (2,2)）下采样，过 12 个 Conformer 块（hidden=1536，8 头，块内 attention logit cap=50，chunk=12/左上下文 13，残差权重 0.5），时间维再 ÷4（`conf_reduction_factor=4`），得到至多 188 帧 ×1536；经 embed_audio（RMSNorm→Linear 1536→2048→RMSNorm）升维，不足 188 用最后一条硬嵌入补齐（`modeling:1448-1526, 2167`）。

## 7 特殊 Token 与词表布局（vocab 262400）

| token 字段 | ID | 含义 |
|---|---|---|
| boi_token_id | `255999` | 图像开始（begin of image） |
| boa_token_id | `256000` | 音频开始（begin of audio） |
| eoi_token_id | `262144` | 图像结束 = 视觉硬嵌入区间起点（vocab_offset=262144） |
| image_token_id | `262145` | 图像特征占位，每图 256 个，运行时被视觉特征替换 |
| eoa_token_id | `262272` | 音频结束 = 音频硬嵌入区间起点（audio_config.vocab_offset=262272） |
| audio_token_id | `262273` | 音频特征占位，每段最长 188 个 |
| eos_token_id | `[1, 106]` | 结束符列表（106 为 IT 对话结束） |

> 词表区间：普通文本 [0, 262144)；视觉硬嵌入 [262144, 262272)（128 条）；音频硬嵌入 [262272, 262400)（128 条）。视觉/音频软特征经 masked_scatter 直接填入 hidden 序列，不经过主 embedding 查表（`modeling:1976-2012`）。

## 8 演进视角：Gemma 3 → Gemma 3n

| 维度 | Gemma 3（官方口径） | Gemma 3n（E2B/E4B，本页 config 核对） |
|---|---|---|
| 定位 | 云端/通用，1B–27B | 端侧（手机/笔记本），E2B/E4B 有效≈2B/≈4B |
| 模态 | 文本 + 图像（SigLIP/SoViT） | **文本 + 图像(MobileNetV5) + 音频(Conformer)**，原生三路 |
| 上下文 | 128K | 32K（max_position_embeddings=32768） |
| FFN | 标准 dense GeGLU | MatFormer 逐层中间维数组 + 前 10 层高斯 top-k 激活稀疏 |
| 嵌套/残差技巧 | 无 | **AltUp 四路线 · LAuReL(rank64) · PLE 逐层嵌入** |
| KV 共享 | 无 | 末段共享（E2B 后10层 / E4B 后15层） |
| 混合注意力 | 滑窗 512 + 每 6 层 1 full | 滑窗 512 + 每 5 层 1 full；双 RoPE θ=1e6/1e4 |
| 视觉编码器 | ViT 系 SigLIP/SoViT | 轻量 MobileNetV5-300M（CNN），256 soft token/图 |
| vocab | 262144 | 262400（+256 多模态特殊 token） |

> 主线：Gemma 3n 把「端侧高效」做到极致——用 MatFormer 嵌套 + AltUp/LAuReL/PLE + 激活稀疏 + KV 共享，在约 2B/4B 有效参数下获得接近更大模型的质量，并首次原生并入音频；代价是上下文从 128K 回收到 32K。Gemma 3 侧数字为官方口径（128K、文本+图像、SigLIP），Gemma 3n 侧均为本地 config + 源码核对。

## 9 来源

1. 本地检查点 config.json（E2B 权威基准）
   `E:\models\model\gemma-3n-E2B-it\config.json（text_config / vision_config / audio_config 逐字段核对）`
2. E4B config.json（ModelScope 下载核验，2026-10-22）
   `google/gemma-3n-E4B-it/config.json：hidden 2048 / 35 层 / intermediate 16384×35 / kv_shared 15 / 稀疏 [0.95]×10+[0.0]×25`
3. transformers 源码 modeling_gemma3n.py（本地 E:\transformers）
   `RMSNorm:130；ScaledEmbedding:918；LAuReL:932-947；MLP+高斯top-k:950-985；AltUp:988-1084；TextAttention:1162-1270；DecoderLayer:1273-1346；Rotary:1529-1588；TextModel:1592-1799；MultimodalEmbedder:1883-1930；Model/图像特征:1939-1974；ForConditionalGeneration+softcap:2194-2300`
4. transformers 源码 configuration_gemma3n.py
   `intermediate 逐层校验:131-139；默认层型每5层full:141-144；默认稀疏前10层0.95:146-150；双 RoPE 映射:182-200`
5. Gemma 3n Technical Report（arXiv 2505.05146）
   `https://arxiv.org/abs/2505.05146（E2B/E4B 有效参数、MatFormer 嵌套、PLE、端侧效率）`
6. Gemma 3n 模型卡片（Google AI for Developers）
   `https://ai.google.dev/gemma/docs/core/model_card_3n（参数量与部署规格）`

> 数据口径：所有数字来自上述来源并在页内标注；E2B/E4B 核心差异（层数 30→35、FFN 8192→16384、KV 共享 10→15）逐项对到 config 原文。访问与核验日期：2026-10-22。

---

*页脚：Gemma 3n 家族架构详解（E2B / E4B）· 数据来源：本地 config + ModelScope E4B config + transformers 源码 + arXiv 2505.05146*
