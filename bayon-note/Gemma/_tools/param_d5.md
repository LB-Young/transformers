# Dimension 5: table structure
source: E:\transformers\bayon-note\Gemma\gemma-4-26B-A4B-it-param-breakdown.docx
location_base: /word/document.xml/w:body/
syntax: [T<n>] starts a table, [R<n>] a row, and [C<n>] a cell; listed defaults apply when omitted; null means unset/inherited.
table_defaults: grid=[{"w":"3173"},{"w":"3173"},{"w":"3173"}], nesting_level=0, part="word/document.xml", section_index=1, style="LightGrid-Accent1", width={"type":"auto","w":"0"}
row_defaults: none
cell_defaults: paragraphs=1, valign="center"
definitions: @v1 cell.width={"type":"dxa","w":"1701"}; @v2 cell.shading={"val":"clear","fill":"1F4E79"}; @v3 cell.width={"type":"dxa","w":"4819"}; @v4 cell.width={"type":"dxa","w":"2551"}; @v5 cell.width={"type":"dxa","w":"1984"}; @v6 cell.width={"type":"dxa","w":"4252"}; @v7 cell.width={"type":"dxa","w":"2268"}; @v8 cell.width={"type":"dxa","w":"5669"}; @v9 cell.width={"type":"dxa","w":"2835"}
text_policy: full text is emitted without truncation.
omitted: record_type=table; repeated defaults; table text because it is the ordered concatenation of cell text; raw table XML.
normalized: loc is relative to location_base; @v references definitions; border and margin XML are losslessly compacted by tag and attributes.

[T1 loc=5]

[R1]
[C1 width=@v1 shading=@v2]
符号
[C2 width=@v3 shading=@v2]
含义
[C3 width=@v4 shading=@v2]
取值

[R2]
[C1 width=@v1]
H
[C2 width=@v3]
hidden_size
[C3 width=@v4]
2816

[R3]
[C1 width=@v1]
L
[C2 width=@v3]
num_hidden_layers
[C3 width=@v4]
30

[R4]
[C1 width=@v1]
V
[C2 width=@v3]
vocab_size
[C3 width=@v4]
262,144

[R5]
[C1 width=@v1]
n_head
[C2 width=@v3]
num_attention_heads
[C3 width=@v4]
16

[R6]
[C1 width=@v1]
n_kv (sliding)
[C2 width=@v3]
num_key_value_heads（sliding 层）
[C3 width=@v4]
8

[R7]
[C1 width=@v1]
n_kv (full)
[C2 width=@v3]
num_global_key_value_heads（full 层）
[C3 width=@v4]
2

[R8]
[C1 width=@v1]
d_head (sliding)
[C2 width=@v3]
head_dim（sliding 层）
[C3 width=@v4]
256

[R9]
[C1 width=@v1]
d_head (full)
[C2 width=@v3]
global_head_dim（full 层）
[C3 width=@v4]
512

[R10]
[C1 width=@v1]
I_d
[C2 width=@v3]
intermediate_size（shared/dense MLP）
[C3 width=@v4]
2112

[R11]
[C1 width=@v1]
I_e
[C2 width=@v3]
moe_intermediate_size（每个 routed expert）
[C3 width=@v4]
704

[R12]
[C1 width=@v1]
N_e
[C2 width=@v3]
num_experts
[C3 width=@v4]
128

[R13]
[C1 width=@v1]
K
[C2 width=@v3]
top_k_experts
[C3 width=@v4]
8

[R14]
[C1 width=@v1]
L_s / L_f
[C2 width=@v3]
sliding / full 层数
[C3 width=@v4]
25 / 5

[R15]
[C1 width=@v1]
—
[C2 width=@v3]
tie_word_embeddings
[C3 width=@v4]
true

[R16]
[C1 width=@v1]
—
[C2 width=@v3]
hidden_size_per_layer_input（PLE）
[C3 width=@v4]
0（关闭）

[R17]
[C1 width=@v1]
—
[C2 width=@v3]
num_kv_shared_layers
[C3 width=@v4]
0

[R18]
[C1 width=@v1]
—
[C2 width=@v3]
attention_k_eq_v
[C3 width=@v4]
true（full 层 K=V）

[T2 loc=22]

[R1]
[C1 width=@v5 shading=@v2]
算子
[C2 width=@v6 shading=@v2]
形状
[C3 width=@v7 shading=@v2]
参数量

[R2]
[C1 width=@v5]
q_proj
[C2 width=@v6]
2816 × (16·256) = 2816 × 4096
[C3 width=@v7]
11,534,336

[R3]
[C1 width=@v5]
k_proj
[C2 width=@v6]
2816 × (8·256) = 2816 × 2048
[C3 width=@v7]
5,767,168

[R4]
[C1 width=@v5]
v_proj
[C2 width=@v6]
2816 × (8·256) = 2816 × 2048
[C3 width=@v7]
5,767,168

[R5]
[C1 width=@v5]
o_proj
[C2 width=@v6]
4096 × 2816
[C3 width=@v7]
11,534,336

[R6]
[C1 width=@v5]
q_norm / k_norm
[C2 width=@v6]
各 d_h=256
[C3 width=@v7]
512

[R7]
[C1 width=@v5]
v_norm
[C2 width=@v6]
with_scale=False
[C3 width=@v7]
0

[T3 loc=26]

[R1]
[C1 width=@v5 shading=@v2]
算子
[C2 width=@v6 shading=@v2]
形状
[C3 width=@v7 shading=@v2]
参数量

[R2]
[C1 width=@v5]
q_proj
[C2 width=@v6]
2816 × (16·512) = 2816 × 8192
[C3 width=@v7]
23,068,672

[R3]
[C1 width=@v5]
k_proj
[C2 width=@v6]
2816 × (2·512) = 2816 × 1024
[C3 width=@v7]
2,883,584

[R4]
[C1 width=@v5]
v_proj
[C2 width=@v6]
无（v = k）
[C3 width=@v7]
0

[R5]
[C1 width=@v5]
o_proj
[C2 width=@v6]
8192 × 2816
[C3 width=@v7]
23,068,672

[R6]
[C1 width=@v5]
q_norm / k_norm
[C2 width=@v6]
各 512
[C3 width=@v7]
1,024

[T4 loc=48]

[R1]
[C1 width=@v6 shading=@v2]
模块
[C2 width=@v4 shading=@v2]
参数量
[C3 width=@v1 shading=@v2]
占比

[R2]
[C1 width=@v6]
Token Embedding（共享 lm_head）
[C2 width=@v4]
0.738 B
[C3 width=@v1]
2.9%

[R3]
[C1 width=@v6]
Attention（25 sliding + 5 full）
[C2 width=@v4]
1.110 B
[C3 width=@v1]
4.3%

[R4]
[C1 width=@v6]
Shared Dense MLP（30 层）
[C2 width=@v4]
0.535 B
[C3 width=@v1]
2.1%

[R5]
[C1 width=@v6]
Router（30 层）
[C2 width=@v4]
0.011 B
[C3 width=@v1]
0.04%

[R6]
[C1 width=@v6]
Routed Experts（30 × 128）
[C2 width=@v4]
22.838 B
[C3 width=@v1]
88.6%

[R7]
[C1 width=@v6]
各种 RMSNorm
[C2 width=@v4]
~0.001 B
[C3 width=@v1]
~0%

[R8]
[C1 width=@v6]
Text 小计
[C2 width=@v4]
≈ 25.23 B
[C3 width=@v1]
97.9%

[R9]
[C1 width=@v6]
Vision Encoder
[C2 width=@v4]
≈ 0.56 B
[C3 width=@v1]
2.1%

[R10]
[C1 width=@v6]
总计
[C2 width=@v4]
≈ 25.79 B ≈ 26 B
[C3 width=@v1]
100%

[T5 loc=59 grid=[{"w":"4759"},{"w":"4759"}]]

[R1]
[C1 width=@v8 shading=@v2]
项
[C2 width=@v9 shading=@v2]
参数量

[R2]
[C1 width=@v8]
Embedding + Attention + Dense MLP + Router（恒激活）
[C2 width=@v9]
2.39 B

[R3]
[C1 width=@v8]
8 / 128 个 Experts 被激活
[C2 width=@v9]
1.43 B

[R4]
[C1 width=@v8]
每 token 实际激活
[C2 width=@v9]
≈ 3.82 B ≈ 4 B
