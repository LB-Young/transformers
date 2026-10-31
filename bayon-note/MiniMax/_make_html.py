# -*- coding: utf-8 -*-
"""Generate self-contained index.html for each MiniMax model dir."""
import os, json

ROOT = r"E:\transformers\bayon-note\MiniMax"

CSS = open(r"E:\transformers\bayon-note\Gemma\gemma-3\index.html", encoding="utf-8").read()
CSS = CSS[CSS.index("<style>"):CSS.index("</style>")+8]  # exact <style>...</style> block
# Inline local echarts (CDN unreliable in mainland); per task spec fallback
_ECH = open(r"E:\transformers\bayon-note\Qwen\qwen3\_tools\echarts.min.js", encoding="utf-8").read()
ECH_SCRIPT = "<script>\n" + _ECH + "\n</script>"

def page(model_dir, title, brand, badges, lead, sections_html, chart_id, chart_option, sources, mm_nav=False):
    mm_cls = " mm-link" if mm_nav else ""
    nav = f'''<nav class="top"><div class="wrap">
  <div class="brand"><svg viewBox="0 0 24 24" width="20" height="20"><rect x="2" y="2" width="20" height="20" rx="5" fill="#1d4ed8"/><text x="12" y="16" font-size="11" font-weight="bold" fill="#fff" text-anchor="middle" font-family="Arial">M</text></svg>{brand}</div>
  <a class="link" href="#family">家族全景</a>
  <a class="link" href="#overview">端到端</a>
  <a class="link" href="#layer">解码层</a>
  <a class="link" href="#mech">核心机制</a>
  <a class="link" href="#attn">注意力</a>
  {f'<a class="link mm-link" href="#mm">多模态</a>' if mm_nav else ''}
  <a class="link" href="#tokens">Token</a>
  <a class="link" href="#evo">演进</a>
  <a class="link" href="#sources">来源</a>
</div></nav>'''
    badge_html = "".join(f'<span class="badge{" mm" if i==1 and mm_nav else ""}">{b}</span>' for i,b in enumerate(badges))
    src_items = "".join(f"<li>{t}<span class='u'>{u}</span></li>" for t,u in sources)
    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect x='2' y='2' width='28' height='28' rx='7' fill='%231d4ed8'/%3E%3Ctext x='16' y='22' font-size='15' font-family='Arial' font-weight='bold' fill='%23fff' text-anchor='middle'%3EM%3C/text%3E%3C/svg%3E">
{CSS}
{ECH_SCRIPT}
</head>
<body>
{nav}
<div class="wrap">
<header class="hero">
  <h1>{title}</h1>
  <div class="badges">{badge_html}</div>
  <p class="lead">{lead}</p>
</header>
{sections_html}
<section id="sources">
  <h2><span class="secno">{'9' if mm_nav else '8'}</span>参考文献与依据</h2>
  <ol class="src">{src_items}</ol>
</section>
<footer>MiniMax 系列架构笔记 · 依据：远端官方 config（hf-mirror）+ transformers 源码（E:\\transformers）+ 官方论文/model card · 数字逐条标源</footer>
</div>
<div id="tip"></div>
<script>
var chart = echarts.init(document.getElementById('{chart_id}'));
chart.setOption({json.dumps(chart_option, ensure_ascii=False)});
window.addEventListener('resize', function(){{ chart.resize(); }});
</script>
</body></html>'''
    out = os.path.join(ROOT, model_dir, "index.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print("html:", out)

# ---------- shared table helper ----------
def cfg_table(rows, widths="18%,30%,52%"):
    tr = "".join(f"<tr><td class='k'>{a}</td><td>{b}</td><td class='note'>{c}</td></tr>" for a,b,c in rows)
    return f'''<div class="table-scroll"><table class="cmp"><colgroup><col style="width:{widths.split(',')[0]}"><col style="width:{widths.split(',')[1]}"><col style="width:{widths.split(',')[2]}"></colgroup>
<thead><tr><th>维度</th><th>官方取值</th><th>来源</th></tr></thead><tbody>{tr}</tbody></table></div>'''

def img(src, cap):
    return f'''<div class="diagram-shell"><img src="_arch/{src}" alt="{cap}" style="width:100%;max-width:1100px;display:block;margin:0 auto;"></div><p class="note" style="margin-top:6px">{cap}</p>'''

# ================= Text-01 =================
t01_rows = [
 ("总参 / 激活","456B 总参 / 约 45.9B（≈A46B）每 token","论文 arXiv:2501.08313 摘要"),
 ("层数 / hidden","80 层 / 6144","远端 config"),
 ("注意力头 / KV / head_dim","64 / 8 / 128（GQA 8:1）","远端 config"),
 ("专家 / top-k / 共享专家","32 / 2 / 无（shared=0）","远端 config"),
 ("专家中间维","9216","远端 config + 源码"),
 ("注意力配比","70 Lightning + 10 全 Softmax（每 8 层 1 个）","attn_type_list 逐条对齐"),
 ("RoPE","部分旋转 rotary_dim=64，θ=1e7","远端 config"),
 ("门控残差","α=3.5566，β=1.0","远端 config"),
 ("词表 / 上下文","200064 / 训练1M 外推4M","远端 config + 论文"),
]
t01_sec = f'''
<section id="family"><h2><span class="secno">1</span>家族全景</h2>
<p class="note">MiniMax-Text-01 是 2025-01 开源的 456B MoE 基座，核心是 Lightning 线性注意力:全 Softmax = 7:1 的混合架构。「456B-A46B」= 总参 456B、每 token 激活约 45.9B。</p>
{cfg_table(t01_rows)}
<div id="chart1" style="width:100%;height:340px;margin-top:16px"></div></section>
<section id="overview"><h2><span class="secno">2</span>端到端结构</h2>{img("arch_overview.png","图：Text-01 端到端——词嵌入→80 层堆叠（70 闪电+10 全注意力）→RMSNorm→LM Head")}</section>
<section id="layer"><h2><span class="secno">3</span>一层解码器</h2>
<p>源码 <span class="num">MiniMaxDecoderLayer.forward</span>：两次门控残差。</p>
<div class="formula">x = norm(x);  x = self_attn(x);  x = residual*alpha + x*beta   <span class="cm"># α≈3.5566</span>
x = norm(x);  x = moe(x);       x = residual*alpha + x*beta</div></section>
<section id="mech"><h2><span class="secno">4</span>核心机制：Lightning 块递归线性注意力</h2>
{img("arch_mech.png","图：块内因果 + 块间固定状态递归 + 逐头指数衰减")}
<div class="formula">S &lt;- S*gamma_block + (K*key_decay)^T * V      <span class="cm"># 块间状态，固定 B×H×d×d</span>
out = (Q*query_decay) @ S                       <span class="cm"># 读历史摘要</span>
gamma_block = exp(-slope_rate * block_size)     <span class="cm"># block_size=256</span></div></section>
<section id="attn"><h2><span class="secno">5</span>注意力：7:1 混合</h2>
<p>70 层 Lightning（无 RoPE）+ 10 层全 Softmax GQA（每 8 层 1 个，部分 RoPE 64/128）。</p></section>
<section id="tokens"><h2><span class="secno">6</span>Token 与上下文</h2><p>词表 200064，无滑动窗口；训练 1M、推理外推 4M。</p></section>
<section id="evo"><h2><span class="secno">7</span>演进坐标</h2><p>Text-01(2025-01) → VL-01 加视觉塔 → M1 同构权重+RL推理 → M2 改全Softmax。</p></section>
'''
t01_chart = {
 "tooltip":{"trigger":"axis"},
 "legend":{"data":["层数"]},
 "grid":{"left":50,"right":20,"top":40,"bottom":30},
 "xAxis":{"type":"category","data":["Lightning 线性注意力层","全 Softmax 注意力层"]},
 "yAxis":{"type":"value","name":"层数"},
 "series":[{"type":"bar","data":[{"value":70,"itemStyle":{"color":"#1d4ed8"}},{"value":10,"itemStyle":{"color":"#b45309"}}],"label":{"show":True,"position":"top"}}]
}
page("minimax_text_01","MiniMax-Text-01 家族架构详解",
 "MiniMax Text-01",
 ["456B 总参 / 45.9B 激活（A46B）","80 层 · 32 专家 top-2","Lightning : Softmax = 7:1","部分 RoPE 64/128 · θ=1e7","训练 1M / 外推 4M"],
 "本页以<b>远端官方 config</b>（HF MiniMaxAI/MiniMax-Text-01）+ <b>transformers 源码 models/minimax</b> 为依据。Lightning 块递归线性注意力 + MoE 是核心。",
 t01_sec, "chart1", t01_chart,
 [("论文：MiniMax-01: Scaling Foundation Models with Lightning Attention","https://arxiv.org/abs/2501.08313"),
  ("远端官方 config","https://huggingface.co/MiniMaxAI/MiniMax-Text-01"),
  ("transformers 原生转换仓","https://huggingface.co/MiniMaxAI/MiniMax-Text-01-hf"),
  ("源码：models/minimax/configuration_minimax.py · modeling_minimax.py","E:\\transformers\\src\\transformers\\models\\minimax")])

# ================= VL-01 =================
vl_rows = [
 ("文本塔","= Text-01（80层·7:1闪电·32专家top2）","远端 text_config"),
 ("文本塔上下文","8192（按图像块切分）","text_config.max_position=8192"),
 ("视觉编码器","CLIP ViT · patch14 · 336px","vision_config"),
 ("视觉塔 hidden/层/头","1024 / 24 / 16","vision_config"),
 ("投影维","6144（对齐文本塔）","projection_dim"),
 ("每瓦 patch 数","24×24 = 576","patch14@336"),
 ("动态分辨率","grid pinpoints 336~2016","image_grid_pinpoints"),
 ("<image> id","200025","config"),
 ("续训","512B 视觉-语言 token","论文"),
]
vl_sec = f'''
<section id="family"><h2><span class="secno">1</span>家族全景</h2>
<p class="note">MiniMax-VL-01 = Text-01 文本塔 + CLIP 视觉塔，512B 视觉-语言 token 续训。任务清单「Visual-01」即官方仓 MiniMax-VL-01。</p>
{cfg_table(vl_rows)}
<div id="chart1" style="width:100%;height:340px;margin-top:16px"></div></section>
<section id="overview"><h2><span class="secno">2</span>端到端</h2>{img("arch_overview.png","图：文本/图像双流汇入同一 80 层文本塔")}</section>
<section id="layer"><h2><span class="secno">3</span>解码层</h2><p>文本塔每层与 Text-01 完全一致（门控残差 + 32 专家 top-2）。</p></section>
<section id="mech"><h2><span class="secno">4</span>核心机制</h2>{img("arch_mech.png","图：文本塔 7:1 闪电 + MoE（继承 Text-01）")}</section>
<section id="attn"><h2><span class="secno">5</span>注意力</h2><p>70 Lightning + 10 全 Softmax，RoPE 仅全注意力层。</p></section>
<section id="mm" class="mm"><h2 class="mm"><span class="secno">6</span>多模态结构</h2>
{img("arch_vision.png","图：视觉塔 token 流——336 瓦片→patch14→576 patch→CLIP→投影→拼入序列")}
<p class="note">grid pinpoints 动态切块：原图按最近的 336/672/…/2016 网格切瓦片，每瓦独立编码后展平拼回。</p></section>
<section id="tokens"><h2><span class="secno">7</span>Token</h2><p>词表 200064，新增 &lt;image&gt;=200025；文本块上下文 8192。</p></section>
<section id="evo"><h2><span class="secno">8</span>演进坐标</h2><p>Text-01 → VL-01（首开开源 VLM）→ M1/M2/M3 多模态分化。</p></section>
'''
vl_chart = {
 "tooltip":{"trigger":"axis"},
 "grid":{"left":60,"right":20,"top":30,"bottom":40},
 "xAxis":{"type":"category","data":["336(1瓦)","672","1008","1344","1680","2016(最大)"],"axisLabel":{"interval":0}},
 "yAxis":{"type":"value","name":"视觉 token/图 (≈576×瓦数)"},
 "series":[{"type":"line","data":[576,2304,5184,9216,13824,19353],"areaStyle":{},"itemStyle":{"color":"#0e7490"}}]
}
page("minimax_visual_01","MiniMax-Visual-01（VL-01）家族架构详解",
 "MiniMax VL-01",
 ["Text-01 文本塔 + CLIP 视觉塔","patch14 · 336px · 24层","动态分辨率 grid pinpoints","投影 1024→6144","512B 视觉-语言 token 续训"],
 "以<b>远端官方 config</b>（MiniMaxAI/MiniMax-VL-01）+ 论文为依据；本地无独立模块，按「官方实现+config」双来源。",
 vl_sec, "chart1", vl_chart,
 [("论文：MiniMax-01（含 VL-01）","https://arxiv.org/abs/2501.08313"),
  ("远端官方 config","https://huggingface.co/MiniMaxAI/MiniMax-VL-01")], mm_nav=True)

# ================= M1 =================
m1_rows = [
 ("总参 / 激活","456B / 约 45.9B（与 Text-01 相同）","论文 arXiv:2506.13585"),
 ("层数/hidden/头","80 / 6144 / 64Q·8KV","远端 config"),
 ("专家/top-k","32 / 2（与 Text-01 逐字段一致）","远端 config"),
 ("全注意力层位置","每 8 层 1 个（同 Text-01）","attn_type_list 对齐"),
 ("原生上下文","1,000,000（1M）","论文"),
 ("思维预算","40K / 80K 两版","论文"),
 ("差异点","RL 后训练（CISPO），非改架构","论文"),
]
m1_sec = f'''
<section id="family"><h2><span class="secno">1</span>家族全景</h2>
<p class="note">M1 与 Text-01 <b>逐字段同构</b>——网络结构零改动，差异在大规模 RL 训练做长链推理。</p>
{cfg_table(m1_rows)}
<div id="chart1" style="width:100%;height:340px;margin-top:16px"></div></section>
<section id="overview"><h2><span class="secno">2</span>端到端</h2>{img("arch_overview.png","图：1M 输入 → 同 Text-01 堆叠 → 长思维链输出")}</section>
<section id="layer"><h2><span class="secno">3</span>解码层</h2><p>与 Text-01 完全相同。</p></section>
<section id="mech"><h2><span class="secno">4</span>核心机制：测试时算力近线性扩展</h2>
{img("arch_mech.png","图：长输入被 Lightning 压进固定状态，长思维链生成省算力")}
<p>论文口径：100K 生成长度 FLOPs 约为 DeepSeek-R1 的 25%；CISPO 裁剪重要性采样权重。</p></section>
<section id="attn"><h2><span class="secno">5</span>注意力</h2><p>70 Lightning + 10 全 Softmax，部分 RoPE 64/128。</p></section>
<section id="tokens"><h2><span class="secno">6</span>Token</h2><p>原生 1M 上下文；40K/80K 思维预算。</p></section>
<section id="evo"><h2><span class="secno">7</span>演进坐标</h2><p>Text-01 基座 → M1 同构+RL → M2 路线分叉。</p></section>
'''
m1_chart = {
 "tooltip":{"trigger":"axis"},
 "grid":{"left":60,"right":20,"top":30,"bottom":30},
 "xAxis":{"type":"category","data":["DeepSeek-R1","MiniMax-M1"]},
 "yAxis":{"type":"value","name":"相对 FLOPs（R1=100）"},
 "series":[{"type":"bar","data":[{"value":100,"itemStyle":{"color":"#94a3b8"}},{"value":25,"itemStyle":{"color":"#1d4ed8"}}],"label":{"show":True,"position":"top"}}]
}
page("minimax_m1","MiniMax-M1 家族架构详解",
 "MiniMax M1",
 ["456B / 45.9B（同 Text-01 架构）","首个开源大规模混合注意力推理模型","原生 1M 上下文","40K / 80K 思维预算","CISPO RL 算法"],
 "以<b>远端 config</b>（MiniMaxAI/MiniMax-M1）+ 论文为依据。M1 是「同构权重 + RL」，不是新架构。",
 m1_sec, "chart1", m1_chart,
 [("论文：MiniMax-M1: Scaling Test-Time Compute with Lightning Attention","https://arxiv.org/abs/2506.13585"),
  ("官方公告","https://www.minimaxi.com/news/minimaxm1"),
  ("远端官方 config","https://huggingface.co/MiniMax-M1")])

# ================= M2 =================
m2_rows = [
 ("总参 / 激活","约 229.9B / 9.8B（≈230B-A10B）","技术报告口径 + config 推算"),
 ("层数 / hidden","62 / 3072","远端 config"),
 ("注意力","全 Softmax GQA（48Q/8KV，head128），无 Lightning","config + 源码"),
 ("专家 / top-k","256 / 8","远端 config"),
 ("专家中间维","1536","config + 源码"),
 ("路由","sigmoid + e_score 校正偏置","config + 源码"),
 ("QK-Norm","q/k 后各 RMSNorm（per-layer）","config + 源码"),
 ("RoPE","rotary_dim 64，θ=5e6","远端 config"),
 ("MTP","use_mtp，3 个模块","远端 config"),
 ("上下文 / 权重","196608 / FP8(e4m3fn)","远端 config"),
]
m2_sec = f'''
<section id="family"><h2><span class="secno">1</span>家族全景</h2>
<p class="note">M2 是路线反转：放弃 Lightning，62 层全 Softmax，专家从 32→256、top-k 2→8。</p>
{cfg_table(m2_rows)}
<div id="chart1" style="width:100%;height:340px;margin-top:16px"></div></section>
<section id="overview"><h2><span class="secno">2</span>端到端</h2>{img("arch_overview.png","图：词嵌入→62 层全 Softmax+256专家→RMSNorm→LM Head+3 MTP")}</section>
<section id="layer"><h2><span class="secno">3</span>解码层</h2>
<div class="formula">x = norm(x); x = self_attn(x); x = residual + x   <span class="cm"># 朴素残差，取消 α 门控</span>
x = norm(x); x = moe256(x);   x = residual + x</div></section>
<section id="mech"><h2><span class="secno">4</span>核心机制</h2>
{img("arch_mech.png","图：sigmoid 路由+校正偏置 · QK-Norm · MTP")}
<p class="note">sigmoid 评分替代 softmax，加可学习 e_score_correction_bias 缓解专家负载。</p></section>
<section id="attn"><h2><span class="secno">5</span>注意力</h2><p>62 层全部标准 Softmax GQA（无 Lightning），q/k 后 RMSNorm，部分 RoPE 64/128。</p></section>
<section id="tokens"><h2><span class="secno">6</span>Token</h2><p>词表 200064，bos=200034，eos=200020，上下文 196608。</p></section>
<section id="evo"><h2><span class="secno">7</span>演进坐标</h2><p>Text-01/M1（Lightning 路线）→ M2（全 Softmax+256专家）→ M3（MSA 块稀疏+多模态）。</p></section>
'''
m2_chart = {
 "tooltip":{"trigger":"axis"},
 "legend":{"data":["专家总数","每 token 激活专家"]},
 "grid":{"left":60,"right":20,"top":40,"bottom":30},
 "xAxis":{"type":"category","data":["Text-01/M1","M2"]},
 "yAxis":{"type":"value","name":"专家数"},
 "series":[{"type":"bar","name":"专家总数","data":[32,256],"itemStyle":{"color":"#1d4ed8"}},
           {"type":"bar","name":"每 token 激活专家","data":[2,8],"itemStyle":{"color":"#b45309"}}]
}
page("minimax_m2","MiniMax-M2 家族架构详解",
 "MiniMax M2",
 ["约 230B / 9.8B 激活","62 层全 Softmax GQA（无 Lightning）","256 专家 top-8 · sigmoid 路由","QK-Norm · MTP ×3","FP8 发布 · 192K 上下文"],
 "以<b>远端 config</b>（MiniMaxAI/MiniMax-M2）+ <b>源码 models/minimax_m2</b> 为依据。",
 m2_sec, "chart1", m2_chart,
 [("远端官方 config","https://huggingface.co/MiniMaxAI/MiniMax-M2"),
  ("源码：models/minimax_m2/configuration_minimax_m2.py · modeling_minimax_m2.py","E:\\transformers\\src\\transformers\\models\\minimax_m2")])

# ================= M3 =================
m3_rows = [
 ("层数 / hidden","60 / 6144","远端 text_config"),
 ("层结构","前 3 层 Dense + 后 57 层 MoE","mlp_layer_types"),
 ("专家 / top-k / 共享","128 / 4 / 1 共享专家","远端 config"),
 ("注意力","前3层全注意力 + 后57层 MSA 块稀疏","sparse_attention_config"),
 ("稀疏块 / top-k 块","128 / 16 + 1 局部","config"),
 ("激活 / Norm","SwiGLU-OAI(α1.702) · GemmaNorm+per_head QK-Norm","config"),
 ("MTP 模块","7","config"),
 ("上下文","1,048,576（1M）","config"),
 ("视觉塔","CLIP · patch14 · 2016 · 3D RoPE · 投影6144","vision_config"),
]
m3_sec = f'''
<section id="family"><h2><span class="secno">1</span>家族全景</h2>
<p class="note">M3 原生多模态旗舰：前 3 层 Dense、后 57 层 MoE+MSA 块稀疏注意力，128 专家带 1 共享专家。</p>
{cfg_table(m3_rows)}
<div id="chart1" style="width:100%;height:340px;margin-top:16px"></div></section>
<section id="overview"><h2><span class="secno">2</span>端到端</h2>{img("arch_overview.png","图：文本+图像/视频 → 3 层 Dense → 57 层 MoE+MSA → LM Head+7 MTP")}</section>
<section id="layer"><h2><span class="secno">3</span>解码层</h2><p>前 3 层全注意力 + SwiGLU(12288)；后 57 层 MSA + 128 专家 top-4 + 共享专家。</p></section>
<section id="mech"><h2><span class="secno">4</span>核心机制：MSA 块稀疏注意力</h2>
{img("arch_msa.png","图：Index 分支选 top-16 块，Main 分支只在选中块算 GQA")}
<p class="note">官方口径：1M 上下文单 token 算力约为 M2 的 1/20，预填充提速约 9 倍。</p></section>
<section id="attn"><h2><span class="secno">5</span>注意力</h2><p>前 3 层全 Softmax，后 57 层 MSA 块稀疏（block128，top-16 块+1 局部）。</p></section>
<section id="mm" class="mm"><h2 class="mm"><span class="secno">6</span>多模态结构</h2>
{img("arch_vision.png","图：视觉塔 3D RoPE + patch merge（空间×2 时间×2）")}</section>
<section id="tokens"><h2><span class="secno">7</span>Token</h2><p>&lt;image&gt;=200025，&lt;video&gt;=200026，每图 576 token；上下文 1M。</p></section>
<section id="evo"><h2><span class="secno">8</span>演进坐标</h2><p>Text-01(M1) Lightning → M2 全 Softmax → M3 MSA 块稀疏+原生多模态。「MiniMax-Text-04」未检索到官方仓，规格待确认。</p></section>
'''
m3_chart = {
 "tooltip":{"trigger":"axis"},
 "legend":{"data":["Dense 层","MoE 稀疏层"]},
 "grid":{"left":60,"right":20,"top":40,"bottom":30},
 "xAxis":{"type":"category","data":["前 3 层","后 57 层"]},
 "yAxis":{"type":"value","name":"层数"},
 "series":[{"type":"bar","stack":"t","data":[3,0],"name":"Dense 层","itemStyle":{"color":"#1d4ed8"}},
           {"type":"bar","stack":"t","data":[0,57],"name":"MoE 稀疏层","itemStyle":{"color":"#0e7490"}}]
}
page("minimax_m3_vl","MiniMax-M3（M3-VL）家族架构详解",
 "MiniMax M3",
 ["原生多模态（图像+视频）","60 层：3 Dense + 57 MoE","MSA 块稀疏注意力","128 专家 top-4 + 1 共享","1M 上下文"],
 "以<b>远端 config</b>（MiniMaxAI/MiniMax-M3）+ <b>源码 models/minimax_m3_vl</b> 为依据。",
 m3_sec, "chart1", m3_chart,
 [("官方博客：MiniMax M3","https://www.minimaxi.com/blog/minimax-m3"),
  ("vLLM Day-0 技术解析","https://vllm-project.github.io/2026/06/12/minimax-m3-vllm.html"),
  ("远端官方 config","https://huggingface.co/MiniMaxAI/MiniMax-M3")], mm_nav=True)

print("ALL HTML DONE")
