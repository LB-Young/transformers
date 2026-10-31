# -*- coding: utf-8 -*-
"""生成 GLM 早期代三份自包含 index.html，CSS 逐字沿用 gemma-3 标杆。"""
import os

ROOT = r"E:\transformers\bayon-note\GLM"

CSS = """
  :root{
    --bg:#faf9f6; --panel:#ffffff; --ink:#23272f; --sub:#66707b; --line:#e5e4df;
    --accent:#1d4ed8; --accent-ink:#1e40af; --accent-soft:#e2eafb; --accent-line:#c0d0f4;
    --node:#fffdf9; --edge:#c9c4ba; --mono:#3b5390;
    --sliding:#dbe7fc; --sliding-edge:#a9c2f3; --full:#1d4ed8;
    --mm:#0e7490; --mm-ink:#155e75; --mm-soft:#e0f2f7; --mm-line:#a5d8e6;
    --radius:12px;
  }
  *{box-sizing:border-box; margin:0; padding:0;}
  html{scroll-behavior:smooth;}
  body{background:var(--bg); color:var(--ink); font-family:system-ui,-apple-system,"Segoe UI","Microsoft YaHei","PingFang SC","Noto Sans SC",sans-serif; line-height:1.65; font-size:15px;}
  .wrap{max-width:1220px; margin:0 auto; padding:0 16px;}
  code,.num{font-family:ui-monospace,"Cascadia Mono",Consolas,"Courier New",monospace; font-size:.92em;}
  .num{color:var(--mono); background:var(--accent-soft); padding:0 4px; border-radius:4px; white-space:nowrap;}
  a{color:var(--accent-ink);}
  nav.top{position:sticky; top:0; z-index:50; background:rgba(250,249,246,.94); backdrop-filter:blur(6px); border-bottom:1px solid var(--line);}
  nav.top .wrap{display:flex; align-items:center; gap:14px; padding-top:10px; padding-bottom:10px; overflow-x:auto; white-space:nowrap;}
  nav.top .brand{display:flex; align-items:center; gap:8px; font-weight:700; color:var(--accent-ink);}
  nav.top a.link{color:var(--ink); text-decoration:none; font-size:.9rem; padding:6px 10px; border-radius:8px;}
  nav.top a.link:hover{background:var(--accent-soft); color:var(--accent-ink);}
  header.hero{padding:36px 0 8px;}
  .badges{display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; margin:12px 0;}
  @media (min-width:820px){.badges{grid-template-columns:repeat(4,minmax(0,1fr));}}
  .badge{font-size:.78rem; padding:3px 10px; border-radius:999px; border:1px solid var(--accent-line); background:var(--accent-soft); color:var(--accent-ink);}
  h1{font-size:clamp(1.6rem,4vw,2.3rem); letter-spacing:.5px;}
  .lead{color:var(--sub); max-width:1000px; margin-top:8px;}
  .lead b{color:var(--ink);}
  section{margin:36px 0;}
  h2{font-size:1.35rem; display:flex; align-items:center; gap:10px; margin-bottom:14px;}
  h2 .secno{font-size:.85rem; color:var(--accent-ink); background:var(--accent-soft); border:1px solid var(--accent-line); width:28px; height:28px; border-radius:8px; display:inline-flex; align-items:center; justify-content:center; flex:none;}
  h3{font-size:1.02rem; margin:12px 0 6px;}
  .note{font-size:.85rem; color:var(--sub);}
  .tip{font-size:.85rem; color:var(--sub); background:var(--accent-soft); border:1px dashed var(--accent-line); padding:8px 12px; border-radius:10px; margin-top:8px;}
  .diagram-shell{background:var(--panel); border:1px solid var(--line); border-radius:var(--radius); padding:14px; overflow-x:auto;}
  table.cmp{width:100%; min-width:860px; border-collapse:collapse; background:var(--panel); border:1px solid var(--line); border-radius:var(--radius); overflow:hidden; table-layout:fixed;}
  .table-scroll{overflow-x:auto; -webkit-overflow-scrolling:touch;}
  table.cmp th,table.cmp td{padding:7px 10px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; font-size:.8rem; word-wrap:break-word;}
  table.cmp thead th{background:var(--accent-soft); color:var(--accent-ink); font-weight:700; font-size:.84rem;}
  table.cmp tbody tr:nth-child(even){background:#f5f8ff;}
  table.cmp td.k{font-weight:600;}
  table.cmp td small{color:var(--sub); display:block; margin-top:2px;}
  table.cmp col.c1{width:22%;}
  table.cmp col.cx{width:19.5%;}
  table.cmp tr.hl td{background:#eaf1fe;}
  .cards{display:grid; grid-template-columns:minmax(0,1fr); gap:14px;}
  @media (min-width:820px){.cards{grid-template-columns:repeat(2,minmax(0,1fr));}}
  .card{background:var(--panel); border:1px solid var(--line); border-radius:var(--radius); padding:16px 18px; min-width:0; overflow-wrap:anywhere;}
  .card h3{margin-top:0; font-size:.98rem;}
  .card p{font-size:.88rem; margin-top:6px;}
  .formula{background:#1f2937; color:#bfd3ff; font-family:ui-monospace,Consolas,monospace; font-size:.82rem; padding:10px 14px; border-radius:8px; margin-top:8px; overflow-x:auto; white-space:pre; line-height:1.6;}
  #chart{width:100%; height:340px; background:var(--panel); border:1px solid var(--line); border-radius:var(--radius); margin-top:10px;}
  ol.src{list-style:none; counter-reset:s; background:var(--panel); border:1px solid var(--line); border-radius:var(--radius); padding:16px 18px;}
  ol.src li{counter-increment:s; padding:8px 0; border-bottom:1px dashed var(--line); font-size:.86rem;}
  ol.src li:last-child{border-bottom:none;}
  ol.src li::before{content:"[" counter(s) "] "; color:var(--accent-ink); font-weight:700; font-family:ui-monospace,Consolas,monospace;}
  ol.src .u{display:block; color:var(--sub); word-break:break-all; font-size:.78rem;}
  footer{margin:44px 0 24px; color:var(--sub); font-size:.8rem; text-align:center;}
  @media print{nav.top{display:none;} body{background:#fff;} .diagram-shell{overflow:visible; border:none; padding:0;}}
"""

NAV = """
<nav class="top"><div class="wrap">
  <div class="brand">{brand}</div>
  <a class="link" href="#family">家族全景</a>
  <a class="link" href="#overview">数据流总览</a>
  <a class="link" href="#layer">解码层解剖</a>
  <a class="link" href="#mech">核心机制</a>
  <a class="link" href="#attn">注意力</a>
  <a class="link" href="#tokens">特殊 Token</a>
  <a class="link" href="#evo">演进视角</a>
  <a class="link" href="#sources">来源</a>
</div></nav>
"""

HTML_TMPL = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect x='2' y='2' width='28' height='28' rx='7' fill='%231d4ed8'/%3E%3Ctext x='16' y='22' font-size='14' font-family='Arial' font-weight='bold' fill='%23fff' text-anchor='middle'%3EG%3C/text%3E%3C/svg%3E">
<style>{css}</style>
<script src="https://fastly.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
</head>
<body>
{nav}
<div class="wrap">
<header class="hero">
  <h1>{h1}</h1>
  <div class="badges">{badges}</div>
  <p class="lead">{lead}</p>
</header>
{body}
<section id="chartsec"><h2><span class="secno">9</span>早期代关键尺寸对比（真实 config 复算）</h2>
<div id="chart"></div>
<p class="note">数据来自各模型官方 config / 论文：层数、hidden、FFN、词表（千）等字段逐值复算。</p></section>
<footer>{footer}</footer>
</div>
<script>
const chart = echarts.init(document.getElementById('chart'));
chart.setOption({chart_option});
window.addEventListener('resize', ()=>chart.resize());
</script>
</body>
</html>
"""


def badge(t):
    return f'<span class="badge">{t}</span>'


def section(no, sid, title, inner):
    return (f'<section id="{sid}"><h2><span class="secno">{no}</span>{title}</h2>'
            f'{inner}</section>')


def table(headers, rows, widths):
    col = "".join(f'<col style="width:{w}%">' for w in widths)
    th = "".join(f"<th>{h}</th>" for h in headers)
    trs = []
    for r in rows:
        tds = "".join(f'<td class="{c[0]}" >{c[1]}</td>' if isinstance(c, tuple) else f"<td>{c}</td>" for c in r)
        trs.append(f"<tr>{tds}</tr>")
    return (f'<div class="table-scroll"><table class="cmp"><colgroup>{col}</colgroup>'
            f"<thead><tr>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table></div>")


def card(title, paras):
    ps = "".join(f"<p>{p}</p>" for p in paras)
    return f'<div class="card"><h3>{title}</h3>{ps}</div>'


def src_ol(items):
    lis = "".join(f"<li>{t}<span class='u'>{u}</span></li>" for t, u in items)
    return f"<ol class='src'>{lis}</ol>"


# ---------- 每模型内容 ----------
def glm130b():
    badges = "".join(badge(t) for t in [
        "骨干：双向 GLM（blank infilling），非 GPT 单向",
        "70 层 · hidden 12288 · 96 头 · head_dim 128",
        "DeepNorm Post-LN · GeGLU FFN 32768 · RoPE",
        "上下文 2048 · icetk 词表 150000 · 无公开 HF 权重",
    ])
    family = section(1, "family", "家族全景：GLM-130B 核心配置",
        '<p class="note">单一稠密 130B 尺寸。数字来源：官方论文 arXiv:2210.02414 §2 与附录 Table 11。</p>'
        + table(["字段", "取值", "来源"], [
            [("k", "参数量 / 形态"), "≈130B 稠密双语（中英）", "论文 §1/§2.1"],
            [("k", "层数"), "70（9×8−2 流水线均衡）", "论文 §2.3"],
            [("k", "hidden_size"), "12288", "论文 §2.3 / T11"],
            [("k", "注意力头数"), "96（head_dim=128）", "公开口径（待确认）"],
            [("k", "FFN hidden"), "32768（=8/3·12288）", "论文 T11/§B.3"],
            [("k", "归一化"), "DeepNorm Post-LN，α=(2N)^(1/4)", "论文 §2.1"],
            [("k", "FFN 激活"), "GeGLU（GLU+GeLU）", "论文 §B.3"],
            [("k", "位置编码"), "RoPE", "论文 §B.3"],
            [("k", "最大序列"), "2048", "论文 §2.3"],
            [("k", "分词器 / 词表"), "icetk（sentencepiece）/ 150000", "论文 §B.1"],
        ], [34, 42, 24]))
    overview = section(2, "overview", "端到端数据流总览",
        '<p class="note">破坏 span→[MASK]/[gMASK] → icetk 嵌入 → 70×GLM 层（双向/自回归混合注意力+GeGLU）→ 最终 LN → softmax。</p>'
        '<div class="diagram-shell"><img src="_arch/01_end_to_end.png" style="width:100%;min-width:820px" alt="GLM-130B 端到端主干"></div>')
    layer = section(3, "layer", "解码层解剖：DeepNorm 残差块",
        '<div class="formula">DeepNorm(x) = LayerNorm( α·x + SubLayer(x) ),  α = (2N)^(1/4)\n# N=70 → α≈3.44；FFN/v_proj/out_proj 初始化缩放 (2N)^(-1/4)</div>'
        '<p class="note">相比 Pre-LN，DeepNorm Post-LN 把深层主分支 value 尺度约束在有界范围，是 70 层稳定训练的关键。</p>')
    mech = section(4, "mech", "核心机制",
        '<div class="diagram-shell"><img src="_arch/02_core_mech.png" style="width:100%;min-width:820px" alt="GLM-130B 核心机制"></div>'
        '<div class="cards">'
        + card("① GLM blank infilling", ["采样 span 替换为单个 mask 标记，自回归重建；span 间可见性由随机排列决定。", "[MASK] 单 span（窗512拼4）；[gMASK] 多 span（窗2048）。"])
        + card("② DeepNorm + EGS", ["反直觉地选 Post-LN 稳定训练；Embedding Gradient Shrink 把嵌入梯度缩到 0.1，抑制 loss spike。"])
        + card("③ GeGLU", ["FFN=(GeLU(xW1)⊙xV)W2；引入 V 后 d_ffn 由 4·d_h 降到 8/3·d_h 保持参数量。"])
        + card("④ INT4 缩放律", ["权重分布形状良好，无需量化训练即可 INT4 推理，4×3090 可跑。"])
        + "</div>")
    attn = section(5, "attn", "注意力机制：双向 prefix 注意力",
        '<div class="formula">Attn(Q,K,V) = softmax( QKᵀ/√d_k + mask_GLM ) V\n# prefix 上下文双向；被掩码 span 内自回归；96 头 × 128 维</div>'
        '<p class="tip">与 GPT 式因果注意力的根本差别：未掩码上下文双向可见。头数 96、head_dim 128 由官方口径与 hidden 复算，附录 config 未逐字段列出，标「待确认」。</p>')
    tokens = section(6, "tokens", "特殊 Token 与词表",
        table(["项", "取值"], [
            [("k", "分词器"), "icetk 图像-文本统一（前 20000 为图像 token）"],
            [("k", "词表"), "150000"],
            [("k", "破坏标记"), "[MASK] / [gMASK] 两类目标"],
            [("k", "上下文窗"), "gMASK=2048；MASK/MIP=512（4 样本拼接）"],
        ], [28, 72]))
    evo = section(7, "evo", "演进视角：GLM-130B → ChatGLM-6B",
        table(["维度", "GLM-130B", "ChatGLM-6B"], [
            [("k", "规模"), "130B 双向基座", "6.2B 对话"],
            [("k", "层/hidden"), "70 / 12288", "28 / 4096"],
            [("k", "位置编码"), "RoPE 1D", "2D-RoPE"],
            [("k", "归一化"), "DeepNorm Post-LN", "LayerNorm"],
            [("k", "上下文"), "2048", "2048"],
            [("k", "词表"), "icetk 150000", "130528"],
        ], [22, 39, 39]))
    sources = section(8, "sources", "参考文献", src_ol([
        ("GLM-130B 官方论文 arXiv:2210.02414", "https://arxiv.org/abs/2210.02414"),
        ("论文 HTML 全文（§2 / 附录 Table 11）", "https://arxiv.org/html/2210.02414v2"),
        ("官方 GitHub THUDM/GLM-130B", "https://github.com/THUDM/GLM-130B"),
        ("清华 KEG 官方博客", "https://keg.cs.tsinghua.edu.cn/glm-130b/zh/posts/glm-130b/"),
        ("ChatGLM 家族综述 arXiv:2406.12793", "https://arxiv.org/abs/2406.12793"),
    ]))
    chart = """{
      backgroundColor:'transparent',
      tooltip:{},
      legend:{bottom:0},
      grid:{left:60,right:30,top:40,bottom:60},
      xAxis:{type:'category',data:['层数','hidden','FFN','词表(千)']},
      yAxis:{type:'value',name:'数值'},
      series:[
        {name:'GLM-130B',type:'bar',data:[70,12288,32768,150]},
        {name:'ChatGLM-6B',type:'bar',data:[28,4096,16384,130.5]},
        {name:'ChatGLM2/3',type:'bar',data:[28,4096,13696,65]}
      ]
    }"""
    return badges, family + overview + layer + mech + attn + tokens + evo + sources, chart


def chatglm6b():
    badges = "".join(badge(t) for t in [
        "model_type=chatglm · ChatGLMModel · trust_remote_code",
        "28 层 · hidden 4096 · 32 头（head_dim 128）· FFN 16384",
        "prefix-LM · 2D-RoPE（position_encoding_2d=true）",
        "LayerNorm · 上下文 2048 · 词表 130528",
    ])
    family = section(1, "family", "家族全景：官方 config 实参表",
        '<p class="note">来源：HF THUDM/chatglm-6b 远端 config（hf-mirror 拉取，2026-10-31，留档 _tools/_cfgpull/）。</p>'
        + table(["config 字段", "取值", "含义"], [
            [("k", "model_type / architectures"), "chatglm / ChatGLMModel", "需 trust_remote_code"],
            [("k", "hidden_size"), "4096", "隐层"],
            [("k", "num_layers"), "28", "层数"],
            [("k", "num_attention_heads"), "32", "head_dim=128"],
            [("k", "inner_hidden_size"), "16384", "=4×4096"],
            [("k", "position_encoding_2d"), "true", "2D 旋转位置编码"],
            [("k", "max_sequence_length"), "2048", "上下文"],
            [("k", "vocab_size"), "130528", "ice tokenizer"],
            [("k", "layernorm_epsilon"), "1e-5", "LayerNorm"],
            [("k", "bos/eos/mask/gmask/pad"), "130004/130005/130000/130001/3", "对话控制标记"],
        ], [34, 34, 32]))
    overview = section(2, "overview", "端到端数据流总览",
        '<div class="diagram-shell"><img src="_arch/01_end_to_end.png" style="width:100%;min-width:820px" alt="ChatGLM-6B 端到端主干"></div>')
    layer = section(3, "layer", "解码层解剖",
        '<div class="formula">x = LayerNorm(x)\na = SelfAttn(Q/K/V, prefix_mask) + 2D-RoPE\nx = x + a\nx = LayerNorm(x)\nx = x + FFN(x)   # inner 16384</div>')
    mech = section(4, "mech", "核心机制",
        '<div class="diagram-shell"><img src="_arch/02_core_mech.png" style="width:100%;min-width:820px" alt="ChatGLM-6B 核心机制"></div>'
        '<div class="cards">'
        + card("① 2D-RoPE", ["两组旋转位置 id（序列位置 + span 内相对位置），分别作用于 Q/K 两半。"])
        + card("② prefix-LM 掩码", ["prefix 双向、generation 因果三角，同一模型兼顾理解与生成。"])
        + card("③ 特殊标记", ["mask=130000/gmask=130001 沿用 GLM；约 1T token 中英预训练 + SFT。"])
        + card("④ 部署", ["INT4 量化后消费级显卡可本地运行（6.2B）。"])
        + "</div>")
    attn = section(5, "attn", "注意力机制",
        table(["项", "取值"], [
            [("k", "Q 头数"), "32"],
            [("k", "KV 头数"), "32（标准 MHA，无分组字段）"],
            [("k", "head_dim"), "4096/32=128"],
            [("k", "位置编码"), "2D-RoPE（position_encoding_2d=true）"],
            [("k", "类型"), "prefix-LM"],
        ], [30, 70])
        + '<p class="tip">对比：ChatGLM2 才引入 multi_query_attention=true / group_num=2；本代为全 MHA。</p>')
    tokens = section(6, "tokens", "特殊 Token 与词表",
        table(["token", "ID"], [
            [("k", "mask_token_id"), "130000"],
            [("k", "gmask_token_id"), "130001"],
            [("k", "bos / eos"), "130004 / 130005"],
            [("k", "pad"), "3"],
            [("k", "vocab_size"), "130528"],
        ], [50, 50]))
    evo = section(7, "evo", "演进视角：GLM-130B → ChatGLM-6B → ChatGLM2",
        table(["维度", "GLM-130B", "ChatGLM-6B", "ChatGLM2"], [
            [("k", "层/hidden"), "70/12288", "28/4096", "28/4096"],
            [("k", "注意力"), "双向 GLM", "prefix 2D-RoPE", "MQA 1D-RoPE"],
            [("k", "归一化"), "DeepNorm", "LayerNorm", "RMSNorm"],
            [("k", "上下文"), "2048", "2048", "32768"],
            [("k", "词表"), "150000", "130528", "65024"],
        ], [22, 26, 26, 26]))
    sources = section(8, "sources", "参考文献", src_ol([
        ("远端 config HF THUDM/chatglm-6b（hf-mirror）", "https://hf-mirror.com/THUDM/chatglm-6b/resolve/main/config.json"),
        ("官方仓库 THUDM/ChatGLM-6B（trust_remote_code）", "https://github.com/THUDM/ChatGLM-6B"),
        ("ChatGLM 家族综述 arXiv:2406.12793", "https://arxiv.org/abs/2406.12793"),
        ("GLM 2D 位置编码论文 arXiv:2103.10360", "https://arxiv.org/abs/2103.10360"),
    ]))
    chart = """{
      backgroundColor:'transparent',
      tooltip:{},
      legend:{bottom:0},
      grid:{left:60,right:30,top:40,bottom:60},
      xAxis:{type:'category',data:['hidden','FFN','层数','上下文']},
      yAxis:{type:'value'},
      series:[{name:'ChatGLM-6B',type:'bar',data:[4096,16384,28,2048]}]
    }"""
    return badges, family + overview + layer + mech + attn + tokens + evo + sources, chart


def chatglm23():
    badges = "".join(badge(t) for t in [
        "model_type=chatglm · ChatGLMModel · trust_remote_code",
        "28 层 · hidden 4096 · 32Q/2KV（MQA）· kv_channels 128",
        "RMSNorm(post) · SwiGLU FFN 13696 · 1D-RoPE(original)",
        "词表 65024 · 两版架构数字完全一致（仅 seq_length 异）",
    ])
    family = section(1, "family", "家族全景：两代 config 逐字段对照",
        '<p class="note">来源：HF THUDM/chatglm2-6b 与 chatglm3-6b 远端 config（hf-mirror，2026-10-31 留档）。逐字段比对：架构数字全部相同，唯一差异 seq_length。</p>'
        + table(["字段", "ChatGLM2", "ChatGLM3", "说明"], [
            [("k", "hidden_size"), "4096", "4096", "相同"],
            [("k", "num_layers"), "28", "28", "相同"],
            [("k", "num_attention_heads"), "32", "32", "Q 头"],
            [("k", "kv_channels"), "128", "128", "head_dim"],
            [("k", "multi_query_group_num"), "2", "2", "KV 组"],
            [("k", "ffn_hidden_size"), "13696", "13696", "SwiGLU"],
            [("k", "rmsnorm / post_layer_norm"), "true/true", "true/true", "RMSNorm"],
            [("k", "original_rope"), "true", "true", "1D RoPE"],
            [("k", "add_qkv_bias"), "true", "true", "仅 QKV 带 bias"],
            [("hl", ("k", "seq_length")), ("hl", "32768"), ("hl", "8192"), "唯一差异"],
            [("k", "padded_vocab_size"), "65024", "65024", "相同"],
            [("k", "tie_word_embeddings"), "false", "false", "解绑"],
        ], [30, 18, 18, 34]))
    overview = section(2, "overview", "端到端数据流总览（两代同构）",
        '<div class="diagram-shell"><img src="_arch/01_end_to_end.png" style="width:100%;min-width:820px" alt="ChatGLM2/3 端到端主干"></div>')
    layer = section(3, "layer", "解码层解剖",
        '<div class="formula">x = RMSNorm(x)\na = SelfAttn(QKV_bias, kv_groups=2, rope=1d)\nx = x + a   # post_layer_norm\nx = RMSNorm(x)\nx = x + SwiGLU(x)  # ffn 13696</div>')
    mech = section(4, "mech", "核心机制",
        '<div class="diagram-shell"><img src="_arch/02_core_mech.png" style="width:100%;min-width:820px" alt="ChatGLM2/3 核心机制"></div>'
        '<div class="cards">'
        + card("① MQA/GQA", ["32 Q 头共享 2 组 KV，KV cache 降为 1/16；官方称推理提速约 42%。"])
        + card("② RMSNorm+SwiGLU", ["rmsnorm=true 取代 LayerNorm；ffn 13696≈10/3·4096 补偿 GQA 减参。"])
        + card("③ 1D-RoPE", ["original_rope=true，移除 ChatGLM-6B 的 2D-RoPE。"])
        + card("④ 上下文", ["ChatGLM2 config=32768（FlashAttention 32K）；ChatGLM3 config=8192。"])
        + "</div>")
    attn = section(5, "attn", "注意力机制",
        '<div class="formula">Attn = softmax_fp32( QKᵀ/√128 + mask ) V   # K,V 仅 2 组（repeat_kv 到 32 头）</div>'
        + table(["项", "取值"], [
            [("k", "Q 头 / KV 组"), "32 / 2"],
            [("k", "kv_channels"), "128"],
            [("k", "位置编码"), "1D-RoPE"],
            [("k", "QKV bias"), "true"],
        ], [32, 68]))
    tokens = section(6, "tokens", "特殊 Token 与词表",
        table(["项", "取值"], [
            [("k", "padded_vocab_size"), "65024"],
            [("k", "eos / pad"), "2 / 0"],
            [("k", "tie_word_embeddings"), "false"],
            [("k", "ChatGLM3 新增"), "function call / Agent 角色标记（chat template）"],
        ], [36, 64]))
    evo = section(7, "evo", "演进视角：ChatGLM-6B → ChatGLM2 → ChatGLM3",
        table(["维度", "ChatGLM-6B", "ChatGLM2", "ChatGLM3"], [
            [("k", "注意力"), "MHA 32", "MQA 32/2", "MQA 32/2"],
            [("k", "位置编码"), "2D-RoPE", "1D-RoPE", "1D-RoPE"],
            [("k", "归一化"), "LayerNorm", "RMSNorm", "RMSNorm"],
            [("k", "FFN"), "16384", "13696 SwiGLU", "13696 SwiGLU"],
            [("k", "config 上下文"), "2048", "32768", "8192"],
            [("k", "词表"), "130528", "65024", "65024"],
        ], [24, 25, 25, 26]))
    sources = section(8, "sources", "参考文献", src_ol([
        ("config HF THUDM/chatglm2-6b（hf-mirror）", "https://hf-mirror.com/THUDM/chatglm2-6b/resolve/main/config.json"),
        ("config HF THUDM/chatglm3-6b（hf-mirror）", "https://hf-mirror.com/THUDM/chatglm3-6b/resolve/main/config.json"),
        ("官方仓库 ChatGLM2-6B / ChatGLM3", "https://github.com/THUDM/ChatGLM3"),
        ("ChatGLM 家族综述 arXiv:2406.12793", "https://arxiv.org/abs/2406.12793"),
    ]))
    chart = """{
      backgroundColor:'transparent',
      tooltip:{},
      legend:{bottom:0},
      grid:{left:60,right:30,top:40,bottom:60},
      xAxis:{type:'category',data:['层数','hidden','FFN','KV 组']},
      yAxis:{type:'value'},
      series:[
        {name:'ChatGLM2-6B',type:'bar',data:[28,4096,13696,2]},
        {name:'ChatGLM3-6B',type:'bar',data:[28,4096,13696,2]}
      ]
    }"""
    return badges, family + overview + layer + mech + attn + tokens + evo + sources, chart


MODELS = {
    "glm_130b": dict(
        title="GLM-130B 家族架构详解", brand="GLM-130B", h1="GLM-130B 家族架构详解",
        lead="本页以官方论文 arXiv:2210.02414 与官方 GitHub 为唯一依据。GLM-130B 无公开 HF 权重、transformers 无原生模块，数字逐字段来自论文 §2 与附录 Table 11。",
        footer="GLM-130B 家族架构详解 · 官方论文 arXiv:2210.02414 + 官方 GitHub/KEG · 仅供技术学习",
        fn=glm130b),
    "chatglm_6b": dict(
        title="ChatGLM-6B 家族架构详解", brand="ChatGLM-6B", h1="ChatGLM-6B 家族架构详解",
        lead="本页以 HF THUDM/chatglm-6b 远端 config（hf-mirror）与 arXiv:2406.12793 为唯一依据。transformers v5 无原生 chatglm 模块，需 trust_remote_code 加载官方实现。",
        footer="ChatGLM-6B 家族架构详解 · 远端官方 config + 官方仓库 + arXiv 2406.12793 · 仅供技术学习",
        fn=chatglm6b),
    "chatglm2_3": dict(
        title="ChatGLM2 / ChatGLM3 家族架构详解", brand="ChatGLM2/3", h1="ChatGLM2 / ChatGLM3 家族架构详解",
        lead="逐字段比对确认两代 config 架构数字完全一致（仅 seq_length 不同），故合并本页。来源：HF THUDM/chatglm2-6b 与 chatglm3-6b 远端 config（hf-mirror）。",
        footer="ChatGLM2/3 家族架构详解 · 远端官方 config + 官方仓库 + arXiv 2406.12793 · 仅供技术学习",
        fn=chatglm23),
}

for key, m in MODELS.items():
    badges, body, chart = m["fn"]()
    nav = NAV.format(brand=m["brand"])
    html = HTML_TMPL.format(title=m["title"], css=CSS, nav=nav, h1=m["h1"],
                            badges=badges, lead=m["lead"], body=body,
                            footer=m["footer"], chart_option=chart)
    out = os.path.join(ROOT, key, "index.html")
    with open(out, "w", encoding="utf-8", newline="") as f:
        f.write(html)
    print("wrote", out, len(html))
print("ALL HTML DONE")
