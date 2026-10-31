# -*- coding: utf-8 -*-
import os, re
ROOT = r"E:\transformers\bayon-note\MiniMax"
JOBS = {
 "minimax_text_01": ("MiniMax-Text-01", "MiniMax-Text-01 家族架构详解",
   "MiniMaxForCausalLM · 456B总参/45.9B激活 · 80层 · Lightning:Softmax=7:1 · 32专家top2"),
 "minimax_visual_01": ("MiniMax-Visual-01", "MiniMax-Visual-01（VL-01）家族架构详解",
   "MiniMaxVL01ForConditionalGeneration · Text-01文本塔+CLIP视觉塔 · patch14/336 · 动态分辨率"),
 "minimax_m1": ("MiniMax-M1", "MiniMax-M1 家族架构详解",
   "MiniMaxM1ForCausalLM · 456B/45.9B（同Text-01架构）· RL长链推理 · 1M上下文"),
 "minimax_m2": ("MiniMax-M2", "MiniMax-M2 家族架构详解",
   "MiniMaxM2ForCausalLM · 约230B/9.8B · 62层全Softmax · 256专家top8 · QK-Norm · MTP · FP8"),
 "minimax_m3_vl": ("MiniMax-M3", "MiniMax-M3（M3-VL）家族架构详解",
   "MiniMaxM3SparseForConditionalGeneration · 60层 · 3Dense+57MoE · MSA块稀疏 · 128专家+1共享 · 原生多模态"),
}
for d,(outname,title,sub) in JOBS.items():
    p = os.path.join(ROOT, d, "_tools", "md2docx.py")
    t = open(p, encoding="utf-8").read()
    t = re.sub(r'BASE = r".*?"', lambda m: f'BASE = r"{ROOT}"', t, count=1)
    models_block = f'MODELS = [\n    ("{d}", "{title}", "{sub}"),\n]'
    t = re.sub(r'MODELS = \[.*?\]', lambda m: models_block, t, count=1, flags=re.S)
    t = re.sub(r'OUT_NAME = \{.*?\}', lambda m: f'OUT_NAME = {{"{d}": "{outname}"}}', t, count=1, flags=re.S)
    open(p,"w",encoding="utf-8").write(t)
    print("patched", d)
