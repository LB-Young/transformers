# -*- coding: utf-8 -*-
"""Driver: build docx for all 5 MiniMax model dirs via one md2docx core."""
import os, sys, io, importlib.util
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = r"E:\transformers\bayon-note\MiniMax"
core_path = os.path.join(ROOT, "minimax_text_01", "_tools", "md2docx.py")
spec = importlib.util.spec_from_file_location("md2docx_core", core_path)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

JOBS = [
 ("minimax_text_01", "MiniMax-Text-01 家族架构详解",
  "MiniMaxForCausalLM · 456B 总参/45.9B 激活 · 80层 · Lightning:Softmax=7:1 · 32专家top2"),
 ("minimax_visual_01", "MiniMax-Visual-01（VL-01）家族架构详解",
  "MiniMaxVL01ForConditionalGeneration · Text-01 文本塔 + CLIP 视觉塔 · patch14/336 · 动态分辨率"),
 ("minimax_m1", "MiniMax-M1 家族架构详解",
  "MiniMaxM1ForCausalLM · 456B/45.9B（同 Text-01 架构）· RL 长链推理 · 1M 上下文"),
 ("minimax_m2", "MiniMax-M2 家族架构详解",
  "MiniMaxM2ForCausalLM · 约230B/9.8B · 62层全Softmax · 256专家top8 · QK-Norm · MTP · FP8"),
 ("minimax_m3_vl", "MiniMax-M3（M3-VL）家族架构详解",
  "MiniMaxM3SparseForConditionalGeneration · 60层 · 3Dense+57MoE · MSA块稀疏 · 128专家+1共享 · 原生多模态"),
]
OUT_NAME = {
 "minimax_text_01": "MiniMax-Text-01",
 "minimax_visual_01": "MiniMax-Visual-01",
 "minimax_m1": "MiniMax-M1",
 "minimax_m2": "MiniMax-M2",
 "minimax_m3_vl": "MiniMax-M3",
}

for model, title, subtitle in JOBS:
    d = os.path.join(ROOT, model)
    md_path = os.path.join(d, "index.md")
    img_dir = os.path.join(d, "_arch")
    out = os.path.join(d, f"{OUT_NAME[model]}-家族架构详解.docx")
    try:
        figs = core.build_docx(d, title, subtitle, md_path, img_dir, out)
        print(f"{model}: OK  figs={figs}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"{model}: FAILED  {e}")
