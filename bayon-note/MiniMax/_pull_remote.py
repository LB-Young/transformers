# -*- coding: utf-8 -*-
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import hf_hub_download

out = r"E:\transformers\bayon-note\MiniMax\_remote_src"
os.makedirs(out, exist_ok=True)
files = [
    ("MiniMaxAI/MiniMax-M2", "configuration_minimax_m2.py"),
    ("MiniMaxAI/MiniMax-Text-01", "configuration_minimax_text_01.py"),
    ("MiniMaxAI/MiniMax-M1", "configuration_minimax_m1.py"),
    ("MiniMaxAI/MiniMax-VL-01", "configuration_minimax_vl_01.py"),
]
for repo, fn in files:
    try:
        p = hf_hub_download(repo, fn, local_dir=out)
        print("OK", repo, fn, "->", p)
    except Exception as e:
        print("MISS", repo, fn, str(e)[:120])
