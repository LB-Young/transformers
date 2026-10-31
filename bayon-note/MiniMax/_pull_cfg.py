# -*- coding: utf-8 -*-
import os, json, sys
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import hf_hub_download, list_repo_files, HfApi

api = HfApi(endpoint="https://hf-mirror.com")

targets = [
    ("MiniMaxAI/MiniMax-Text-01", "minimax_text_01"),
    ("MiniMaxAI/MiniMax-Text-01-hf", "minimax_text_01_hf_probe"),
    ("MiniMaxAI/MiniMax-Visual-01", "minimax_visual_01"),
    ("MiniMaxAI/MiniMax-M1", "minimax_m1"),
    ("MiniMaxAI/MiniMax-M1-80k", "minimax_m1_80k_probe"),
    ("MiniMaxAI/MiniMax-M2", "minimax_m2"),
    ("MiniMaxAI/MiniMax-M2-hf", "minimax_m2_hf_probe"),
    ("MiniMaxAI/MiniMax-Text-04", "minimax_text_04"),
    ("MiniMaxAI/MiniMax-M3", "minimax_m3_probe"),
    ("MiniMaxAI/MiniMax-M3-VL", "minimax_m3_vl"),
    ("MiniMaxAI/MiniMax-VL-01", "minimax_vl01_probe"),
]

out_root = r"E:\transformers\bayon-note\MiniMax"
probe_lines = []

for repo, tag in targets:
    try:
        files = api.list_repo_files(repo, repo_type="model")
        probe_lines.append(f"[OK] {repo} exists, files count={len(files)}")
        has_cfg = "config.json" in files
        probe_lines.append(f"     config.json present: {has_cfg}; sample: {files[:12]}")
        if has_cfg:
            dst_dir = os.path.join(out_root, tag, "_tools", "_cfgpull")
            os.makedirs(dst_dir, exist_ok=True)
            p = hf_hub_download(repo, "config.json", local_dir=os.path.join(out_root, tag, "_tmp_" + tag))
            cfg_dst = os.path.join(dst_dir, "config.json")
            with open(p, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            with open(cfg_dst, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            probe_lines.append(f"     saved -> {cfg_dst}")
            probe_lines.append(f"     model_type={cfg.get('model_type')} arch={cfg.get('architectures')}")
    except Exception as e:
        probe_lines.append(f"[MISS] {repo}: {type(e).__name__} {str(e)[:160]}")

print("\n".join(probe_lines))
with open(os.path.join(out_root, "_pull_probe.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(probe_lines))
