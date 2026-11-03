# -*- coding: utf-8 -*-
import os, json
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import hf_hub_download, HfApi

api = HfApi(endpoint="https://hf-mirror.com")

targets = [
    ("MiniMaxAI/MiniMax-M2.1", "minimax_m2_1"),
    ("MiniMaxAI/MiniMax-M2.5", "minimax_m2_5"),
    ("MiniMaxAI/MiniMax-M2.7", "minimax_m2_7"),
    ("MiniMaxAI/MiniMax-M3.1-Flash", "minimax_m3_1_flash"),
]

out_root = r"E:\transformers\bayon-note\MiniMax"
lines = []

for repo, tag in targets:
    try:
        files = api.list_repo_files(repo, repo_type="model")
        lines.append(f"[OK] {repo} exists, files count={len(files)}")
        has_cfg = "config.json" in files
        lines.append(f"     config.json present: {has_cfg}")
        if has_cfg:
            dst_dir = os.path.join(out_root, tag, "_tools", "_cfgpull")
            os.makedirs(dst_dir, exist_ok=True)
            p = hf_hub_download(repo, "config.json",
                                 local_dir=os.path.join(out_root, tag, "_tmp_" + tag))
            cfg_dst = os.path.join(dst_dir, "config.json")
            with open(p, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            with open(cfg_dst, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            lines.append(f"     saved -> {cfg_dst}")
            lines.append(f"     model_type={cfg.get('model_type')} arch={cfg.get('architectures')}")
    except Exception as e:
        lines.append(f"[MISS] {repo}: {type(e).__name__} {str(e)[:200]}")

print("\n".join(lines))
with open(os.path.join(out_root, "_pull_probe_new.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
