# -*- coding: utf-8 -*-
"""一次性校验：launch.json(JSONC) 可解析 + Qwen 配置与 debug_qwen.py 注册表一致。"""
import json
import os
import sys

sys.path.insert(0, r"E:\transformers\bayon-note\Qwen")
from debug_qwen import MODEL_REGISTRY  # noqa: E402

raw = open(r"E:\transformers\.vscode\launch.json", encoding="utf-8").read()
clean = [ln for ln in raw.splitlines() if ln.strip() and not ln.strip().startswith("//")]
cfg = json.loads("\n".join(clean))

qwen_confs = [c for c in cfg["configurations"] if c["name"].startswith("Qwen")]
print(f"configurations total={len(cfg['configurations'])}  qwen={len(qwen_confs)}")

problems = []
for c in qwen_confs:
    args = c.get("args", [])
    if "--model" in args:
        key = args[args.index("--model") + 1]
        if key not in MODEL_REGISTRY:
            problems.append(f"[{c['name']}] unknown model key: {key}")
        elif not os.path.isdir(MODEL_REGISTRY[key]["dir"]):
            problems.append(f"[{c['name']}] missing dir: {MODEL_REGISTRY[key]['dir']}")
    if not c.get("python") or not os.path.isfile(c["python"]):
        problems.append(f"[{c['name']}] bad python: {c.get('python')}")
    if not c.get("program") or not os.path.isfile(c["program"]):
        problems.append(f"[{c['name']}] bad program: {c.get('program')}")
    if "--image" in args:
        img = args[args.index("--image") + 1]
        if not os.path.isfile(img):
            problems.append(f"[{c['name']}] missing image: {img}")

for key, reg in MODEL_REGISTRY.items():
    if not os.path.isdir(reg["dir"]):
        problems.append(f"[registry] {key} dir missing: {reg['dir']}")

print("PROBLEMS:", problems if problems else "NONE - all OK")
