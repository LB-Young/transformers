# -*- coding: utf-8 -*-
"""Create Kimi dir tree and pull official config.json from hf-mirror."""
import os, json, urllib.request, ssl

ROOT = r"E:\transformers\bayon-note\Kimi"
REPOS = {
    "moonshot_8b": "MoonshotAI/Moonshot-8B",
    "moonshot_32b_a32b": "MoonshotAI/Moonshot-32B-A32B",
    "kimi_k2_0905": "MoonshotAI/Kimi-K2-0905",
}
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

for sub, repo in REPOS.items():
    d = os.path.join(ROOT, sub)
    for sd in ["_arch", "_shots", os.path.join("_tools", "_cfgpull")]:
        os.makedirs(os.path.join(d, sd), exist_ok=True)
    url = f"https://hf-mirror.com/{repo}/resolve/main/config.json"
    out = os.path.join(d, "_tools", "_cfgpull", "config.json")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
            data = r.read()
        with open(out, "wb") as f:
            f.write(data)
        cfg = json.loads(data.decode("utf-8"))
        print(f"=== {sub} ({repo}) OK bytes={len(data)} ===")
        print(json.dumps(cfg, ensure_ascii=False, indent=1)[:3000])
        print()
    except Exception as e:
        print(f"=== {sub} ({repo}) FAILED: {e} ===")
