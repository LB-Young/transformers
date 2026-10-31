# -*- coding: utf-8 -*-
import json, ssl, urllib.request
ctx = ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=45, context=ctx).read()

# 1) try huggingface.co API listing
for base in ["https://huggingface.co", "https://hf-mirror.com"]:
    try:
        d = json.loads(get(f"{base}/api/models?author=MoonshotAI&limit=200&full=false").decode())
        print(f"[{base}] count={len(d)}")
        for m in sorted(d, key=lambda x: x["id"]):
            print("   ", m["id"])
        break
    except Exception as e:
        print(f"[{base}] API fail: {e}")

# 2) probe candidate configs
cands = [
    "MoonshotAI/Moonshot-8B",
    "MoonshotAI/Moonshot-8B-Base",
    "MoonshotAI/Moonshot-32B-A32B",
    "MoonshotAI/Kimi-K2-0905",
    "MoonshotAI/Kimi-K2",
]
for repo in cands:
    for base in ["https://hf-mirror.com", "https://huggingface.co"]:
        url = f"{base}/{repo}/resolve/main/config.json"
        try:
            raw = get(url)
            print(f"OK  {url}  bytes={len(raw)}")
            break
        except Exception as e:
            print(f"ERR {url}  -> {e}")
