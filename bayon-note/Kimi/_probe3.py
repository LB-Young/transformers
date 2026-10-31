# -*- coding: utf-8 -*-
import json, ssl, urllib.request, os
ctx = ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
def get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=timeout, context=ctx).read()

cands = [
 "moonshotai/Moonshot-8B","moonshotai/Moonshot-8B-Instruct","moonshotai/Moonshot-8B-Base",
 "moonshotai/Moonshot-32B-A32B","moonshotai/Moonshot-32B-A32B-Instruct",
 "moonshotai/Moonshot-32B","moonshotai/Moonshot-32B-Base",
 "moonshotai/Moonlight-16B-A3B","moonshotai/Kimi-K2-Base",
]
for repo in cands:
    url=f"https://hf-mirror.com/{repo}/resolve/main/config.json"
    try:
        raw=get(url); print(f"OK  {repo} bytes={len(raw)}")
    except Exception as e:
        print(f"ERR {repo} -> {e}")

# save the real K2-0905 config
out=r"E:\transformers\bayon-note\Kimi\kimi_k2_0905\_tools\_cfgpull\config.json"
raw=get("https://hf-mirror.com/moonshotai/Kimi-K2-Instruct-0905/resolve/main/config.json")
open(out,"wb").write(raw)
print("saved K2-0905 config", len(raw))
print(json.dumps(json.loads(raw.decode()), ensure_ascii=False, indent=1))
