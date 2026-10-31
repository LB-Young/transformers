# -*- coding: utf-8 -*-
import json, ssl, urllib.request
ctx = ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
def get(url, timeout=40):
    req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=timeout, context=ctx).read()

cands = [
    "moonshotai/Kimi-K2",
    "moonshotai/Kimi-K2-0905",
    "moonshotai/Kimi-K2-Instruct-0905",
    "moonshotai/Moonshot-8B",
    "moonshotai/Moonshot-32B-A32B",
    "moonshotai/Moonshot-32B",
]
for repo in cands:
    url = f"https://hf-mirror.com/{repo}/resolve/main/config.json"
    try:
        raw = get(url); print(f"OK  hf-mirror {repo} bytes={len(raw)}")
    except Exception as e:
        print(f"ERR hf-mirror {repo} -> {e}")

print("---- ModelScope ----")
for repo in cands:
    # ModelScope file resolve
    url = f"https://modelscope.cn/api/v1/models/{repo}/repo?FilePath=config.json"
    try:
        raw = get(url); txt=raw.decode("utf-8","replace")
        head = txt[:200].replace("\n"," ")
        print(f"?? MS {repo} bytes={len(raw)} head={head}")
    except Exception as e:
        print(f"ERR MS {repo} -> {e}")
