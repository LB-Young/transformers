# -*- coding: utf-8 -*-
import ssl, urllib.request, json, re
ctx = ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
def get(url, timeout=30):
    req=urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=timeout, context=ctx).read().decode("utf-8","replace")

for q in ["Moonshot-8B","Moonshot-32B","Moonshot-8","Moonshot 8B","Moonshot-32B-A32B"]:
    try:
        h=get(f"https://hf-mirror.com/api/models?search={urllib.parse.quote(q)}&limit=30")
        ids=re.findall(r'"id":"([^"]+)"', h)
        print(f"== search '{q}': {len(ids)} ==")
        for i in ids: print("   ", i)
    except Exception as e:
        print("ERR", q, e)

# save reference configs
import os
def save(repo, subdir):
    url=f"https://hf-mirror.com/{repo}/resolve/main/config.json"
    try:
        raw=get(url); os.makedirs(os.path.dirname(subdir), exist_ok=True)
        open(subdir,"wb").write(raw); print("saved", repo, len(raw))
    except Exception as e:
        print("ERR save", repo, e)

save("moonshotai/Moonlight-16B-A3B", r"E:\transformers\bayon-note\Kimi\moonshot_32b_a32b\_tools\_cfgpull\Moonlight-16B-A3B.config.json")
save("moonshotai/Kimi-K2-Base", r"E:\transformers\bayon-note\Kimi\kimi_k2_0905\_tools\_cfgpull\Kimi-K2-Base.config.json")
