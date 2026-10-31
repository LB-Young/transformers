# -*- coding: utf-8 -*-
import json, ssl, urllib.request
ctx = ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
for url in [
    "https://hf-mirror.com/api/models?author=MoonshotAI&limit=200",
]:
    req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
    data = json.loads(urllib.request.urlopen(req, timeout=60, context=ctx).read().decode())
    print("URL", url, "count", len(data))
    for m in sorted(data, key=lambda x: x["id"]):
        print(m["id"])
