# -*- coding: utf-8 -*-
import ssl, urllib.request, re
ctx = ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
def get(url, timeout=40):
    req=urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=timeout, context=ctx).read().decode("utf-8","replace")

for url in [
  "https://hf-mirror.com/api/models?author=moonshotai&limit=300",
]:
    try:
        h=get(url)
        print("RAW len", len(h))
        # extract model ids
        ids=re.findall(r'"id":"([^"]+)"', h)
        print("count", len(ids))
        for i in sorted(set(ids)): print("  ", i)
    except Exception as e:
        print("ERR", e)
