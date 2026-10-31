# -*- coding: utf-8 -*-
import ssl, urllib.request, json
ctx = ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
def get(url, timeout=40):
    req=urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=timeout, context=ctx).read().decode("utf-8","replace")

urls = [
 "huggingface.co/moonshotai/Moonshot-8B",
 "huggingface.co/moonshotai/Moonshot-32B-A32B",
 "huggingface.co/moonshotai/Moonshot-8B-Base",
 "modelscope.cn/models/moonshotai/Moonshot-8B",
]
for u in urls:
    api=f"http://archive.org/wayback/available?url={u}"
    try:
        print("===", u)
        print("  ", get(api)[:500])
    except Exception as e:
        print("ERR", u, e)
