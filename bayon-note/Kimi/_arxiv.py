# -*- coding: utf-8 -*-
import ssl, urllib.request, re
ctx = ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
def get(url, timeout=40):
    req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=timeout, context=ctx).read().decode("utf-8","replace")

for aid in ["2501.08504", "2508.05149", "2507.20534", "2501.12599"]:
    for host in ["https://arxiv.org/abs/", "https://export.arxiv.org/abs/"]:
        try:
            html = get(host+aid)
            t = re.search(r"<title>(.*?)</title>", html, re.S)
            ab = re.search(r'<blockquote class="abstract[^>]*>(.*?)</blockquote>', html, re.S)
            print(f"=== {host}{aid} ===")
            print("TITLE:", (t.group(1).strip() if t else "?")[:200])
            if ab:
                txt = re.sub(r"<[^>]+>","",ab.group(1)).strip()
                print("ABS:", txt[:400])
            print()
            break
        except Exception as e:
            print(f"ERR {host}{aid}: {e}")
