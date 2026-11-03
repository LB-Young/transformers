# -*- coding: utf-8 -*-
"""Try GitHub proxy mirrors / jsdelivr for the K3 tech report PDF."""
import os, urllib.request, ssl, time

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

out = r"E:\transformers\bayon-note\Kimi\kimi_k3\_tools\_cfgpull\k3_tech_report.pdf"
SHA = "3cb39dfd32e51c3328e2e4b4af21341247d06c43"
RAW = f"https://raw.githubusercontent.com/MoonshotAI/Kimi-K3/{SHA}/k3_tech_report.pdf"

urls = [
    f"https://cdn.jsdelivr.net/gh/MoonshotAI/Kimi-K3@{SHA}/k3_tech_report.pdf",
    f"https://ghfast.top/{RAW}",
    f"https://ghproxy.net/{RAW}",
    f"https://mirror.ghproxy.com/{RAW}",
    f"https://gh.llkk.cc/{RAW}",
    f"https://gh-proxy.com/{RAW}",
]

for url in urls:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=90, context=ctx) as r:
            data = r.read()
        if len(data) < 100000:
            print(f"too small ({len(data)}), skip: {url}")
            continue
        with open(out, "wb") as f:
            f.write(data)
        print(f"OK bytes={len(data)} <- {url}")
        raise SystemExit(0)
    except Exception as e:
        print(f"FAILED: {url} :: {e}")
        time.sleep(2)
print("ALL PROXY VARIANTS FAILED")
