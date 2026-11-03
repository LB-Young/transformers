# -*- coding: utf-8 -*-
"""Retry K3 tech report PDF download with several URL variants."""
import os, urllib.request, ssl, time

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

out = r"E:\transformers\bayon-note\Kimi\kimi_k3\_tools\_cfgpull\k3_tech_report.pdf"

urls = [
    "https://raw.githubusercontent.com/MoonshotAI/Kimi-K3/3cb39dfd32e51c3328e2e4b4af21341247d06c43/k3_tech_report.pdf",
    "https://github.com/MoonshotAI/Kimi-K3/raw/3cb39dfd32e51c3328e2e4b4af21341247d06c43/k3_tech_report.pdf",
    "https://hf-mirror.com/moonshotai/Kimi-K3/resolve/main/k3_tech_report.pdf",
]

for url in urls:
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120, context=ctx) as r:
                data = r.read()
            if len(data) < 10000:
                print(f"too small ({len(data)}), skip: {url}")
                break
            with open(out, "wb") as f:
                f.write(data)
            print(f"OK bytes={len(data)} <- {url}")
            raise SystemExit(0)
        except Exception as e:
            print(f"attempt {attempt+1} FAILED: {url} :: {e}")
            time.sleep(3)
print("ALL URL VARIANTS FAILED")
