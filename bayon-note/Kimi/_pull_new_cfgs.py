# -*- coding: utf-8 -*-
"""Pull K2.6 / K2.7-Code / K3 configs + K3 tech report PDF from hf-mirror/github."""
import os, json, urllib.request, ssl

ROOT = r"E:\transformers\bayon-note\Kimi"
REPOS = {
    "kimi_k2_6": "moonshotai/Kimi-K2.6",
    "kimi_k2_7_code": "moonshotai/Kimi-K2.7-Code",
    "kimi_k3": "moonshotai/Kimi-K3",
}
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120, context=ctx) as r:
        return r.read()

for sub, repo in REPOS.items():
    d = os.path.join(ROOT, sub)
    os.makedirs(os.path.join(d, "_tools", "_cfgpull"), exist_ok=True)
    url = f"https://hf-mirror.com/{repo}/resolve/main/config.json"
    out = os.path.join(d, "_tools", "_cfgpull", "config.json")
    try:
        data = get(url)
        with open(out, "wb") as f:
            f.write(data)
        cfg = json.loads(data.decode("utf-8"))
        print(f"=== {sub} ({repo}) OK bytes={len(data)} ===")
        print(json.dumps(cfg, ensure_ascii=False, indent=1))
        print()
    except Exception as e:
        print(f"=== {sub} ({repo}) CONFIG FAILED: {e} ===")

# K3 tech report PDF
pdf_url = "https://raw.githubusercontent.com/MoonshotAI/Kimi-K3/3cb39dfd32e51c3328e2e4b4af21341247d06c43/k3_tech_report.pdf"
pdf_out = os.path.join(ROOT, "kimi_k3", "_tools", "_cfgpull", "k3_tech_report.pdf")
try:
    data = get(pdf_url)
    with open(pdf_out, "wb") as f:
        f.write(data)
    print(f"=== K3 PDF OK bytes={len(data)} -> {pdf_out} ===")
except Exception as e:
    print(f"=== K3 PDF FAILED: {e} ===")
