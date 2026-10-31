# -*- coding: utf-8 -*-
"""用 Chrome headless 把 _arch_html/*.html 重新截图成 _arch/*.png（2x 分辨率）"""
import os, re, io, sys, subprocess
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE = r"E:\transformers\bayon-note\Gemma"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
MODELS = ["gemma-1", "gemma-2", "gemma-3", "gemma-3n", "gemma-4"]

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, timeout=120)
    return r.returncode, (r.stderr or r.stdout).decode("utf-8", "ignore")

ok = fail = 0
for m in MODELS:
    d = os.path.join(BASE, m)
    arch_html = os.path.join(d, "_arch_html")
    arch = os.path.join(d, "_arch")
    if not os.path.isdir(arch_html):
        continue
    for fn in sorted(os.listdir(arch_html)):
        if not fn.endswith(".html"):
            continue
        src = os.path.join(arch_html, fn)
        # 从 html 内 viewBox 取逻辑尺寸
        html = open(src, encoding="utf-8").read()
        vb = re.search(r'viewBox="[\d.]+ [\d.]+ ([\d.]+) ([\d.]+)"', html)
        if not vb:
            continue
        w, h = int(float(vb.group(1))), int(float(vb.group(2)))
        out = os.path.join(arch, fn[:-5] + ".png")
        cmd = [CHROME, "--headless", "--disable-gpu", "--hide-scrollbars",
               "--force-device-scale-factor=2",
               f"--window-size={w},{h}",
               f"--screenshot={out}",
               "file:///" + src.replace("\\", "/")]
        rc, err = run(cmd)
        if rc == 0 and os.path.exists(out):
            sz = os.path.getsize(out)
            print(f"OK  {m}/{fn[:-5]}.png  {w}x{h} -> {os.path.getsize(out)//1024}KB")
            ok += 1
        else:
            print(f"FAIL {m}/{fn} rc={rc} {err[:200]}")
            fail += 1
print(f"done ok={ok} fail={fail}")
