# -*- coding: utf-8 -*-
"""用 Chrome headless 把各模型 svg_pages 渲染为 2x PNG，供 docx 插入。"""
import json
import os
import subprocess
import sys

BASE = r"E:\transformers\bayon-note"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

def main():
    with open(os.path.join(BASE, "_arch_manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    for model, entries in manifest.items():
        for e in entries:
            page = e["page"]
            out_png = os.path.join(BASE, model, "_arch", f"{e['idx']:02d}_{e['name']}.png")
            w_css = e["w"] + 32
            h_css = e["h"] + 32
            cmd = [
                CHROME,
                "--headless=new", "--disable-gpu", "--hide-scrollbars",
                "--force-device-scale-factor=2",
                f"--window-size={w_css},{h_css}",
                f"--screenshot={out_png}",
                "file:///" + page.replace("\\", "/"),
            ]
            r = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=90)
            ok = os.path.exists(out_png) and os.path.getsize(out_png) > 5000
            print(f"[{model}] {e['name']} -> {'OK' if ok else 'FAIL'} ({os.path.getsize(out_png) if os.path.exists(out_png) else 0} B)")
            if not ok:
                print("  stderr:", r.stderr[:500])

if __name__ == "__main__":
    main()
