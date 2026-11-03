# -*- coding: utf-8 -*-
"""Diff K2.5 vs K2.6 vs K2.7-Code configs; print K3 layer-list stats."""
import json, os

BASE = r"E:\transformers\bayon-note\Kimi"

def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)

k25 = load(os.path.join(BASE, "kimi_k2_5", "_tools", "_cfgpull", "config.Kimi-K2.5.json"))
k26 = load(os.path.join(BASE, "kimi_k2_6", "_tools", "_cfgpull", "config.json"))
k27 = load(os.path.join(BASE, "kimi_k2_7_code", "_tools", "_cfgpull", "config.json"))
k3 = load(os.path.join(BASE, "kimi_k3", "_tools", "_cfgpull", "config.json"))

def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat(v, key + "."))
        else:
            out[key] = v
    return out

def diff(a, b, na, nb):
    fa, fb = flat(a), flat(b)
    keys = sorted(set(fa) | set(fb))
    print(f"===== DIFF {na} vs {nb} =====")
    for k in keys:
        va, vb = fa.get(k, "<MISSING>"), fb.get(k, "<MISSING>")
        if va != vb:
            print(f"  {k}: {na}={va!r}  ->  {nb}={vb!r}")
    print()

diff(k25, k26, "K2.5", "K2.6")
diff(k26, k27, "K2.6", "K2.7-Code")

# K3 stats
lc = k3["text_config"]["linear_attn_config"]
fa = lc["full_attn_layers"]
kda = lc["kda_layers"]
print("===== K3 layer stats =====")
print("num_hidden_layers:", k3["text_config"]["num_hidden_layers"])
print("full_attn_layers count:", len(fa), fa)
print("kda_layers count:", len(kda))
print("covered:", len(set(fa) | set(kda)), " dense layer0, remaining:", set(range(0, k3["text_config"]["num_hidden_layers"])) - set(fa) - set(kda))
