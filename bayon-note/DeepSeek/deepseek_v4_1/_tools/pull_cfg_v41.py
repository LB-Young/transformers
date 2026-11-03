# -*- coding: utf-8 -*-
"""Pull official config.json / generation_config.json / tokenizer_config.json from hf-mirror for DeepSeek-V4.1-Flash."""
import os, sys, io, shutil
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import hf_hub_download

OUT = r"E:\transformers\bayon-note\DeepSeek\deepseek_v4_1\_tools\_cfgpull"
os.makedirs(OUT, exist_ok=True)
REPO = "deepseek-ai/DeepSeek-V4.1-Flash"
FILES = ["config.json", "generation_config.json", "tokenizer_config.json"]
tag = REPO.split("/")[-1]
for fn in FILES:
    try:
        p = hf_hub_download(repo_id=REPO, filename=fn)
        dst = os.path.join(OUT, f"{tag}.{fn}")
        shutil.copyfile(p, dst)
        print("OK", REPO, fn, "->", dst)
    except Exception as e:
        print("FAIL", REPO, fn, repr(e)[:300])
print("DONE")
