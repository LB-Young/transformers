# -*- coding: utf-8 -*-
"""Pull official config.json / generation_config.json / tokenizer_config.json from hf-mirror for DeepSeek repos."""
import os, sys, io, shutil
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import hf_hub_download

V1 = r"E:\transformers\bayon-note\DeepSeek\deepseek_v1\_tools\_cfgpull"
V2 = r"E:\transformers\bayon-note\DeepSeek\deepseek_v2\_tools\_cfgpull"

TASKS = [
    ("deepseek-ai/deepseek-llm-7b-base", V1),
    ("deepseek-ai/deepseek-llm-67b-base", V1),
    ("deepseek-ai/DeepSeek-V2", V2),
    ("deepseek-ai/DeepSeek-V2-Chat", V2),
    ("deepseek-ai/DeepSeek-V2.5", V2),
]
FILES = ["config.json", "generation_config.json", "tokenizer_config.json"]

for repo, outdir in TASKS:
    tag = repo.split("/")[-1]
    for fn in FILES:
        try:
            p = hf_hub_download(repo_id=repo, filename=fn)
            dst = os.path.join(outdir, f"{tag}.{fn}")
            shutil.copyfile(p, dst)
            print("OK", repo, fn, "->", dst)
        except Exception as e:
            print("FAIL", repo, fn, repr(e)[:200])
print("DONE")
