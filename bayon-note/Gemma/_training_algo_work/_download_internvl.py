# -*- coding: utf-8 -*-
"""从 ModelScope 下载 InternVL 各版本最小尺寸模型到 E:\\models\\model。"""
import os
import sys
import time

MODEL_DIR = r"E:\models\model"
LOG = r"E:\transformers\bayon-note\Gemma\_training_algo_work\_download_internvl.log"

MODELS = [
    # (版本, ModelScope ID, 目标目录名)
    ("1.5", "OpenGVLab/Mini-InternVL-Chat-2B-V1-5", "Mini-InternVL-Chat-2B-V1-5"),
    ("2.0", "OpenGVLab/InternVL2-1B",                "InternVL2-1B"),
    ("2.5", "OpenGVLab/InternVL2_5-1B",              "InternVL2_5-1B"),
    ("3.0", "OpenGVLab/InternVL3-1B-hf",             "InternVL3-1B-hf"),
    ("3.5", "OpenGVLab/InternVL3_5-1B-HF",           "InternVL3_5-1B-HF"),
]

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def main():
    from modelscope import snapshot_download
    os.makedirs(MODEL_DIR, exist_ok=True)
    log("=== 开始下载 InternVL 系列（共 %d 个）===" % len(MODELS))
    for ver, mid, dirname in MODELS:
        dest = os.path.join(MODEL_DIR, dirname)
        if os.path.isdir(dest) and any(fname.endswith(".safetensors") for _, _, fname in os.walk(dest) for fname in []):
            pass
        # 简单判定：目录存在且非空则认为已下载过
        if os.path.isdir(dest) and len(os.listdir(dest)) > 3:
            log(f"[SKIP] v{ver} {mid} -> 已存在 {dest}（{len(os.listdir(dest))} 个文件）")
            continue
        log(f"[开始] v{ver} {mid} -> {dest}")
        for attempt in range(1, 4):
            try:
                snapshot_download(model_id=mid, local_dir=dest)
                log(f"[完成] v{ver} {mid} 下载完成")
                break
            except Exception as e:
                log(f"[失败] v{ver} {mid} 第{attempt}次: {type(e).__name__}: {str(e)[:120]}")
                if attempt < 3:
                    time.sleep(5)
        else:
            log(f"[放弃] v{ver} {mid} 三次尝试均失败")
    log("=== 全部结束 ===")

if __name__ == "__main__":
    main()
