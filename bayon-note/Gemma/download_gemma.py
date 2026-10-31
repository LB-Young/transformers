# -*- coding: utf-8 -*-
"""download_gemma.py —— 从 ModelScope(魔搭) 下载五个 Gemma 最小模型

用法：python download_gemma.py
下载目标：E:\\models\\model\\<模型名>\\
来源：ModelScope（国内直连，无需 HuggingFace token）
"""
import sys

from modelscope import snapshot_download

# (ModelScope 模型 ID, 本地输出目录)
MODELS = [
    ("LLM-Research/gemma-2b-it",      r"E:\models\model\gemma-2b-it"),      # Gemma 1 最小 2B
    ("LLM-Research/gemma-2-2b-it",    r"E:\models\model\gemma-2-2b-it"),    # Gemma 2 最小 2B
    ("google/gemma-3-270m-it",        r"E:\models\model\gemma-3-270m-it"),  # Gemma 3 最小 270M
    ("google/gemma-3n-E2B-it",        r"E:\models\model\gemma-3n-E2B-it"),  # Gemma 3n 最小 E2B(实际5B)
    ("google/gemma-4-E2B-it",         r"E:\models\model\gemma-4-E2B-it"),   # Gemma 4 最小 E2B(实际~5B)
]


def main():
    failed = []
    for model_id, local_dir in MODELS:
        print(f"=== 开始下载 {model_id} -> {local_dir}", flush=True)
        try:
            snapshot_download(model_id, local_dir=local_dir)
            print(f"=== 完成 {model_id}", flush=True)
        except Exception as e:
            print(f"=== 失败 {model_id}: {e}", flush=True)
            failed.append(model_id)
    print("==== 全部结束 ====")
    if failed:
        print("失败的模型:", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
