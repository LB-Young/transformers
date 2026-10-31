#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
debug_gemma_multi_mode.py —— Gemma 多模态（图像输入）源码调试入口脚本
=====================================================================

用途
----
    在本地 conda 环境（transformers-gemma）中，用 E:\\transformers 的源码
    加载支持图像输入的多模态 Gemma 模型（Gemma 3 1B+/Gemma 3n/Gemma 4/Gemma 4-unified），
    让模型"看图说话"，并支持断点调试进入视觉编码器源码：
      1. 只打印 config（--config-only）
      2. 打印模型结构 + 每层参数数量（--list-model / --params）
      3. 逐层追踪前向传播的输入/输出 shape，含视觉编码器（--trace-shapes）
      4. 在指定模块自动打断点进入 pdb（--breakpoints / --debug）
      5. 定位每个模块对应的 transformers 源码文件（--locate）
      6. 多模态生成验证：输入一张或多张图片 + 文本指令（默认）

与 debug_gemma.py 的区别
------------------------
    * debug_gemma.py 只走 tokenizer -> model.generate（纯文本）；
    * 本脚本走 processor -> model.generate（文本 + 图像），可以断点进
      modeling_gemma3n.py / modeling_gemma4.py 的 Gemma3nVisionTransformer /
      Gemma4VisionTransformer 等视觉模块。

支持图像输入的模型（本机五个模型里只有下面两类是多模态）
--------------------------------------------------------
    gemma-3n  : Gemma3nForConditionalGeneration  （本地 gemma-3n-E2B-it）
    gemma-4   : Gemma4ForConditionalGeneration   （本地 gemma-4-E2B-it）
    gemma-3   : 仅 1b/4b/12b/27b 多模态（270m 是 gemma3_text 纯文本，不支持）
    gemma-4-unified : 12b 统一多模态（含音频）

典型用法
--------
    # 用本地权重 + 离线，让 Gemma 3n 描述一张图片
    python debug_gemma_multi_mode.py --model gemma-3n --local-dir E:\\models\\model\\gemma-3n-E2B-it --offline \
        --image C:\\Users\\LBY\\Downloads\\Gemma模型版本数量.png \
        --prompt "请详细介绍这张图片的结构和内容。" --max-new-tokens 128

    # 追踪视觉编码器每个模块的 shape
    python debug_gemma_multi_mode.py --model gemma-4 --local-dir E:\\models\\model\\gemma-4-E2B-it --offline \
        --image 图片.png --trace-shapes --depth 2

    # 在视觉 transformer 打断点，进入 pdb 交互调试
    python debug_gemma_multi_mode.py --model gemma-3n --local-dir E:\\models\\model\\gemma-3n-E2B-it --offline \
        --image 图片.png --breakpoints vision

    # 多张图片
    python debug_gemma_multi_mode.py --model gemma-4 --local-dir E:\\models\\model\\gemma-4-E2B-it --offline \
        --image 图1.png,图2.png --prompt "对比这两张图。"
"""

import argparse
import json
import os
import sys
import traceback

import torch
from PIL import Image

# ---------------------------------------------------------------------------
# 多模态模型注册表：模型名 -> 条件生成类 + HF ID 建议值
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "gemma-3": {
        "cls": "Gemma3ForConditionalGeneration",       # 多模态条件生成（1b/4b/12b/27b）
        "variants": {
            "1b":  "google/gemma-3-1b-it",
            "4b":  "google/gemma-3-4b-it",
            "12b": "google/gemma-3-12b-it",
            "27b": "google/gemma-3-27b-it",
        },
        "default_variant": "1b",
        "note": "270m 是纯文本（gemma3_text），不支持图像；本注册表只列多模态变体",
    },
    "gemma-3n": {
        "cls": "Gemma3nForConditionalGeneration",      # 移动端多模态
        "variants": {
            "e2b": "google/gemma-3n-E2B-it",
            "e4b": "google/gemma-3n-E4B-it",
        },
        "default_variant": "e2b",
    },
    "gemma-4": {
        "cls": "Gemma4ForConditionalGeneration",       # 多模态条件生成
        "variants": {
            "e2b":     "google/gemma-4-E2B-it",
            "e4b":     "google/gemma-4-E4B-it",
            "31b":     "google/gemma-4-31B-it",
            "26b-a4b": "google/gemma-4-26B-A4B-it",
        },
        "default_variant": "e2b",
    },
    "gemma-4-unified": {
        "cls": "Gemma4UnifiedForConditionalGeneration", # 统一多模态（图像+音频+视频）
        "variants": {
            "12b": "google/gemma-4-12B-unified-it",
        },
        "default_variant": "12b",
    },
}

# 纯文本模型提示（本机已下载的纯文本目录）
TEXT_ONLY_HINTS = {
    "gemma-1": "Gemma 1 是纯文本 LLM（GemmaForCausalLM），不支持图像输入。",
    "gemma-2": "Gemma 2 是纯文本 LLM（Gemma2ForCausalLM），不支持图像输入。",
    "gemma-3-270m": "Gemma 3 270M 是纯文本版（model_type=gemma3_text），不支持图像；请用 1b/4b/12b/27b。",
}


# ---------------------------------------------------------------------------
# 简单的 tee 日志
# ---------------------------------------------------------------------------
class Tee:
    def __init__(self, log_path=None):
        self.file = open(log_path, "w", encoding="utf-8") if log_path else None

    def emit(self, text):
        print(text)
        if self.file:
            self.file.write(text + "\n")
            self.file.flush()

    def close(self):
        if self.file:
            self.file.close()


# ---------------------------------------------------------------------------
# shape 描述
# ---------------------------------------------------------------------------
def describe_shape(obj, max_items=6):
    if obj is None:
        return "None"
    if isinstance(obj, torch.Tensor):
        return str(tuple(obj.shape))
    if isinstance(obj, (tuple, list)):
        parts = [describe_shape(x) for x in obj[:max_items]]
        if len(obj) > max_items:
            parts.append("...")
        return "(" + ", ".join(parts) + ")"
    if isinstance(obj, dict):
        items = list(obj.items())[:max_items]
        return "{" + ", ".join(f"{k}:{describe_shape(v)}" for k, v in items) + "}"
    return type(obj).__name__


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Gemma 多模态（图像输入）源码调试入口脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例见文件头注释。",
    )
    # ---- 模型选择 ----
    p.add_argument("--model", choices=list(MODEL_REGISTRY.keys()),
                   help="多模态模型：gemma-3(1b+) / gemma-3n / gemma-4 / gemma-4-unified")
    p.add_argument("--variant", default=None)
    p.add_argument("--model-id", default=None,
                   help="直接指定 HF 模型 ID 或本地目录（优先级最高）")
    p.add_argument("--local-dir", default=None,
                   help="本地权重目录（如 E:\\models\\model\\gemma-3n-E2B-it）")
    # ---- 运行环境 ----
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--offline", action="store_true",
                   help="离线模式（HF_HUB_OFFLINE=1），配合 --local-dir 使用")
    # ---- 多模态输入 ----
    p.add_argument("--image", default=None, required=True,
                   help="图片路径，逗号分隔可传多张（必填）")
    p.add_argument("--prompt", default="请详细介绍这张图片的结构和内容，包括画面元素、布局与传递的信息。",
                   help="文本指令")
    p.add_argument("--max-new-tokens", type=int, default=128,
                   help="生成的最大新 token 数")
    # ---- 调试动作（可叠加）----
    p.add_argument("--config-only", action="store_true")
    p.add_argument("--list-model", action="store_true")
    p.add_argument("--params", action="store_true")
    p.add_argument("--trace-shapes", action="store_true")
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--breakpoints", default=None,
                   help="逗号分隔的模块名关键字，命中即 breakpoint() 进入 pdb，如 vision,attention")
    p.add_argument("--debug", action="store_true",
                   help="在模型前向调用前 breakpoint() 进入 pdb")
    p.add_argument("--locate", action="store_true")
    # ---- 其他 ----
    p.add_argument("--log", default=None)
    return p.parse_args(argv)


def resolve_model_id(args):
    if args.model_id:
        return args.model_id, None
    if args.local_dir:
        if args.model:
            return args.local_dir, MODEL_REGISTRY[args.model]["cls"]
        return args.local_dir, None
    reg = MODEL_REGISTRY[args.model]
    variant = args.variant or reg["default_variant"]
    if variant not in reg["variants"]:
        raise SystemExit(f"[错误] 模型 {args.model} 没有规格 {variant}，可选：{list(reg['variants'])}")
    return reg["variants"][variant], reg["cls"]


# ---------------------------------------------------------------------------
# 多模态能力检查：根据 config 判断该模型是否真的支持图像
# ---------------------------------------------------------------------------
def check_vision_support(config, log):
    model_type = getattr(config, "model_type", "")
    vision_config = getattr(config, "vision_config", None)
    if vision_config is not None:
        log.emit(f"[视觉] 检测到 vision_config（{vision_config.model_type}）→ 支持图像输入")
        return
    # 没有 vision_config：逐个对照纯文本提示，给出可执行建议
    hints = list(TEXT_ONLY_HINTS.values())
    raise SystemExit(
        f"[错误] 模型 model_type={model_type} 没有 vision_config，不支持图像输入。\n"
        f"  可能原因与建议：\n"
        + "\n".join("  - " + h for h in hints)
    )


# ---------------------------------------------------------------------------
# 加载组件
# ---------------------------------------------------------------------------
def load_components(model_id, cls_hint, args, log):
    from transformers import AutoConfig, AutoProcessor

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    log.emit(f"[加载] config        <- {model_id}")
    config = AutoConfig.from_pretrained(model_id)
    log.emit(f"[配置] architecture  = {getattr(config, 'architectures', None)}")
    log.emit(f"[配置] model_type    = {config.model_type}")
    if args.config_only:
        log.emit(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))
        return None, None, None

    # 多模态能力检查（在下载权重前拦截纯文本模型）
    check_vision_support(config, log)

    log.emit(f"[加载] processor     <- {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)
    log.emit(f"[配置] processor 类  = {processor.__class__.__module__}.{processor.__class__.__name__}")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    device_map = args.device if args.device != "auto" else "auto"

    if cls_hint:
        model_cls = getattr(__import__("transformers", fromlist=[cls_hint]), cls_hint)
        log.emit(f"[加载] 显式类       = {model_cls.__module__}.{model_cls.__name__}")
        model = model_cls.from_pretrained(
            model_id, dtype=dtype, device_map=device_map,
            low_cpu_mem_usage=True, trust_remote_code=False,
        )
    else:
        from transformers import AutoModelForImageTextToText
        log.emit("[加载] 使用 AutoModelForImageTextToText 自动映射")
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=dtype, device_map=device_map,
            low_cpu_mem_usage=True, trust_remote_code=False,
        )
        log.emit(f"[加载] 实际类       = {model.__class__.__module__}.{model.__class__.__name__}")

    return config, processor, model


# ---------------------------------------------------------------------------
# 图片加载：PIL -> RGB
# ---------------------------------------------------------------------------
def load_images(image_arg, log):
    paths = [p.strip() for p in image_arg.split(",") if p.strip()]
    if not paths:
        raise SystemExit("[错误] --image 不能为空")
    images = []
    for path in paths:
        if not os.path.isfile(path):
            raise SystemExit(f"[错误] 图片不存在：{path}")
        img = Image.open(path).convert("RGB")
        log.emit(f"[图片] {path}  size={img.size}  mode={img.mode}")
        images.append(img)
    return images


# ---------------------------------------------------------------------------
# 动作：模块树 / shape 追踪 / 断点 / 源码定位（与 debug_gemma.py 一致）
# ---------------------------------------------------------------------------
def dump_model_tree(model, args, log):
    total = 0
    for name, module in model.named_modules():
        if not name:
            continue
        level = name.count(".")
        n_params = sum(p.numel() for p in module.parameters())
        total += n_params
        indent = "  " * level
        if args.params:
            log.emit(f"{indent}{name}  [{module.__class__.__name__}]  params={n_params:,}")
        else:
            log.emit(f"{indent}{name}  [{module.__class__.__name__}]")
    log.emit(f"\n[统计] 总参数 = {total:,}")


def install_shape_trace(model, args, log):
    handles = []

    def make_hook(name, level):
        def hook(module, args_in, output):
            indent = "  " * level
            in_shape = describe_shape(args_in[0]) if args_in else "None"
            log.emit(f"{indent}[{name}] in={in_shape}  out={describe_shape(output)}")
        return hook

    for name, module in model.named_modules():
        if not name:
            continue
        level = name.count(".")
        if level > args.depth:
            continue
        handles.append(module.register_forward_hook(make_hook(name, level)))
    log.emit(f"[追踪] 已注册 {len(handles)} 个 forward hook（depth<={args.depth}）")
    return handles


def install_breakpoints(model, keywords, log):
    handles = []

    def make_hook(name):
        def hook(module, args_in, output):
            log.emit(f">>> 命中断点：{name}（{module.__class__.__name__}）")
            breakpoint()
            return output
        return hook

    for name, module in model.named_modules():
        if any(k in name for k in keywords):
            handles.append(module.register_forward_hook(make_hook(name)))
    log.emit(f"[断点] 关键字 {keywords} 命中 {len(handles)} 个模块")
    return handles


def locate_source(model, log):
    import inspect
    seen = set()
    for name, module in model.named_modules():
        cls = module.__class__
        if cls in seen:
            continue
        seen.add(cls)
        try:
            src = inspect.getsourcefile(cls)
        except TypeError:
            src = "(builtin)"
        log.emit(f"{name or '<root>':<42} {cls.__name__:<32} {src}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    log = Tee(args.log)

    if not (args.model or args.model_id or args.local_dir):
        log.emit("[错误] 请用 --model / --model-id / --local-dir 至少指定一个模型。")
        log.emit("例如：python debug_gemma_multi_mode.py --model gemma-3n --local-dir <目录> --image 图.png")
        raise SystemExit(2)

    log.emit("=" * 70)
    log.emit("Gemma 多模态源码调试入口")
    log.emit(f"  model      : {args.model or '(自定义)'}")
    log.emit(f"  variant    : {args.variant or '(默认/自定义)'}")
    log.emit(f"  local-dir  : {args.local_dir or '(无)'}")
    log.emit(f"  image      : {args.image}")
    log.emit(f"  device     : {args.device}   dtype: {args.dtype}")
    log.emit("=" * 70)

    try:
        model_id, cls_hint = resolve_model_id(args)
        config, processor, model = load_components(model_id, cls_hint, args, log)

        if config is None:                     # --config-only
            log.emit("[完成] config-only 模式结束。")
            return 0

        # ---- 图片 ----
        images = load_images(args.image, log)

        # ---- 调试动作 ----
        if args.locate:
            log.emit("\n[源码定位] 主要模块 -> 源码文件：")
            locate_source(model, log)
        if args.list_model:
            log.emit("\n[结构] 模型模块树：")
            dump_model_tree(model, args, log)
        trace_handles = []
        if args.trace_shapes:
            log.emit("\n[追踪] 前向传播 shape 输出：")
            trace_handles = install_shape_trace(model, args, log)
        bp_handles = []
        if args.breakpoints:
            keywords = [k.strip() for k in args.breakpoints.split(",") if k.strip()]
            bp_handles = install_breakpoints(model, keywords, log)
        if args.debug:
            log.emit("\n[debug] 已停在 processor 前，可检查 model / processor / images ...")
            breakpoint()

        # ---- 多模态输入构造 ----
        # Gemma3n/Gemma4 的 processor 要求文本中包含图像占位符 token（如 <|image|>），
        # 否则 input_ids 里没有 image token，与视觉特征数量不匹配。
        # 标准做法：用 chat template 渲染 messages（content 里声明 image 类型），
        # 占位符由 tokenizer 自动插入。
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": args.prompt}]
        messages = [{"role": "user", "content": content}]
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        log.emit(f"\n[输入] chat template 渲染文本 = {prompt_text!r}")
        inputs = processor(text=prompt_text, images=images, return_tensors="pt")
        log.emit(f"[输入] processor 输出键：{list(inputs.keys())}")
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                log.emit(f"        {k}: {tuple(v.shape)}  dtype={v.dtype}")

        device = next(model.parameters()).device
        inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                  for k, v in inputs.items()}

        # ---- 生成 ----
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=getattr(processor.tokenizer, "pad_token_id", None)
                           or getattr(processor.tokenizer, "eos_token_id", None),
            )
        text = processor.decode(out_ids[0], skip_special_tokens=True)
        log.emit(f"\n[生成] 结果：\n{text}")

        log.emit("\n[完成] 运行结束。")
        return 0

    except Exception as e:
        log.emit("\n[异常] " + traceback.format_exc())
        msg = str(e)
        if "401" in msg or "gated" in msg or "access" in msg.lower():
            log.emit("[建议] Gemma 是 gated 模型：请先到 https://huggingface.co/google 接受协议，"
                     "再运行 huggingface-cli login（或设置环境变量 HF_TOKEN）。")
        if "OutOfMemory" in msg or "out of memory" in msg.lower():
            log.emit("[建议] 显存不足：换更小的变体，或 --dtype float16，或 --device cpu。")
        return 1
    finally:
        log.close()


if __name__ == "__main__":
    sys.exit(main())
