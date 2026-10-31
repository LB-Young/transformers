#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
debug_qwen.py —— Qwen 系列（纯文本 + 多模态）源码调试入口脚本
=====================================================================

用途
----
    在本地 conda 环境（transformers-gemma）中，用 E:\\transformers 的源码
    加载 E:\\models\\qwen 下已下载的 Qwen 系列模型，支持断点调试进入
    transformers 源码（VS Code 里 justMyCode:false 即可）：

      1. 纯文本模型：AutoTokenizer + AutoModelForCausalLM
         （Qwen1 / Qwen1.5 / Qwen2 / Qwen2.5 / Qwen3）
      2. 多模态模型：AutoProcessor + AutoModelForImageTextToText
         （Qwen2-VL / Qwen2.5-VL / Qwen3-VL / Qwen3.5，输入一张或多张图 + 文本）
      3. GPTQ 量化模型：Qwen3.5-35B-A3B-GPTQ-Int4（权重已校验，加载需 gptqmodel）

调试动作（可叠加）
------------------
    --config-only        只打印 config.json 后退出
    --list-model         打印模型模块树（--params 附带每模块参数数）
    --trace-shapes       注册 forward hook 逐层打印输入/输出 shape
    --breakpoints <kw>   关键字命中模块时 breakpoint() 进入 pdb（如 vision,delta,attention,moe）
    --fn-breakpoints <fn> 函数级断点：transformers 的 modeling_*.py 中同名函数被调用时
                          breakpoint()（如 compute_3d_position_ids），用于调试普通方法
    --forward-only       不调 model.generate()，直接 model(**inputs) 一次完整前向。
                          注意：Qwen2VL 的 generate() 会在 _prepare_position_ids_for_generation
                          预计算 position_ids（1359 行），导致 forward 里
                          "if position_ids is None" 不成立、compute_3d_position_ids 不被调用；
                          用 --forward-only 时 position_ids 保持 None，必进该函数（1095 行定义）。
    --debug              在构造输入前 breakpoint()
    --locate             打印每个模块对应的源码文件

典型用法
--------
    # 纯文本：Qwen3-0.6B 跑一遍 + 打印结构
    python debug_qwen.py --model qwen3-0.6b --offline --list-model --params --trace-shapes --depth 2

    # 多模态：让 Qwen3-VL-2B 描述测试图（默认图见 DEFAULT_IMAGE）
    python debug_qwen.py --model qwen3-vl-2b --offline --prompt "请描述这张图片。" --max-new-tokens 128

    # 在视觉编码器打断点
    python debug_qwen.py --model qwen2.5-vl-3b --offline --breakpoints vision

    # 在 Gated DeltaNet / MoE 路由打断点（Qwen3.5 系列）
    python debug_qwen.py --model qwen3.5-35b-a3b-gptq --offline --breakpoints delta,moe

模型目录已在下方 MODEL_REGISTRY 内配置好，也可用 --local-dir 覆盖。
"""

import argparse
import json
import os
import sys
import traceback

import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Qwen 模型注册表：模型 key -> 本地目录 / 模态 / 特殊标记
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    # ---- 纯文本（AutoTokenizer + AutoModelForCausalLM）----
    "qwen1-1_8b": {
        "dir": r"E:\models\qwen\Qwen-1_8B",
        "modal": "text",
        "trust_remote_code": True,
        "note": "旧版 Qwen1（QWenLMHeadModel），需 trust_remote_code；tokenizer 需 pip install tiktoken",
    },
    "qwen1.5-0.5b": {
        "dir": r"E:\models\qwen\Qwen1.5-0.5B-Chat",
        "modal": "text",
    },
    "qwen2-0.5b": {
        "dir": r"E:\models\qwen\Qwen2-0.5B-Instruct",
        "modal": "text",
    },
    "qwen2.5-0.5b": {
        "dir": r"E:\models\qwen\Qwen2.5-0.5B-Instruct",
        "modal": "text",
    },
    "qwen3-0.6b": {
        "dir": r"E:\models\qwen\Qwen3-0.6B",
        "modal": "text",
    },
    # ---- 多模态（AutoProcessor + AutoModelForImageTextToText）----
    "qwen2-vl-2b": {
        "dir": r"E:\models\qwen\Qwen2-VL-2B-Instruct",
        "modal": "multimodal",
    },
    "qwen2.5-vl-3b": {
        "dir": r"E:\models\qwen\Qwen2.5-VL-3B-Instruct",
        "modal": "multimodal",
    },
    "qwen3-vl-2b": {
        "dir": r"E:\models\qwen\Qwen3-VL-2B-Instruct",
        "modal": "multimodal",
    },
    "qwen3.5-0.8b": {
        "dir": r"E:\models\qwen\Qwen3.5-0.8B-Instruct",
        "modal": "multimodal",
        "note": "官方仓库名实为 Qwen/Qwen3.5-0.8B（无 -Instruct 后缀）；带 vision_config，是多模态模型",
    },
    "qwen3.5-35b-a3b-gptq": {
        "dir": r"E:\models\qwen\Qwen3.5-35B-A3B-GPTQ-Int4",
        "modal": "multimodal",
        "gptq": True,
        "note": "GPTQ-Int4 22.74GB / 14 分片，权重已逐字节校验；加载需要 gptqmodel（Windows 暂无预编译 wheel）",
    },
}

# 多模态模型未指定 --image 时的默认测试图（几何图形，可随时替换成自己的图）
DEFAULT_IMAGE = r"E:\transformers\bayon-note\Qwen\qwen_test_image.png"

# 默认中文提示词
DEFAULT_PROMPT = "请详细介绍这张图片的结构和内容，包括画面元素、布局与传递的信息。"


# ---------------------------------------------------------------------------
# 简单的 tee 日志（与 debug_gemma_*.py 一致）
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
        description="Qwen 系列（纯文本 + 多模态）源码调试入口脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例见文件头注释。",
    )
    p.add_argument("--model", choices=list(MODEL_REGISTRY.keys()),
                   help="模型 key，见 MODEL_REGISTRY（如 qwen3-0.6b / qwen3-vl-2b / qwen3.5-35b-a3b-gptq）")
    p.add_argument("--local-dir", default=None,
                   help="本地权重目录（覆盖注册表默认目录）")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                   help="GPTQ 模型忽略此参数，自动用 torch_dtype=auto")
    p.add_argument("--offline", action="store_true",
                   help="离线模式（HF_HUB_OFFLINE=1），配合本地权重使用")
    p.add_argument("--image", default=None,
                   help="图片路径，逗号分隔可传多张；仅多模态模型使用，缺省用默认测试图")
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="文本指令")
    p.add_argument("--max-new-tokens", type=int, default=128)
    # ---- 调试动作（可叠加）----
    p.add_argument("--config-only", action="store_true")
    p.add_argument("--list-model", action="store_true")
    p.add_argument("--params", action="store_true")
    p.add_argument("--trace-shapes", action="store_true")
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--breakpoints", default=None,
                   help="逗号分隔的模块名关键字，命中即 breakpoint() 进入 pdb，如 vision,delta,attention,moe")
    p.add_argument("--fn-breakpoints", default=None,
                   help="逗号分隔的函数名，transformers 的 modeling_*.py 中同名函数被调用时即 breakpoint()，"
                        "如 compute_3d_position_ids（用于非 nn.Module 的普通方法）")
    p.add_argument("--debug", action="store_true", help="在构造输入前 breakpoint()")
    p.add_argument("--forward-only", action="store_true",
                   help="不调 generate，直接 model(**inputs) 一次完整前向；此时 position_ids=None，"
                        "Qwen2-VL 会进入 compute_3d_position_ids 分支（generate 路径不会调用该函数）")
    p.add_argument("--locate", action="store_true")
    p.add_argument("--log", default=None)
    return p.parse_args(argv)


def resolve_model_dir(args):
    if args.local_dir:
        return args.local_dir, None
    if not args.model:
        raise SystemExit("[错误] 请用 --model 指定模型（--list-model 可看注册表），或用 --local-dir 直接指定目录。")
    reg = MODEL_REGISTRY[args.model]
    return reg["dir"], reg


# ---------------------------------------------------------------------------
# 加载组件
# ---------------------------------------------------------------------------
def load_components(model_dir, reg, args, log):
    from transformers import AutoConfig

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    trust_remote_code = bool(reg and reg.get("trust_remote_code"))

    log.emit(f"[加载] config        <- {model_dir}")
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    log.emit(f"[配置] architecture  = {getattr(config, 'architectures', None)}")
    log.emit(f"[配置] model_type    = {config.model_type}")
    log.emit(f"[配置] vision_config = {'有' if getattr(config, 'vision_config', None) else '无'}")
    if args.config_only:
        log.emit(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))
        return None, None, None

    # 设备与精度
    device_map = args.device if args.device != "auto" else "auto"
    is_gptq = bool(reg and reg.get("gptq"))
    dtype = "auto" if is_gptq else {
        "bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32,
    }[args.dtype]

    modal = reg["modal"] if reg else ("multimodal" if getattr(config, "vision_config", None) else "text")
    log.emit(f"[加载] 模态          = {modal}" + ("（GPTQ 量化，torch_dtype=auto）" if is_gptq else ""))

    if modal == "multimodal":
        from transformers import AutoProcessor, AutoModelForImageTextToText
        log.emit(f"[加载] processor     <- {model_dir}")
        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
        log.emit(f"[配置] processor 类  = {processor.__class__.__module__}.{processor.__class__.__name__}")
        log.emit(f"[加载] model         <- {model_dir} (AutoModelForImageTextToText)")
        model = AutoModelForImageTextToText.from_pretrained(
            model_dir, torch_dtype=dtype, device_map=device_map,
            low_cpu_mem_usage=True, trust_remote_code=trust_remote_code,
        )
        log.emit(f"[配置] 实际模型类    = {model.__class__.__module__}.{model.__class__.__name__}")
        return config, processor, model
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        log.emit(f"[加载] tokenizer     <- {model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
        log.emit(f"[加载] model         <- {model_dir} (AutoModelForCausalLM)")
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=dtype, device_map=device_map,
            low_cpu_mem_usage=True, trust_remote_code=trust_remote_code,
        )
        log.emit(f"[配置] 实际模型类    = {model.__class__.__module__}.{model.__class__.__name__}")
        return config, tokenizer, model


# ---------------------------------------------------------------------------
# 图片加载：PIL -> RGB
# ---------------------------------------------------------------------------
def load_images(image_arg, log):
    paths = [p.strip() for p in image_arg.split(",") if p.strip()]
    if not paths:
        raise SystemExit("[错误] 图片列表为空")
    images = []
    for path in paths:
        if not os.path.isfile(path):
            raise SystemExit(f"[错误] 图片不存在：{path}")
        img = Image.open(path).convert("RGB")
        log.emit(f"[图片] {path}  size={img.size}  mode={img.mode}")
        images.append(img)
    return images


# ---------------------------------------------------------------------------
# 动作：模块树 / shape 追踪 / 断点 / 源码定位
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


def make_fn_tracer(fn_targets, log):
    """函数级断点：transformers 的 modeling_*.py 中目标函数被调用时 breakpoint()。

    通过 sys.settrace 只监听 call 事件（开销极小，不跟踪行级）；命中后 breakpoint()
    暂停在调用点，在 VS Code 调试器的调用栈中选择目标函数帧即可查看局部变量，
    继续单步即进入函数体，适合调试 compute_3d_position_ids 这类普通方法。
    """
    def tracer(frame, event, arg):
        if event == "call":
            code = frame.f_code
            if code.co_name in fn_targets:
                fname = code.co_filename.replace("\\", "/")
                if "/transformers/models/" in fname and os.path.basename(fname).startswith("modeling_"):
                    log.emit(f"\n>>> 命中函数断点：{fname}:{code.co_firstlineno} → {code.co_name}")
                    log.emit("[提示] 在调试器调用栈中选择该函数帧查看局部变量，继续单步即进入函数体。")
                    breakpoint()
        return None
    return tracer


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

    log.emit("=" * 70)
    log.emit("Qwen 系列源码调试入口")
    log.emit(f"  model     : {args.model or '(自定义)'}")
    log.emit(f"  local-dir : {args.local_dir or '(注册表默认)'}")
    log.emit(f"  device    : {args.device}   dtype: {args.dtype}")
    log.emit("=" * 70)

    fn_trace_set = False
    try:
        model_dir, reg = resolve_model_dir(args)
        if reg and reg.get("note"):
            log.emit(f"[提示] {reg['note']}")
        config, tokenizer_or_processor, model = load_components(model_dir, reg, args, log)

        if config is None:                       # --config-only
            log.emit("[完成] config-only 模式结束。")
            return 0

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
        if args.fn_breakpoints:
            if sys.gettrace() is not None:
                log.emit("[提示] 检测到调试器正在运行（debugpy），--fn-breakpoints 依赖 sys.settrace 会与调试器冲突；"
                         "请改为直接在目标函数行打 VS Code 断点（justMyCode:false 已支持进入源码）。")
            else:
                fn_targets = [f.strip() for f in args.fn_breakpoints.split(",") if f.strip()]
                sys.settrace(make_fn_tracer(fn_targets, log))
                fn_trace_set = True
                log.emit(f"[断点] 函数断点：{fn_targets}（命中 transformers modeling_*.py 同名函数即 breakpoint()）")
        if args.debug:
            log.emit("\n[debug] 已停在输入构造前，可检查 model / processor ...")
            breakpoint()

        # ---- 输入构造（文本 vs 多模态）----
        is_multimodal = hasattr(tokenizer_or_processor, "apply_chat_template") is False \
            or "Processor" in tokenizer_or_processor.__class__.__name__
        # 更稳妥的判断：processor 一定存在 image_processor；tokenizer 则没有
        processor = tokenizer_or_processor if is_multimodal else None

        if is_multimodal:
            image_arg = args.image or DEFAULT_IMAGE
            if not os.path.isfile(image_arg):
                raise SystemExit(
                    f"[错误] 图片不存在：{image_arg}。多模态模型需要图片输入，请用 --image 指定（逗号分隔可多张）。"
                )
            images = load_images(image_arg, log)
            content = [{"type": "image"} for _ in images] + [{"type": "text", "text": args.prompt}]
            messages = [{"role": "user", "content": content}]
            prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            log.emit(f"\n[输入] chat template 渲染文本 = {prompt_text!r}")
            inputs = processor(text=prompt_text, images=images, return_tensors="pt")
            decoder = processor
        else:
            tokenizer = tokenizer_or_processor
            messages = [{"role": "user", "content": args.prompt}]
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            log.emit(f"\n[输入] chat template 渲染文本 = {prompt_text!r}")
            inputs = tokenizer(prompt_text, return_tensors="pt")
            decoder = tokenizer

        log.emit(f"[输入] 键：{list(inputs.keys())}")
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                log.emit(f"        {k}: {tuple(v.shape)}  dtype={v.dtype}")

        device = next(model.parameters()).device
        inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                  for k, v in inputs.items()}

        # ---- 生成（generate 或一次完整前向）----
        with torch.no_grad():
            if args.forward_only:
                log.emit("\n[前向] 一次完整 forward（未走 generate）：position_ids 保持 None，"
                         "Qwen2-VL 会进入 compute_3d_position_ids")
                outputs = model(**inputs)
                log.emit(f"[输出] logits: {tuple(outputs.logits.shape)}")
                next_id = outputs.logits[0, -1].argmax().item()
                log.emit(f"[输出] 末位 argmax token id = {next_id}")
            else:
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=getattr(decoder, "pad_token_id", None)
                               or getattr(decoder, "eos_token_id", None),
                )
                text = decoder.decode(out_ids[0], skip_special_tokens=True)
                log.emit(f"\n[生成] 结果：\n{text}")

        log.emit("\n[完成] 运行结束。")
        return 0

    except Exception as e:
        log.emit("\n[异常] " + traceback.format_exc())
        msg = str(e)
        low = msg.lower()
        if "tiktoken" in low or "import tiktoken" in low:
            log.emit("[建议] Qwen-1_8B 的 tokenizer 需要 tiktoken：pip install tiktoken")
        if "gptqmodel" in low or "gptq" in low:
            log.emit("[建议] GPTQ 模型需要 gptqmodel 运行时：Windows 暂无预编译 wheel，"
                     "安装需 VS Build Tools 编译；当前 transformers/torch 环境未改动，"
                     "可用 --config-only 或直接读权重文件继续调试。")
        if "OutOfMemory" in msg or "out of memory" in low:
            log.emit("[建议] 显存不足：换更小的模型，或 --dtype float16，或 --device cpu，"
                     "35B-A3B 请确保双卡（device_map=auto 自动分片）。")
        return 1
    finally:
        if fn_trace_set:
            sys.settrace(None)
        log.close()


if __name__ == "__main__":
    sys.exit(main())
