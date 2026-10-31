#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
debug_gemma.py —— Gemma 五模型 transformers 源码调试入口脚本
============================================================

用途
----
    在本地 conda 环境（transformers-gemma）中，用 E:\\transformers 的源码
    加载 Gemma 1 / 2 / 3 / 3n / 4（及 4-unified），并支持：
      1. 只打印 config（--config-only，不下载权重，最快）
      2. 打印模型结构 + 每层参数数量（--list-model / --params）
      3. 逐层追踪前向传播的输入/输出 shape（--trace-shapes）
      4. 在指定模块自动打断点进入 pdb（--breakpoints / --debug）
      5. 定位每个模块对应的 transformers 源码文件（--locate）
      6. 最小生成验证（--max-new-tokens）

典型用法
--------
    # 只查看 config（不需要权重、不需要登录）
    python debug_gemma.py --model gemma-3 --variant 1b --config-only

    # 加载 gemma-3-1b-it，打印结构并追踪前 3 层 shape（需要 HF token）
    python debug_gemma.py --model gemma-3 --variant 1b --list-model --trace-shapes --depth 3

    # 在注意力模块打断点，进入 pdb 交互调试
    python debug_gemma.py --model gemma-3 --variant 1b --breakpoints self_attn

    # 用本地已下载的权重目录（离线可加 --offline）
    python debug_gemma.py --model gemma-2 --local-dir E:\\models\\gemma-2-2b-it --offline

    # 自定义任意 HF 模型 ID
    python debug_gemma.py --model-id google/gemma-2-9b-it --config-only

详细说明见同目录 README.md（使用说明）与 DEBUG_GUIDE.md（调试指南）。
"""

import argparse
import inspect
import json
import os
import sys
import traceback

import torch

# ---------------------------------------------------------------------------
# 五个模型的注册表：模型名 -> transformers 类名 + HF 模型 ID 建议值
# ---------------------------------------------------------------------------
# 说明：
#   * variants 里的 HF ID 是社区常用 ID；HuggingFace 上的实际名称/大小写可能
#     调整，脚本允许用 --model-id 覆盖，或把 --variant 换成自己的值。
#   * 下载需要接受 Google 的许可协议并用 HF token 登录（见 README.md）。
MODEL_REGISTRY = {
    "gemma-1": {
        "cls": "GemmaForCausalLM",                    # transformers.models.gemma.modeling_gemma
        "variants": {
            "2b":    "google/gemma-2b-it",            # 2B 指令版
            "2b-pt": "google/gemma-2b",               # 2B 预训练版
            "7b":    "google/gemma-7b-it",            # 7B 指令版
            "7b-pt": "google/gemma-7b",               # 7B 预训练版
        },
        "default_variant": "2b",
    },
    "gemma-2": {
        "cls": "Gemma2ForCausalLM",                   # transformers.models.gemma2.modeling_gemma2
        "variants": {
            "2b":  "google/gemma-2-2b-it",
            "9b":  "google/gemma-2-9b-it",
            "27b": "google/gemma-2-27b-it",
        },
        "default_variant": "2b",
    },
    "gemma-3": {
        "cls": "Gemma3ForCausalLM",                   # transformers.models.gemma3.modeling_gemma3
        "variants": {
            "270m": "google/gemma-3-270m-it",
            "1b":   "google/gemma-3-1b-it",
            "4b":   "google/gemma-3-4b-it",
            "12b":  "google/gemma-3-12b-it",
            "27b":  "google/gemma-3-27b-it",
        },
        "default_variant": "1b",                      # 默认用最小的 1B，便于快速验证
    },
    "gemma-3n": {
        "cls": "Gemma3nForConditionalGeneration",     # 多模态条件生成（非 ForCausalLM！）
        "variants": {
            "e2b": "google/gemma-3n-E2B-it",          # 移动端 2B（实际约 5B 参数）
            "e4b": "google/gemma-3n-E4B-it",          # 移动端 4B（实际约 8B 参数）
        },
        "default_variant": "e2b",
    },
    "gemma-4": {
        "cls": "Gemma4ForConditionalGeneration",      # 多模态条件生成（非 ForCausalLM！）
        "variants": {
            "e2b":       "google/gemma-4-E2B-it",     # 端侧 2B
            "e4b":       "google/gemma-4-E4B-it",     # 端侧 4B
            "31b":       "google/gemma-4-31B-it",     # 稠密 31B
            "26b-a4b":   "google/gemma-4-26B-A4B-it", # MoE 26B(激活 4B)
        },
        "default_variant": "e2b",
    },
    "gemma-4-unified": {
        "cls": "Gemma4UnifiedForConditionalGeneration", # 统一多模态（非 ForCausalLM！）
        "variants": {
            "12b": "google/gemma-4-12B-unified-it",   # 12B 统一多模态
        },
        "default_variant": "12b",
    },
}


# ---------------------------------------------------------------------------
# 简单的 tee 日志：同时输出到控制台和文件（可选）
# ---------------------------------------------------------------------------
class Tee:
    """把 print 的内容同时写入 log 文件与标准输出。"""
    def __init__(self, log_path=None):
        self.file = open(log_path, "w", encoding="utf-8") if log_path else None

    def emit(self, text):
        print(text)                                    # 控制台
        if self.file:
            self.file.write(text + "\n")               # 文件
            self.file.flush()

    def close(self):
        if self.file:
            self.file.close()


# ---------------------------------------------------------------------------
# shape 描述：把 Tensor / tuple / dict / Cache 等对象转成可读字符串
# ---------------------------------------------------------------------------
def describe_shape(obj, max_items=6):
    """递归地把一个输出对象转成 shape 摘要字符串。

    覆盖前向输出常见的四种形态：
      - torch.Tensor      -> (batch, seq, dim)
      - tuple / list      -> 逐元素递归，超长截断
      - dict              -> key:shape 对
      - 其他（如 KV Cache）-> 返回类型名，避免把内部对象整个打印出来
    """
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
        parts = [f"{k}:{describe_shape(v)}" for k, v in items]
        return "{" + ", ".join(parts) + "}"
    return type(obj).__name__


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Gemma 五模型 transformers 源码调试入口脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例见文件头注释。",
    )
    # ---- 模型选择（三选一：--model / --model-id / 默认值）----
    p.add_argument("--model", choices=list(MODEL_REGISTRY.keys()),
                   help="选择五个模型之一：gemma-1 / gemma-2 / gemma-3 / gemma-3n / gemma-4 / gemma-4-unified")
    p.add_argument("--variant", default=None,
                   help="该模型下的规格（如 gemma-3 的 1b/4b/12b/27b；gemma-4 的 e2b/e4b/31b/26b-a4b）")
    p.add_argument("--model-id", default=None,
                   help="直接指定 HF 模型 ID 或本地目录名（优先级高于 --model/--variant）")
    p.add_argument("--local-dir", default=None,
                   help="本地权重目录（例如 E:\\models\\gemma-3-4b-it），存在时不走网络")
    # ---- 运行环境 ----
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                   help="运行设备，默认 auto（有 GPU 用 GPU）")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                   help="加载精度；5060Ti(Blackwell) 建议 bfloat16")
    p.add_argument("--offline", action="store_true",
                   help="离线模式（HF_HUB_OFFLINE=1），配合 --local-dir 使用")
    # ---- 调试动作（可叠加）----
    p.add_argument("--config-only", action="store_true",
                   help="只打印 config.json 的完整内容，不下载权重（无需登录）")
    p.add_argument("--list-model", action="store_true",
                   help="打印模型完整模块树")
    p.add_argument("--params", action="store_true",
                   help="打印每个子模块的参数数量与总参数量")
    p.add_argument("--trace-shapes", action="store_true",
                   help="注册 forward hook，逐层打印输入/输出 shape")
    p.add_argument("--depth", type=int, default=3,
                   help="--trace-shapes 追踪的最大层深度（按模块名中的点号计）")
    p.add_argument("--breakpoints", default=None,
                   help="逗号分隔的模块名关键字，命中即在该模块前向时 breakpoint() 进入 pdb")
    p.add_argument("--debug", action="store_true",
                   help="在模型前向调用前 breakpoint() 进入 pdb")
    p.add_argument("--locate", action="store_true",
                   help="打印各主要模块类对应的 transformers 源码文件路径")
    # ---- 生成验证 ----
    p.add_argument("--prompt", default="Write a short haiku about transformers.",
                   help="生成验证的输入文本")
    p.add_argument("--max-new-tokens", type=int, default=16,
                   help="生成的最大新 token 数")
    # ---- 其他 ----
    p.add_argument("--log", default=None,
                   help="把全部输出同时写入该文件（UTF-8）")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# 解析模型选择，得到最终 HF ID 或本地目录
# ---------------------------------------------------------------------------
def resolve_model_id(args):
    """返回 (model_id_or_local_path, 实际类名)。"""
    # 1) --model-id 优先级最高
    if args.model_id:
        return args.model_id, None
    # 2) --local-dir 作为加载路径；若同时给了 --model，保留注册表的类名
    if args.local_dir:
        if args.model:
            reg = MODEL_REGISTRY[args.model]
            return args.local_dir, reg["cls"]
        return args.local_dir, None
    # 3) 注册表：--model + --variant
    reg = MODEL_REGISTRY[args.model]
    variant = args.variant or reg["default_variant"]
    if variant not in reg["variants"]:
        raise SystemExit(f"[错误] 模型 {args.model} 没有规格 {variant}，可选：{list(reg['variants'])}")
    return reg["variants"][variant], reg["cls"]


# ---------------------------------------------------------------------------
# 加载组件（含源码定位打印）
# ---------------------------------------------------------------------------
def load_components(model_id, cls_hint, args, log):
    """加载 AutoConfig / AutoTokenizer / AutoModelForCausalLM。"""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    # 离线模式：禁止访问网络，强制从本地缓存/目录读取
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    # ---- config 先行：所有模式都需要（--config-only 到此为止）----
    log.emit(f"[加载] config        <- {model_id}")
    config = AutoConfig.from_pretrained(model_id)
    log.emit(f"[配置] architecture  = {getattr(config, 'architectures', None)}")
    log.emit(f"[配置] model_type    = {config.model_type}")
    if args.config_only:
        # 把 config 序列化成普通 dict 打印，便于对照源码
        log.emit(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))
        return None, None, None

    # ---- tokenizer ----
    log.emit(f"[加载] tokenizer     <- {model_id}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        log.emit(f"[警告] tokenizer 加载失败（可忽略，仍可看结构）：{e}")
        tokenizer = None

    # ---- 模型主体：优先显式类（能精确定位源码），否则 Auto 自动映射 ----
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    device_map = args.device if args.device != "auto" else "auto"

    if cls_hint:
        # 根据类名从 transformers 动态导入对应类，保证走源码
        model_cls = getattr(__import__("transformers", fromlist=[cls_hint]), cls_hint)
        log.emit(f"[加载] 显式类       = {model_cls.__module__}.{model_cls.__name__}")
        model = model_cls.from_pretrained(
            model_id, dtype=dtype, device_map=device_map,
            low_cpu_mem_usage=True, trust_remote_code=False,
        )
    else:
        log.emit("[加载] 使用 AutoModelForCausalLM 自动映射")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map=device_map,
            low_cpu_mem_usage=True, trust_remote_code=False,
        )
        log.emit(f"[加载] 实际类       = {model.__class__.__module__}.{model.__class__.__name__}")

    return config, tokenizer, model


# ---------------------------------------------------------------------------
# 动作 1：打印模块树 + 参数统计
# ---------------------------------------------------------------------------
def dump_model_tree(model, args, log):
    """按层级缩进打印 named_modules，可选附带每个模块的参数数量。"""
    total = 0
    total_trainable = 0
    for name, module in model.named_modules():
        if not name:                                   # 根模块
            continue
        level = name.count(".")                        # 层数 = 名字中点号数
        indent = "  " * level
        n_params = sum(p.numel() for p in module.parameters())
        total += n_params
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total_trainable += trainable
        if args.params:
            log.emit(f"{indent}{name}  [{module.__class__.__name__}]  params={n_params:,}")
        else:
            log.emit(f"{indent}{name}  [{module.__class__.__name__}]")
    log.emit(f"\n[统计] 总参数 = {total:,}  可训练 = {total_trainable:,}")


# ---------------------------------------------------------------------------
# 动作 2：逐层 shape 追踪（forward hook）
# ---------------------------------------------------------------------------
def install_shape_trace(model, args, log):
    """给深度 <= args.depth 的模块注册 forward hook，打印输入/输出 shape。"""
    handles = []

    def make_hook(name, level):
        def hook(module, args_in, output):
            indent = "  " * level
            in_shape = describe_shape(args_in[0]) if args_in else "None"
            out_shape = describe_shape(output)
            log.emit(f"{indent}[{name}] in={in_shape}  out={out_shape}")
        return hook

    for name, module in model.named_modules():
        if not name:
            continue
        level = name.count(".")
        if level > args.depth:                         # 只追踪到指定深度
            continue
        handles.append(module.register_forward_hook(make_hook(name, level)))
    log.emit(f"[追踪] 已注册 {len(handles)} 个 forward hook（depth<={args.depth}）")
    return handles


# ---------------------------------------------------------------------------
# 动作 3：按关键字打断点（进入 pdb）
# ---------------------------------------------------------------------------
def install_breakpoints(model, keywords, log):
    """在模块名包含任一关键字的模块前向时调用 breakpoint()。

    进入 pdb 后：
      - up / down  在调用栈中上下移动，可跳到 modeling_*.py 源码帧
      - p hidden_states  打印当前帧变量
      - c 继续执行到下一个断点
    """
    handles = []

    def make_hook(name):
        def hook(module, args_in, output):
            log.emit(f">>> 命中断点：{name}（{module.__class__.__name__}）")
            breakpoint()                               # 进入 pdb 交互
            return output
        return hook

    for name, module in model.named_modules():
        if any(k in name for k in keywords):
            handles.append(module.register_forward_hook(make_hook(name)))
    log.emit(f"[断点] 关键字 {keywords} 命中 {len(handles)} 个模块")
    return handles


# ---------------------------------------------------------------------------
# 动作 4：定位源码文件
# ---------------------------------------------------------------------------
def locate_source(model, log):
    """打印主要模块类对应的源码文件，方便在 VSCode 中直接打开断点。"""
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
        if name:
            log.emit(f"{name:<42} {cls.__name__:<28} {src}")
        else:
            log.emit(f"{'<root>':<42} {cls.__name__:<28} {src}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    log = Tee(args.log)

    # 校验：必须给出一种模型选择方式
    if not (args.model or args.model_id or args.local_dir):
        log.emit("[错误] 请用 --model / --model-id / --local-dir 至少指定一个模型。")
        log.emit("例如：python debug_gemma.py --model gemma-3 --variant 1b --config-only")
        raise SystemExit(2)

    # 打印本次运行摘要
    log.emit("=" * 70)
    log.emit("Gemma 源码调试入口")
    log.emit(f"  model      : {args.model or '(自定义)'}")
    log.emit(f"  variant    : {args.variant or '(默认/自定义)'}")
    log.emit(f"  model-id   : {args.model_id or '(由注册表解析)'}")
    log.emit(f"  local-dir  : {args.local_dir or '(无)'}")
    log.emit(f"  device     : {args.device}")
    log.emit(f"  dtype      : {args.dtype}")
    log.emit("=" * 70)

    try:
        model_id, cls_hint = resolve_model_id(args)
        config, tokenizer, model = load_components(model_id, cls_hint, args, log)

        # --config-only 已在上一步返回，直接结束
        if config is None:
            log.emit("[完成] config-only 模式结束。")
            return 0

        # 可选：定位源码文件
        if args.locate:
            log.emit("\n[源码定位] 主要模块 -> 源码文件：")
            locate_source(model, log)

        # 可选：打印结构树
        if args.list_model:
            log.emit("\n[结构] 模型模块树：")
            dump_model_tree(model, args, log)

        # 可选：追踪 shape（hooks 在 forward 期间生效）
        trace_handles = []
        if args.trace_shapes:
            log.emit("\n[追踪] 前向传播 shape 输出：")
            trace_handles = install_shape_trace(model, args, log)

        # 可选：按关键字打断点
        bp_handles = []
        if args.breakpoints:
            keywords = [k.strip() for k in args.breakpoints.split(",") if k.strip()]
            bp_handles = install_breakpoints(model, keywords, log)

        # 可选：整体 debug 断点（在构造输入之前）
        if args.debug:
            log.emit("\n[debug] 已停在模型前向之前，可检查 model / tokenizer / device...")
            breakpoint()

        # ---- 前向 + 生成验证 ----
        if tokenizer is not None:
            log.emit(f"\n[生成] prompt = {args.prompt!r}")
            inputs = tokenizer(args.prompt, return_tensors="pt")
            # 把输入搬到模型所在设备
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
            log.emit(f"[生成] 结果：{text}")
        else:
            log.emit("\n[生成] 跳过（tokenizer 不可用）")

        log.emit("\n[完成] 运行结束。")
        return 0

    except Exception as e:
        log.emit("\n[异常] " + traceback.format_exc())
        # 针对最常见的两类错误给出可执行建议
        msg = str(e)
        if "401" in msg or "gated" in msg or "access" in msg.lower():
            log.emit("[建议] Gemma 是 gated 模型：请先到 https://huggingface.co/google 接受协议，"
                     "再运行 huggingface-cli login（或设置环境变量 HF_TOKEN）。")
        if "OutOfMemory" in msg or "out of memory" in msg.lower():
            log.emit("[建议] 显存不足：换更小的 variant（如 1b/e2b），或 --dtype float16，或 --device cpu。")
        return 1
    finally:
        log.close()


if __name__ == "__main__":
    sys.exit(main())
