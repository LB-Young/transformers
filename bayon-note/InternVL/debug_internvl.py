#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
debug_internvl.py —— InternVL 系列（多模态/图像输入）源码调试入口脚本
=====================================================================

用途
----
    在本地 conda 环境（transformers-gemma）中加载 InternVL 各版本模型，
    让模型"看图说话"，并支持断点调试：

      1. 只打印 config（--config-only）
      2. 打印模型结构 + 每层参数数量（--list-model / --params）
      3. 逐层追踪前向传播的输入/输出 shape（--trace-shapes）
      4. 在指定模块自动打断点进入 pdb（--breakpoints / --debug）
      5. 定位每个模块对应的源码文件（--locate）
      6. 多模态生成验证：输入一张或多张图片 + 文本指令（默认）

本地 5 个 InternVL 模型与加载方式
---------------------------------
    * InternVL 1.5  : Mini-InternVL-Chat-2B-V1-5   legacy 格式（官方远程代码，trust_remote_code）
    * InternVL 2.0  : InternVL2-1B                 legacy 格式（官方远程代码）
    * InternVL 2.5  : InternVL2_5-1B               legacy 格式（官方远程代码）
    * InternVL 3.0  : InternVL3-1B-hf              transformers 原生（InternVLForConditionalGeneration）
    * InternVL 3.5  : InternVL3_5-1B-HF            transformers 原生（InternVLForConditionalGeneration）

    legacy 格式：断点会进入模型目录内的官方远程代码（modeling_internvl_chat.py），
                 走 model.chat 高层接口（chat 模板、多轮对话由远程代码实现）；
    hf 格式    ：断点进入 E:\\transformers\\src\\transformers\\models\\internvl\\ 源码，
                 走 processor + model.generate 标准接口。

典型用法
--------
    # InternVL 3.5（transformers 原生）描述一张图片
    python debug_internvl.py --model internvl-3_5 --image 图.png --max-new-tokens 128

    # InternVL 2.5（legacy）多模态 + 追踪视觉编码器 shape
    python debug_internvl.py --model internvl-2_5 --image 图.png --trace-shapes --depth 2

    # 在视觉 transformer 打断点
    python debug_internvl.py --model internvl-3 --image 图.png --breakpoints vision

    # 多张图片
    python debug_internvl.py --model internvl-3_5 --image 图1.png,图2.png --prompt "对比这两张图。"
"""

import argparse
import json
import os
import sys
import traceback

import torch
from PIL import Image

# ---------------------------------------------------------------------------
# 本地模型注册表：模型名 -> (目录名, 格式, 家族)
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "internvl-1_5": {
        "dir": "Mini-InternVL-Chat-2B-V1-5", "fmt": "legacy",
        "note": "InternVL 1.5 最小模型，legacy 远程代码格式",
    },
    "internvl-2": {
        "dir": "InternVL2-1B", "fmt": "legacy",
        "note": "InternVL 2.0 最小模型，legacy 远程代码格式",
    },
    "internvl-2_5": {
        "dir": "InternVL2_5-1B", "fmt": "legacy",
        "note": "InternVL 2.5 最小模型，legacy 远程代码格式",
    },
    "internvl-3": {
        "dir": "InternVL3-1B-hf", "fmt": "hf",
        "note": "InternVL 3.0 最小模型，transformers 原生格式",
    },
    "internvl-3_5": {
        "dir": "InternVL3_5-1B-HF", "fmt": "hf",
        "note": "InternVL 3.5 最小模型，transformers 原生格式",
    },
}

MODEL_DIR = r"E:\models\model"
IMAGE_SIZE = 448  # InternVL 视觉塔输入分辨率（legacy 分支固定单 tile 用）


# ---------------------------------------------------------------------------
# 简单 tee 日志
# ---------------------------------------------------------------------------
class Tee:
    def __init__(self, log_path=None):
        self.file = open(log_path, "w", encoding="utf-8") if log_path else None

    def emit(self, text):
        print(text, flush=True)
        if self.file:
            self.file.write(text + "\n")
            self.file.flush()

    def close(self):
        if self.file:
            self.file.close()


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
        description="InternVL 多模态（图像输入）源码调试入口脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例见文件头注释。",
    )
    p.add_argument("--model", choices=list(MODEL_REGISTRY.keys()),
                   help="模型：internvl-1_5 / internvl-2 / internvl-2_5 / internvl-3 / internvl-3_5")
    p.add_argument("--local-dir", default=None,
                   help="自定义模型目录（覆盖注册表默认路径）")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--offline", action="store_true",
                   help="离线模式（HF_HUB_OFFLINE=1）")
    p.add_argument("--image", default=None, required=True,
                   help="图片路径，逗号分隔可传多张（必填）")
    p.add_argument("--prompt", default="请详细介绍这张图片的结构和内容，包括画面元素、布局与传递的信息。",
                   help="文本指令")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--no-image-prefix", action="store_true",
                   help="legacy 分支不自动在问题前加 <image> 占位符")
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
    p.add_argument("--log", default=None)
    return p.parse_args(argv)


def resolve_model(args):
    reg = MODEL_REGISTRY[args.model]
    local_dir = args.local_dir or os.path.join(MODEL_DIR, reg["dir"])
    if not os.path.isdir(local_dir):
        raise SystemExit(f"[错误] 模型目录不存在：{local_dir}")
    return local_dir, reg


# ---------------------------------------------------------------------------
# 图片加载
# ---------------------------------------------------------------------------
def load_images_pil(image_arg, log):
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


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """官方 InternVL load_image 的宽高比匹配（含 tie-break）。"""
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0 and 2 * width < 3 * height and ratio[0] * ratio[1] < best_ratio[0] * best_ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """官方 InternVL dynamic tile：把任意宽高比图片切成 image_size 方块（可加全局缩略图）。"""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i + j <= max_num + 1
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_legacy_pixel_values(path, dtype, min_num=1, max_num=12, use_thumbnail=True, image_size=448):
    """legacy 分支图像加载：官方 dynamic tile 预处理，输出 (T, 3, image_size, image_size)。"""
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(path).convert("RGB")
    tiles = dynamic_preprocess(img, min_num=min_num, max_num=max_num,
                               image_size=image_size, use_thumbnail=use_thumbnail)
    pixel_values = torch.stack([transform(t) for t in tiles]).to(dtype)
    return pixel_values


# ---------------------------------------------------------------------------
# 调试动作（legacy 与 hf 通用）
# ---------------------------------------------------------------------------
def dump_model_tree(model, args, log):
    total = 0
    for name, module in model.named_modules():
        if not name:
            continue
        n_params = sum(p.numel() for p in module.parameters())
        total += n_params
        indent = "  " * name.count(".")
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
# legacy 分支：官方远程代码 + model.chat
# ---------------------------------------------------------------------------
def run_legacy(args, local_dir, reg, images, log):
    from transformers import AutoModel, AutoTokenizer

    log.emit(f"[加载] legacy 远程代码模式，目录：{local_dir}")

    # 兼容补丁：transformers 5.21 加载收尾会访问 model.all_tied_weights_keys，
    # 旧版远程代码模型类（InternVLChatModel）没有该属性，导致 AttributeError。
    # 在 PreTrainedModel 基类上包装该方法，缺属性时补空 dict（hf 分支不受影响）。
    import transformers.modeling_utils as _mu
    _orig_move = _mu.PreTrainedModel._move_missing_keys_from_meta_to_device

    def _safe_move(self, *args, **kwargs):
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
            log.emit("[补丁] 已为模型补 all_tied_weights_keys={}（兼容 transformers 5.21）")
        return _orig_move(self, *args, **kwargs)

    _mu.PreTrainedModel._move_missing_keys_from_meta_to_device = _safe_move

    # 兼容补丁：transformers 5.21 的 standardize_rope_params 会把旧格式 rope_scaling
    # {type, factor} 自动补成含 rope_type/rope_theta 的 4 字段；旧版远程 InternLM2Config
    # 的 _rope_scaling_validation 只接受两字段，导致 InternVL 1.5 加载失败。
    # 仅在 legacy 分支禁用该扩展（hf 分支不受影响）。
    from transformers import modeling_rope_utils as _r

    def _noop_standardize(self, *a, **k):
        return None

    _r.RotaryEmbeddingConfigMixin.standardize_rope_params = _noop_standardize
    log.emit("[补丁] 禁用 standardize_rope_params 自动扩展（兼容旧 InternLM2Config rope 校验）")

    model = AutoModel.from_pretrained(
        local_dir,
        trust_remote_code=True,
        dtype=args.dtype,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
    log.emit(f"[加载] 模型类 = {model.__class__.__module__}.{model.__class__.__name__}")
    log.emit(f"[加载] tokenizer 类 = {tokenizer.__class__.__module__}.{tokenizer.__class__.__name__}")

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (
        "cuda" if args.device == "cuda" else "cpu")
    model = model.to(device).eval()
    log.emit(f"[设备] {device}  dtype={args.dtype}")

    # 兼容补丁：transformers 5.21 的生成循环（DynamicCache / chunked prefill / cache_position）
    # 与旧版远程 generate 的 prepare_inputs_for_generation 相互纠缠（prefill 用 inputs_embeds、
    # decode 步 5.x 透传 DynamicCache、position_ids 被 5.x 提前累加等），表现为 decode 步
    # Attention 形状错乱。稳妥做法：绕开 5.x 生成循环，手动实现贪心 decode——
    # prefill 一次 inputs_embeds 前向，随后每步喂单 token + 远程 tuple KV cache。
    def _manual_generate(self, pixel_values=None, input_ids=None, attention_mask=None,
                         visual_features=None, generation_config=None,
                         output_hidden_states=None, **generate_kwargs):
        img_ctx_id = self.img_context_token_id
        if img_ctx_id is None:
            raise ValueError("img_context_token_id 未设置（需先调用 model.chat 初始化）")
        # ---- 复刻远程 generate：把 vit 特征替换进文本嵌入 ----
        if pixel_values is not None:
            vit_embeds = visual_features if visual_features is not None else self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)
            ids = input_ids.reshape(B * N)
            selected = ids == img_ctx_id
            n_sel = int(selected.sum().item())
            if n_sel == 0:
                raise ValueError("query 中没有找到 <IMG_CONTEXT> 占位符")
            vit = vit_embeds.reshape(-1, C)
            log.emit(f"[图像] 占位符={n_sel}  vit_embeds={vit.shape[0]}  匹配={n_sel == vit.shape[0]}  "
                     f"vit mean={vit_embeds.mean().item():.4f} std={vit_embeds.std().item():.4f}")
            if n_sel != vit.shape[0]:
                log.emit(f"[警告] 占位符 {n_sel} != vit_embeds {vit.shape[0]}，按占位符数截断")
                input_embeds[selected] = input_embeds[selected] * 0.0 + vit[:n_sel].to(input_embeds.device)
            else:
                input_embeds[selected] = vit.to(input_embeds.device)
            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        max_new = 128
        eos = None
        if isinstance(generation_config, dict):
            max_new = int(generation_config.get("max_new_tokens", max_new))
            eos = generation_config.get("eos_token_id")
        max_new = int(generate_kwargs.get("max_new_tokens", max_new))
        eos = generate_kwargs.get("eos_token_id", eos)
        eos_set = {eos} if isinstance(eos, int) else (set(eos) if eos else set())

        lm = self.language_model
        with torch.no_grad():
            out = lm(input_ids=None, inputs_embeds=input_embeds,
                     attention_mask=attention_mask, use_cache=True)
            past = out.past_key_values
            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            mask = attention_mask
            gen = []
            for _ in range(max_new):
                if next_id.item() in eos_set:
                    break
                gen.append(next_id)
                if len(gen) >= max_new:
                    break
                mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=-1)
                out = lm(input_ids=next_id, past_key_values=past,
                         attention_mask=mask, use_cache=True)
                past = out.past_key_values
                next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        out_ids = torch.cat(gen, dim=1) if gen else torch.zeros((1, 0), dtype=torch.long, device=input_ids.device)
        return out_ids

    if hasattr(model, "language_model"):
        import types
        model.generate = types.MethodType(_manual_generate, model)
        log.emit("[补丁] 手动贪心 decode 接管 model.generate（绕开 transformers 5.21 生成循环）")
        lm_type = model.language_model.__class__.__name__
        if lm_type == "InternLM2ForCausalLM":
            log.emit("[提示] 该模型的 LLM 为远程 InternLM2ForCausalLM（InternVL 1.5），其远程代码基于旧版"
                     " transformers 编写，与 5.21 存在数值不兼容，推理输出可能退化。结构/断点调试不受影响；"
                     "如需正常推理输出，建议使用 transformers 4.x 环境。")

    if args.config_only:
        log.emit(json.dumps(model.config.to_dict(), indent=2, ensure_ascii=False))
        return 0

    # 图像张量（官方 dynamic tile 预处理：1.5/2.0/2.5 均为 dynamic + use_thumbnail）
    cfg = model.config
    min_num = int(getattr(cfg, "min_dynamic_patch", 1))
    max_num = int(getattr(cfg, "max_dynamic_patch", 12))
    use_thumb = bool(getattr(cfg, "use_thumbnail", True))
    image_paths = [p.strip() for p in args.image.split(",") if p.strip()]
    if len(image_paths) > 1:
        log.emit("[警告] legacy 分支 model.chat 仅支持单图，已取第一张")
    pixel_values = load_legacy_pixel_values(
        image_paths[0], getattr(torch, args.dtype),
        min_num=min_num, max_num=max_num, use_thumbnail=use_thumb,
    )
    log.emit(f"[图像] dynamic tiles={pixel_values.shape[0]}  形状={tuple(pixel_values.shape)}  dtype={pixel_values.dtype}")

    # 问题构造：legacy 需要 <image> 占位（远程 chat 内部做字符串替换，1 张图只放 1 个）
    question = args.prompt
    if not args.no_image_prefix:
        question = "<image>\n" + question
    log.emit(f"\n[输入] question = {question!r}")

    # 调试动作
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
        log.emit("\n[debug] 已停在 model.chat 前，可检查 model / tokenizer / pixel_values ...")
        breakpoint()

    generation_config = dict(max_new_tokens=args.max_new_tokens, do_sample=False)
    with torch.no_grad():
        # 远程模板摘要（与 model.chat 内部构造一致，供 debug 参考）
        try:
            import importlib.util as _ilu
            conv_path = os.path.join(local_dir, "conversation.py")
            if os.path.isfile(conv_path):
                _spec = _ilu.spec_from_file_location("_conv_probe", conv_path)
                _conv = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_conv)
                _tpl = _conv.get_conv_template(model.template)
                log.emit(f"[模板] name={model.template!r} sep={_tpl.sep!r} "
                         f"num_image_token={model.num_image_token} system={model.system_message!r}")
        except Exception as _e:
            log.emit(f"[模板] 摘要失败：{_e}")
        response, history = model.chat(
            tokenizer,
            pixel_values=pixel_values.to(device),
            question=question,
            generation_config=generation_config,
            history=None,
            return_history=True,
        )
    log.emit(f"\n[生成] 结果：\n{response}")
    return 0


# ---------------------------------------------------------------------------
# hf 分支：transformers 原生 + processor + generate
# ---------------------------------------------------------------------------
def run_hf(args, local_dir, reg, images, log):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    log.emit(f"[加载] transformers 原生模式，目录：{local_dir}")
    processor = AutoProcessor.from_pretrained(local_dir)
    log.emit(f"[加载] processor 类 = {processor.__class__.__module__}.{processor.__class__.__name__}")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    device_map = args.device if args.device != "auto" else "auto"
    model = AutoModelForImageTextToText.from_pretrained(
        local_dir, dtype=dtype, device_map=device_map, low_cpu_mem_usage=True,
    )
    log.emit(f"[加载] 模型类 = {model.__class__.__module__}.{model.__class__.__name__}")

    if args.config_only:
        log.emit(json.dumps(model.config.to_dict(), indent=2, ensure_ascii=False))
        return 0

    # 图像占位符：取 processor 的 image token（context_image_token / <image>）
    image_token = getattr(processor.tokenizer, "context_image_token", None) or "<image>"
    log.emit(f"[配置] image_token = {image_token!r}")
    prompt_text = (image_token + "\n") * len(images) + args.prompt
    log.emit(f"\n[输入] prompt_text = {prompt_text!r}")

    # 调试动作
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

    inputs = processor(text=prompt_text, images=images, return_tensors="pt")
    log.emit(f"[输入] processor 输出键：{list(inputs.keys())}")
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            log.emit(f"        {k}: {tuple(v.shape)}  dtype={v.dtype}")

    device = next(model.parameters()).device
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
              for k, v in inputs.items()}

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
    return 0


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    log = Tee(args.log)

    if not args.model:
        log.emit("[错误] 请用 --model 指定模型：internvl-1_5 / internvl-2 / internvl-2_5 / internvl-3 / internvl-3_5")
        raise SystemExit(2)

    log.emit("=" * 70)
    log.emit("InternVL 多模态源码调试入口")
    log.emit(f"  model      : {args.model}")
    log.emit(f"  image      : {args.image}")
    log.emit(f"  device     : {args.device}   dtype: {args.dtype}")
    log.emit("=" * 70)

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    try:
        local_dir, reg = resolve_model(args)
        log.emit(f"[配置] 模型目录 = {local_dir}  格式 = {reg['fmt']}  {reg.get('note', '')}")
        images = load_images_pil(args.image, log)

        if reg["fmt"] == "legacy":
            return run_legacy(args, local_dir, reg, images, log)
        return run_hf(args, local_dir, reg, images, log)

    except Exception as e:
        log.emit("\n[异常] " + traceback.format_exc())
        msg = str(e)
        if "trust_remote_code" in msg.lower() or "custom code" in msg.lower():
            log.emit("[建议] legacy 模型需要 trust_remote_code=True（脚本已默认开启）。")
        if "OutOfMemory" in msg or "out of memory" in msg.lower():
            log.emit("[建议] 显存不足：换更小变体，或 --dtype float16，或 --device cpu。")
        return 1
    finally:
        log.close()


if __name__ == "__main__":
    sys.exit(main())
