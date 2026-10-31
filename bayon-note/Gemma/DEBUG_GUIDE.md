# Gemma 五模型源码调试指南（DEBUG_GUIDE）

本文档面向「想在 `E:\transformers` 源码里真正打断点、看每一步计算」的场景。
配合 `debug_gemma.py` 使用，共四种调试方式，由浅入深。

---

## 0. 调试目标总览

五个模型在 transformers 源码中的位置（**断点就打在下面的文件里**）：

| 模型            | 核心源码文件                                                          | 关键类                                                                                                                   |
| --------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Gemma 1         | `src/transformers/models/gemma/modeling_gemma.py`                   | `GemmaModel` → `GemmaDecoderLayer` → `GemmaAttention`                                                            |
| Gemma 2         | `src/transformers/models/gemma2/modeling_gemma2.py`                 | `Gemma2Model` → `Gemma2DecoderLayer` → `Gemma2Attention`（交替局部/全局）                                        |
| Gemma 3         | `src/transformers/models/gemma3/modeling_gemma3.py`                 | `Gemma3Model` / `Gemma3TextModel`、视觉 `Gemma3VisionModel`（SigLIP）                                              |
| Gemma 3n        | `src/transformers/models/gemma3n/modeling_gemma3n.py`               | **`Gemma3nForConditionalGeneration`** → `Gemma3nModel`（MatFormer 嵌套 + PLE + KV 共享 + MobileNetV5 视觉塔） |
| Gemma 4         | `src/transformers/models/gemma4/modeling_gemma4.py`                 | **`Gemma4ForConditionalGeneration`** → `Gemma4Model` / MoE（`Gemma4SparseMoeBlock`）                        |
| Gemma 4 Unified | `src/transformers/models/gemma4_unified/modeling_gemma4_unified.py` | **`Gemma4UnifiedForConditionalGeneration`**（文本+视觉+音频编码器）                                              |

> ⚠️ Gemma 3n / 4 / 4-unified 的多模态入口类是 `ForConditionalGeneration`，
> 用 `AutoModelForCausalLM` 加载会报错；`debug_gemma.py` 已按正确类名加载。

建议顺序：**先跑通 1B / E2B 小模型**，再逐步放大；先把方式 A/B 跑熟，再上 C/D。

---

## 1. 方式 A：静态查看（不打断点，先建立全局认识）

```powershell
# 1) 只看 config：超参数一览（层数、维度、头数、RoPE theta、词表……）
python debug_gemma.py --model gemma-3 --variant 1b --config-only

# 2) 结构树 + 每模块参数量：理解模块嵌套关系
python debug_gemma.py --model gemma-3 --variant 1b --list-model --params

# 3) 定位每个模块类对应的源码文件
python debug_gemma.py --model gemma-3 --variant 1b --locate
```

配合架构图 HTML（本目录 `gemma-x/index.html`）对照看：
config 里的数字 ↔ 架构图里的规格 ↔ 源码里的 `config.xxx` 读取点。

**读源码技巧**：从 `modeling_gemma3.py` 的 `Gemma3ForCausalLM.forward()` 入手，
一路追 `Gemma3TextModel.forward()` → `Gemma3DecoderLayer.forward()` →
`Gemma3Attention.forward()` / `Gemma3MLP.forward()`，整条数据流就清晰了。

---

## 2. 方式 B：shape 追踪（不动源码，看每一步的张量形状）

脚本自动给每个模块挂 forward hook，前向时打印输入/输出 shape：

```powershell
python debug_gemma.py --model gemma-3 --variant 1b --list-model --trace-shapes --depth 3
```

输出示例（示意）：

```
[model] in=(1, 7)  out=(1, 7, 1152)          # token ids -> hidden_states
  [model.layers] in=(1, 7, 1152)  out=(1, 7, 1152)
    [model.layers.0] in=(1, 7, 1152)  out=(1, 7, 1152)
      [model.layers.0.self_attn] ...
      [model.layers.0.mlp] ...
  [model.embed_tokens] in=(1, 7)  out=(1, 7, 1152)
  [model.norm] ...
[lm_head] in=(1, 7, 1152)  out=(1, 7, 256000)
```

要点：

- **`--depth` 控制粒度**：`0` 只看根；`2` 看到每个 DecoderLayer；`4~5` 可看到
  `self_attn` 内部的 `q_proj`/`k_proj`/`o_proj`、`mlp` 的 `gate_proj`/`up_proj`/`down_proj`。
- **shape 的 `(batch, seq, dim)`** 一眼能确认：RoPE 旋转在哪一步、GQA 的
  kv 头压缩在哪一步、MoE 的 router 输出在哪里分流。
- KV Cache 是新版 transformers 的 `Cache` 对象，shape 追踪会显示为类型名
  （如 `DynamicCache`）；想深入 cache 内部，用方式 C 在 `self_attn` 断点后 `p cache`。

---

## 3. 方式 C：pdb 交互断点（不改源码，按关键字自动停）

```powershell
python debug_gemma.py --model gemma-3 --variant 1b --breakpoints self_attn
```

脚本会把名字包含 `self_attn` 的所有模块都装上断点，前向到第一个注意力层时停住，
进入 pdb。常用命令：

| 命令                                            | 作用                                                                                            |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `up` / `down`                               | 在调用栈中上下移动（`up` 到 `modeling_gemma3.py` 源码帧）                                   |
| `p hidden_states`                             | 打印变量（注意：hook 帧里参数名是 `args_in[0]`，`up` 一层后才是源码里的 `hidden_states`） |
| `p module` / `p module.k_proj.weight.shape` | 查看当前模块与权重                                                                              |
| `l`                                           | 列出当前位置附近源码                                                                            |
| `c`                                           | 继续运行到下一个断点                                                                            |
| `q`                                           | 退出调试                                                                                        |
| `h`                                           | 查看全部 pdb 帮助                                                                               |

> 一次性看所有层的话，`--breakpoints` 会命中每一层同名模块，`c` 会一层层停。
> 只想停第一层：`--breakpoints model.layers.0.self_attn`。

**整体断点**：`--debug` 会在构造输入之后、前向之前停一次，
适合检查 `model`、`inputs`、`device`、dtype 是否就绪。

---

## 4. 方式 D：VSCode 图形化断点（推荐，真正“源码级 debug”）

**前置**：VSCode 安装 Python 扩展，并让调试器使用本 conda 环境。

### 4.1 选择解释器

`Ctrl+Shift+P` → `Python: Select Interpreter` → 选择
`transformers-gemma`（路径 `C:\Users\LBY\.conda\envs\transformers-gemma\python.exe`）。

### 4.2 写 launch.json

在 `E:\transformers\.vscode\launch.json` 中写入（若已有文件则合并）：

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug Gemma (script)",
      "type": "debugpy",
      "request": "launch",
      "program": "E:/transformers/bayon-note/Gemma/debug_gemma.py",
      "console": "integratedTerminal",
      "args": ["--model", "gemma-3", "--variant", "1b", "--list-model"],
      "cwd": "E:/transformers/bayon-note/Gemma",
      "justMyCode": false,
      "python": "C:/Users/LBY/.conda/envs/transformers-gemma/python.exe"
    },
    {
      "name": "Debug Gemma (source entry)",
      "type": "debugpy",
      "request": "launch",
      "program": "E:/transformers/scripts/debug_gemma_src.py",
      "console": "integratedTerminal",
      "cwd": "E:/transformers",
      "justMyCode": false,
      "python": "C:/Users/LBY/.conda/envs/transformers-gemma/python.exe"
    }
  ]
}
```

> `justMyCode: false` 关键：允许断点进入 `site-packages/transformers`
> 与 `E:\transformers\src` 的源码帧（默认只停用户代码）。

### 4.3 在源码里打断点

打开例如 `src/transformers/models/gemma3/modeling_gemma3.py`，在
`Gemma3Attention.forward` 的 `query_states` 计算行、`attn_weights` 行、
`output` 行各打一个断点，按 **F5**。调试器会显示每一步的局部变量、
张量 shape 和调用栈，并支持条件断点（右键断点 → Expression，如
`layer_idx == 2`）。

### 4.4 写一个最小的源码入口（可选）

若不想经过 debug_gemma.py，可新建 `scripts/debug_gemma_src.py`（上面对应）：

```python
# scripts/debug_gemma_src.py —— 直接进源码的最小入口
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "google/gemma-3-1b-it"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto",
    low_cpu_mem_usage=True,
)
inputs = tok("Explain the Gemma 3 architecture.", return_tensors="pt").to("cuda")

# 在这里打断点：F9
outputs = model(**inputs)
print(outputs.logits.shape)
```

---

## 5. 常见调试问题排查

| 问题                           | 处理                                                                                                      |
| ------------------------------ | --------------------------------------------------------------------------------------------------------- |
| 断点停不住（F5 直接跑完）      | 检查 `justMyCode: false` 是否生效；确认断点行被实际执行（如 `q_proj` 行被 `attn` 分支跳过时换一行） |
| pdb 里看不到 `hidden_states` | hook 帧与源码帧不同，先 `up` 一到两层                                                                   |
| 显存 OOM                       | 换小模型/`--dtype float16`/`--device cpu`；或 `device_map="auto"` 双卡分载                          |
| KV cache 想看内部              | 断点停在 `self_attn`，`p cache` / `p cache.get_seq_length()`                                        |
| 想对比两个模型同层输出         | 各跑一次 `--trace-shapes --log out_gemma3.txt`，再 diff 两份日志                                        |
| 加载时反复报 gated             | 见 README 第 3 节登录步骤；`--config-only` 不需登录                                                     |
| 源码改动了不生效               | `pip install -e` 安装的源码即时生效；若曾普通安装过，先 `pip uninstall transformers`                  |

---

## 6. 推荐的上手路径

1. `--config-only` 看 gemma-3-1b 的 config（5 分钟，建立超参数直觉）；
2. `--list-model --params` 看模块树与参数量分布（10 分钟）；
3. `--trace-shapes --depth 5` 跑一遍前向，对照架构图确认每条数据流（15 分钟）；
4. VSCode 在 `Gemma3DecoderLayer.forward` 打断点，`F5` 单步跟一层
   （30 分钟，此后换 gemma2/gemma4 只是换文件）；
5. 用 `--breakpoints` 在 MoE router / PLE / 视觉编码器等特色模块定点深挖。

按此路径，五个模型的架构差异（GQA 演进、局部/全局注意力、MoE、多模态路径）
都能在源码层面得到验证。
