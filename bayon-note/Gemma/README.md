# Gemma 五模型源码调试 · 使用说明

本目录是「用 HuggingFace `transformers` 源码 debug Gemma 系列模型」的工作区。
包含一个入口脚本与两份文档：

```
bayon-note/Gemma/
├── debug_gemma.py      # 入口脚本（加载模型 + 结构查看 + shape 追踪 + 断点调试）
├── README.md           # 本文件：环境与脚本使用说明
├── DEBUG_GUIDE.md      # 调试指南：四种调试方法与 VSCode 图形化断点
├── gemma-1/            # Gemma 1 架构图（index.html）
├── gemma-2/            # Gemma 2 架构图
├── gemma-3/            # Gemma 3 架构图
├── gemma-3n/           # Gemma 3n 架构图
└── gemma-4/            # Gemma 4 架构图
```

---

## 1. 环境

已创建 conda 虚拟环境 **`transformers-gemma`**（Python 3.11.17）：

| 项目 | 值 |
|---|---|
| 环境名 | `transformers-gemma` |
| Python | 3.11.17（位于 `C:\Users\LBY\.conda\envs\transformers-gemma\`） |
| PyTorch | 2.11.0+cu128（CUDA 12.8，适配 RTX 5060 Ti Blackwell 架构） |
| transformers | 源码可编辑安装（`E:\transformers`，版本 `5.21.0.dev0`） |
| accelerate | 1.15.0（`device_map="auto"` 双卡分载必需） |
| sentencepiece | 0.2.2（Gemma 的 SentencePiece tokenizer 必需） |

**已验证状态**（2026-10-20 环境搭建时）：

- [x] `import transformers` 指向 `E:\transformers\src\transformers\__init__.py`
- [x] 五模型类均可导入：`GemmaForCausalLM` / `Gemma2ForCausalLM` / `Gemma3ForCausalLM` / `Gemma3nForCausalLM` / `Gemma4ForCausalLM` / `Gemma4UnifiedForCausalLM`
- [x] 小模型 GPU 前向通过：`Gemma1`（随机权重）在 `cuda:0` 上 bfloat16 前向成功
- [ ] 真实权重加载：需先完成第 3 节 HuggingFace 登录

**激活环境**（在任意终端）：

```powershell
conda activate transformers-gemma
```

或直接使用绝对路径的 python（无需激活）：

```powershell
C:\Users\LBY\.conda\envs\transformers-gemma\python.exe 脚本参数...
```

> 注：`E:\transformers` 以 `pip install -e` 可编辑模式安装，`import transformers`
> 指向的就是源码目录。改动源码后立即生效，无需重装。

---

## 2. 五个模型与脚本的对应关系

脚本内置注册表，`--model` 可选：`gemma-1` / `gemma-2` / `gemma-3` / `gemma-3n` / `gemma-4` / `gemma-4-unified`。

| 脚本 `--model` | transformers 类 | 源码文件 | 可选 `--variant` |
|---|---|---|---|
| `gemma-1` | `GemmaForCausalLM` | `src/transformers/models/gemma/modeling_gemma.py` | `2b` `7b` |
| `gemma-2` | `Gemma2ForCausalLM` | `.../gemma2/modeling_gemma2.py` | `2b` `9b` `27b` |
| `gemma-3` | `Gemma3ForCausalLM` | `.../gemma3/modeling_gemma3.py` | `270m` `1b` `4b` `12b` `27b` |
| `gemma-3n` | `Gemma3nForConditionalGeneration` | `.../gemma3n/modeling_gemma3n.py` | `e2b` `e4b` |
| `gemma-4` | `Gemma4ForConditionalGeneration` | `.../gemma4/modeling_gemma4.py` | `e2b` `e4b` `31b` `26b-a4b` |
| `gemma-4-unified` | `Gemma4UnifiedForConditionalGeneration` | `.../gemma4_unified/modeling_gemma4_unified.py` | `12b` |

> ⚠️ **注意**：Gemma 3n / 4 / 4-unified 是多模态模型，主类为 `ForConditionalGeneration`
> （**不是** `ForCausalLM`），脚本已按此修正；用 `AutoModelForCausalLM` 加载它们会报错。

`--variant` 对应模型 ID（脚本内置，可用 `--model-id` 覆盖），例如：

- `gemma-3 --variant 1b` → `google/gemma-3-1b-it`
- `gemma-4 --variant 26b-a4b` → `google/gemma-4-26B-A4B-it`

---

## 3. 首次使用：登录 HuggingFace（必需）

Gemma 是 **gated 模型**：必须先在网页接受许可协议，再用 token 登录，才能下载权重。

1. 打开 <https://huggingface.co/google> ，找到对应模型页，点击 **Agree and access repository**（接受协议）；
2. 创建 Access Token：<https://huggingface.co/settings/tokens> （`Read` 权限即可）；
3. 本机登录：

```powershell
conda activate transformers-gemma
huggingface-cli login
# 粘贴刚才的 token，回车
```

> 不想登录也可以跑 `--config-only`（只读 config，不下载权重），或把权重
> 下载到本地后用 `--local-dir` 加载。

---

## 4. 常用命令

### 4.1 只读 config（最快、无需登录）

```powershell
python debug_gemma.py --model gemma-3 --variant 1b --config-only
```

### 4.2 加载模型并打印结构 + 参数统计

```powershell
python debug_gemma.py --model gemma-3 --variant 1b --list-model --params
```

### 4.3 逐层追踪前向 shape（调试核心功能）

```powershell
python debug_gemma.py --model gemma-3 --variant 1b --list-model --trace-shapes --depth 3
```

`--depth 3` 表示只追踪名字中点号数 ≤3 的模块（如 `model.layers.0`）。
调大 depth 可以看更细的算子（如 `model.layers.0.self_attn`）。

### 4.4 在指定模块自动打断点（pdb）

```powershell
python debug_gemma.py --model gemma-3 --variant 1b --breakpoints self_attn
# 命中断点后输入 pdb 命令：up / p hidden_states / c ...
```

### 4.5 用本地权重 + 离线模式（推荐，已下载到本地）

五个最小模型已从 **ModelScope（魔搭）** 下载到 `E:\models\model\` 下
（`gemma-2b-it` / `gemma-2-2b-it` / `gemma-3-270m-it` / `gemma-3n-E2B-it` / `gemma-4-E2B-it`，
共约 30GB，无需 HuggingFace token）。直接离线加载：

```powershell
python debug_gemma.py --model gemma-3 --local-dir E:\models\model\gemma-3-270m-it --offline --list-model --trace-shapes
python debug_gemma.py --model gemma-3n --local-dir E:\models\model\gemma-3n-E2B-it --offline --list-model --params
python debug_gemma.py --model gemma-4 --local-dir E:\models\model\gemma-4-E2B-it --offline --breakpoints self_attn
```

> 重新下载/补下其他变体：运行同目录 `download_gemma.py`（改脚本里的模型列表即可）。

### 4.6 定位每个模块的源码文件

```powershell
python debug_gemma.py --model gemma-3 --variant 1b --locate
```

输出每类模块对应的 `.py` 文件路径，方便去 VSCode 里打断点。

### 4.7 生成验证

```powershell
python debug_gemma.py --model gemma-3 --variant 1b --prompt "介绍一下 transformers" --max-new-tokens 32
```

---

## 5. 参数速查

| 参数 | 默认 | 说明 |
|---|---|---|
| `--model` | 无 | 五个模型名之一（必填其一，或 `--model-id` / `--local-dir`） |
| `--variant` | 各模型默认 | 规格，见第 2 节表 |
| `--model-id` | 无 | 直接指定 HF ID，覆盖注册表 |
| `--local-dir` | 无 | 本地权重目录 |
| `--device` | `auto` | `auto` / `cuda` / `cpu` |
| `--dtype` | `bfloat16` | `bfloat16` / `float16` / `float32` |
| `--offline` | 关 | 离线模式（需本地权重） |
| `--config-only` | 关 | 只打印 config，不加载权重 |
| `--list-model` | 关 | 打印模块树 |
| `--params` | 关 | 打印各模块参数数量 |
| `--trace-shapes` | 关 | 逐层打印输入/输出 shape |
| `--depth` | 3 | shape 追踪的最大深度 |
| `--breakpoints` | 无 | 逗号分隔的模块名关键字，命中即断点 |
| `--debug` | 关 | 前向调用前进入 pdb |
| `--locate` | 关 | 打印各模块源码文件路径 |
| `--prompt` | 见脚本 | 生成验证文本 |
| `--max-new-tokens` | 16 | 生成 token 数 |
| `--log` | 无 | 输出同时写入该文件（UTF-8） |

---

## 6. 显存参考（双 RTX 5060 Ti 16GB）

单卡 16GB 足够跑以下规格（bfloat16）：

| 模型规格 | 显存占用（约） | 备注 |
|---|---|---|
| gemma-3-1b / gemma-3n-E2B / gemma-4-E2B | ~3–5 GB | 首选调试对象 |
| gemma-2-2b / gemma-3-4b / gemma-3n-E4B | ~6–10 GB | 单卡轻松 |
| gemma-3-12b / gemma-2-9b / gemma-4-E4B | ~14–20 GB | 可单卡，紧张时换 float16 |
| gemma-3-27b / gemma-2-27b / gemma-4-31b / 26b-a4b | >16 GB | 需双卡 `device_map="auto"` 分载，或 CPU |

> 双卡并行加载由 `transformers` 的 `device_map="auto"` 自动处理
> （需要 `accelerate`，环境已随 transformers 源码安装依赖一并安装）。

---

## 7. 常见问题（FAQ）

| 现象 | 原因与解决 |
|---|---|
| `401 Unauthorized` / gated | 未登录或未接受协议，见第 3 节 |
| `ImportError: cannot import name 'Gemma4ForCausalLM'` | transformers 版本过旧；本环境装的是 `E:\transformers` 源码（v5.21.dev），已包含全部五个模型 |
| `CUDA out of memory` | 换更小 variant、`--dtype float16`，或加 `--device cpu` |
| 下载太慢/失败 | 先 `huggingface-cli download` 或网页手动下载到本地，再用 `--local-dir --offline` |
| 中文乱码 | 终端用 UTF-8（`chcp 65001`）；脚本输出本身是 UTF-8 |
| Windows 提示路径过长 | 权重目录尽量短，例如 `E:\models\g3-1b` |

更多调试细节见 **[DEBUG_GUIDE.md](./DEBUG_GUIDE.md)**。
