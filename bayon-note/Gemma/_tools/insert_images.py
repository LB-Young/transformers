# -*- coding: utf-8 -*-
"""把 _arch/*.png 架构图插入各 index.md 的对应小节标题之后。"""
import os, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
BASE = r"E:\transformers\bayon-note\Gemma"

# 模型 -> [(png 文件名, md 锚点标题行(前缀或全等), 图片说明)]
PLAN = {
 "gemma-1": [
   ("01_2端到端数据流总览（以本地_2B_为主尺寸）.png", "### 2.1 数据流图（SVG 结构化重述）", "端到端数据流总览（2B 为主尺寸，7B 同构）"),
   ("02_3单个_GemmaDecoderLayer_解剖.png", "### 3.1 解码层展开图（SVG 结构化重述）", "单个 GemmaDecoderLayer 展开（MQA 广播 / GeGLU / 双 RMSNorm）"),
 ],
 "gemma-2": [
   ("01_2端到端数据流总览（以本地_2B_为基准尺寸）.png", "### 2.1 数据流图（SVG 结构化重述）", "端到端数据流总览（2B 基准，9B/27B 同构）"),
   ("02_3解码层解剖_Gemma2DecoderLaye.png", "### 3.1 解码层展开图（SVG 结构化重述）", "Gemma2DecoderLayer 展开（交替注意力 / softcap / 双 RMSNorm）"),
 ],
 "gemma-3": [
   ("01_2端到端数据流总览.png", "### 2.1 数据流图（SVG 结构化重述）", "端到端数据流总览"),
   ("02_3解码层解剖_Gemma_2_式双_RMSNor.png", "### 3.1 解码层展开图（SVG 结构化重述）", "解码层解剖（双 RMSNorm 残差块）"),
 ],
 "gemma-3n": [
   ("01_2端到端数据流总览（文本_+_图像_+_音频三路.png", "### 2.1 数据流图（SVG 结构化重述）", "端到端数据流总览（文本 + 图像 + 音频三路）"),
   ("02_3解码层解剖_Gemma3nTextDecode.png", "### 解码层展开图（SVG 结构化重述）", "Gemma3nTextDecoderLayer 展开（LAuReL / PLE / 共享 KV）"),
 ],
 "gemma-4": [
   ("01_2端到端数据流总览.png", "### 2.1 数据流图（SVG 结构化重述）", "端到端数据流总览（masked_scatter 融合 / MoE 并行分支 / softcap 30）"),
   ("02_3.1_dense_解码层（E2B_E4B_12.png", "### 3.1 dense 解码层（E2B/E4B/12B/31B 通用）", "dense 解码层展开（QKV Norm / GeGLU）"),
   ("03_3.2_MoE_解码层（26B-A4B_的每一层.png", "### 3.2 MoE 解码层（26B-A4B 的每一层）", "MoE 解码层展开（dense MLP ‖ 128 专家并行 / Router 输入取残差）"),
 ],
}

for model, items in PLAN.items():
    md_path = os.path.join(BASE, model, "index.md")
    with open(md_path, encoding='utf-8') as f:
        lines = f.read().split('\n')
    out = []
    inserted = set()
    for line in lines:
        out.append(line)
        stripped = line.strip()
        for png, anchor, desc in items:
            if anchor not in inserted and stripped.startswith(anchor):
                img_line = f"\n![架构图：{desc}](./_arch/{png})"
                out.append(img_line)
                inserted.add(anchor)
    # 校验全部锚点命中
    missing = [a for _, a, _ in items if a not in inserted]
    with open(md_path, 'w', encoding='utf-8', newline='') as f:
        f.write('\n'.join(out))
    print(f"{model}: 插入 {len(inserted)}/{len(items)} {'缺失: ' + str(missing) if missing else ''}")
