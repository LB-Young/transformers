# -*- coding: utf-8 -*-
import ssl, urllib.request, json, os
ctx = ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
def getb(url, timeout=30):
    req=urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=timeout, context=ctx).read()
def save(repo, path):
    raw=getb(f"https://hf-mirror.com/{repo}/resolve/main/config.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path,"wb").write(raw)
    cfg=json.loads(raw.decode())
    print("===",repo,"===")
    for k in ["model_type","architectures","hidden_size","num_hidden_layers","num_attention_heads",
              "num_key_value_heads","intermediate_size","moe_intermediate_size","n_routed_experts",
              "n_shared_experts","num_experts_per_tok","first_k_dense_replace","moe_layer_freq",
              "kv_lora_rank","q_lora_rank","qk_nope_head_dim","qk_rope_head_dim","v_head_dim",
              "vocab_size","max_position_embeddings","rope_theta","tie_word_embeddings","torch_dtype"]:
        if k in cfg: print(f"  {k} = {cfg[k]}")

save("moonshotai/Moonlight-16B-A3B", r"E:\transformers\bayon-note\Kimi\moonshot_32b_a32b\_tools\_cfgpull\Moonlight-16B-A3B.config.json")
save("moonshotai/Kimi-K2-Base", r"E:\transformers\bayon-note\Kimi\kimi_k2_0905\_tools\_cfgpull\Kimi-K2-Base.config.json")
