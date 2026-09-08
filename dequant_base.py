#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Dequantize the NF4 base (NO adapter) to a clean bf16 model dir.

This is the lossless base for GGUF conversion + runtime-LoRA in llama.cpp:
convert this directory to GGUF once, then swap adapters at serve time instead
of re-merging for every experiment.

Usage:
  MODEL_PATH=Qwen/Qwen3-14B BASE_BF16_DIR=models/qwen3_14b_base_bf16 \
  python dequant_base.py
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig

BASE = os.environ.get("MODEL_PATH", "Qwen/Qwen3-14B")
OUT = os.environ.get("BASE_BF16_DIR", "models/qwen3_14b_base_bf16")

print("Loading base (NF4)...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
    device_map="auto", trust_remote_code=True, attn_implementation="sdpa")

import bitsandbytes as bnb
from safetensors import safe_open
idx = Path(BASE) / "model.safetensors.index.json"
if idx.exists():
    weight_map = json.load(open(idx))["weight_map"]
    fixed = 0
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            if (getattr(module.weight, "quant_state", None) is None and
                    getattr(module, "quant_state", None) is None):
                wkey = f"{name}.weight"
                if wkey not in weight_map:
                    continue
                with safe_open(str(Path(BASE) / weight_map[wkey]),
                               framework="pt", device="cpu") as sf:
                    orig = sf.get_tensor(wkey)
                if orig.dtype in (torch.bfloat16, torch.float16, torch.float32):
                    module.weight = bnb.nn.Params4bit(
                        orig.to(torch.bfloat16), requires_grad=False,
                        quant_type="nf4", compress_statistics=True,
                        quant_storage=torch.uint8, module=module).to("cuda:0")
                    if module.quant_state is None:
                        module.quant_state = module.weight.quant_state
                    fixed += 1
                del orig
                torch.cuda.empty_cache()
    print(f"Re-quantized {fixed} bf16 layers", flush=True)

print("Dequantizing all Linear4bit -> bf16 (streaming to CPU)...", flush=True)
sd_out = {}
n = 0
for name, module in model.named_modules():
    if isinstance(module, bnb.nn.Linear4bit):
        qs = getattr(module.weight, "quant_state", None) or module.quant_state
        w = bnb.functional.dequantize_4bit(module.weight.data, qs)
        sd_out[name + ".weight"] = w.to(torch.bfloat16).cpu()
        if module.bias is not None:
            sd_out[name + ".bias"] = module.bias.data.to(torch.bfloat16).cpu()
        n += 1
        del w
        if n % 40 == 0:
            torch.cuda.empty_cache()
            print(f"  {n} layers", flush=True)
for name, param in model.named_parameters():
    if name not in sd_out and "quant" not in name:
        sd_out[name] = param.data.to(torch.bfloat16).cpu()
print(f"dequantized {n} layers; entries {len(sd_out)}", flush=True)

cfg = AutoConfig.from_pretrained(BASE, trust_remote_code=True)
del model
torch.cuda.empty_cache()
with torch.device("meta"):
    clean = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16)
missing, unexpected = clean.load_state_dict(sd_out, assign=True, strict=False)
missing = [m for m in missing if not m.endswith("inv_freq")]
assert not missing and not unexpected, (missing[:5], unexpected[:5])
if clean.config.tie_word_embeddings and "lm_head.weight" not in sd_out:
    clean.tie_weights()
os.makedirs(OUT, exist_ok=True)
clean.save_pretrained(OUT, safe_serialization=True, max_shard_size="4GB")
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
tok.save_pretrained(OUT)
# strip stale quantization_config
p = os.path.join(OUT, "config.json")
c = json.load(open(p, encoding="utf-8"))
c.pop("quantization_config", None)
json.dump(c, open(p, "w", encoding="utf-8"), indent=2)
print(f"Saved bf16 base -> {OUT}", flush=True)
