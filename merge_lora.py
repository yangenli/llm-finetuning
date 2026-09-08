#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Merge the contract-classifier LoRA into the Qwen3-14B base -> bf16 model dir.

Loads the base exactly as in training (bnb NF4 + the re-quantize fix), attaches
the adapter, then merge_and_unload() — peft dequantizes NF4 layer-by-layer and
folds in BA, so the merged bf16 weights match what the adapter was trained
against. Saved model feeds vLLM (and optional AWQ quantization) downstream.

Needs ~28 GB of free VRAM during the merge. Output is ~28 GB of bf16
safetensors, which is what vLLM (and the optional AWQ step) consumes.

Usage:
  MODEL_PATH=Qwen/Qwen3-14B \
  ADAPTER=runs/qwen3_14b_contracts_<stamp>/lora_adapter \
  MERGED_DIR=models/qwen3_14b_contracts_merged \
  python merge_lora.py
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE = os.environ.get("MODEL_PATH", "Qwen/Qwen3-14B")
ADAPTER = os.environ["ADAPTER"]            # <run dir>/lora_adapter
OUT = os.environ.get("MERGED_DIR", "models/qwen3_14b_contracts_merged")

print("Loading base (NF4)...")
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
    device_map="auto", trust_remote_code=True, attn_implementation="sdpa")

# Unsloth pre-quantized checkpoint fix (same as training)
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
    if fixed:
        print(f"Re-quantized {fixed} bf16 layers")

print("Attaching adapter...")
model = PeftModel.from_pretrained(model, ADAPTER)
print("Merging (dequantize + fold LoRA)...")
model = model.merge_and_unload()
print(f"GPU memory after merge: {torch.cuda.memory_allocated()/1e9:.1f} GB")

# peft re-quantizes merged weights back to NF4, so dequantize every Linear4bit
# to bf16 manually, stream to CPU, then pour into a clean skeleton and save
# (save_pretrained on the quantized model itself hits transformers 5.5
# reverse_transform NotImplementedError).
print("Dequantizing merged weights to bf16 (streaming to CPU)...")
sd_out = {}
n_deq = 0
for name, module in model.named_modules():
    if isinstance(module, bnb.nn.Linear4bit):
        qs = getattr(module.weight, "quant_state", None) or module.quant_state
        w = bnb.functional.dequantize_4bit(module.weight.data, qs)
        sd_out[name + ".weight"] = w.to(torch.bfloat16).cpu()
        if module.bias is not None:
            sd_out[name + ".bias"] = module.bias.data.to(torch.bfloat16).cpu()
        n_deq += 1
        del w
        if n_deq % 40 == 0:
            torch.cuda.empty_cache()
            print(f"  dequantized {n_deq} layers")
for name, param in model.named_parameters():
    if name not in sd_out and "quant" not in name:
        sd_out[name] = param.data.to(torch.bfloat16).cpu()
print(f"Dequantized {n_deq} Linear4bit layers; state dict entries: {len(sd_out)}")

from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(BASE, trust_remote_code=True)
del model
torch.cuda.empty_cache()
with torch.device("meta"):
    clean = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16)
missing, unexpected = clean.load_state_dict(sd_out, assign=True, strict=False)
missing = [m for m in missing if not m.endswith("inv_freq")]
print("missing:", missing[:10], " unexpected:", unexpected[:10])
assert not missing and not unexpected, "state dict mismatch"
if clean.config.tie_word_embeddings and "lm_head.weight" not in sd_out:
    clean.tie_weights()
os.makedirs(OUT, exist_ok=True)
clean.save_pretrained(OUT, safe_serialization=True, max_shard_size="4GB")
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
tok.save_pretrained(OUT)
print(f"Saved merged model -> {OUT}")
