#!/usr/bin/env python3
"""AWQ-quantize the merged model to W4A16 for vLLM (~28 GB bf16 -> ~10 GB).

Calibration is a JSONL of real prompts from your own corpus, one object per
line with "system" and "user" keys; 256 documents is plenty. Calibrating on
in-domain text matters considerably more than the sample count.

Usage:
  MERGED_DIR=models/qwen3_14b_contracts_merged \
  CALIB_JSONL=data/awq_calib.jsonl \
  AWQ_DIR=models/qwen3_14b_contracts_awq \
  python quantize_awq.py
"""
import json
import os

from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

SRC = os.environ.get("MERGED_DIR", "models/qwen3_14b_contracts_merged")
CAL = os.environ.get("CALIB_JSONL", "data/awq_calib.jsonl")
OUT = os.environ.get("AWQ_DIR", "models/qwen3_14b_contracts_awq")

print("Loading tokenizer + calibration data...")
tok = AutoTokenizer.from_pretrained(SRC)
rows = [json.loads(l) for l in open(CAL, encoding="utf-8")]
texts = [tok.apply_chat_template(
    [{"role": "system", "content": r["system"]},
     {"role": "user", "content": r["user"]}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False)
    for r in rows]
ds = Dataset.from_dict({"text": texts})

print("Loading bf16 model (cpu, layer-wise onload during calibration)...")
model = AutoModelForCausalLM.from_pretrained(SRC, dtype="bfloat16")

recipe = AWQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["lm_head"],
)

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=2304,
    num_calibration_samples=len(ds),
    output_dir=OUT,
)
tok.save_pretrained(OUT)
print(f"Saved AWQ model -> {OUT}")
