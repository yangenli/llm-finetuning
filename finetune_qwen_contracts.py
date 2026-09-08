#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Fine-tune Qwen3-14B (QLoRA) to classify SEC exhibits as executive contracts.

Task : exhibit head text -> JSON {contract, types, person, title, ceo,
       executed, amendment}
Data : one parquet of labeled examples (schema in docs/data-format.md)

Fits on a single 32 GB GPU at batch 4 / accum 2; a 24 GB card works at
batch 2 / accum 4.

Usage:
  MODEL_PATH=Qwen/Qwen3-14B DATA_PATH=data/labeled.parquet \
  OUTPUT_DIR=runs python finetune_qwen_contracts.py

  SMOKE=1 python finetune_qwen_contracts.py   # 120 train / 20 test, 20 steps
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from contract_prompt import SYSTEM_PROMPT, USER_TEMPLATE, TYPES, build_target_json

# --- LoRA ---
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# --- Training ---
LEARNING_RATE = 2e-4
NUM_EPOCHS = 2
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "cosine"
MAX_SEQ_LENGTH = 2304        # system ~700 tok + input ~1200 tok + target ~120 tok
TEST_SIZE = 0.1
RANDOM_SEED = 42

DATA_PATH = Path(os.environ.get("DATA_PATH", "data/labeled.parquet"))
SMOKE = os.environ.get("SMOKE", "") == "1"   # tiny run: verify env/model/training loop
MODEL_NAME = os.environ.get("MODEL_PATH", "Qwen/Qwen3-14B")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
GRAD_ACCUM_STEPS = int(os.environ.get("GRAD_ACCUM_STEPS", "2"))
_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "runs")) / f"qwen3_14b_contracts_{_stamp}"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
config = dict(lora_r=LORA_R, lora_alpha=LORA_ALPHA, lr=LEARNING_RATE,
              epochs=NUM_EPOCHS, batch=BATCH_SIZE, accum=GRAD_ACCUM_STEPS,
              max_len=MAX_SEQ_LENGTH, model=MODEL_NAME, data=str(DATA_PATH))
with open(OUTPUT_DIR / "training_config.json", "w") as f:
    json.dump(config, f, indent=2)
print("=" * 60)
print("Fine-tune Qwen3-14B: SEC executive-contract classifier")
for k, v in config.items():
    print(f"  {k:10}: {v}")

# ============================== 1. DATA ==============================
print(f"\n{'='*60}\n1. Loading data\n{'='*60}")
df = pd.read_parquet(DATA_PATH)
df = df[(df["parse_ok"] == 1) & (df["input_text"].fillna("") != "")]
df["target"] = df.apply(build_target_json, axis=1)
print(f"Usable labeled samples: {len(df)}")

df["strata"] = df["stratum"] + "_" + df["primary"].astype(str)
vc = df["strata"].value_counts()
df.loc[df["strata"].isin(vc[vc < 2].index), "strata"] = "rare"
train_df, test_df = train_test_split(df, test_size=TEST_SIZE,
                                     random_state=RANDOM_SEED,
                                     stratify=df["strata"])
if SMOKE:
    train_df = train_df.sample(n=min(120, len(train_df)), random_state=RANDOM_SEED)
    test_df = test_df.sample(n=min(20, len(test_df)), random_state=RANDOM_SEED)
    print("SMOKE MODE: 120 train / 20 test, 20 steps")
print(f"Train: {len(train_df)}, Test: {len(test_df)}")
test_df.to_parquet(OUTPUT_DIR / "test_set.parquet", index=False)

# ============================== 2. MODEL ==============================
print(f"\n{'='*60}\n2. Loading model\n{'='*60}")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config, device_map="auto",
    trust_remote_code=True,
    attn_implementation="sdpa",   # no flash_attn wheel on this machine
)
model.config.use_cache = False
print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.1f} GB")

# Unsloth pre-quantized checkpoint fix: re-quantize bf16 layers lacking
# quant_state (same issue/fix as the individualism project).
import bitsandbytes as bnb
from safetensors import safe_open

_index_path = Path(MODEL_NAME) / "model.safetensors.index.json"
if _index_path.exists():
    with open(_index_path) as _f:
        _weight_map = json.load(_f)["weight_map"]
    _fixed = 0
    for _name, _module in model.named_modules():
        if isinstance(_module, bnb.nn.Linear4bit):
            if (getattr(_module.weight, "quant_state", None) is None and
                    getattr(_module, "quant_state", None) is None):
                _wkey = f"{_name}.weight"
                if _wkey not in _weight_map:
                    continue
                _shard = str(Path(MODEL_NAME) / _weight_map[_wkey])
                with safe_open(_shard, framework="pt", device="cpu") as _sf:
                    _orig = _sf.get_tensor(_wkey)
                if _orig.dtype in (torch.bfloat16, torch.float16, torch.float32):
                    _module.weight = bnb.nn.Params4bit(
                        _orig.to(torch.bfloat16), requires_grad=False,
                        quant_type="nf4", compress_statistics=True,
                        quant_storage=torch.uint8, module=_module,
                    ).to("cuda:0")
                    if _module.quant_state is None:
                        _module.quant_state = _module.weight.quant_state
                    _fixed += 1
                del _orig
                torch.cuda.empty_cache()
    if _fixed:
        print(f"Re-quantized {_fixed} bf16 layers to NF4")

# ============================== 3. LoRA ==============================
print(f"\n{'='*60}\n3. Adding LoRA\n{'='*60}")
from peft import LoraConfig, get_peft_model

for p in model.parameters():
    p.requires_grad = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM"))
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {trainable:,}")

# ============================== 4. DATASET ==============================
print(f"\n{'='*60}\n4. Preparing dataset\n{'='*60}")
from datasets import Dataset

def format_example(row):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(text=row["input_text"])},
        {"role": "assistant", "content": row["target"]},
    ]
    return {"text": tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
        enable_thinking=False)}

train_dataset = Dataset.from_list(
    [format_example(r) for _, r in train_df.iterrows()])
lens = [len(tokenizer(train_dataset[i]["text"])["input_ids"])
        for i in range(min(200, len(train_dataset)))]
print(f"Token lengths: mean={np.mean(lens):.0f} p95={np.percentile(lens, 95):.0f} "
      f"max={np.max(lens)} (cap {MAX_SEQ_LENGTH})")

# ============================== 5. TRAIN ==============================
print(f"\n{'='*60}\n5. Training\n{'='*60}")
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir=str(OUTPUT_DIR / "checkpoints"),
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type=LR_SCHEDULER,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,
    bf16=True, logging_steps=20, save_strategy="epoch",
    seed=RANDOM_SEED, report_to="none", optim="adamw_8bit",
    max_grad_norm=1.0, gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_length=MAX_SEQ_LENGTH, dataset_text_field="text", packing=False,
    **({"max_steps": 20} if SMOKE else {}),
)
trainer = SFTTrainer(model=model, processing_class=tokenizer,
                     train_dataset=train_dataset, args=training_args)
t0 = time.time()
result = trainer.train()
print(f"\nTrain done in {(time.time()-t0)/60:.1f} min, "
      f"loss={result.training_loss:.4f}, "
      f"peak GPU {torch.cuda.max_memory_allocated()/1e9:.1f} GB")

# ============================== 6. EVAL ==============================
print(f"\n{'='*60}\n6. Evaluating\n{'='*60}")
model.eval()
preds, errors = [], 0
for i, (_, row) in enumerate(test_df.iterrows()):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(text=row["input_text"])},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True, enable_thinking=False).to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=160, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()
    try:
        p = json.loads(resp)
        preds.append({"doc_key": row["doc_key"], "resp": resp,
                      "p_contract": int(p["contract"]),
                      "p_types": set(t for t in p.get("types", []) if t in TYPES),
                      "p_ceo": int(p.get("ceo") or 0),
                      "p_person": p.get("person"),
                      "t_contract": int(row["contract"]),
                      "t_types": set(row["types"]),
                      "t_ceo": int(row["ceo"]),
                      "t_person": row["person"]})
    except Exception:
        errors += 1
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(test_df)}")

pv = pd.DataFrame(preds)
print(f"\nParse errors: {errors}/{len(test_df)}")
acc_c = (pv.p_contract == pv.t_contract).mean()
print(f"contract accuracy: {acc_c:.3f}")
tp = ((pv.p_contract == 1) & (pv.t_contract == 1)).sum()
prec = tp / max(1, (pv.p_contract == 1).sum())
rec = tp / max(1, (pv.t_contract == 1).sum())
print(f"contract precision: {prec:.3f}  recall: {rec:.3f}")
inter = pv.apply(lambda r: len(r.p_types & r.t_types), axis=1).sum()
micro_p = inter / max(1, pv.p_types.map(len).sum())
micro_r = inter / max(1, pv.t_types.map(len).sum())
print(f"types micro-P/R: {micro_p:.3f}/{micro_r:.3f}")
sub = pv[pv.t_contract == 1]
print(f"ceo accuracy (on contracts): {(sub.p_ceo == sub.t_ceo).mean():.3f}")
def _norm_name(x):
    return x.strip().lower() if isinstance(x, str) else ""
name_match = sub.apply(lambda r: _norm_name(r.t_person) == _norm_name(r.p_person),
                       axis=1).mean()
print(f"person exact-match (on contracts): {name_match:.3f}")
pv.drop(columns=["p_types", "t_types"]).to_parquet(
    OUTPUT_DIR / "test_predictions.parquet", index=False)

# ============================== 7. SAVE ==============================
model.save_pretrained(str(OUTPUT_DIR / "lora_adapter"))
tokenizer.save_pretrained(str(OUTPUT_DIR / "lora_adapter"))
print(f"\nSaved LoRA adapter -> {OUTPUT_DIR / 'lora_adapter'}")
