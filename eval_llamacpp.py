#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Evaluate a GGUF model served by llama.cpp's llama-server against the held-out
test set. Measures F1 and steady-state throughput.

Prompts are built with the HF tokenizer's chat template (enable_thinking=False)
and posted RAW to /completion, so they are byte-identical to training and do
not depend on llama.cpp's own chat-template handling — the usual reason a GGUF
build scores worse than the same weights under transformers.

Requires llama-server to be listening already.
  TEST_SET=runs/<stamp>/test_set.parquet \
  TOKENIZER_DIR=models/qwen3_14b_contracts_merged \
  python -u eval_llamacpp.py [N_DOCS] [CONCURRENCY]
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contract_prompt import SYSTEM_PROMPT, USER_TEMPLATE

TEST = os.environ["TEST_SET"]              # <run dir>/test_set.parquet
BASE = os.environ.get("TOKENIZER_DIR", "models/qwen3_14b_contracts_merged")
URL = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080/completion")
RAW_LOG = os.environ.get("RAW_LOG", "eval_llamacpp_raw.jsonl")

N_DOCS = int(sys.argv[1]) if len(sys.argv) > 1 else 600
CONC = int(sys.argv[2]) if len(sys.argv) > 2 else 32

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(BASE)

df = pd.read_parquet(TEST)
df = df[df["input_text"].fillna("") != ""].head(N_DOCS).reset_index(drop=True)
print(f"test docs: {len(df)}  concurrency: {CONC}", flush=True)

MAX_PROMPT_TOK = 2900
prompts = []
for t in df["input_text"]:
    s = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": USER_TEMPLATE.format(text=t)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompts.append(tok(s, add_special_tokens=False).input_ids[:MAX_PROMPT_TOK])


def call(p):
    r = requests.post(URL, json={
        "prompt": p, "n_predict": 160, "temperature": 0.0,
        "cache_prompt": True}, timeout=600)
    r.raise_for_status()
    return r.json()["content"].strip()


# warmup
with ThreadPoolExecutor(max_workers=CONC) as ex:
    list(ex.map(call, prompts[:8]))
t0 = time.time()
with ThreadPoolExecutor(max_workers=CONC) as ex:
    resps = list(ex.map(call, prompts[8:]))
dt = time.time() - t0
print(f"STEADY-STATE: {len(resps)} docs in {dt:.0f}s = {len(resps)/dt:.2f} doc/s",
      flush=True)

resps = [None] * 8 + resps
with open(RAW_LOG, "w", encoding="utf-8") as fo:
    for i, r_ in enumerate(resps):
        fo.write(json.dumps({"i": i, "resp": r_}, ensure_ascii=False) + "\n")
ok = tp = fp = fn = tn = 0
ceo_ok = ceo_n = per_ok = per_n = 0
for i, (_, row) in enumerate(df.iterrows()):
    if resps[i] is None:
        continue
    try:
        p = json.loads(resps[i])
        assert isinstance(p, dict)
        ok += 1
    except Exception:
        continue
    pc, tc = int(p.get("contract") or 0), int(row["contract"])
    if pc and tc: tp += 1
    elif pc and not tc: fp += 1
    elif not pc and tc: fn += 1
    else: tn += 1
    if tc == 1 and pc == 1:
        ceo_n += 1
        ceo_ok += int(int(p.get("ceo") or 0) == int(row["ceo"]))
        per_n += 1
        pa = p.get("person")
        a = pa.strip().lower() if isinstance(pa, str) else ""
        b = row["person"].strip().lower() if isinstance(row["person"], str) else ""
        per_ok += int(a == b)
prec = tp / max(1, tp + fp)
rec = tp / max(1, tp + fn)
f1 = 2 * prec * rec / max(1e-9, prec + rec)
print(f"parse_ok: {ok}/{len(df)-8}")
print(f"contract: P={prec:.3f} R={rec:.3f} F1={f1:.3f} acc={(tp+tn)/max(1,tp+tn+fp+fn):.3f}")
print(f"ceo acc: {ceo_ok/max(1,ceo_n):.3f}  person exact: {per_ok/max(1,per_n):.3f}")
