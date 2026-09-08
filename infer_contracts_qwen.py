#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Full-corpus inference with the fine-tuned adapter, using plain transformers
(left-padded batched generate, length-sorted batches). Nothing extra to
install — but roughly an order of magnitude slower than infer_vllm.py, so use
this for small corpora or as a correctness reference for the faster runtimes.

  pool     : ALL EX-10 documents by default — no keyword gate, so a screen's
             misses cannot propagate into the sample. --include-ex99 widens it.
  layout   : reads  $CORPUS_ROOT/{year}/QTR{q}/{day}.parquet
             writes $OUT_ROOT/{year}/QTR{q}/{day}.parquet
             One output file per input file, so a run is resumable: an existing
             non-empty output is skipped.
  multi-GPU: run one instance per GPU on disjoint years, e.g.
             CUDA_VISIBLE_DEVICES=0 python infer_contracts_qwen.py --years 2005-2011 --adapter <run_dir>
             CUDA_VISIBLE_DEVICES=1 python infer_contracts_qwen.py --years 2012-2018 --adapter <run_dir>
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import argparse
import glob
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from contract_prompt import (SYSTEM_PROMPT, USER_TEMPLATE, TYPES,
                             doc_input_text)

IN_ROOT = os.environ.get("CORPUS_ROOT", "data/corpus")
OUT_ROOT = os.environ.get("OUT_ROOT", "output/contracts_llm")
BASE_MODEL = os.environ.get("MODEL_PATH", "Qwen/Qwen3-14B")

META_COLS = ["cik", "year", "date", "form_type", "accession", "web_url",
             "exhibit", "doc_type", "sequence", "filename", "description"]


def parse_years(s):
    if "-" in s:
        a, b = s.split("-")
        return set(range(int(a), int(b) + 1))
    return {int(s)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True,
                    help="run dir (uses <dir>/lora_adapter) or direct adapter/checkpoint path")
    ap.add_argument("--years", default="2005-2026")
    ap.add_argument("--include-ex99", action="store_true")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--max-files", type=int, default=0)
    args = ap.parse_args()

    adapter = Path(args.adapter)
    if (adapter / "lora_adapter").exists():
        adapter = adapter / "lora_adapter"
    years = parse_years(args.years)

    # ---------------- model ----------------
    print("Loading model...")
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"       # CRITICAL for batched generation
    tokenizer.truncation_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True),
        device_map="auto", trust_remote_code=True,
        attn_implementation="sdpa")
    model.config.use_cache = True

    # Unsloth pre-quantized checkpoint fix (see finetune script)
    import bitsandbytes as bnb
    from safetensors import safe_open
    _index_path = Path(BASE_MODEL) / "model.safetensors.index.json"
    if _index_path.exists():
        with open(_index_path) as f:
            _weight_map = json.load(f)["weight_map"]
        _fixed = 0
        for _name, _module in model.named_modules():
            if isinstance(_module, bnb.nn.Linear4bit):
                if (getattr(_module.weight, "quant_state", None) is None and
                        getattr(_module, "quant_state", None) is None):
                    _wkey = f"{_name}.weight"
                    if _wkey not in _weight_map:
                        continue
                    with safe_open(str(Path(BASE_MODEL) / _weight_map[_wkey]),
                                   framework="pt", device="cpu") as sf:
                        _orig = sf.get_tensor(_wkey)
                    if _orig.dtype in (torch.bfloat16, torch.float16, torch.float32):
                        _module.weight = bnb.nn.Params4bit(
                            _orig.to(torch.bfloat16), requires_grad=False,
                            quant_type="nf4", compress_statistics=True,
                            quant_storage=torch.uint8, module=_module).to("cuda:0")
                        if _module.quant_state is None:
                            _module.quant_state = _module.weight.quant_state
                        _fixed += 1
                    del _orig
                    torch.cuda.empty_cache()
        if _fixed:
            print(f"Re-quantized {_fixed} bf16 layers")

    model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    print(f"Adapter: {adapter}")
    print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    MAX_PROMPT_LEN = 2200

    def classify_batch(texts):
        prompts = [tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": USER_TEMPLATE.format(text=t)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
            for t in texts]
        inputs = tokenizer(prompts, padding=True, truncation=True,
                           max_length=MAX_PROMPT_LEN,
                           return_tensors="pt").to("cuda")
        plen = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=160, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        return [tokenizer.decode(o[plen:], skip_special_tokens=True).strip()
                for o in out]

    # ---------------- files ----------------
    src_files = sorted(glob.glob(os.path.join(IN_ROOT, "**", "*.parquet"),
                                 recursive=True))
    src_files = [f for f in src_files
                 if int(os.path.basename(f)[:4]) in years]
    if args.max_files:
        src_files = src_files[:args.max_files]
    print(f"Day-files in {args.years}: {len(src_files)}")

    t0 = time.time()
    n_docs_total = 0
    for fi, src in enumerate(src_files):
        rel = os.path.relpath(src, IN_ROOT)
        out = os.path.join(OUT_ROOT, rel)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)

        df = pq.read_table(src).to_pandas()
        if not args.include_ex99:
            df = df[df.exhibit == "EX-10"]
        df = df.reset_index(drop=True)
        if len(df) == 0:
            pd.DataFrame(columns=META_COLS).to_parquet(out, index=False)
            continue

        texts = [doc_input_text(t) for t in df["text"]]
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]),
                       reverse=True)
        results = [None] * len(df)
        for bs in range(0, len(order), args.batch_size):
            idx = order[bs:bs + args.batch_size]
            # empty-text docs (binary) skip the model
            live = [i for i in idx if texts[i]]
            resps = classify_batch([texts[i] for i in live]) if live else []
            rmap = dict(zip(live, resps))
            for i in idx:
                resp = rmap.get(i, "")
                rec = {"raw_response": resp, "no_text": int(not texts[i])}
                try:
                    p = json.loads(resp)
                    rec.update(
                        contract=int(p["contract"]),
                        types=",".join(t for t in p.get("types", []) if t in TYPES),
                        person=p.get("person"), title=p.get("title"),
                        ceo=int(p.get("ceo") or 0),
                        executed=int(p.get("executed") or 0),
                        amendment=int(p.get("amendment") or 0), parse_ok=1)
                except Exception:
                    rec.update(contract=None, types=None, person=None,
                               title=None, ceo=None, executed=None,
                               amendment=None, parse_ok=0)
                results[i] = rec

        res = pd.concat([df[META_COLS], pd.DataFrame(results)], axis=1)
        tmp = out + ".part"
        res.to_parquet(tmp, index=False)
        os.replace(tmp, out)

        n_docs_total += len(df)
        el = time.time() - t0
        rate = n_docs_total / el if el > 0 else 0
        print(f"[{fi+1}/{len(src_files)}] {os.path.basename(src)}: {len(df)} docs "
              f"| cum {n_docs_total:,} @ {rate:.1f} doc/s", flush=True)

    print(f"\nDONE. {n_docs_total:,} docs in {(time.time()-t0)/3600:.2f} h")


if __name__ == "__main__":
    main()
