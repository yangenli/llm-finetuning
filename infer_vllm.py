#!/usr/bin/env python3
"""Full-corpus inference with vLLM — the fast path.

Mirrors infer_contracts_qwen.py (same document pool, same resumable per-file
outputs, same output schema) but uses vLLM continuous batching plus prefix
caching, which matters here because the ~700-token system prompt is identical
for every document and is therefore computed once rather than per document.

vLLM is Linux-only; on Windows run this under WSL.

Usage (one instance per GPU, on disjoint years):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python infer_vllm.py \
    --years 2005-2012 [--gpu-mem 0.90] [--max-files N] [--out-root DIR]
"""
import argparse
import glob
import json
import os
import sys
import time

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contract_prompt import SYSTEM_PROMPT, USER_TEMPLATE, TYPES, doc_input_text

MODEL = os.environ.get("SERVE_MODEL", "models/qwen3_14b_contracts_awq")
IN_ROOT = os.environ.get("CORPUS_ROOT", "data/corpus")

META_COLS = ["cik", "year", "date", "form_type", "accession", "web_url",
             "exhibit", "doc_type", "sequence", "filename", "description"]


def parse_years(s):
    if "-" in s:
        a, b = s.split("-")
        return set(range(int(a), int(b) + 1))
    return {int(s)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2005-2026")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out-root",
                    default=os.environ.get("OUT_ROOT", "output/contracts_llm"))
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--quant", default=None,
                    help="e.g. fp8 for on-the-fly FP8 weight quantization")
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--include-ex99", action="store_true")
    args = ap.parse_args()
    years = parse_years(args.years)

    from vllm import LLM, SamplingParams, TokensPrompt
    print("Loading vLLM engine...", flush=True)
    llm = LLM(model=args.model, max_model_len=4096,
              gpu_memory_utilization=args.gpu_mem,
              enable_prefix_caching=True,
              **({"quantization": args.quant} if args.quant else {}))
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=160)

    def build_prompt(text):
        return tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": USER_TEMPLATE.format(text=text)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    src_files = sorted(glob.glob(os.path.join(IN_ROOT, "**", "*.parquet"),
                                 recursive=True))
    src_files = [f for f in src_files
                 if int(os.path.basename(f)[:4]) in years]
    if args.max_files:
        src_files = src_files[:args.max_files]
    print(f"Day-files in {args.years}: {len(src_files)}", flush=True)

    t0 = time.time()
    n_total = 0
    for fi, src in enumerate(src_files):
        rel = os.path.relpath(src, IN_ROOT)
        out = os.path.join(args.out_root, rel)
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
        live_idx = [i for i, t in enumerate(texts) if t]
        # tokenize + hard-cap so no request can exceed max_model_len
        prompts = [TokensPrompt(prompt_token_ids=tok(
                       build_prompt(texts[i]),
                       add_special_tokens=False).input_ids[:3900])
                   for i in live_idx]
        outs = llm.generate(prompts, sp, use_tqdm=False) if prompts else []
        rmap = {i: o.outputs[0].text.strip() for i, o in zip(live_idx, outs)}

        results = []
        for i in range(len(df)):
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
                rec.update(contract=None, types=None, person=None, title=None,
                           ceo=None, executed=None, amendment=None, parse_ok=0)
            results.append(rec)

        res = pd.concat([df[META_COLS], pd.DataFrame(results)], axis=1)
        tmp = out + ".part"
        res.to_parquet(tmp, index=False)
        os.replace(tmp, out)

        n_total += len(df)
        el = time.time() - t0
        rate = n_total / el if el > 0 else 0
        print(f"[{fi+1}/{len(src_files)}] {os.path.basename(src)}: {len(df)} docs "
              f"| cum {n_total:,} @ {rate:.1f} doc/s", flush=True)

    print(f"DONE. {n_total:,} docs in {(time.time()-t0)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
