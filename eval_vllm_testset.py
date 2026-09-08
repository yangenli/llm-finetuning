#!/usr/bin/env python3
"""Score a served model against the held-out test set, and measure
steady-state throughput at the same time.

TEST_SET is <run dir>/test_set.parquet, which finetune_qwen_contracts.py
writes out BEFORE training. The split is fixed by RANDOM_SEED, so every
runtime — transformers, vLLM bf16, AWQ, GGUF — is scored on identical
documents and the numbers are comparable.

Usage:
  TEST_SET=runs/<stamp>/test_set.parquet \
  SERVE_MODEL=models/qwen3_14b_contracts_merged \
  [QUANT=fp8] [N_DOCS=500] python eval_vllm_testset.py
"""
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contract_prompt import SYSTEM_PROMPT, USER_TEMPLATE, TYPES

TEST = os.environ["TEST_SET"]              # <run dir>/test_set.parquet
MODEL = os.environ.get("SERVE_MODEL", "models/qwen3_14b_contracts_merged")


def main():
    df = pd.read_parquet(TEST)
    df = df[df["input_text"].fillna("") != ""].reset_index(drop=True)
    print(f"test docs: {len(df)}", flush=True)

    from vllm import LLM, SamplingParams
    quant = os.environ.get("QUANT") or None
    n_docs = int(os.environ.get("N_DOCS", "0"))
    if n_docs:
        df = df.head(n_docs)
    llm = LLM(model=MODEL, max_model_len=4096,
              gpu_memory_utilization=float(os.environ.get("GPU_MEM", "0.92")),
              enable_prefix_caching=True,
              **({"quantization": quant} if quant else {}))
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=160)
    texts = [tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": USER_TEMPLATE.format(text=t)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for t in df["input_text"]]
    # tokenize + hard-cap at 3900 tokens so no request can exceed max_model_len
    from vllm import TokensPrompt
    prompts = [TokensPrompt(prompt_token_ids=tok(t, add_special_tokens=False).input_ids[:3900])
               for t in texts]

    # warmup on first 8 (compile/graphs), then timed run on the rest
    _ = llm.generate(prompts[:8], sp, use_tqdm=False)
    t0 = time.time()
    outs = llm.generate(prompts[8:], sp, use_tqdm=False)
    dt = time.time() - t0
    rate = len(outs) / dt
    print(f"STEADY-STATE: {len(outs)} docs in {dt:.0f}s = {rate:.2f} doc/s", flush=True)

    resps = [None] * 8 + [o.outputs[0].text.strip() for o in outs]
    ok = tp = fp = fn = tn = 0
    ceo_ok = ceo_n = 0
    per_ok = per_n = 0
    for i, (_, row) in enumerate(df.iterrows()):
        if resps[i] is None:
            continue
        try:
            p = json.loads(resps[i])
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
            a = (p.get("person") or "").strip().lower()
            b = (row["person"] or "").strip().lower() if isinstance(row["person"], str) else ""
            per_ok += int(a == b)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    print(f"parse_ok: {ok}/{len(df)-8}")
    print(f"contract: P={prec:.3f} R={rec:.3f} F1={f1:.3f} acc={(tp+tn)/max(1,tp+tn+fp+fn):.3f}")
    print(f"ceo acc (both-contract): {ceo_ok/max(1,ceo_n):.3f}")
    print(f"person exact: {per_ok/max(1,per_n):.3f}")


if __name__ == "__main__":
    main()
