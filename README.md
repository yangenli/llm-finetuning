# QLoRA fine-tuning for structured document classification

The working pipeline I use to fine-tune **Qwen3-14B** to read a document and
emit a **structured JSON record**, and then to run that model over a corpus of
millions of documents on consumer GPUs.

The concrete task here is SEC filing exhibits: given the head of an EX-10 /
EX-99 document, decide whether it is an executive employment or compensation
contract, and extract who it is with, their title, whether they are the CEO,
whether it is an executed agreement or a form, and whether it amends an earlier
one. But nothing below is specific to that task — swap the prompt and the
target schema in `contract_prompt.py` and the same pipeline trains a classifier
or extractor for any document corpus.

**This repository is code only.** No training data, no labels, no corpus, no
model weights. Every path is configured through environment variables, so you
point the scripts at your own data. See
[`docs/data-format.md`](docs/data-format.md) for the schema your data has to be
in.

---

## The shape of the problem

Text-as-data research increasingly needs a judgment made on *every* document in
a large corpus — not a keyword count, and not a sample small enough to send to a
commercial API. Three constraints collide:

1. **A keyword or regex screen is not a measurement.** Whatever it misses is
   silently absent from the sample, and you cannot characterize the miss.
2. **A frontier API on millions of documents** is expensive, slow, non-local
   (a problem for licensed data), and not stable across model versions — a
   revision mid-project breaks comparability.
3. **A small open model prompted zero-shot** is not reliable enough on a
   domain-specific taxonomy.

The way out is distillation: label a few tens of thousands of documents well —
with a frontier model through its batch API, with research assistants, or both —
then fine-tune a mid-size open model on those labels and run it locally over
everything. The fine-tuned model is a fixed artifact: it does not change under
you, it costs nothing per document after training, and the data never leaves
your machine. That is what this code does.

**Input** — the stripped-HTML head of one document (first 4,000 characters:
title, parties, recitals):

```
EMPLOYMENT AGREEMENT This Employment Agreement (this "Agreement") is entered
into as of January 15, 2014, by and between ACME CORP., a Delaware corporation
(the "Company"), and Jane Q. Doe ("Executive")...
```

**Output** — one JSON object, nothing else:

```json
{"contract": 1, "types": ["employment"], "person": "Jane Q. Doe",
 "title": "President and Chief Executive Officer", "ceo": 1,
 "executed": 1, "amendment": 0}
```

Training the model to emit JSON rather than a label id is what makes one pass
produce a whole record: a binary decision, a multi-label taxonomy of 21 contract
types, and two extracted strings, all at once.

---

## Pipeline

```
   base model (Hugging Face)        your labeled parquet
              |                              |
              +--------------+---------------+
                             v
              finetune_qwen_contracts.py            QLoRA, 4-bit NF4
                             |                      ~2 epochs
              +--------------+---------------+
              v                              v
        lora_adapter/                 test_set.parquet
              |                              |  (frozen before training)
      +-------+--------+                     |
      v                v                     |
   path A           path B                   |
   use as-is        merge_lora.py            |
      |             -> bf16 merged           |
      |                v                     |
      |             quantize_awq.py          |
      |             -> W4A16 (~10 GB)        |
      v                v                     v
  infer_contracts   infer_vllm.py     eval_vllm_testset.py
    _qwen.py                          eval_llamacpp.py
      |                |                     |
      +--------+-------+                     v
               v                      P / R / F1 + throughput,
       per-file parquet,              identical documents for
       resumable, multi-GPU           every runtime
```

Path **A** (adapter on the 4-bit base, plain transformers) needs nothing beyond
the training environment and is the right choice for a corpus of thousands.
Path **B** (merge → quantize → vLLM) is what you want for millions: continuous
batching plus prefix caching, and the ~700-token system prompt is identical for
every document, so it is computed once instead of per document.

| Script | What it does |
|---|---|
| `contract_prompt.py` | System prompt, 21-type taxonomy, text prep, target JSON builder. **One source of truth** for labeling, training, inference, and eval. |
| `finetune_qwen_contracts.py` | QLoRA fine-tune. Writes the adapter, the frozen test split, config, and test predictions. |
| `merge_lora.py` | Fold the adapter into the base → bf16 model directory for vLLM. |
| `dequant_base.py` | Dequantize the base with *no* adapter → bf16, for GGUF + runtime-LoRA in llama.cpp. |
| `quantize_awq.py` | AWQ W4A16 quantization of the merged model (~28 GB → ~10 GB). |
| `infer_contracts_qwen.py` | Corpus inference, plain transformers, resumable, multi-GPU by year range. |
| `infer_vllm.py` | Same, on vLLM. The fast path. |
| `eval_vllm_testset.py` | Score a served model on the held-out set + measure throughput. |
| `eval_llamacpp.py` | Same against a llama.cpp `llama-server`, prompts posted raw. |

---

## Requirements

**Hardware.** Fine-tuning a 14B model in 4-bit with LoRA fits in 32 GB of VRAM
at batch 4 / accumulation 2, and in 24 GB at batch 2 / accumulation 4.
`merge_lora.py` wants ~28 GB free during the merge. Inference in AWQ W4A16 fits
comfortably in 24 GB. Everything here was developed on consumer cards.

**Software.** Python 3.11+, CUDA 12.x. `pip install -r requirements.txt`.
Versions the pipeline currently runs on:

```
torch 2.10  transformers 5.5  peft 0.19  trl 1.8  datasets 5.0
bitsandbytes 0.49  accelerate 1.14
```

vLLM and `llmcompressor` (the AWQ step) are Linux-only — on Windows run those
two steps under WSL. Training and the transformers inference path run natively
on Windows.

---

## Step 0a — download the base model from Hugging Face

The scripts never download anything implicitly. Pull the base model once,
explicitly, to a directory you control, and point `MODEL_PATH` at it.

```bash
pip install -U huggingface_hub          # provides the `hf` CLI (hub >= 1.0)
hf auth login                           # only needed for gated repos
```

**Option A — full-precision base (recommended).** ~28 GB, quantized to 4-bit on
the fly by `bitsandbytes` when the script loads it:

```bash
hf download Qwen/Qwen3-14B --local-dir models/Qwen3-14B
export MODEL_PATH=models/Qwen3-14B
```

**Option B — pre-quantized 4-bit base.** ~11 GB, so a much smaller download and
a faster load. This is the checkpoint family that needs the re-quantization fix
described in the notes at the bottom — which every script here already carries,
so it just works:

```bash
hf download unsloth/Qwen3-14B-unsloth-bnb-4bit --local-dir models/Qwen3-14B-bnb-4bit
export MODEL_PATH=models/Qwen3-14B-bnb-4bit
```

Either way, verify you got the weights and not just the metadata — a partial
download fails much later, in the middle of loading:

```bash
ls models/Qwen3-14B          # expect model-0000N-of-0000M.safetensors,
                             # model.safetensors.index.json, tokenizer.json,
                             # tokenizer_config.json, config.json
```

Notes:

- `MODEL_PATH` also accepts a bare hub id (`Qwen/Qwen3-14B`), in which case
  `transformers` downloads to `~/.cache/huggingface` on first use. Downloading
  to an explicit `--local-dir` is worth it: a 28 GB cache in the home directory
  is easy to lose track of, and on Windows you usually want it on another drive.
- Point the cache elsewhere with `HF_HOME=/path/to/cache` if you do use the
  implicit path.
- Resume an interrupted download by rerunning the same command; `hf` skips
  completed shards.
- Behind a firewall or on a mirror, set `HF_ENDPOINT` before downloading
  (for example `export HF_ENDPOINT=https://hf-mirror.com`).
- On `huggingface_hub` < 1.0 the command is `huggingface-cli download` with the
  same arguments.

Other sizes drop in unchanged — `Qwen/Qwen3-8B` trains in ~16 GB of VRAM,
`Qwen/Qwen3-32B` needs ~40 GB and a smaller batch. The pipeline is not tied to
Qwen either; any causal-LM checkpoint with a chat template works, though you
will need to drop `enable_thinking=False` for models that do not accept it.

---

## Step 0b — bring your own labels

The scripts read one parquet of labeled examples. Columns and dtypes are in
[`docs/data-format.md`](docs/data-format.md).

How I produce them, in case it is useful: draw a stratified sample from the
corpus, label it with a frontier model through its **batch API** (half price,
and a 20k-document job returns overnight) using *exactly* the `SYSTEM_PROMPT`
in `contract_prompt.py`, then hand-audit a few hundred to measure the label
noise you are about to distill. Two things matter more than the sample size:

- **Stratify.** If you sample uniformly from a corpus where the positive rate
  is a few percent, most of the labeling budget buys you obvious negatives.
  Stratify on a cheap screen — keyword hits, form type, exhibit type — and keep
  the stratum on each row (`stratum`), so the train/test split can be stratified
  too and you can report accuracy per stratum. Aggregate accuracy hides exactly
  where a model fails.
- **Label under the deployment prompt.** If the labeling prompt and the
  inference prompt differ at all, you are training the model to answer a
  question you will never ask it. Importing both from one module is the whole
  reason `contract_prompt.py` exists.

Around 20,000 labeled documents was enough here. Fewer may be plenty for a
simpler schema.

---

## Step 1 — fine-tune

```bash
MODEL_PATH=models/Qwen3-14B \
DATA_PATH=data/labeled.parquet \
OUTPUT_DIR=runs \
BATCH_SIZE=4 GRAD_ACCUM_STEPS=2 \
python finetune_qwen_contracts.py
```

Smoke-test the whole loop first — 120 train / 20 test documents, 20 steps, a
couple of minutes. It catches every environment problem before you spend a
night on a real run:

```bash
SMOKE=1 python finetune_qwen_contracts.py
```

A run writes `runs/qwen3_14b_contracts_<timestamp>/`:

```
lora_adapter/            the trained adapter (~150 MB) — this is the artifact
test_set.parquet         the held-out split, written BEFORE training
training_config.json     every hyperparameter, for the appendix
test_predictions.parquet raw model output on the test set, per document
checkpoints/             per-epoch checkpoints
```

`test_set.parquet` is written before training starts and the split is fixed by
`RANDOM_SEED`, so every later runtime — transformers, vLLM bf16, AWQ, GGUF — is
scored on byte-identical documents and the numbers are comparable across all of
them. It is also what tells you whether quantization cost you anything.

The script prints contract accuracy, precision/recall, micro-P/R over the
multi-label types, CEO-flag accuracy, and exact-match on the extracted person
name at the end of training.

**Defaults, and when to move them.** `r=16, alpha=32, dropout=0.05` on all
seven attention and MLP projections; `lr=2e-4`, cosine schedule, 5% warmup,
2 epochs, `adamw_8bit`, bf16. This is the standard QLoRA recipe and it is a
reasonable starting point for most classification/extraction tasks. Raise `r`
to 32–64 only if training loss plateaus high — for a task like this one, LoRA
capacity is rarely the binding constraint; label quality is. Three epochs
starts to overfit 20k examples. `MAX_SEQ_LENGTH` (2304) should be set from the
token-length distribution the script prints, not guessed: p95 plus headroom.

---

## Step 2 — run it over the corpus

**Small corpus / no extra runtime:**

```bash
CORPUS_ROOT=data/corpus OUT_ROOT=output/contracts_llm \
python infer_contracts_qwen.py --adapter runs/qwen3_14b_contracts_<stamp> \
                               --years 2005-2026 --batch-size 24
```

**Large corpus — merge, quantize, serve on vLLM:**

```bash
# 1. fold the adapter into the base weights
MODEL_PATH=models/Qwen3-14B \
ADAPTER=runs/qwen3_14b_contracts_<stamp>/lora_adapter \
MERGED_DIR=models/contracts_merged python merge_lora.py

# 2. (optional) AWQ W4A16: ~28 GB -> ~10 GB, small accuracy cost
MERGED_DIR=models/contracts_merged CALIB_JSONL=data/awq_calib.jsonl \
AWQ_DIR=models/contracts_awq python quantize_awq.py

# 3. run
SERVE_MODEL=models/contracts_awq CORPUS_ROOT=data/corpus \
python infer_vllm.py --years 2005-2012 --gpu-mem 0.90
```

Both inference scripts write **one output parquet per input file** and skip an
output that already exists, so a run is resumable after a crash and can be
split across GPUs by year range with no coordination:

```bash
CUDA_VISIBLE_DEVICES=0 python infer_vllm.py --years 2005-2011 &
CUDA_VISIBLE_DEVICES=1 python infer_vllm.py --years 2012-2018 &
CUDA_VISIBLE_DEVICES=2 python infer_vllm.py --years 2019-2026 &
```

Every output row keeps the raw model response and a `parse_ok` flag alongside
the parsed fields. Nothing is dropped: a document whose response would not parse
is still a row, and a document with no extractable text is marked `no_text=1`
rather than silently disappearing. If you cannot account for every document in
the corpus, you cannot characterize your sample.

---

## Step 3 — evaluate

```bash
TEST_SET=runs/qwen3_14b_contracts_<stamp>/test_set.parquet \
SERVE_MODEL=models/contracts_merged \
python eval_vllm_testset.py            # add QUANT=fp8 to compare quantizations
```

```bash
TEST_SET=runs/.../test_set.parquet TOKENIZER_DIR=models/contracts_merged \
python -u eval_llamacpp.py 600 32      # llama-server must already be listening
```

Both report parse rate, precision / recall / F1 on the binary decision,
CEO-flag accuracy, person exact-match, and steady-state throughput after a
warm-up. Run the same script against bf16, AWQ, FP8 and GGUF builds to see what
each quantization actually costs you on your task — the answer is often
"nothing measurable", but it is worth knowing rather than assuming.

---

## Configuration reference

| Variable | Used by | Meaning |
|---|---|---|
| `MODEL_PATH` | finetune, merge, dequant, infer | Base model: HF hub id or local directory |
| `DATA_PATH` | finetune | Labeled parquet |
| `OUTPUT_DIR` | finetune | Parent for timestamped run directories |
| `BATCH_SIZE`, `GRAD_ACCUM_STEPS` | finetune | Per-device batch and accumulation |
| `SMOKE` | finetune | `1` = 120/20 docs, 20 steps |
| `ADAPTER` | merge | `<run dir>/lora_adapter` |
| `MERGED_DIR` | merge, quantize | bf16 merged model directory |
| `CALIB_JSONL`, `AWQ_DIR` | quantize | AWQ calibration prompts, output |
| `BASE_BF16_DIR` | dequant | bf16 base, no adapter |
| `CORPUS_ROOT`, `OUT_ROOT` | infer | Corpus root, inference output root |
| `SERVE_MODEL` | infer_vllm, eval_vllm | Model directory vLLM loads |
| `TEST_SET` | eval | `<run dir>/test_set.parquet` |
| `TOKENIZER_DIR`, `LLAMA_URL` | eval_llamacpp | Tokenizer source, llama-server endpoint |
| `QUANT`, `N_DOCS`, `GPU_MEM` | eval_vllm | On-the-fly quantization, doc cap, VRAM fraction |

---

## Adapting this to your own task

1. Rewrite `SYSTEM_PROMPT`, `TYPES` and `build_target_json` in
   `contract_prompt.py`. Keep the contract that the assistant turn is **only**
   a JSON object — that is what makes the output parseable at scale.
2. Rewrite `doc_input_text` for your document format. `INPUT_CHARS` is the real
   design decision: it sets sequence length, which sets training time and VRAM.
   Feed the model the part of the document that carries the signal, not the
   whole thing.
3. Point `DATA_PATH` at your labeled parquet and adjust the column names in
   `format_example` / `build_target_json`.
4. Replace the metrics block at the end of `finetune_qwen_contracts.py` with
   metrics for your schema.

The rest — QLoRA setup, the quantization fix, merging, serving, the resumable
corpus loop — is task-independent.

---

## Notes from actually running this

Small things that cost me real time, in case they save you some.

- **A pre-quantized 4-bit checkpoint may ship some layers as bf16.** Several
  publishers keep selected MLP layers in bf16 for accuracy. `bitsandbytes`
  wraps them as `Linear4bit` anyway, with no `quant_state`, and `matmul_4bit`
  then fails at the first forward pass. Every script here carries the same fix:
  walk the modules, find `Linear4bit` with no `quant_state`, reload the original
  tensor from the safetensors shard, and re-quantize it to NF4 properly. If you
  start from an official checkpoint quantized on the fly, the block is a no-op —
  leave it in.
- **Set `tokenizer.pad_token = eos_token` explicitly.** Some Qwen3 checkpoints
  default to `<|vision_pad|>`, which is not what you want in a text batch.
- **`enable_thinking=False`, everywhere.** Qwen3 emits a reasoning block by
  default. It must be off identically in training, inference and evaluation —
  a mismatch changes the prompt prefix and quietly degrades the model.
- **`padding_side="left"` for batched generation.** With right padding, batched
  `generate` produces subtly wrong output that still parses as valid JSON.
- **Merging a LoRA into a 4-bit base gives you re-quantized weights.** `peft`
  dequantizes, folds in BA, then re-quantizes. `merge_lora.py` therefore
  dequantizes every `Linear4bit` back to bf16 by hand, streams to CPU, and
  pours the state dict into a clean skeleton — `save_pretrained` on the
  quantized model raises `NotImplementedError` in current transformers.
- **Post prompts raw to llama.cpp.** Build the string with the HF chat template
  and POST to `/completion`, not `/v1/chat/completions`. Otherwise llama.cpp
  applies its own template and your GGUF build "loses accuracy" that it never
  actually lost.
- **`packing=False`.** Packing several short examples into one sequence is a
  throughput win for general SFT, but here each example is one document and one
  answer; packing lets the loss bleed across document boundaries.
- **Log the token-length distribution before training, not after.** The script
  prints mean / p95 / max for 200 examples. `MAX_SEQ_LENGTH` set below p95
  truncates the answer off the end of your longest training examples, and the
  failure looks like a model that "sometimes doesn't finish the JSON".

---

## License

MIT — see [LICENSE](LICENSE). The prompts and taxonomy in `contract_prompt.py`
are released under the same terms; if you use them in academic work, a citation
is appreciated.

Yangen Li · [yangenli.github.io](https://yangenli.github.io)
