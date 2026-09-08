# Data format

No data ships with this repository. These are the schemas the scripts expect,
so you can build the same files from your own corpus.

Everything is parquet. Nothing is required to be in a particular directory —
paths come from the environment variables listed in the README.

---

## 1. The labeled training file (`DATA_PATH`)

One row per labeled document. This is the only file `finetune_qwen_contracts.py`
reads.

| Column | Type | Required | Meaning |
|---|---|---|---|
| `doc_key` | str | yes | Stable unique id for the document. Used to join predictions back, and to keep train/test disjoint. |
| `input_text` | str | yes | Exactly what the model sees — already stripped and truncated by `doc_input_text()`. Do **not** store raw HTML here. |
| `parse_ok` | int | yes | `1` if the document yielded usable text. Rows with `0` are dropped before training. |
| `stratum` | str | yes | Sampling stratum. Used to stratify the train/test split, and worth reporting accuracy by. |
| `primary` | str/int | yes | Second stratification key (here: the primary contract type). Combined with `stratum` into the split key. |
| `contract` | int (0/1) | yes | The binary target. |
| `types` | list[str] | yes | Multi-label, from `TYPES` in `contract_prompt.py`. Empty list when `contract=0`. |
| `person` | str or null | yes | Extracted counterparty name, `null` if none. |
| `title` | str or null | yes | Extracted job title, `null` if not stated. |
| `ceo` | int (0/1) | yes | Whether the counterparty is the CEO. |
| `executed` | int (0/1) | yes | Real agreement vs. a form/plan/schedule. |
| `amendment` | int (0/1) | yes | Amends or restates an earlier agreement. |

Notes:

- The seven label columns are turned into the assistant turn by
  `build_target_json()`. Change the schema there and here together.
- `types` must be a genuine list, not a comma-joined string — parquet stores it
  as a list column and `build_target_json` calls `list()` on it.
- Missing strings should be real nulls. `build_target_json` guards against
  pandas turning them into `float('nan')`, which is truthy and would otherwise
  serialize as the non-standard token `NaN`, breaking every strict JSON parser
  downstream.
- Strata with fewer than 2 rows are folded into a `"rare"` bucket before the
  split, since `train_test_split` cannot stratify on a singleton.

Minimal example of one row:

```python
{
  "doc_key": "0000320193-14-000008|5|ex10-1.htm",
  "input_text": "EMPLOYMENT AGREEMENT This Employment Agreement ...",
  "parse_ok": 1,
  "stratum": "keyword_hit",
  "primary": "employment",
  "contract": 1,
  "types": ["employment", "noncompete"],
  "person": "Jane Q. Doe",
  "title": "President and Chief Executive Officer",
  "ceo": 1,
  "executed": 1,
  "amendment": 0,
}
```

---

## 2. The corpus (`CORPUS_ROOT`)

What the inference scripts walk. Layout:

```
$CORPUS_ROOT/{year}/QTR{q}/{yyyymmdd}.parquet
```

The scripts glob `**/*.parquet` and read the year from the **first four
characters of the filename**, so a file must be named `20140115.parquet`, not
`jan-15-2014.parquet`. The intermediate directory names are otherwise free —
one output file is written per input file, mirroring the relative path, which
is what makes a run resumable and shardable across GPUs.

| Column | Type | Meaning |
|---|---|---|
| `text` | str | Raw document body. `doc_input_text()` strips and truncates it at read time. |
| `exhibit` | str | Filter key (`"EX-10"` by default; `--include-ex99` widens it). |
| `cik`, `year`, `date`, `form_type`, `accession`, `web_url`, `doc_type`, `sequence`, `filename`, `description` | — | Carried through to the output unchanged, so a result row is traceable back to the filing. |

If your metadata columns differ, edit `META_COLS` in both inference scripts.

---

## 3. Inference output (`OUT_ROOT`)

Written one file per input file, same relative path. Every `META_COLS` column,
plus:

| Column | Type | Meaning |
|---|---|---|
| `raw_response` | str | The model's exact output. Keep it: it is how you debug a parse failure or re-parse after a schema change without re-running the model. |
| `parse_ok` | int (0/1) | Whether `raw_response` parsed as JSON. |
| `no_text` | int (0/1) | `1` when the document had no extractable text (a uuencoded PDF or image), so it never reached the model. |
| `contract`, `types`, `person`, `title`, `ceo`, `executed`, `amendment` | | Parsed fields; all null when `parse_ok=0`. `types` is comma-joined here rather than a list. |

Every input document produces exactly one output row. A document that could not
be read and a document the model refused to parse are both still rows, flagged.
That is deliberate: a corpus-scale measurement is only interpretable if you can
account for the documents that did *not* make it into the sample.

---

## 4. AWQ calibration file (`CALIB_JSONL`)

Only needed for the optional `quantize_awq.py` step. JSONL, one object per line:

```json
{"system": "<SYSTEM_PROMPT>", "user": "<USER_TEMPLATE filled with a real document>"}
```

256 documents drawn from your own corpus is enough. In-domain calibration text
matters much more than the count — calibrating on generic web text costs you
accuracy on a task with a long, fixed system prompt like this one.
