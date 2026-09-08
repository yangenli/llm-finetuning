#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared prompt / schema / text-prep for the executive-contract classifier.

ONE source of truth, imported by the fine-tuning, inference and evaluation
scripts alike, so the prompt the labels were produced under, the prompt the
model is trained on, and the prompt it is served with can never drift apart.
Change the taxonomy or the system prompt here and everything downstream moves
together — but note that changing it after labeling invalidates the labels.
"""
import json
import math
import re

# ------------------------------------------------------------------ text prep
TAG = re.compile(r'<[^>]+>')
ENT = re.compile(r'&(#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);?')
WS = re.compile(r'\s+')

HEAD_RAW = 150_000   # raw chars scanned (HTML is tag-heavy)
INPUT_CHARS = 4_000  # stripped chars fed to the model (title + parties + recitals)


def doc_input_text(body):
    """Model input: stripped-HTML head of the exhibit document."""
    t = body[:HEAD_RAW]
    if t.lstrip()[:6] == "begin ":      # uuencoded binary (pdf/image) — no text
        return ""
    t = TAG.sub(' ', t)
    t = ENT.sub(' ', t)
    return WS.sub(' ', t).strip()[:INPUT_CHARS]


def doc_key(row):
    """Stable unique id for one exhibit document."""
    return f"{row['accession']}|{row['sequence']}|{row['filename']}"


# ------------------------------------------------------------------ taxonomy
TYPES = ["employment", "offer_letter", "severance", "cic", "noncompete",
         "separation", "retention", "consulting", "indemnification",
         "equity_award", "bonus", "deferred_comp", "serp", "salary_continuation",
         "split_dollar", "confidentiality", "relocation", "clawback",
         "service_agreement", "comp_arrangement", "other"]

# ------------------------------------------------------------------ prompt
SYSTEM_PROMPT = """You classify SEC filing exhibit documents (EX-10 / EX-99). You see the beginning of one document (HTML stripped, possibly truncated). Decide whether it is an EXECUTIVE employment-related contract or compensation arrangement, and extract structured fields.

## contract (0 or 1)
1 = the document is a contract/arrangement between a company and an individual executive, officer, or director about their employment, compensation, separation, or post-employment restrictions — OR a compensatory plan/schedule covering executives.
0 = anything else (credit agreements, M&A, leases, licenses, supply/customer contracts, press releases about earnings, certifications, fund documents, ...).

## types (list of strings, empty if contract=0)
All that apply, from EXACTLY this vocabulary:
- "employment": employment agreement/contract (incl. "Executive Agreement")
- "offer_letter": offer letter / employment offer ("we are pleased to offer you...")
- "severance": severance agreement, plan, or pay arrangement
- "cic": change-in-control / management continuity / termination protection / golden parachute
- "noncompete": non-compete, non-solicitation, restrictive covenant agreement
- "separation": separation, transition, retirement, resignation agreement; release of claims on departure
- "retention": retention agreement / stay bonus
- "consulting": consulting or independent-contractor agreement with an individual (often a former executive); firm-to-firm consulting engagements are NOT contracts
- "indemnification": officer/director indemnification agreement
- "equity_award": individual equity grant (stock option, RSU, restricted stock, performance shares, SAR, phantom stock)
- "bonus": bonus agreement/letter, sign-on bonus
- "deferred_comp": nonqualified deferred compensation agreement/plan
- "serp": supplemental executive retirement plan/agreement, excess benefit
- "salary_continuation": salary continuation agreement
- "split_dollar": split-dollar life insurance agreement
- "confidentiality": confidentiality / NDA / proprietary information & invention assignment with an employee
- "relocation": relocation benefits
- "clawback": compensation recovery / clawback agreement or policy
- "service_agreement": UK-style director's/executive's service agreement
- "comp_arrangement": compensation schedule/summary (base salaries, director compensation, bonus plan descriptions)
- "other": executive-related contract that fits none of the above

## person (string or null)
Full name of the individual counterparty (the executive/employee), exactly as written. null if the document is a plan/form/schedule with no single named counterparty.

## title (string or null)
The individual's job title as stated in the document (e.g. "President and Chief Executive Officer"). null if not stated.

## ceo (0 or 1)
1 = the individual counterparty is (or is becoming) the Chief Executive Officer / CEO / co-CEO, per the document text. 0 = someone else, or no individual counterparty, or cannot tell.

## executed (0 or 1)
1 = an actual agreement with named parties (even if the visible part is unsigned).
0 = a template ("Form of ..."), a plan document, or a schedule/summary.

## amendment (0 or 1)
1 = an amendment, restatement, extension, or waiver of an earlier agreement.

## Rules
- Judge ONLY from the text shown. Do not guess from the company name.
- The text may be truncated mid-document; that is normal.
- If the text is empty or garbled, return contract=0.

## Output
ONLY a JSON object, nothing else:
{"contract": <0|1>, "types": [...], "person": <str|null>, "title": <str|null>, "ceo": <0|1>, "executed": <0|1>, "amendment": <0|1>}"""

USER_TEMPLATE = "Classify this SEC exhibit document:\n\n{text}"


def _text_or_null(v):
    """Normalize an optional text field to a string or None.

    A plain truthiness test is not enough here: labels arrive through pandas,
    which represents a missing string as float('nan'), and NaN is truthy. It
    would therefore pass an `if v` guard and reach json.dumps, which emits the
    non-standard token NaN -- accepted by Python's json but rejected by every
    strict parser.
    """
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    s = str(v).strip()
    return s if s else None


def build_target_json(label_row):
    """Render one labeled row back to the canonical assistant JSON string
    (used as the finetune target). label_row: dict with the 7 fields."""
    return json.dumps({
        "contract": int(label_row["contract"]),
        "types": list(label_row["types"]),
        "person": _text_or_null(label_row["person"]),
        "title": _text_or_null(label_row["title"]),
        "ceo": int(label_row["ceo"]),
        "executed": int(label_row["executed"]),
        "amendment": int(label_row["amendment"]),
    }, ensure_ascii=False, allow_nan=False)
