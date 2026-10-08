"""
Grammatical-error-correction (GEC) features.

The rubric is essentially about how many grammatical mistakes a speaker makes.
A GEC model (CoEdIT-large, a T5 model fine-tuned for text editing) rewrites each
sentence into correct English; the amount of rewriting it needs is a direct
measure of the errors.

Before correcting, fillers ("um", "uh", ...) and immediate word repeats are
removed, so the edits count grammar rather than disfluency (fluency already has
its own features). Each transcript is split into sentences, corrected in
batches, and compared word by word with difflib.

Features per clip:
  gec_edits_per_100w      word-level edits (insert / delete / replace) per 100 words
  gec_ins / del / rep_per_100w   the same split by edit type
  gec_sent_changed        share of sentences the model changed
  gec_char_dist           normalised character edit distance (1 - similarity)
  gec_edits_per_sent      mean edits per sentence

Output: artifacts/features/{split}_gec.parquet
"""
import argparse
import difflib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

from log_utils import get_logger

ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPTS = ROOT / "artifacts" / "transcripts"
OUT = ROOT / "artifacts" / "features"
FILLERS = r"\b(um+|uh+|erm*|hmm+|mm+|ah+)\b[,.]?\s*"
PROMPT = "Fix grammatical errors in this sentence: "


def clean(text):
    t = re.sub(FILLERS, "", text, flags=re.I)
    t = re.sub(r"\b(\w+)(?:[\s,]+\1\b)+", r"\1", t, flags=re.I)  # "there are, there are" -> "there are"
    return re.sub(r"\s+", " ", t).strip()


def sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p for p in parts if len(p.split()) >= 3]


def word_edits(a, b):
    wa, wb = a.lower().split(), b.lower().split()
    ins = dele = rep = 0
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, wa, wb).get_opcodes():
        if op == "insert":
            ins += j2 - j1
        elif op == "delete":
            dele += i2 - i1
        elif op == "replace":
            rep += max(i2 - i1, j2 - j1)
    return ins, dele, rep, len(wa)


@torch.no_grad()
def correct(model, tok, sents, batch, device):
    out = []
    for i in range(0, len(sents), batch):
        enc = tok([PROMPT + s for s in sents[i:i + batch]], return_tensors="pt", padding=True,
                  truncation=True, max_length=128).to(device)
        gen = model.generate(**enc, max_new_tokens=128, num_beams=1)
        out.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return out


def features(orig_sents, corr_sents):
    ins = dele = rep = words = changed = 0
    per_sent = []
    for a, b in zip(orig_sents, corr_sents):
        i, d, r, n = word_edits(a, b)
        ins, dele, rep, words = ins + i, dele + d, rep + r, words + n
        per_sent.append(i + d + r)
        changed += (i + d + r) > 0
    words = max(words, 1)
    joined_a, joined_b = " ".join(orig_sents), " ".join(corr_sents)
    return {
        "gec_edits_per_100w": 100 * (ins + dele + rep) / words,
        "gec_ins_per_100w": 100 * ins / words,
        "gec_del_per_100w": 100 * dele / words,
        "gec_rep_per_100w": 100 * rep / words,
        "gec_sent_changed": changed / max(len(orig_sents), 1),
        "gec_edits_per_sent": float(np.mean(per_sent)) if per_sent else 0.0,
        "gec_char_dist": 1 - difflib.SequenceMatcher(None, joined_a, joined_b).ratio(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="grammarly/coedit-large")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    log = get_logger("gec_features")
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = T5ForConditionalGeneration.from_pretrained(args.model, dtype=torch.float16).to(device).eval()
    log.info("model=%s batch=%d", args.model, args.batch)

    for split in ["train", "test"]:
        recs = [json.loads(l) for l in open(TRANSCRIPTS / f"{split}.jsonl")]
        t0 = time.time()
        # correct all sentences of the split in one sorted pass (similar lengths batch better)
        per_clip = [sentences(clean(r["text"])) or [clean(r["text"]) or "."] for r in recs]
        flat = [(ci, s) for ci, ss in enumerate(per_clip) for s in ss]
        order = sorted(range(len(flat)), key=lambda k: len(flat[k][1]))
        corrected = [None] * len(flat)
        for start in range(0, len(order), 512):
            idx = order[start:start + 512]
            for k, c in zip(idx, correct(model, tok, [flat[k][1] for k in idx], args.batch, device)):
                corrected[k] = c
            log.info("%s %d/%d sentences (%.1f min)", split, min(start + 512, len(order)), len(order), (time.time() - t0) / 60)

        rows, pos = [], 0
        for ss in per_clip:
            rows.append(features(ss, corrected[pos:pos + len(ss)]))
            pos += len(ss)
        df = pd.DataFrame(rows, index=[r["filename"] for r in recs])
        df.to_parquet(OUT / f"{split}_gec.parquet")
        log.info("%s done %s | mean edits per 100 words %.2f (%.1f min)", split, df.shape,
                 df.gec_edits_per_100w.mean(), (time.time() - t0) / 60)

        if split == "train":  # a few examples in the log, to sanity-check the corrections
            for k in range(0, min(len(flat), 2000), 400):
                log.info("  example | %s  ->  %s", flat[k][1][:120], corrected[k][:120])


if __name__ == "__main__":
    main()
