"""
Build features from the cached transcripts.

Three groups:
  1. handcrafted  - fluency / speech stats from word timestamps, plus simple
                    lexical stats (length, vocabulary, fillers, repetitions)
  2. grammar      - per-sentence acceptability from a classifier fine-tuned on
                    CoLA (Corpus of Linguistic Acceptability), aggregated per clip
  3. embedding    - sentence-transformer embedding of the full transcript

Output: artifacts/features/{split}_{group}.parquet, indexed by filename
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from log_utils import get_logger

ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPTS = ROOT / "artifacts" / "transcripts"
OUT = ROOT / "artifacts" / "features"

FILLERS = {"um", "umm", "uh", "uhh", "hmm", "mm", "er", "erm", "ah"}
CORRECTION_MARKERS = ["i mean", "sorry", "no wait", "rather"]
COLA_MODEL = "textattack/roberta-base-CoLA"
EMB_MODEL = "sentence-transformers/all-mpnet-base-v2"


def load_transcripts(split):
    with open(TRANSCRIPTS / f"{split}.jsonl") as f:
        recs = [json.loads(line) for line in f]
    return {r["filename"]: r for r in recs}


def tokenize(text):
    return re.findall(r"[a-zA-Z']+", text.lower())


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if len(tokenize(p)) >= 2]


def handcrafted(rec):
    text = rec["text"]
    toks = tokenize(text)
    n = len(toks)
    words = rec["words"]
    duration = max(rec["duration"], 1e-6)
    sents = split_sentences(text)

    # pauses between consecutive words
    gaps = np.array([b["start"] - a["end"] for a, b in zip(words, words[1:])]) if len(words) > 1 else np.array([0.0])
    gaps = np.clip(gaps, 0, None)
    speech_time = sum(w["end"] - w["start"] for w in words)

    # immediate word repetitions ("I I went", "the the")
    reps = sum(1 for a, b in zip(toks, toks[1:]) if a == b)
    # cut-off words that Whisper writes as "wh-" / "I-"
    cutoffs = len(re.findall(r"\b\w+-(?=\s|$)", text))

    segs = rec["segments"]
    return {
        "n_words": n,
        "n_unique": len(set(toks)),
        "ttr": len(set(toks)) / n if n else 0,
        "words_per_min": n / duration * 60,
        "articulation_rate": n / speech_time if speech_time > 0 else 0,
        "n_sentences": len(sents),
        "mean_sent_len": np.mean([len(tokenize(s)) for s in sents]) if sents else 0,
        "max_sent_len": max([len(tokenize(s)) for s in sents]) if sents else 0,
        "mean_word_len": np.mean([len(t) for t in toks]) if toks else 0,
        "filler_ratio": sum(t in FILLERS for t in toks) / n if n else 0,
        "repetition_ratio": reps / n if n else 0,
        "cutoff_ratio": cutoffs / n if n else 0,
        "correction_count": sum(text.lower().count(m) for m in CORRECTION_MARKERS),
        "pause_mean": gaps.mean(),
        "pause_long_count": int((gaps > 0.5).sum()),
        "pause_long_per_min": (gaps > 0.5).sum() / duration * 60,
        "speech_ratio": speech_time / duration,
        "word_prob_mean": np.mean([w["prob"] for w in words]) if words else 0,
        "word_prob_low_ratio": np.mean([w["prob"] < 0.5 for w in words]) if words else 0,
        "avg_logprob": np.mean([s["avg_logprob"] for s in segs]) if segs else -5,
        "no_speech_prob": np.mean([s["no_speech_prob"] for s in segs]) if segs else 1,
        "compression_ratio": np.mean([s["compression_ratio"] for s in segs]) if segs else 0,
    }


@torch.no_grad()
def grammar_scores(texts, device):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(COLA_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(COLA_MODEL).to(device).eval()

    rows = []
    for text in texts:
        sents = split_sentences(text) or [text or "."]
        enc = tok(sents, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        p = torch.softmax(model(**enc).logits, -1)[:, 1].cpu().numpy()  # P(acceptable)
        rows.append({
            "cola_mean": p.mean(),
            "cola_min": p.min(),
            "cola_std": p.std(),
            "cola_bad_ratio": (p < 0.5).mean(),
        })
    return rows


def embeddings(texts, device):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMB_MODEL, device=device)
    return model.encode([t or "" for t in texts], batch_size=16, show_progress_bar=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log = get_logger("features")
    log.info("device=%s", device)

    for split in ["train", "test"]:
        recs = load_transcripts(split)
        names = list(recs)
        texts = [recs[n]["text"] for n in names]
        log.info("%s: %d transcripts", split, len(names))

        hc = pd.DataFrame([handcrafted(recs[n]) for n in names], index=names)
        hc.to_parquet(OUT / f"{split}_handcrafted.parquet")
        log.info("%s handcrafted %s", split, hc.shape)

        gr = pd.DataFrame(grammar_scores(texts, device), index=names)
        gr.to_parquet(OUT / f"{split}_grammar.parquet")
        log.info("%s grammar (CoLA) %s | mean acceptability %.3f", split, gr.shape, gr["cola_mean"].mean())

        emb = embeddings(texts, device)
        pd.DataFrame(emb, index=names, columns=[f"emb_{i}" for i in range(emb.shape[1])]).to_parquet(
            OUT / f"{split}_embedding.parquet"
        )
        log.info("%s embedding %s", split, emb.shape)


if __name__ == "__main__":
    main()
