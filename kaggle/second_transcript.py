"""
Kaggle-notebook version of the second-transcript experiment (needs 2 GPUs, e.g. T4 x2).

1. Transcribes every clip with Whisper large-v3 (same verbatim settings as
   src/transcribe.py), the two GPUs working in parallel.
2. Fine-tunes DeBERTa-v3-base on the new transcripts (same settings as
   src/finetune_text.py), topic-grouped folds on one GPU and random folds on the
   other, at the same time.

Outputs in /kaggle/working:
  transcripts_v3/{train,test}.jsonl
  {train,test}_deberta_v3t_topic.parquet, {train,test}_deberta_v3t.parquet
Copy the transcripts to artifacts/transcripts_v3/ and the parquet files to artifacts/features/.

Needs the competition data and harsh3008/shl-asr-transcripts (for train_topic.parquet).
Notebook settings: Accelerator = GPU T4 x2, Internet = On.
"""
import glob
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "faster-whisper"], check=True)

import librosa
import numpy as np
import pandas as pd
import torch
from faster_whisper import WhisperModel
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

OUT = Path("/kaggle/working")
TX = OUT / "transcripts_v3"
VERBATIM_PROMPT = "Umm, so, uh, I- I was like, you know... I goes there and, hmm, we was talking."
MODEL = "microsoft/deberta-v3-base"
FOLDS, EPOCHS, LR, HEAD_LR, BATCH, MAX_LEN = 5, 5, 2e-5, 1e-3, 8, 320
SEEDS = [42, 7, 2024]
_lock = threading.Lock()


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} | {msg}"
    with _lock:
        print(line, flush=True)
        with open(OUT / "second_transcript.log", "a") as f:
            f.write(line + "\n")


def find(pattern):
    hits = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    if not hits:
        raise FileNotFoundError(f"{pattern} not found under /kaggle/input")
    return Path(hits[0])


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def strat_bins(y):
    b = np.round(np.asarray(y) * 2).astype(int)
    b[b < 4] = 4
    return b


# ---------------------------------------------------------------- transcription

def transcribe_file(model, path):
    audio, _ = librosa.load(path, sr=16000, mono=True)
    segments, info = model.transcribe(audio, language="en", beam_size=5, initial_prompt=VERBATIM_PROMPT,
                                      condition_on_previous_text=False, word_timestamps=True, vad_filter=False)
    segs, words = [], []
    for s in segments:
        segs.append({"start": s.start, "end": s.end, "text": s.text.strip(), "avg_logprob": s.avg_logprob,
                     "no_speech_prob": s.no_speech_prob, "compression_ratio": s.compression_ratio})
        for w in s.words or []:
            words.append({"word": w.word, "start": w.start, "end": w.end, "prob": w.probability})
    return {"text": " ".join(s["text"] for s in segs).strip(), "duration": info.duration,
            "segments": segs, "words": words}


def transcribe_all(data):
    TX.mkdir(exist_ok=True)
    # one model replica per GPU; two threads keep both busy
    model = WhisperModel("large-v3", device="cuda", device_index=[0, 1], compute_type="float16", num_workers=2)
    for split in ["train", "test"]:
        files = pd.read_csv(data / f"{split}.csv").filename.tolist()
        t0, done = time.time(), [0]

        def work(fname):
            rec = transcribe_file(model, data / split / fname)
            rec["filename"] = fname
            done[0] += 1
            if done[0] % 50 == 0:
                log(f"{split} {done[0]}/{len(files)} | {(time.time() - t0) / 60:.1f} min")
            return rec

        with ThreadPoolExecutor(2) as ex:
            recs = list(ex.map(work, files))
        with open(TX / f"{split}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        log(f"{split} transcribed in {(time.time() - t0) / 60:.1f} min")
    del model


# ---------------------------------------------------------------- DeBERTa

class TextDS(Dataset):
    def __init__(self, texts, y=None):
        self.texts, self.y = texts, y

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], np.float32(self.y[i] if self.y is not None else 0)


class Regressor(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name, dtype=torch.float32)
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(self.enc.config.hidden_size, 1))

    def forward(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        return self.head((h * m).sum(1) / m.sum(1).clamp(min=1)).squeeze(-1)


@torch.no_grad()
def predict(model, texts, collate, device):
    model.eval()
    out = []
    for i in range(0, len(texts), 32):
        ids, mask = collate(texts[i:i + 32])
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(ids.to(device), mask.to(device)).float().cpu().numpy())
    return np.concatenate(out)


def finetune(name, topic_cv, device, train, test, X, Xte, y, groups, tok):
    mu = float(y.mean())
    collate = lambda texts: (lambda e: (e["input_ids"], e["attention_mask"]))(
        tok(list(texts), padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt"))
    oof, test_pred = np.zeros(len(y)), np.zeros(len(Xte))
    train_fit, train_cnt = np.zeros(len(y)), np.zeros(len(y))

    for seed in SEEDS:
        torch.manual_seed(seed)
        seed_oof = np.zeros(len(y))
        cv = (StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=seed) if topic_cv
              else StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed))
        for fold, (tr, va) in enumerate(cv.split(X, strat_bins(y), groups if topic_cv else None), 1):
            t0 = time.time()
            model = Regressor(MODEL).to(device)
            opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": LR},
                                     {"params": model.head.parameters(), "lr": HEAD_LR}], weight_decay=0.01)
            loader = DataLoader(TextDS([X[i] for i in tr], y[tr] - mu), batch_size=BATCH, shuffle=True)
            steps = len(loader) * EPOCHS
            sched = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)
            scaler = torch.amp.GradScaler()
            va_texts = [X[i] for i in va]
            for ep in range(1, EPOCHS + 1):
                model.train()
                losses = []
                for texts, target in loader:
                    ids, mask = collate(texts)
                    with torch.autocast("cuda", dtype=torch.float16):
                        pred = model(ids.to(device), mask.to(device))
                    loss = nn.functional.mse_loss(pred.float(), target.to(device))
                    opt.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    sched.step()
                    losses.append(loss.item())
                p_va = np.clip(predict(model, va_texts, collate, device) + mu, 0, 5)
                log(f"{name} seed {seed} fold {fold} epoch {ep} | train MSE {np.mean(losses):.4f} "
                    f"| val RMSE {rmse(y[va], p_va):.4f} | val Pearson {pearsonr(y[va], p_va)[0]:.4f}")
            seed_oof[va] = p_va
            oof[va] += p_va / len(SEEDS)
            test_pred += np.clip(predict(model, Xte, collate, device) + mu, 0, 5) / (FOLDS * len(SEEDS))
            train_fit[tr] += np.clip(predict(model, [X[i] for i in tr], collate, device) + mu, 0, 5)
            train_cnt[tr] += 1
            log(f"{name} seed {seed} fold {fold} done ({(time.time() - t0) / 60:.1f} min)")
            del model, opt
            torch.cuda.empty_cache()
        log(f"{name} seed {seed} OOF RMSE {rmse(y, seed_oof):.4f} | OOF Pearson {pearsonr(y, seed_oof)[0]:.4f}")

    train_fit /= train_cnt
    log(f"{name} ({len(SEEDS)} seed avg) train RMSE {rmse(y, train_fit):.4f} | OOF RMSE {rmse(y, oof):.4f} "
        f"| OOF Pearson {pearsonr(y, oof)[0]:.4f}")
    pd.DataFrame({name: oof, f"{name}_train": train_fit}, index=train.filename.values).to_parquet(OUT / f"train_{name}.parquet")
    pd.DataFrame({name: test_pred}, index=test.filename.values).to_parquet(OUT / f"test_{name}.parquet")



def main():
    data = find("train.csv").parent
    log(f"data: {data} | GPUs: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    transcribe_all(data)

    train = pd.read_csv(data / "train.csv")
    train = train[train.label > 0].reset_index(drop=True)  # zero-score clips are non-speech noise
    test = pd.read_csv(data / "test.csv")
    tx = {s: {r["filename"]: r["text"] for r in map(json.loads, open(TX / f"{s}.jsonl"))} for s in ["train", "test"]}
    X = [tx["train"][f] for f in train.filename]
    Xte = [tx["test"][f] for f in test.filename]
    y = train.label.values.astype(np.float32)
    groups = pd.read_parquet(find("train_topic.parquet")).loc[train.filename, "topic"].values
    tok = AutoTokenizer.from_pretrained(MODEL)

    jobs = [("deberta_v3t_topic", True, "cuda:0"), ("deberta_v3t", False, "cuda:1")]
    with ThreadPoolExecutor(2) as ex:
        for f in [ex.submit(finetune, n, t, d, train, test, X, Xte, y, groups, tok) for n, t, d in jobs]:
            f.result()
    log("saved transcripts and train/test parquet files to /kaggle/working")


if __name__ == "__main__":
    main()
