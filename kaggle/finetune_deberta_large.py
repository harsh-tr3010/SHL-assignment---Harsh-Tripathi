"""
Kaggle-notebook version of src/finetune_text.py for DeBERTa-v3-large (needs a 16 GB GPU).

Self-contained: reads the competition labels from /kaggle/input, the Whisper
transcripts (artifacts/transcripts/{train,test}.jsonl, uploaded as a private
dataset), fine-tunes microsoft/deberta-v3-large as a regressor in 5-fold CV for
several seeds, and writes out-of-fold / test predictions to /kaggle/working.
Copy the parquet files into artifacts/features/ and train.py picks them up as
`deberta_large`.

Notebook settings: Accelerator = GPU (T4 or P100), Internet = On.
"""
import glob
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

MODEL = "microsoft/deberta-v3-large"
NAME = "deberta_large"
FOLDS, EPOCHS, LR, HEAD_LR, BATCH, ACCUM, MAX_LEN = 5, 5, 1e-5, 5e-4, 4, 2, 320
SEEDS = [42, 7, 2024]
OUT = Path("/kaggle/working")


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} | {msg}"
    print(line, flush=True)
    with open(OUT / f"{NAME}.log", "a") as f:
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
        h = self.enc.config.hidden_size
        # LayerNorm keeps the pooled features well scaled for the fp16 head
        self.head = nn.Sequential(nn.LayerNorm(h), nn.Dropout(0.1), nn.Linear(h, 1))

    def forward(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        return self.head((h * m).sum(1) / m.sum(1).clamp(min=1)).squeeze(-1)


@torch.no_grad()
def predict(model, texts, collate, device):
    model.eval()
    out = []
    for i in range(0, len(texts), 16):
        ids, mask = collate(texts[i:i + 16])
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(ids.to(device), mask.to(device)).float().cpu().numpy())
    return np.concatenate(out)


def main():
    device = "cuda"
    data = find("train.csv").parent
    tx_dir = find("train.jsonl").parent
    log(f"data: {data} | transcripts: {tx_dir} | GPU: {torch.cuda.get_device_name(0)}")

    train = pd.read_csv(data / "train.csv")
    train = train[train.label > 0].reset_index(drop=True)  # zero-score clips are non-speech noise
    test = pd.read_csv(data / "test.csv")
    tx = {s: {r["filename"]: r["text"] for r in map(json.loads, open(tx_dir / f"{s}.jsonl"))} for s in ["train", "test"]}
    X = [tx["train"][f] for f in train.filename]
    Xte = [tx["test"][f] for f in test.filename]
    y = train.label.values.astype(np.float32)
    mu = float(y.mean())

    tok = AutoTokenizer.from_pretrained(MODEL)
    collate = lambda texts: (lambda e: (e["input_ids"], e["attention_mask"]))(
        tok(list(texts), padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt"))

    oof, test_pred = np.zeros(len(y)), np.zeros(len(Xte))
    train_fit, train_cnt = np.zeros(len(y)), np.zeros(len(y))

    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        seed_oof = np.zeros(len(y))
        skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(skf.split(X, strat_bins(y)), 1):
            t0 = time.time()
            model = Regressor(MODEL).to(device)
            opt = torch.optim.AdamW([
                {"params": model.enc.parameters(), "lr": LR},
                {"params": model.head.parameters(), "lr": HEAD_LR},
            ], weight_decay=0.01)
            loader = DataLoader(TextDS([X[i] for i in tr], y[tr] - mu), batch_size=BATCH, shuffle=True)
            steps = len(loader) * EPOCHS // ACCUM
            sched = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)
            scaler = torch.amp.GradScaler()
            va_texts = [X[i] for i in va]

            for ep in range(1, EPOCHS + 1):
                model.train()
                losses = []
                opt.zero_grad()
                for step, (texts, target) in enumerate(loader, 1):
                    ids, mask = collate(texts)
                    with torch.autocast("cuda", dtype=torch.float16):
                        pred = model(ids.to(device), mask.to(device))
                    loss = nn.functional.mse_loss(pred.float(), target.to(device))
                    scaler.scale(loss / ACCUM).backward()
                    losses.append(loss.item())
                    if step % ACCUM == 0 or step == len(loader):
                        scaler.unscale_(opt)
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(opt)
                        scaler.update()
                        sched.step()
                        opt.zero_grad()
                p_va = np.clip(predict(model, va_texts, collate, device) + mu, 0, 5)
                log(f"seed {seed} fold {fold} epoch {ep} | train MSE {np.mean(losses):.4f} | val RMSE {rmse(y[va], p_va):.4f} "
                    f"| val Pearson {pearsonr(y[va], p_va)[0]:.4f} | {(time.time() - t0) / 60:.1f} min")

            seed_oof[va] = p_va
            oof[va] += p_va / len(SEEDS)
            test_pred += np.clip(predict(model, Xte, collate, device) + mu, 0, 5) / (FOLDS * len(SEEDS))
            train_fit[tr] += np.clip(predict(model, [X[i] for i in tr], collate, device) + mu, 0, 5)
            train_cnt[tr] += 1
            log(f"seed {seed} fold {fold} final val RMSE {rmse(y[va], p_va):.4f} ({(time.time() - t0) / 60:.1f} min)")
            del model, opt
            torch.cuda.empty_cache()
        log(f"seed {seed} OOF RMSE {rmse(y, seed_oof):.4f} | OOF Pearson {pearsonr(y, seed_oof)[0]:.4f}")

    train_fit /= train_cnt
    log(f"{NAME} ({len(SEEDS)} seed avg) train RMSE {rmse(y, train_fit):.4f} | OOF RMSE {rmse(y, oof):.4f} "
        f"| OOF Pearson {pearsonr(y, oof)[0]:.4f}")
    pd.DataFrame({NAME: oof, f"{NAME}_train": train_fit}, index=train.filename.values).to_parquet(OUT / f"train_{NAME}.parquet")
    pd.DataFrame({NAME: test_pred}, index=test.filename.values).to_parquet(OUT / f"test_{NAME}.parquet")
    log("saved train/test parquet files to /kaggle/working")


if __name__ == "__main__":
    main()
