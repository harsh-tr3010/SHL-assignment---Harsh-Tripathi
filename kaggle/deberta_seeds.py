"""
Kaggle-notebook script: more DeBERTa-v3-base seeds on the original (turbo) transcripts.

The text model is the noisiest part of the blend and was averaged over only three
seeds (42, 7, 2024). Same settings as src/finetune_text.py.

  SEED_SET=1: seven more seeds, random folds (GPU 0) and topic-grouped folds (GPU 1);
              with the first three, OOF and test predictions come from ten seeds.
              -> {train,test}_deberta_s7.parquet, {train,test}_deberta_topic_s7.parquet
  SEED_SET=2: ten more random-fold seeds (GPU 1), and ten models trained on all
              training clips for the test predictions (GPU 0).
              -> {train,test}_deberta_s10b.parquet, test_deberta_full.parquet
Needs the competition data and harsh3008/shl-asr-transcripts.
Notebook settings: Accelerator = GPU T4 x2, Internet = On.
"""
import glob
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

OUT = Path("/kaggle/working")
MODEL = "microsoft/deberta-v3-base"
FOLDS, EPOCHS, LR, HEAD_LR, BATCH, MAX_LEN = 5, 5, 2e-5, 1e-3, 8, 320
SEED_SET = 1
SEEDS = [1, 2, 3, 4, 5, 6, 8] if SEED_SET == 1 else [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
FULL_SEEDS = list(range(100, 110))
_lock = threading.Lock()


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} | {msg}"
    with _lock:
        print(line, flush=True)
        with open(OUT / "deberta_seeds.log", "a") as f:
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


def train_one(X_tr, y_tr, mu, collate, device):
    model = Regressor(MODEL).to(device)
    opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": LR},
                             {"params": model.head.parameters(), "lr": HEAD_LR}], weight_decay=0.01)
    loader = DataLoader(TextDS(X_tr, y_tr - mu), batch_size=BATCH, shuffle=True)
    steps = len(loader) * EPOCHS
    sched = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler()
    for ep in range(EPOCHS):
        model.train()
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
    return model


def full_fit(name, device, test, X, Xte, y, tok):
    """Models trained on every training clip (same schedule); test predictions only."""
    mu = float(y.mean())
    collate = lambda texts: (lambda e: (e["input_ids"], e["attention_mask"]))(
        tok(list(texts), padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt"))
    test_pred = np.zeros(len(Xte))
    for seed in FULL_SEEDS:
        t0 = time.time()
        torch.manual_seed(seed)
        model = train_one(X, y, mu, collate, device)
        test_pred += np.clip(predict(model, Xte, collate, device) + mu, 0, 5) / len(FULL_SEEDS)
        p_tr = np.clip(predict(model, X, collate, device) + mu, 0, 5)
        log(f"{name} seed {seed} | train RMSE {rmse(y, p_tr):.4f} | {(time.time() - t0) / 60:.1f} min")
        del model
        torch.cuda.empty_cache()
    pd.DataFrame({name: test_pred}, index=test.filename.values).to_parquet(OUT / f"test_{name}.parquet")
    log(f"{name}: saved test predictions ({len(FULL_SEEDS)} seeds)")


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
    tx_dir = find("train.jsonl").parent
    log(f"data: {data} | transcripts: {tx_dir} | GPUs: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    train = pd.read_csv(data / "train.csv")
    train = train[train.label > 0].reset_index(drop=True)  # zero-score clips are non-speech noise
    test = pd.read_csv(data / "test.csv")
    tx = {s: {r["filename"]: r["text"] for r in map(json.loads, open(tx_dir / f"{s}.jsonl"))} for s in ["train", "test"]}
    X = [tx["train"][f] for f in train.filename]
    Xte = [tx["test"][f] for f in test.filename]
    y = train.label.values.astype(np.float32)
    groups = pd.read_parquet(find("train_topic.parquet")).loc[train.filename, "topic"].values
    tok = AutoTokenizer.from_pretrained(MODEL)

    with ThreadPoolExecutor(2) as ex:
        if SEED_SET == 1:
            futures = [ex.submit(finetune, "deberta_s7", False, "cuda:0", train, test, X, Xte, y, groups, tok),
                       ex.submit(finetune, "deberta_topic_s7", True, "cuda:1", train, test, X, Xte, y, groups, tok)]
        else:
            futures = [ex.submit(full_fit, "deberta_full", "cuda:0", test, X, Xte, y, tok),
                       ex.submit(finetune, "deberta_s10b", False, "cuda:1", train, test, X, Xte, y, groups, tok)]
        for f in futures:
            f.result()
    log("saved train/test parquet files to /kaggle/working")


if __name__ == "__main__":
    main()
