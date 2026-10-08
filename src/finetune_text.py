"""
Fine-tune DeBERTa-v3-base as a regressor on the transcripts, inside K-fold CV.

The frozen features (CoLA, embeddings) only measure grammar indirectly. Here the
text model is trained on the labels directly, so it can learn which errors the
raters actually penalise. With ~730 clips the model is small (base size), the
schedule short and the learning rate low.

Every fold trains for a fixed number of epochs (no per-fold early stopping, which
would leak the validation labels into the OOF score). The final model of each
fold predicts its held-out clips (-> out-of-fold predictions) and the test set;
test predictions are averaged over folds. Validation metrics are still logged
after every epoch to choose --epochs.

Output: artifacts/features/{train,test}_deberta.parquet  (column: deberta)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from log_utils import get_logger
from train import SEED, load_labels, rmse, strat_bins

ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPTS = ROOT / "artifacts" / "transcripts"
OUT = ROOT / "artifacts" / "features"


class TextDS(Dataset):
    def __init__(self, texts, y=None, w=None):
        self.texts, self.y, self.w = texts, y, w

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return (self.texts[i], np.float32(self.y[i]) if self.y is not None else np.float32(0),
                np.float32(self.w[i]) if self.w is not None else np.float32(1))


class Regressor(nn.Module):
    def __init__(self, name):
        super().__init__()
        # keep master weights in fp32; autocast handles fp16 compute
        self.enc = AutoModel.from_pretrained(name, dtype=torch.float32)
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(self.enc.config.hidden_size, 1))

    def forward(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)  # mean pooling over real tokens
        return self.head(pooled).squeeze(-1)


def predict(model, loader, collate, device):
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for texts, _, _ in loader:
            ids, mask = collate(texts)
            out.append(model(ids.to(device), mask.to(device)).float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/deberta-v3-base")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=320)
    ap.add_argument("--seeds", type=int, nargs="+", default=[SEED],
                    help="train the full CV once per seed and average (smooths out run-to-run noise)")
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--grad_ckpt", action="store_true", help="gradient checkpointing (needed for -large on 6 GB)")
    ap.add_argument("--name", default="deberta", help="output column / file name")
    ap.add_argument("--topic_cv", action="store_true",
                    help="group the folds by topic cluster (artifacts/features/train_topic.parquet)")
    ap.add_argument("--sample_weights", default=None,
                    help="parquet of per-clip training weights (e.g. train_shift_weight) for a weighted MSE loss")
    ap.add_argument("--pseudo", default=None,
                    help="CSV (filename,label) of predicted test scores to add as extra training rows")
    args = ap.parse_args()

    log = get_logger("finetune_text")
    log.info("args: %s", vars(args))
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda"

    y, test_idx = load_labels()
    tx = {s: {r["filename"]: r["text"] for r in map(json.loads, open(TRANSCRIPTS / f"{s}.jsonl"))} for s in ["train", "test"]}
    X = [tx["train"][f] for f in y.index]
    Xte = [tx["test"][f] for f in test_idx]
    yv = y.values.astype(np.float32)
    # center the target so the head starts near the mean
    mu = float(yv.mean())

    # Pseudo-labelling: the test transcripts are competition data; giving them the
    # current ensemble's predicted score lets the text model see more varied
    # speech. They only ever go into the training side of a fold, never validation.
    X_ps, y_ps = [], np.zeros(0, dtype=np.float32)
    if args.pseudo:
        ps = pd.read_csv(args.pseudo).set_index("filename")["label"].loc[test_idx]
        X_ps, y_ps = Xte, ps.values.astype(np.float32)
        log.info("pseudo-labels: %d test transcripts from %s (mean %.2f)", len(X_ps), args.pseudo, y_ps.mean())

    sw = np.ones(len(yv), dtype=np.float32)
    if args.sample_weights:
        sw = pd.read_parquet(OUT / f"{args.sample_weights}.parquet").loc[y.index, "weight"].values.astype(np.float32)
        log.info("weighted loss with %s (min %.2f, max %.2f)", args.sample_weights, sw.min(), sw.max())

    tok = AutoTokenizer.from_pretrained(args.model)
    collate = lambda texts: (lambda e: (e["input_ids"], e["attention_mask"]))(
        tok(list(texts), padding=True, truncation=True, max_length=args.max_len, return_tensors="pt"))
    te_loader = DataLoader(TextDS(Xte), batch_size=32)

    oof = np.zeros(len(yv))
    test_pred = np.zeros(len(Xte))
    train_fit, train_cnt = np.zeros(len(yv)), np.zeros(len(yv))
    n_seeds = len(args.seeds)

    # each seed gets its own fold split and init; OOF / test predictions are averaged over seeds
    groups = pd.read_parquet(OUT / "train_topic.parquet").loc[y.index, "topic"].values if args.topic_cv else None
    make_cv = (lambda seed: StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=seed)) if args.topic_cv         else (lambda seed: StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed))
    splits = [(seed, fold, tr, va)
              for seed in args.seeds
              for fold, (tr, va) in enumerate(make_cv(seed).split(X, strat_bins(yv), groups), 1)]

    for seed, fold, tr, va in splits:
        if fold == 1:
            torch.manual_seed(seed)
            seed_oof = np.zeros(len(yv))
        t0 = time.time()
        model = Regressor(args.model).to(device)
        if args.grad_ckpt:
            model.enc.gradient_checkpointing_enable()
        opt = torch.optim.AdamW([
            {"params": model.enc.parameters(), "lr": args.lr},
            {"params": model.head.parameters(), "lr": 1e-3},
        ], weight_decay=0.01)
        tr_loader = DataLoader(TextDS([X[i] for i in tr] + list(X_ps), np.concatenate([yv[tr], y_ps]) - mu,
                                      np.concatenate([sw[tr], np.ones(len(y_ps), dtype=np.float32)])),
                               batch_size=args.batch, shuffle=True)
        va_loader = DataLoader(TextDS([X[i] for i in va]), batch_size=32)
        steps = len(tr_loader) * args.epochs // args.accum
        sched = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)
        scaler = torch.amp.GradScaler()

        for ep in range(1, args.epochs + 1):
            model.train()
            losses = []
            opt.zero_grad()
            for step, (texts, target, wt) in enumerate(tr_loader, 1):
                ids, mask = collate(texts)
                with torch.autocast("cuda", dtype=torch.float16):
                    pred = model(ids.to(device), mask.to(device))
                wt = wt.to(device)
                loss = (wt * (pred.float() - target.to(device)) ** 2).sum() / wt.sum()
                scaler.scale(loss / args.accum).backward()
                losses.append(loss.item())
                if step % args.accum == 0 or step == len(tr_loader):
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    sched.step()
                    opt.zero_grad()

            p_va = np.clip(predict(model, va_loader, collate, device) + mu, 0, 5)
            r, c = rmse(yv[va], p_va), pearsonr(yv[va], p_va)[0]
            log.info("seed %d fold %d epoch %d | train MSE %.4f | val RMSE %.4f | val Pearson %.4f",
                     seed, fold, ep, np.mean(losses), r, c)

        seed_oof[va] = p_va
        oof[va] += p_va / n_seeds
        test_pred += np.clip(predict(model, te_loader, collate, device) + mu, 0, 5) / (args.folds * n_seeds)
        # predictions on this fold's own training clips (for the training RMSE)
        tr_eval = DataLoader(TextDS([X[i] for i in tr]), batch_size=32)
        train_fit[tr] += np.clip(predict(model, tr_eval, collate, device) + mu, 0, 5)
        train_cnt[tr] += 1
        log.info("seed %d fold %d final val RMSE %.4f (%.1f min)", seed, fold, r, (time.time() - t0) / 60)
        if fold == args.folds:
            log.info("seed %d OOF RMSE %.4f | OOF Pearson %.4f", seed, rmse(yv, seed_oof), pearsonr(yv, seed_oof)[0])
        del model, opt
        torch.cuda.empty_cache()

    train_fit /= train_cnt
    n = args.name
    log.info("%s (%d seed avg) train RMSE %.4f | OOF RMSE %.4f | OOF Pearson %.4f", n, n_seeds,
             rmse(yv, train_fit), rmse(yv, oof), pearsonr(yv, oof)[0])
    pd.DataFrame({n: oof, f"{n}_train": train_fit}, index=y.index).to_parquet(OUT / f"train_{n}.parquet")
    pd.DataFrame({n: test_pred}, index=test_idx).to_parquet(OUT / f"test_{n}.parquet")
    log.info("saved %s OOF / test predictions", n)


if __name__ == "__main__":
    main()
