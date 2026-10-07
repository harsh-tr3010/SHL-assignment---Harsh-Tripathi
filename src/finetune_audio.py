"""
Fine-tune WavLM-base-plus as a regressor on the raw audio, inside K-fold CV.

Frozen WavLM + Ridge was already the strongest single model, so here the speech
model itself is adapted to the grammar labels:
  - the convolutional feature encoder stays frozen (standard for small data);
    the transformer layers are fine-tuned with a low learning rate
  - a learnable softmax-weighted sum over all hidden layers (initialised to
    favour the upper-middle layers that worked best frozen), then mean +
    std pooling over frames, then a small linear head
  - training uses random 15 s crops of each clip (fits a 6 GB GPU and works as
    augmentation); prediction averages over overlapping 15 s windows covering
    the whole clip

Same protocol as finetune_text.py: fixed number of epochs (no per-fold early
stopping), out-of-fold predictions for the training clips, test predictions
averaged over the fold models, validation metrics logged after every epoch.

Output: artifacts/features/{train,test}_{name}.parquet  (column: name)
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import WavLMModel, get_cosine_schedule_with_warmup

from log_utils import get_logger
from train import SEED, load_labels, rmse, strat_bins

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "artifacts" / "features"
SR = 16000


def load_audio(split, names):
    """int16 waveforms kept in memory (~1.7 GB for all clips)."""
    return [sf.read(DATA / split / n, dtype="int16")[0] for n in names]


def normalise(x):
    x = x.astype(np.float32) / 32768.0
    return (x - x.mean()) / (x.std() + 1e-7)


class CropDS(Dataset):
    def __init__(self, waves, y, crop):
        self.waves, self.y, self.crop = waves, y, crop

    def __len__(self):
        return len(self.waves)

    def __getitem__(self, i):
        w = self.waves[i]
        if len(w) > self.crop:
            s = np.random.randint(0, len(w) - self.crop)
            w = w[s:s + self.crop]
        else:
            w = np.pad(w, (0, self.crop - len(w)))
        return torch.from_numpy(normalise(w)), np.float32(self.y[i])


class AudioRegressor(nn.Module):
    def __init__(self, name, init_layer=9):
        super().__init__()
        # layerdrop off: it randomly skips layers in training, which would change the
        # number of hidden states the weighted layer sum sees from step to step
        self.enc = WavLMModel.from_pretrained(name, dtype=torch.float32, layerdrop=0.0)
        self.enc.feature_extractor._freeze_parameters()
        n = self.enc.config.num_hidden_layers + 1
        w = torch.full((n,), -2.0)
        w[max(0, init_layer - 2):init_layer + 2] = 1.0
        self.layer_w = nn.Parameter(w)
        h = self.enc.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(2 * h, 1))

    def forward(self, x):
        hs = torch.stack(self.enc(x, output_hidden_states=True).hidden_states)  # (L, B, T, H)
        h = (torch.softmax(self.layer_w, 0)[:, None, None, None] * hs).sum(0)
        pooled = torch.cat([h.mean(1), h.std(1)], dim=-1)
        return self.head(pooled).squeeze(-1)


@torch.no_grad()
def predict_clip(model, wave, crop, device):
    """Average the prediction over overlapping windows that cover the clip."""
    if len(wave) <= crop:
        starts = [0]
    else:
        starts = list(np.linspace(0, len(wave) - crop, num=int(np.ceil(len(wave) / (crop * 0.75)))).astype(int))
    xs = []
    for s in starts:
        w = wave[s:s + crop]
        xs.append(normalise(np.pad(w, (0, crop - len(w))) if len(w) < crop else w))
    x = torch.from_numpy(np.stack(xs)).to(device)
    with torch.autocast("cuda", dtype=torch.float16):
        return float(model(x).float().mean())


def predict(model, waves, crop, device):
    model.eval()
    return np.array([predict_clip(model, w, crop, device) for w in waves])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/wavlm-base-plus")
    ap.add_argument("--name", default="wavlm_ft")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--crop_s", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    log = get_logger(f"finetune_audio_{args.name}")
    log.info("args: %s", vars(args))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"
    crop = int(args.crop_s * SR)

    y, test_idx = load_labels()
    yv = y.values.astype(np.float32)
    mu = float(yv.mean())
    t0 = time.time()
    tr_waves = load_audio("train", list(y.index))
    te_waves = load_audio("test", list(test_idx))
    log.info("loaded %d train / %d test clips in %.1f min", len(tr_waves), len(te_waves), (time.time() - t0) / 60)

    oof = np.zeros(len(yv))
    test_pred = np.zeros(len(te_waves))
    train_fit, train_cnt = np.zeros(len(yv)), np.zeros(len(yv))
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    for fold, (tr, va) in enumerate(skf.split(np.zeros(len(yv)), strat_bins(yv)), 1):
        t0 = time.time()
        model = AudioRegressor(args.model).to(device)
        opt = torch.optim.AdamW([
            {"params": [p for p in model.enc.parameters() if p.requires_grad], "lr": args.lr},
            {"params": list(model.head.parameters()) + [model.layer_w], "lr": 1e-3},
        ], weight_decay=0.01)
        loader = DataLoader(CropDS([tr_waves[i] for i in tr], yv[tr] - mu, crop),
                            batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)
        steps = len(loader) * args.epochs // args.accum
        sched = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)
        scaler = torch.amp.GradScaler()
        va_waves = [tr_waves[i] for i in va]

        for ep in range(1, args.epochs + 1):
            model.train()
            losses = []
            opt.zero_grad()
            for step, (x, target) in enumerate(loader, 1):
                with torch.autocast("cuda", dtype=torch.float16):
                    pred = model(x.to(device))
                loss = nn.functional.mse_loss(pred.float(), target.to(device))
                scaler.scale(loss / args.accum).backward()
                losses.append(loss.item())
                if step % args.accum == 0 or step == len(loader):
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    sched.step()
                    opt.zero_grad()

            p_va = np.clip(predict(model, va_waves, crop, device) + mu, 0, 5)
            log.info("fold %d epoch %d | train MSE %.4f | val RMSE %.4f | val Pearson %.4f | %.1f min",
                     fold, ep, np.mean(losses), rmse(yv[va], p_va), pearsonr(yv[va], p_va)[0], (time.time() - t0) / 60)

        oof[va] = p_va
        test_pred += np.clip(predict(model, te_waves, crop, device) + mu, 0, 5) / args.folds
        train_fit[tr] += np.clip(predict(model, [tr_waves[i] for i in tr], crop, device) + mu, 0, 5)
        train_cnt[tr] += 1
        top = torch.softmax(model.layer_w.detach(), 0).cpu().numpy()
        log.info("fold %d final val RMSE %.4f (%.1f min) | layer weights peak at layer %d",
                 fold, rmse(yv[va], p_va), (time.time() - t0) / 60, int(top.argmax()))
        del model, opt
        torch.cuda.empty_cache()

    train_fit /= train_cnt
    n = args.name
    log.info("%s train RMSE %.4f | OOF RMSE %.4f | OOF Pearson %.4f", n, rmse(yv, train_fit), rmse(yv, oof), pearsonr(yv, oof)[0])
    pd.DataFrame({n: oof, f"{n}_train": train_fit}, index=y.index).to_parquet(OUT / f"train_{n}.parquet")
    pd.DataFrame({n: test_pred}, index=test_idx).to_parquet(OUT / f"test_{n}.parquet")
    log.info("saved %s OOF / test predictions", n)


if __name__ == "__main__":
    main()
