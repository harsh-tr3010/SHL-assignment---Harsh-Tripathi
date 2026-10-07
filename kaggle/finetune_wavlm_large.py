"""
Kaggle-notebook version of src/finetune_audio.py for WavLM-large (needs ~16 GB GPU).

Self-contained: finds the competition data under /kaggle/input, downloads
microsoft/wavlm-large from the HuggingFace hub, fine-tunes it in 5-fold CV and
writes out-of-fold / test predictions to /kaggle/working. Copy the two parquet
files into artifacts/features/ and train.py picks them up as `wavlm_large_ft`.

Notebook settings: Accelerator = GPU (T4 or P100), Internet = On.
"""
import glob
import os
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

MODEL = "microsoft/wavlm-large"
NAME = "wavlm_large_ft"
FOLDS, EPOCHS, LR, BATCH, ACCUM, CROP_S, SEED = 5, 6, 1e-5, 2, 4, 15.0, 42
HEAD_LR = 2e-4
INIT_LAYER = 21  # best frozen layer for WavLM-large
SR = 16000
OUT = Path("/kaggle/working")


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} | {msg}"
    print(line, flush=True)
    with open(OUT / f"{NAME}.log", "a") as f:
        f.write(line + "\n")


def find_data():
    csv = glob.glob("/kaggle/input/**/train.csv", recursive=True)
    if not csv:
        raise FileNotFoundError("train.csv not found under /kaggle/input - attach the competition data")
    return Path(csv[0]).parent


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def strat_bins(y):
    b = np.round(np.asarray(y) * 2).astype(int)
    b[b < 4] = 4
    return b


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
    def __init__(self, name, init_layer):
        super().__init__()
        # layerdrop off so every forward pass returns the same set of hidden states
        self.enc = WavLMModel.from_pretrained(name, layerdrop=0.0)
        self.enc.feature_extractor._freeze_parameters()
        n = self.enc.config.num_hidden_layers + 1
        w = torch.full((n,), -2.0)
        w[max(0, init_layer - 2):init_layer + 2] = 1.0
        self.layer_w = nn.Parameter(w)
        h = self.enc.config.hidden_size
        # LayerNorm on the pooled stats: WavLM-large hidden states are large in
        # magnitude and without it the fp16 head diverged in the first epochs
        self.head = nn.Sequential(nn.LayerNorm(2 * h), nn.Dropout(0.1), nn.Linear(2 * h, 1))

    def forward(self, x):
        hs = torch.stack(self.enc(x, output_hidden_states=True).hidden_states)
        h = (torch.softmax(self.layer_w, 0)[:, None, None, None] * hs).sum(0)
        return self.head(torch.cat([h.mean(1), h.std(1)], dim=-1)).squeeze(-1)


@torch.no_grad()
def predict(model, waves, crop, device):
    model.eval()
    out = []
    for wave in waves:
        if len(wave) <= crop:
            starts = [0]
        else:
            starts = np.linspace(0, len(wave) - crop, num=int(np.ceil(len(wave) / (crop * 0.75)))).astype(int)
        xs = [normalise(np.pad(wave[s:s + crop], (0, max(0, crop - len(wave[s:s + crop]))))) for s in starts]
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(float(model(torch.from_numpy(np.stack(xs)).to(device)).float().mean()))
    return np.array(out)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda"
    crop = int(CROP_S * SR)
    data = find_data()
    log(f"data dir: {data} | GPU: {torch.cuda.get_device_name(0)}")

    train = pd.read_csv(data / "train.csv")
    train = train[train.label > 0].reset_index(drop=True)  # zero-score clips are non-speech noise
    test = pd.read_csv(data / "test.csv")
    y = train.label.values.astype(np.float32)
    mu = float(y.mean())

    t0 = time.time()
    tr_waves = [sf.read(data / "train" / f, dtype="int16")[0] for f in train.filename]
    te_waves = [sf.read(data / "test" / f, dtype="int16")[0] for f in test.filename]
    log(f"loaded {len(tr_waves)} train / {len(te_waves)} test clips in {(time.time() - t0) / 60:.1f} min")

    oof, test_pred = np.zeros(len(y)), np.zeros(len(te_waves))
    train_fit, train_cnt = np.zeros(len(y)), np.zeros(len(y))
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

    for fold, (tr, va) in enumerate(skf.split(np.zeros(len(y)), strat_bins(y)), 1):
        t0 = time.time()
        model = AudioRegressor(MODEL, INIT_LAYER).to(device)
        model.enc.gradient_checkpointing_enable()  # needed to fit WavLM-large on a 16 GB T4
        opt = torch.optim.AdamW([
            {"params": [p for p in model.enc.parameters() if p.requires_grad], "lr": LR},
            {"params": list(model.head.parameters()) + [model.layer_w], "lr": HEAD_LR},
        ], weight_decay=0.01)
        loader = DataLoader(CropDS([tr_waves[i] for i in tr], y[tr] - mu, crop),
                            batch_size=BATCH, shuffle=True, num_workers=2, drop_last=True)
        steps = len(loader) * EPOCHS // ACCUM
        sched = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)
        scaler = torch.amp.GradScaler()
        va_waves = [tr_waves[i] for i in va]

        for ep in range(1, EPOCHS + 1):
            model.train()
            losses = []
            opt.zero_grad()
            for step, (x, target) in enumerate(loader, 1):
                with torch.autocast("cuda", dtype=torch.float16):
                    pred = model(x.to(device))
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
            p_va = np.clip(predict(model, va_waves, crop, device) + mu, 0, 5)
            log(f"fold {fold} epoch {ep} | train MSE {np.mean(losses):.4f} | val RMSE {rmse(y[va], p_va):.4f} "
                f"| val Pearson {pearsonr(y[va], p_va)[0]:.4f} | {(time.time() - t0) / 60:.1f} min")

        oof[va] = p_va
        test_pred += np.clip(predict(model, te_waves, crop, device) + mu, 0, 5) / FOLDS
        train_fit[tr] += np.clip(predict(model, [tr_waves[i] for i in tr], crop, device) + mu, 0, 5)
        train_cnt[tr] += 1
        log(f"fold {fold} final val RMSE {rmse(y[va], p_va):.4f} ({(time.time() - t0) / 60:.1f} min)")
        # save progress after every fold so a timeout does not lose everything
        np.savez(OUT / f"{NAME}_progress.npz", oof=oof, test_pred=test_pred * FOLDS / fold,
                 train_fit=train_fit, train_cnt=train_cnt, folds_done=fold)
        del model, opt
        torch.cuda.empty_cache()

    train_fit /= train_cnt
    log(f"{NAME} train RMSE {rmse(y, train_fit):.4f} | OOF RMSE {rmse(y, oof):.4f} | OOF Pearson {pearsonr(y, oof)[0]:.4f}")
    pd.DataFrame({NAME: oof, f"{NAME}_train": train_fit}, index=train.filename.values).to_parquet(OUT / f"train_{NAME}.parquet")
    pd.DataFrame({NAME: test_pred}, index=test.filename.values).to_parquet(OUT / f"test_{NAME}.parquet")
    log("saved train/test parquet files to /kaggle/working")


if __name__ == "__main__":
    main()
