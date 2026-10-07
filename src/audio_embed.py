"""
Clip-level speech embeddings from WavLM, a self-supervised speech model.

The transcript-based models only see words. Raters also hear rhythm, hesitation
and how sentences are delivered; WavLM's hidden layers encode that kind of
prosodic / phonetic information. The model is used frozen: audio is split into
10 s windows and every layer's hidden states are pooled over the whole clip
(mean + std over frames).

All layers are cached so the layer range can be chosen by cross-validation
(select_layers.py) instead of by guesswork. The feature group written for the
models is the average over the chosen layers -> one (2 x hidden)-d vector per clip.

Output:
  artifacts/features/{split}_{name}.parquet         (chosen layers, used by train.py)
  artifacts/cache/{split}_{name}_layers.npy         (all layers, n_clips x n_layers x 2*hidden)
"""
import argparse
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from transformers import WavLMModel

from log_utils import get_logger

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "artifacts" / "features"
CACHE = ROOT / "artifacts" / "cache"
SR = 16000
WINDOW = 10 * SR


@torch.no_grad()
def embed_all_layers(model, path, device):
    y, _ = librosa.load(path, sr=SR, mono=True)
    y = (y - y.mean()) / (y.std() + 1e-7)  # WavLM expects zero-mean, unit-variance input
    frames = []
    for s in range(0, len(y), WINDOW):
        chunk = y[s:s + WINDOW]
        if len(chunk) < SR // 2:  # skip tails shorter than 0.5 s
            continue
        x = torch.from_numpy(chunk).float().unsqueeze(0).to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            hs = model(x, output_hidden_states=True).hidden_states
        frames.append(torch.stack(hs)[:, 0].float().cpu())  # (layers, T, H)
    h = torch.cat(frames, dim=1)
    return torch.cat([h.mean(1), h.std(1)], dim=-1).numpy()  # (layers, 2H)


def write_group(name, layers):
    """Average the cached per-layer stats over `layers` and save the feature group."""
    for split in ["train", "test"]:
        arr = np.load(CACHE / f"{split}_{name}_layers.npy")
        names = pd.read_csv(DATA / f"{split}.csv")["filename"].tolist()
        X = arr[:, list(layers)].mean(1)
        h = X.shape[1] // 2
        cols = [f"{name}_mean_{i}" for i in range(h)] + [f"{name}_std_{i}" for i in range(h)]
        pd.DataFrame(X, index=names, columns=cols).to_parquet(OUT / f"{split}_{name}.parquet")
        (OUT / f"{split}_{name}_pca.parquet").unlink(missing_ok=True)  # derived PCs are now stale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/wavlm-base-plus")
    ap.add_argument("--name", default="wavlm", help="feature group name")
    ap.add_argument("--layers", type=int, nargs=2, default=[4, 9], help="first and last layer (inclusive)")
    args = ap.parse_args()

    log = get_logger(f"audio_embed_{args.name}")
    CACHE.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    model = WavLMModel.from_pretrained(args.model).to(device).eval()
    log.info("model=%s name=%s", args.model, args.name)

    for split in ["train", "test"]:
        cache = CACHE / f"{split}_{args.name}_layers.npy"
        if cache.exists():
            log.info("%s: using cached %s", split, cache.name)
            continue
        names = pd.read_csv(DATA / f"{split}.csv")["filename"].tolist()
        t0, rows = time.time(), []
        for i, n in enumerate(names, 1):
            rows.append(embed_all_layers(model, DATA / split / n, device))
            if i % 100 == 0:
                log.info("%s %d/%d (%.1f min)", split, i, len(names), (time.time() - t0) / 60)
        np.save(cache, np.stack(rows).astype(np.float32))
        log.info("%s done (%d clips, %.1f min)", split, len(names), (time.time() - t0) / 60)

    layers = range(args.layers[0], args.layers[1] + 1)
    write_group(args.name, layers)
    log.info("wrote feature group %s from layers %s", args.name, list(layers))


if __name__ == "__main__":
    main()
