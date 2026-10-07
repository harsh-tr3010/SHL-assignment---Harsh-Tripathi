"""
Pick which WavLM layers to pool, by cross-validated Ridge RMSE.

Scores every single layer and a few contiguous ranges, logs the table, and
rewrites the feature group with the best range. Uses only the training labels
inside CV, the same as every other model choice.
"""
import argparse

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audio_embed import CACHE, DATA, write_group
from log_utils import get_logger
from train import cross_validate, load_labels, rmse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="wavlm")
    ap.add_argument("--alpha", type=float, default=3000.0)
    args = ap.parse_args()
    log = get_logger(f"select_layers_{args.name}")

    y, _ = load_labels()
    names = pd.read_csv(DATA / "train.csv")["filename"].tolist()
    arr = np.load(CACHE / f"train_{args.name}_layers.npy")
    pos = [names.index(f) for f in y.index]  # rows of the labelled (non-noise) clips
    arr = arr[pos]
    n_layers = arr.shape[1]

    ridge = lambda: make_pipeline(StandardScaler(), Ridge(alpha=args.alpha))
    results = {}
    candidates = [(i, i) for i in range(n_layers)]
    q = n_layers // 4
    candidates += [(a, b) for a in range(1, n_layers, q) for b in range(a + q - 1, n_layers, q)]
    for a, b in candidates:
        X = arr[:, a:b + 1].mean(1)
        oof, _ = cross_validate(ridge, X, y, repeats=1)
        results[(a, b)] = rmse(y, oof)
        log.info("layers %2d-%2d | CV RMSE %.4f | Pearson %.4f", a, b, results[(a, b)], pearsonr(y, oof)[0])

    best = min(results, key=results.get)
    log.info("best layers %s (CV RMSE %.4f)", best, results[best])
    write_group(args.name, range(best[0], best[1] + 1))
    log.info("rewrote feature group %s", args.name)


if __name__ == "__main__":
    main()
