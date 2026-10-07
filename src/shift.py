"""
Covariate-shift check and importance weights.

Adversarial validation (a classifier asked to tell training clips from test
clips) showed the two sets differ: test clips are shorter (~49 s vs ~56 s),
speakers are faster with fewer long pauses, and the prompts / topics are
distributed differently, while grammatical acceptability itself is not shifted.

Importance weighting corrects for this: each training clip gets the weight
w = p(test | x) / p(train | x), estimated out-of-fold by the same classifier,
so models fit the clips that look like the test set more closely. Weights are
clipped and normalised to mean 1 to keep their variance in check.

Output: artifacts/features/train_shift_weight.parquet (column: weight)
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from log_utils import get_logger
from train import FEAT, load_features, load_labels

GROUPS = ["handcrafted", "audio", "embedding_pca", "wavlm_large_pca"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=float, nargs=2, default=[0.2, 5.0], help="clip raw weights to [lo, hi]")
    # tempering (w ** t, t < 1) trades some bias for lower variance; full weights
    # (t = 1) did better on the public LB than t = 0.5
    ap.add_argument("--temper", type=float, default=1.0)
    ap.add_argument("--out", default="train_shift_weight", help="output file name (without .parquet)")
    args = ap.parse_args()
    log = get_logger("shift")
    log.info("clip=%s temper=%.2f", args.clip, args.temper)
    y, test_idx = load_labels()
    tr = load_features("train", GROUPS, y.index).values
    te = load_features("test", GROUPS, test_idx).values
    X = np.vstack([tr, te])
    is_test = np.r_[np.zeros(len(tr)), np.ones(len(te))]

    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000))
    p = cross_val_predict(clf, X, is_test, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                          method="predict_proba")[:, 1]
    log.info("train-vs-test AUC (features %s): %.3f", GROUPS, roc_auc_score(is_test, p))

    p_tr = np.clip(p[: len(tr)], 1e-3, 1 - 1e-3)
    w = p_tr / (1 - p_tr) * (len(tr) / len(te))
    w = np.clip(w, *args.clip) ** args.temper
    w = w / w.mean()
    ess = w.sum() ** 2 / (w ** 2).sum()
    log.info("weights: min %.2f | median %.2f | max %.2f | effective sample size %.0f of %d",
             w.min(), np.median(w), w.max(), ess, len(w))
    pd.DataFrame({"weight": w}, index=y.index).to_parquet(FEAT / f"{args.out}.parquet")
    log.info("saved %s.parquet", args.out)


if __name__ == "__main__":
    main()
