"""
Topic clusters of the transcripts, for topic-grouped validation.

The clips answer different prompts (a playground, a hobby, a market, an
airport, a flood, social media ...). Clustering the MPNet embeddings of all
transcripts (train + test, no labels involved) recovers those prompts. About
40% of the test clips fall in clusters that are rare or absent in training, so
a plain K-fold, where every fold has seen every topic, is optimistic. Grouping
the CV folds by topic simulates predicting clips on an unseen prompt.

Output: artifacts/features/{train,test}_topic.parquet (column: topic)
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from log_utils import get_logger
from train import FEAT, SEED, load_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()
    log = get_logger("topics")

    y, test_idx = load_labels()
    tr = pd.read_parquet(FEAT / "train_embedding.parquet").loc[y.index]
    te = pd.read_parquet(FEAT / "test_embedding.parquet").loc[test_idx]
    X = normalize(np.vstack([tr.values, te.values]))
    km = KMeans(args.k, n_init=10, random_state=SEED).fit(X)
    lab_tr, lab_te = km.labels_[: len(tr)], km.labels_[len(tr):]

    for c in range(args.k):
        log.info("topic %d | train %3d | test %3d | mean label %.2f", c, (lab_tr == c).sum(), (lab_te == c).sum(),
                 y.values[lab_tr == c].mean() if (lab_tr == c).any() else float("nan"))
    pd.DataFrame({"topic": lab_tr}, index=tr.index).to_parquet(FEAT / "train_topic.parquet")
    pd.DataFrame({"topic": lab_te}, index=te.index).to_parquet(FEAT / "test_topic.parquet")
    log.info("saved topic clusters (k=%d)", args.k)


if __name__ == "__main__":
    main()
