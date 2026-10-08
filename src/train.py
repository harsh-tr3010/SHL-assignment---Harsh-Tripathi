"""
Cross-validated training on the cached features + test predictions.

Models:
  - ridge   : Ridge on transcript embeddings + handcrafted + grammar features
  - svr     : RBF SVR on the same inputs
  - lgbm    : LightGBM on handcrafted + grammar + audio features, 20 embedding
              principal components and (if available) the LLM-judge scores
  - deberta : fine-tuned DeBERTa-v3 (precomputed by finetune_text.py)
Final prediction = weighted blend of the models, weights fitted on out-of-fold
predictions (non-negative least squares), then a linear calibration to undo
the shrinkage towards the mean, clipped to [0, 5].

Covariate shift: the test clips differ from the training clips (shorter,
faster speech, different prompts; see shift.py). By default every model, the
blend and the calibration are fitted with importance weights that up-weight
test-like training clips. `--no_shift_weights` turns this off.

Pseudo-labelling: the pipeline is then refitted once more with the 216 test clips
added to the final fits, labelled with the first run's predictions (weight 0.5),
so the models also see test-condition audio and prompts. `--pseudo_weight 0`
turns this off (v10b).

Topic-aware blend: about half the test clips answer prompts that are rare in the
training set (see topics.py). The blend weights and calibration are therefore
chosen on topic-grouped OOF predictions (whole prompts held out) and applied to
the models above. `--no_topic_blend` keeps the plain-CV weights (v14c).

Validation: repeated stratified K-fold (stratified on the label) so every fold
sees the full score range. Reported metrics: RMSE and Pearson r, the two
leaderboard metrics.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.optimize import nnls
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from log_utils import get_logger

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FEAT = ROOT / "artifacts" / "features"
OUT = ROOT / "artifacts" / "predictions"

SEED = 42
TEXT_GROUPS = ["handcrafted", "grammar", "embedding"]
TABULAR_GROUPS = ["handcrafted", "grammar", "audio", "embedding_pca"]
# models trained elsewhere (finetune_text.py) that only provide OOF + test predictions
PRECOMPUTED = ["deberta", "deberta_w", "deberta_large", "wavlm_ft", "wavlm_large_ft", "deberta_topic"]


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def load_labels():
    """Train labels (noise clips removed) and the test index."""
    train = pd.read_csv(DATA / "train.csv").set_index("filename")
    test = pd.read_csv(DATA / "test.csv").set_index("filename")
    # The 37 clips labelled 0 are not speech (continuous noise: one unbroken
    # energy segment, ~5% voiced frames, very high zero-crossing rate). The same
    # check flags none of the test clips, so they are dropped from training
    # rather than letting them drag the regression towards 0.
    train = train[train["label"] > 0]
    return train["label"], test.index


def make_pca(group, n=20):
    """First n principal components of a high-dimensional feature group
    (`<group>_pca`), so the tree model gets a compact view of it. PCA is
    unsupervised, so fitting it on train + test together uses no labels."""
    if (FEAT / f"train_{group}_pca.parquet").exists():
        return
    from sklearn.decomposition import PCA
    # note: train and test reuse the same file names (audio_192.wav exists in both
    # folders as different recordings), so the two sets are kept apart by position
    tr, te = (pd.read_parquet(FEAT / f"{s}_{group}.parquet") for s in ["train", "test"])
    both = np.vstack([tr.values, te.values])
    mu, sd = both.mean(0), both.std(0) + 1e-8
    pca = PCA(n, random_state=SEED).fit((both - mu) / sd)
    cols = [f"{group}_pc{i}" for i in range(n)]
    for s, d in [("train", tr), ("test", te)]:
        pd.DataFrame(pca.transform((d.values - mu) / sd), index=d.index, columns=cols).to_parquet(FEAT / f"{s}_{group}_pca.parquet")


COUNT_COLS = {
    "handcrafted": ["n_words", "n_unique", "n_sentences", "pause_long_count", "correction_count"],
    "audio": ["n_speech_chunks", "gap_long_count"],
}


def make_rates():
    """Duration-free version of the handcrafted + audio features (group `rates`).

    Test clips are ~7 s shorter than training clips on average, so raw counts
    (words, long pauses, speech chunks, ...) shift with clip length rather than
    with the speaker. Counts are turned into per-minute rates and the clip
    duration itself is dropped."""
    if (FEAT / "train_rates.parquet").exists():
        return
    for s in ["train", "test"]:
        hc = pd.read_parquet(FEAT / f"{s}_handcrafted.parquet")
        au = pd.read_parquet(FEAT / f"{s}_audio.parquet")
        minutes = au["duration"].values / 60.0
        out = []
        for name, d in [("handcrafted", hc), ("audio", au)]:
            d = d.copy()
            for c in COUNT_COLS[name]:
                d[f"{c}_per_min"] = d.pop(c).values / minutes
            out.append(d)
        rates = pd.concat(out, axis=1).drop(columns=["duration"])
        rates = rates.loc[:, ~rates.columns.duplicated()]
        rates.to_parquet(FEAT / f"{s}_rates.parquet")


def load_features(split, groups, index):
    if "rates" in groups:
        make_rates()
    for g in groups:
        if g.endswith("_pca"):
            make_pca(g[:-4])
    parts = [pd.read_parquet(FEAT / f"{split}_{g}.parquet") for g in groups]
    return pd.concat(parts, axis=1).loc[index]


def strat_bins(y):
    # half-point bins; 1.0 and 1.5 have only 4 samples so fold them into 2.0
    b = np.round(np.asarray(y) * 2).astype(int)
    b[b < 4] = 4
    return b


AUDIO_EMBEDDINGS = ["wavlm", "wavlm_large"]


def make_models(use_llm=False, use_wavlm=True, use_svr_audio=False, duration_free=False, use_gec=False):
    # LLM-judge features are off by default: they did not improve CV (or the public LB)
    has = lambda g: (FEAT / f"train_{g}.parquet").exists()
    tab = TABULAR_GROUPS + (["llm"] if use_llm and has("llm") else []) + (["gec"] if use_gec and has("gec") else [])
    if duration_free:
        tab = ["rates"] + [g for g in tab if g not in ("handcrafted", "audio")]
    speech = [g for g in AUDIO_EMBEDDINGS if use_wavlm and has(g)]
    tab = tab + [f"{g}_pca" for g in speech]
    models = {
        "ridge": (TEXT_GROUPS, lambda: make_pipeline(StandardScaler(), Ridge(alpha=300.0))),
        "svr": (TEXT_GROUPS, lambda: make_pipeline(StandardScaler(), SVR(C=3.0, epsilon=0.2))),
        "lgbm": (
            tab,
            lambda: LGBMRegressor(
                n_estimators=600, learning_rate=0.02, num_leaves=7, min_child_samples=15,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                random_state=SEED, verbose=-1,
            ),
        ),
    }
    for g in speech:
        # speech-only model: a separate view of the clip for the blend
        models[f"ridge_{g}"] = ([g], lambda: make_pipeline(StandardScaler(), Ridge(alpha=3000.0)))
        if use_svr_audio:
            # Non-linear version on a compressed (PCA-128) view of the same embedding.
            # Better CV (0.477 vs 0.496 for the blend) but worse public LB (0.3400 vs
            # 0.3308): the RBF kernel seems less robust to train/test recording
            # differences than the linear model, so it is off by default.
            models[f"svr_{g}"] = ([g], lambda: make_pipeline(
                StandardScaler(), PCA(128, random_state=SEED), SVR(C=10.0, epsilon=0.1)))
    return models


def wrmse(y, p, w):
    """RMSE weighted towards test-like clips (importance weights from shift.py)."""
    return float(np.sqrt(np.average((np.asarray(y) - np.asarray(p)) ** 2, weights=w)))


def fit(model, X, y, w=None):
    """Fit with optional sample weights (routed to the last step of a pipeline)."""
    if w is None:
        return model.fit(X, y)
    if hasattr(model, "steps"):
        return model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": w})
    return model.fit(X, y, sample_weight=w)


def cv_splits(X, y, folds, repeats, groups=None):
    """Stratified K-fold, repeated; with `groups`, whole topic clusters are held out
    together (StratifiedGroupKFold) so validation mimics an unseen prompt."""
    if groups is None:
        return list(RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=SEED).split(X, strat_bins(y)))
    return [s for r in range(repeats)
            for s in StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED + r).split(X, strat_bins(y), groups)]


def cross_validate(build, X, y, folds=5, repeats=3, log=None, name="", w=None, groups=None):
    """Repeated stratified K-fold. Returns OOF predictions (averaged over repeats)
    and a per-fold table of train/val metrics."""
    X, y = np.asarray(X), np.asarray(y)
    oof = np.zeros(len(y))
    rows = []
    for k, (tr, va) in enumerate(cv_splits(X, y, folds, repeats, groups), 1):
        m = fit(build(), X[tr], y[tr], None if w is None else w[tr])
        p_tr, p_va = m.predict(X[tr]), m.predict(X[va])
        oof[va] += p_va / repeats
        row = {"model": name, "fold": k, "train_rmse": rmse(y[tr], p_tr),
               "val_rmse": rmse(y[va], p_va), "val_pearson": pearsonr(y[va], p_va)[0]}
        rows.append(row)
        if log:
            log.info("  %s fold %2d | train RMSE %.4f | val RMSE %.4f | val Pearson %.4f",
                     name, k, row["train_rmse"], row["val_rmse"], row["val_pearson"])
    return oof, pd.DataFrame(rows)


def blend_weights(preds, y, sw=None):
    """Non-negative weights (summing to 1) that best combine the model predictions."""
    A, t = np.column_stack(preds), np.asarray(y)
    if sw is not None:  # weighted least squares: scale rows by sqrt(weight)
        s = np.sqrt(sw)
        A, t = A * s[:, None], t * s
    w, _ = nnls(A, t)
    return w / w.sum()


def run(folds=5, repeats=3, log=None, use_llm=False, use_wavlm=True, use_svr_audio=False,
        shift_weights=True, exclude=("wavlm_ft", "deberta_w", "deberta_large", "wavlm_large_ft", "deberta_topic"), duration_free=False, fit_weights="train_shift_weight",
        pseudo=None, pseudo_weight=1.0, use_gec=False, topic_cv=False, drop=()):
    y, test_idx = load_labels()
    if log:
        log.info("train rows after dropping noise clips: %d, test rows: %d", len(y), len(test_idx))
    # importance weights (test-likeness of each training clip); always used for the
    # weighted CV metric if available, and for fitting when shift_weights=True
    wpath = FEAT / "train_shift_weight.parquet"
    iw = pd.read_parquet(wpath).loc[y.index, "weight"].values if wpath.exists() else None
    fpath = FEAT / f"{fit_weights}.parquet"
    sw = pd.read_parquet(fpath).loc[y.index, "weight"].values if shift_weights else None
    if log and shift_weights:
        log.info("fitting with importance weights (covariate-shift correction)")
    groups = pd.read_parquet(FEAT / "train_topic.parquet").loc[y.index, "topic"].values if topic_cv else None
    if topic_cv:
        # topic-grouped validation: use the DeBERTa trained on topic-grouped folds
        exclude = tuple(e for e in exclude if e != "deberta_topic") + ("deberta",)
        if log:
            log.info("topic-grouped CV (%d topics)", len(set(groups)))
    ps = None
    if pseudo is not None:
        ps = (pd.read_csv(pseudo).set_index("filename")["label"].loc[test_idx].values
              if isinstance(pseudo, (str, Path)) else np.asarray(pseudo))
        if log:
            log.info("pseudo-labels for the final refit (weight %.2f)", pseudo_weight)

    oof, test_pred, train_pred, fold_tables = {}, {}, {}, []
    for name, (fgroups, build) in make_models(use_llm, use_wavlm, use_svr_audio, duration_free, use_gec).items():
        fgroups = [g for g in fgroups if g not in drop]
        Xtr = load_features("train", fgroups, y.index)
        Xte = load_features("test", fgroups, test_idx)
        if log:
            log.info("model %s | features %s | X shape %s", name, fgroups, Xtr.shape)

        oof[name], ft = cross_validate(build, Xtr, y, folds, repeats, log, name, w=sw, groups=groups)
        fold_tables.append(ft)

        full = fit(build(), Xtr.values, y.values, sw)  # refit on all training data for test
        train_pred[name] = full.predict(Xtr.values)
        if ps is None:
            test_pred[name] = full.predict(Xte.values)
        else:
            # pseudo-labelling: refit once more with the test clips added, labelled with an
            # earlier submission, so the model also sees test-condition audio and prompts.
            # Only this final refit changes; CV, blend weights and calibration do not.
            Xa = np.vstack([Xtr.values, Xte.values])
            ya = np.r_[y.values, ps]
            wa = np.r_[sw if sw is not None else np.ones(len(y)), np.full(len(ps), pseudo_weight)]
            test_pred[name] = fit(build(), Xa, ya, wa).predict(Xte.values)
        if log:
            log.info("%s | full-fit train RMSE %.4f | OOF RMSE %.4f | OOF Pearson %.4f",
                     name, rmse(y, train_pred[name]), rmse(y, oof[name]), pearsonr(y, oof[name])[0])

    for name in PRECOMPUTED:
        if name in exclude or not (FEAT / f"train_{name}.parquet").exists():
            continue
        tr = pd.read_parquet(FEAT / f"train_{name}.parquet").loc[y.index]
        te = pd.read_parquet(FEAT / f"test_{name}.parquet").loc[test_idx]
        oof[name], test_pred[name] = tr[name].values, te[name].values
        # in-fold predictions on the training clips, if the fine-tuning script saved them
        train_pred[name] = tr.get(f"{name}_train", tr[name]).values
        if log:
            log.info("%s (precomputed) | train RMSE %.4f | OOF RMSE %.4f | OOF Pearson %.4f",
                     name, rmse(y, train_pred[name]), rmse(y, oof[name]), pearsonr(y, oof[name])[0])

    names = list(oof)
    w = blend_weights([oof[n] for n in names], y, sw)
    combine = lambda d: np.column_stack([d[n] for n in names]) @ w
    raw_oof = combine(oof)

    # Blending shrinks predictions towards the mean (low scores come out too high,
    # high scores too low). A linear stretch fitted on the OOF blend undoes part
    # of that; its effect is estimated with an inner K-fold so the reported OOF
    # number is not fitted on itself.
    pw = None if sw is None else np.sqrt(sw)  # np.polyfit weights multiply residuals
    a, b = np.polyfit(raw_oof, y.values, 1, w=pw)
    cal_oof = np.zeros_like(raw_oof)
    for tr_i, va_i in KFold(folds, shuffle=True, random_state=SEED).split(raw_oof):
        ai, bi = np.polyfit(raw_oof[tr_i], y.values[tr_i], 1, w=None if pw is None else pw[tr_i])
        cal_oof[va_i] = ai * raw_oof[va_i] + bi
    calibrate = lambda p: np.clip(a * p + b, 0, 5)

    oof["blend"] = np.clip(cal_oof, 0, 5)
    train_pred["blend"] = calibrate(combine(train_pred))
    test_pred["blend"] = calibrate(combine(test_pred))

    if log:
        log.info("blend weights: %s | calibration: y = %.3f * p + %.3f",
                 dict(zip(names, np.round(w, 3).tolist())), a, b)
        log.info("blend before calibration | OOF RMSE %.4f | OOF Pearson %.4f",
                 rmse(y, np.clip(raw_oof, 0, 5)), pearsonr(y, raw_oof)[0])
        log.info("blend | train RMSE %.4f | OOF RMSE %.4f | OOF Pearson %.4f",
                 rmse(y, train_pred["blend"]), rmse(y, oof["blend"]), pearsonr(y, oof["blend"])[0])
        if iw is not None:
            log.info("blend | test-weighted OOF RMSE %.4f", wrmse(y, oof["blend"], iw))

    return {
        "y": y,
        "oof": pd.DataFrame(oof, index=y.index),
        "train_pred": pd.DataFrame(train_pred, index=y.index),
        "test_pred": pd.DataFrame(test_pred, index=test_idx),
        "folds": pd.concat(fold_tables, ignore_index=True),
        "weights": dict(zip(names, w)),
    }


def topic_blend(res, folds=5, repeats=3, log=None, **kw):
    """Re-choose the blend weights and calibration on topic-grouped OOF predictions
    (whole prompts held out, like the test set, where about half the clips answer
    prompts that are rare in training) and apply them to the test predictions of
    the standard models in `res`. Returns (test predictions, weights)."""
    rt = run(folds, repeats, log, topic_cv=True, **kw)
    oof_t = rt["oof"].drop(columns="blend").rename(columns={"deberta_topic": "deberta"})
    names = list(oof_t.columns)
    y = rt["y"].values
    iw = pd.read_parquet(FEAT / "train_shift_weight.parquet").loc[rt["y"].index, "weight"].values
    w = blend_weights([oof_t[n].values for n in names], y, iw)
    a, b = np.polyfit(oof_t.values @ w, y, 1, w=np.sqrt(iw))
    pred = np.clip(a * (res["test_pred"][names].values @ w) + b, 0, 5)
    if log:
        log.info("topic blend weights: %s | calibration: y = %.3f * p + %.3f",
                 dict(zip(names, np.round(w, 3).tolist())), a, b)
    return pred, dict(zip(names, w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--llm", action="store_true", help="add the LLM-judge features to LightGBM")
    ap.add_argument("--no_wavlm", action="store_true", help="leave out the WavLM audio embeddings")
    ap.add_argument("--svr_audio", action="store_true", help="add RBF-SVR models on the WavLM embeddings (v8)")
    ap.add_argument("--no_shift_weights", action="store_true",
                    help="fit without the covariate-shift importance weights (reproduces v7)")
    ap.add_argument("--duration_free", action="store_true", help="per-minute rates instead of counts for LightGBM")
    ap.add_argument("--fit_weights", default="train_shift_weight", help="weights file used for fitting")
    ap.add_argument("--gec", action="store_true", help="add grammatical-error-correction features to LightGBM")
    ap.add_argument("--topic_cv", action="store_true", help="topic-grouped CV folds (artifacts/features/train_topic.parquet)")
    ap.add_argument("--drop", nargs="*", default=[], help="feature groups to leave out of every model")
    ap.add_argument("--no_topic_blend", action="store_true",
                    help="keep the blend weights from the plain CV instead of the topic-grouped CV (v14c)")
    ap.add_argument("--pseudo", default=None, help="CSV of predicted test scores to add to the final refit")
    ap.add_argument("--pseudo_weight", type=float, default=0.5,
                    help="weight of the pseudo-labelled test clips (0 = no pseudo-labelling)")
    ap.add_argument("--exclude", nargs="*", default=["wavlm_ft", "deberta_w", "deberta_large", "wavlm_large_ft", "deberta_topic"],
                    help="precomputed models to leave out of the blend (fine-tuned WavLM-base: worse public LB; "
                         "importance-weighted DeBERTa: zero blend weight; DeBERTa-large: tie on the public LB "
                         "(0.3203 vs 0.3204); fine-tuned WavLM-large: zero blend weight)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    log = get_logger("train")
    log.info("folds=%d repeats=%d", args.folds, args.repeats)

    kw = dict(use_llm=args.llm, use_wavlm=not args.no_wavlm, use_svr_audio=args.svr_audio,
              shift_weights=not args.no_shift_weights, exclude=args.exclude,
              duration_free=args.duration_free, fit_weights=args.fit_weights, use_gec=args.gec,
              topic_cv=args.topic_cv, drop=tuple(args.drop))
    res = run(args.folds, args.repeats, log, **kw)
    if args.pseudo_weight > 0:
        # Pseudo-labelling: refit once more with the test clips added, labelled with the
        # predictions of the run above (or of --pseudo). Weight 0.5 beat 0 and 1.0 on
        # the public leaderboard (0.3204 vs 0.3215 / 0.3213).
        labels = args.pseudo if args.pseudo else res["test_pred"]["blend"].values
        res = run(args.folds, args.repeats, log, pseudo=labels, pseudo_weight=args.pseudo_weight, **kw)
    res["oof"].assign(label=res["y"]).to_csv(OUT / "oof.csv")
    res["folds"].to_csv(OUT / "fold_metrics.csv", index=False)

    final = res["test_pred"]["blend"].values
    if not args.no_topic_blend and not args.topic_cv:
        # Topic-aware blend (v18): weights chosen for unseen prompts. Public LB 0.3190
        # vs 0.3204 with the plain-CV weights (v14c).
        final, _ = topic_blend(res, args.folds, args.repeats, log, **{k: v for k, v in kw.items() if k != "topic_cv"})

    sub = pd.DataFrame({"filename": res["test_pred"].index, "label": final})
    sub.to_csv(ROOT / "submission.csv", index=False)
    log.info("wrote %s %s", ROOT / "submission.csv", sub.shape)


if __name__ == "__main__":
    main()
