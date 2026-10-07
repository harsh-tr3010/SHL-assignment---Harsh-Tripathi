"""
Signal-level features computed directly from the waveform (no pretrained model).

These don't measure grammar directly, but they capture delivery: how much of the
clip is actual speech, pausing patterns, loudness and pitch variation. They also
help catch near-silent / broken clips, which tend to get very low scores.

Output: artifacts/features/{split}_audio.parquet
"""
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "artifacts" / "features"
SR = 16000


def extract(path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    dur = len(y) / sr
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]

    # speech / silence split with an energy threshold relative to the clip's peak
    intervals = librosa.effects.split(y, top_db=30, frame_length=1024, hop_length=256)
    voiced = sum(e - s for s, e in intervals) / sr
    gaps = np.diff(intervals.reshape(-1))[1::2] / sr if len(intervals) > 1 else np.array([0.0])

    f0, voiced_flag, _ = librosa.pyin(y, fmin=65, fmax=400, sr=sr, frame_length=1024, hop_length=512)
    f0 = f0[~np.isnan(f0)]

    return {
        "duration": dur,
        "rms_mean": rms.mean(),
        "rms_std": rms.std(),
        "rms_max": rms.max(),
        "speech_ratio_sig": voiced / dur if dur else 0,
        "n_speech_chunks": len(intervals),
        "chunks_per_min": len(intervals) / dur * 60 if dur else 0,
        "gap_mean": gaps.mean(),
        "gap_max": gaps.max(),
        "gap_long_count": int((gaps > 1.0).sum()),
        "f0_mean": f0.mean() if len(f0) else 0,
        "f0_std": f0.std() if len(f0) else 0,
        "voiced_frac": np.mean(voiced_flag) if voiced_flag is not None else 0,
        "zcr_mean": librosa.feature.zero_crossing_rate(y)[0].mean(),
        "centroid_mean": librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean(),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ["train", "test"]:
        if (OUT / f"{split}_audio.parquet").exists():
            continue
        names = pd.read_csv(DATA / f"{split}.csv")["filename"].tolist()
        rows = Parallel(n_jobs=-1, verbose=5)(delayed(extract)(DATA / split / n) for n in names)
        df = pd.DataFrame(rows, index=names)
        df.to_parquet(OUT / f"{split}_audio.parquet")
        print(split, df.shape)


if __name__ == "__main__":
    main()
