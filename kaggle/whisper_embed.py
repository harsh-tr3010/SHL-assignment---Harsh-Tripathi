"""
Kaggle-notebook script: layer-wise Whisper encoder embeddings (needs 2 GPUs, e.g. T4 x2).

The Whisper encoder is trained to map speech to text, so its hidden states carry
word- and phrase-level information that WavLM's do not. Same recipe as
src/audio_embed.py: every hidden state is mean/std-pooled over the real (non-padded)
frames of the clip. Clips longer than 30 s are cut into 30 s windows (Whisper's
input size) and the frames of all windows are pooled together.

Outputs in /kaggle/working ({split}_{name}_layers.npy, n_clips x n_layers x 2*hidden,
rows in the order of {split}.csv): copy them to artifacts/cache/ and choose the
layers with src/select_layers.py --name whisper_large / whisper_medium.

Notebook settings: Accelerator = GPU T4 x2, Internet = On.
"""
import glob
import threading
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from transformers import WhisperFeatureExtractor, WhisperModel

OUT = Path("/kaggle/working")
SR, WIN = 16000, 30 * 16000
MODELS = [("openai/whisper-large-v3", "whisper_large", "cuda:0"),
          ("openai/whisper-medium", "whisper_medium", "cuda:1")]
_lock = threading.Lock()


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} | {msg}"
    with _lock:
        print(line, flush=True)
        with open(OUT / "whisper_embed.log", "a") as f:
            f.write(line + "\n")


@torch.no_grad()
def embed(enc, fe, audio, device):
    chunks = [audio[i:i + WIN] for i in range(0, len(audio), WIN)]
    chunks = [c for c in chunks if len(c) >= SR] or chunks[:1]  # drop a tail shorter than 1 s
    feats = fe(chunks, sampling_rate=SR, return_tensors="pt").input_features.to(device, torch.float16)
    hs = enc(feats, output_hidden_states=True).hidden_states  # n_layers x (n_chunks, 1500, hidden)
    # 1500 encoder frames per 30 s window -> 50 frames per second; keep only the real frames
    n_valid = [min(1500, int(np.ceil(len(c) / SR * 50))) for c in chunks]
    out = []
    for h in hs:
        frames = torch.cat([h[k, :n] for k, n in enumerate(n_valid)]).float()
        out.append(torch.cat([frames.mean(0), frames.std(0)]).cpu().numpy())
    return np.stack(out)


def run(model_name, name, device, data):
    fe = WhisperFeatureExtractor.from_pretrained(model_name)
    enc = WhisperModel.from_pretrained(model_name, dtype=torch.float16).encoder.to(device).eval()
    for split in ["train", "test"]:
        files = pd.read_csv(data / f"{split}.csv").filename.tolist()
        t0, rows = time.time(), []
        for i, f in enumerate(files, 1):
            audio, _ = librosa.load(data / split / f, sr=SR, mono=True)
            rows.append(embed(enc, fe, audio, device))
            if i % 100 == 0:
                log(f"{name} {split} {i}/{len(files)} | {(time.time() - t0) / 60:.1f} min")
        arr = np.stack(rows).astype(np.float16)
        np.save(OUT / f"{split}_{name}_layers.npy", arr)
        log(f"{name} {split} done: {arr.shape} in {(time.time() - t0) / 60:.1f} min")


def main():
    data = Path(glob.glob("/kaggle/input/**/train.csv", recursive=True)[0]).parent
    log(f"data: {data} | GPUs: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    errors = []

    def job(*a):
        try:
            run(*a, data)
        except Exception as e:  # keep the other model running
            errors.append(e)
            log(f"{a[1]} failed: {e!r}")

    threads = [threading.Thread(target=job, args=m) for m in MODELS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    log("saved all embeddings to /kaggle/working")


if __name__ == "__main__":
    main()
