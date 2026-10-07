"""
Transcribe all train/test clips with faster-whisper and cache the results.

Whisper tends to clean up speech (drops fillers, smooths restarts), which hides
exactly the errors we want to score. To keep transcripts closer to verbatim:
  - an initial prompt full of disfluencies nudges the decoder to keep them
  - condition_on_previous_text=False avoids it "polishing" based on earlier text
Word timestamps are kept so pause / speaking-rate features can be computed later.

Output: artifacts/transcripts/{train,test}.jsonl  (one record per audio file)
"""
import argparse
import json
import time
from pathlib import Path

import librosa
import pandas as pd
from faster_whisper import WhisperModel

from log_utils import get_logger

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "artifacts" / "transcripts"

VERBATIM_PROMPT = "Umm, so, uh, I- I was like, you know... I goes there and, hmm, we was talking."


def transcribe_file(model, path):
    # decode with librosa (16 kHz mono) instead of faster-whisper's PyAV loader
    audio, _ = librosa.load(path, sr=16000, mono=True)
    segments, info = model.transcribe(
        audio,
        language="en",
        beam_size=5,
        initial_prompt=VERBATIM_PROMPT,
        condition_on_previous_text=False,
        word_timestamps=True,
        vad_filter=False,
    )
    segs, words = [], []
    for s in segments:
        segs.append({
            "start": s.start, "end": s.end, "text": s.text.strip(),
            "avg_logprob": s.avg_logprob, "no_speech_prob": s.no_speech_prob,
            "compression_ratio": s.compression_ratio,
        })
        for w in s.words or []:
            words.append({"word": w.word, "start": w.start, "end": w.end, "prob": w.probability})
    return {
        "text": " ".join(s["text"] for s in segs).strip(),
        "duration": info.duration,
        "segments": segs,
        "words": words,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--compute_type", default="int8_float16")
    args = ap.parse_args()

    log = get_logger("transcribe")
    log.info("model=%s compute_type=%s", args.model, args.compute_type)

    OUT.mkdir(parents=True, exist_ok=True)
    model = WhisperModel(args.model, device="cuda", compute_type=args.compute_type)

    for split in ["train", "test"]:
        df = pd.read_csv(DATA / f"{split}.csv")
        out_path = OUT / f"{split}.jsonl"

        # resume support: skip files already transcribed
        done = set()
        if out_path.exists():
            with open(out_path) as f:
                done = {json.loads(line)["filename"] for line in f}
        todo = [n for n in df["filename"] if n not in done]
        log.info("%s: %d files, %d already done, %d to go", split, len(df), len(done), len(todo))

        t0 = time.time()
        with open(out_path, "a") as f:
            for i, fname in enumerate(todo, 1):
                t = time.time()
                rec = transcribe_file(model, DATA / split / fname)
                rec["filename"] = fname
                f.write(json.dumps(rec) + "\n")
                f.flush()

                eta = (time.time() - t0) / i * (len(todo) - i)
                log.info("%s [%d/%d] %s | %.1fs audio | %d words | %.1fs | eta %.0f min",
                         split, i, len(todo), fname, rec["duration"],
                         len(rec["text"].split()), time.time() - t, eta / 60)
        log.info("%s done in %.1f min", split, (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
