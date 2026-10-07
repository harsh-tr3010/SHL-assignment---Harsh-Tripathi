# Grammar Scoring Engine for Spoken English

SHL hiring challenge: predict a 0–5 grammar score for 45–60 s spoken English clips.
Evaluated with RMSE (leaderboard) and Pearson correlation.

**Final result:** public leaderboard RMSE 0.3204; cross-validated RMSE 0.507 (test-weighted 0.488, Pearson 0.868), training RMSE 0.229.

The full write-up (approach, EDA, CV results, training RMSE, ablations, train/test shift analysis, error analysis,
submission history) is in [`notebooks/grammar_scoring.ipynb`](notebooks/grammar_scoring.ipynb).

Competition data, data-derived artifacts (transcripts, features) and test-set predictions are not included in this
repository; the scripts below regenerate all of them from the competition data.

## Approach

Each clip is looked at two ways: *what* was said and *how* it was said.

1. **Transcribe** with Whisper large-v3-turbo (faster-whisper, fp16). Decoding is pushed towards verbatim output
   (filler-heavy initial prompt, no conditioning on previous text) so grammatical errors, restarts and fillers are kept.
2. **Text model**: DeBERTa-v3-base fine-tuned as a regressor on the transcripts (5-fold CV × 3 seeds, fixed 5 epochs,
   no per-fold early stopping).
3. **Speech model**: frozen WavLM (large: layer 21, base: layer 9, both chosen by CV); hidden states are
   mean/std-pooled over the clip, then a Ridge regressor.
4. **Feature model**: LightGBM on interpretable features — fluency (speaking rate, pauses, fillers, repetitions,
   cut-offs), RoBERTa-CoLA grammatical acceptability, signal-level audio stats and PCA summaries of the text and
   speech embeddings.
5. **Blend**: non-negative least squares on out-of-fold predictions, linear calibration, clip to [0, 5].
6. **Train/test shift correction**: adversarial validation (a classifier separating training from test clips, AUC 0.83)
   showed the test clips are shorter (~49 s vs ~56 s), faster, with fewer long pauses and different prompts. Each training
   clip gets an importance weight p(test|x)/p(train|x), used in every model fit, the blend and the calibration.
7. **Pseudo-labelling**: a final refit adds the 216 test clips, labelled with the first-pass predictions at half weight,
   so the models also see test-condition audio and prompts.

The 37 training clips scored 0 are non-speech noise (no voiced segments, very high zero-crossing rate). No test clip
has that profile, so they are excluded from training. Train and test reuse file names for different recordings, so the
two sets are never joined on file name.

| Version | Change | CV RMSE | Public LB |
|---|---|---|---|
| v1 | frozen text/fluency features + Ridge/SVR/LightGBM | 0.611 | 0.4251 |
| v2 | + fine-tuned DeBERTa | 0.538 | 0.3523 |
| v5 | + WavLM-base speech embeddings | 0.507 | 0.3379 |
| v6 | WavLM layer chosen by CV | 0.505 | 0.3324 |
| v7 | + WavLM-large | 0.496 | 0.3308 |
| v10b | v7 + covariate-shift importance weights | 0.507 (test-weighted 0.488) | 0.3215 |
| **v14c** | **v10b + pseudo-labelled test clips, weight 0.5 (final)** | **0.507 (test-weighted 0.488)** | **0.3204** |

The notebook lists every submitted version, including the ones that did not help (LLM judge, SVR and fine-tuned audio
models, weighted-loss DeBERTa, stronger importance weights).

## Layout

```
src/
  transcribe.py       Whisper transcription                     -> artifacts/transcripts/*.jsonl
  audio_features.py   signal-level audio features                -> artifacts/features/*_audio.parquet
  features.py         fluency, CoLA and MPNet features            -> artifacts/features/
  finetune_text.py    DeBERTa fine-tuning in CV                   -> artifacts/features/*_deberta.parquet
  audio_embed.py      WavLM layer-wise embeddings                 -> artifacts/features/*_wavlm*.parquet
  select_layers.py    choose the WavLM layer by CV
  shift.py            adversarial validation + importance weights -> artifacts/features/train_shift_weight.parquet
  train.py            CV, shift-weighted blend, calibration, pseudo-labelling -> submission.csv
  finetune_audio.py   WavLM end-to-end fine-tuning (tried, not used in the final model)
  llm_judge.py        zero-shot LLM rubric scores (tried, not used in the final model)
  log_utils.py        file + console logging (logs/)
kaggle/
  finetune_wavlm_large.py   WavLM-large fine-tuning for a 16 GB Kaggle GPU (tried)
notebooks/
  grammar_scoring.ipynb
logs/                 per-clip / per-fold / per-epoch logs of the runs
```

## Running

Data goes in `data/` (`train.csv`, `test.csv`, `train/`, `test/`). Tested on Ubuntu (WSL2), Python 3.12, RTX 3050 (6 GB).

```bash
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

python src/transcribe.py --model large-v3-turbo --compute_type float16     # ~40 min
python src/audio_features.py
python src/features.py
python src/finetune_text.py --seeds 42 7 2024                              # ~35 min
python src/audio_embed.py --model microsoft/wavlm-base-plus --name wavlm   # ~9 min
python src/select_layers.py --name wavlm
python src/audio_embed.py --model microsoft/wavlm-large --name wavlm_large # ~25 min
python src/select_layers.py --name wavlm_large
python src/shift.py                                                         # importance weights
python src/train.py                                                         # writes submission.csv (final, v14c)
```

`train.py --pseudo_weight 0` reproduces v10b, and `train.py --pseudo_weight 0 --no_shift_weights` reproduces v7.

Notes for this setup:
- If faster-whisper cannot find `libcublas.so.12`, add the CUDA libraries shipped with the torch wheel
  (`site-packages/nvidia/{cublas,cudnn}/lib`) to `LD_LIBRARY_PATH`.
- If LightGBM cannot find `libgomp.so.1`, install it (`apt install libgomp1`) or point `LD_LIBRARY_PATH` at a bundled copy.
