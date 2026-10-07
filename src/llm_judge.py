"""
Zero-shot grammar rating of each transcript with an instruction-tuned LLM
(Qwen3-4B-Instruct, 4-bit), using the official rubric as the prompt.

Instead of parsing a generated answer, we read the model's next-token
probabilities for "1".."5" right after the prompt. That gives a full
distribution per clip; the expected value and the five probabilities become
features for the regressors (the LLM is never trained on our labels).

Output: artifacts/features/{split}_llm.parquet
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from log_utils import get_logger

ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPTS = ROOT / "artifacts" / "transcripts"
OUT = ROOT / "artifacts" / "features"

RUBRIC = """You are an expert examiner of spoken English grammar. You will read an automatic transcript of a 45-60 second spoken answer. Judge ONLY grammar: sentence structure, syntax and grammatical accuracy. Ignore pronunciation, accent, content, punctuation and capitalisation, and do not penalise obvious speech-recognition slips (odd single words). Fillers and restarts are normal in speech; only count them against the speaker when sentences are left incomplete or broken.

Grammar score rubric:
1 - Struggles with proper sentence structure and syntax; limited control over simple grammatical structures and memorised sentence patterns.
2 - Limited understanding of sentence structure and syntax. Uses simple structures but consistently makes basic sentence-structure and grammatical mistakes. Might leave sentences incomplete.
3 - Decent grasp of sentence structure but makes errors in grammatical structure, or decent grasp of grammatical structure but makes errors in sentence syntax and structure.
4 - Strong understanding of sentence structure and syntax. Consistently good control of grammar. Occasional errors are minor, do not cause misunderstandings, and most are self-corrected.
5 - High grammatical accuracy and adept control of complex grammar. Uses grammar accurately and effectively, seldom makes noticeable mistakes, handles complex structures well and self-corrects when necessary.

Reply with a single digit from 1 to 5."""


def build_prompt(tok, text):
    msgs = [
        {"role": "system", "content": RUBRIC},
        {"role": "user", "content": f"Transcript:\n\"\"\"{text.strip() or '(no speech)'}\"\"\"\n\nGrammar score (1-5):"},
    ]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def score(model, tok, texts, digit_ids, batch, log, split):
    rows = []
    for i in range(0, len(texts), batch):
        prompts = [build_prompt(tok, t) for t in texts[i:i + batch]]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        logits = model(**enc).logits[:, -1, :]  # left padding -> last position is the next token
        p = torch.softmax(logits[:, digit_ids].float(), dim=-1).cpu().numpy()
        for row in p:
            rows.append({**{f"llm_p{k}": row[k - 1] for k in range(1, 6)},
                         "llm_expected": float((row * np.arange(1, 6)).sum()),
                         "llm_argmax": int(row.argmax() + 1)})
        if (i // batch) % 10 == 0:
            log.info("%s %d/%d | last expected score %.2f", split, min(i + batch, len(texts)), len(texts), rows[-1]["llm_expected"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    log = get_logger("llm_judge")
    log.info("model=%s", args.model)

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.float16),
        device_map="cuda",
    ).eval()
    digit_ids = [tok.convert_tokens_to_ids(str(d)) for d in range(1, 6)]
    log.info("digit token ids: %s | GPU memory %.2f GB", digit_ids, torch.cuda.memory_allocated() / 1e9)

    for split in ["train", "test"]:
        recs = [json.loads(l) for l in open(TRANSCRIPTS / f"{split}.jsonl")]
        names = [r["filename"] for r in recs]
        rows = score(model, tok, [r["text"] for r in recs], digit_ids, args.batch, log, split)
        df = pd.DataFrame(rows, index=names)
        df.to_parquet(OUT / f"{split}_llm.parquet")
        log.info("%s done %s | mean expected %.2f", split, df.shape, df.llm_expected.mean())


if __name__ == "__main__":
    main()
