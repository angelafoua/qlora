"""Compare base vs custom tokenizer on average token count over the dataset.

Reports mean tokens/example for each tokenizer and the relative reduction,
which is the headline efficiency metric for the study.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from datasets import load_from_disk
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from train_tokenizer import load_config


def base_lengths(name: str, texts: list[str]) -> list[int]:
    tok = AutoTokenizer.from_pretrained(name)
    return [len(tok(t)["input_ids"]) for t in texts]


def custom_lengths(path: str, texts: list[str]) -> list[int]:
    tok = Tokenizer.from_file(path)
    return [len(tok.encode(t).ids) for t in texts]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="tokenizer/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    ds = load_from_disk(cfg["text_dataset"])
    texts = list(ds["test"][cfg["text_column"]])

    base = base_lengths(cfg["base_tokenizer"], texts)
    custom = custom_lengths(str(Path(cfg["output_dir"]) / "tokenizer.json"), texts)

    base_avg, custom_avg = mean(base), mean(custom)
    reduction = (base_avg - custom_avg) / base_avg if base_avg else 0.0

    metrics = {
        "num_examples": len(texts),
        "base_tokenizer": cfg["base_tokenizer"],
        "base_avg_tokens": round(base_avg, 2),
        "custom_avg_tokens": round(custom_avg, 2),
        "token_reduction_pct": round(100 * reduction, 2),
    }
    with open(cfg["metrics_file"], "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
