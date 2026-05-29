"""Benchmark inference efficiency across four configurations.

  1. base            -> base model + base tokenizer
  2. tokenizer       -> base model + custom tokenizer
  3. qlora           -> QLoRA model + base tokenizer
  4. tokenizer+qlora -> QLoRA model + custom tokenizer

For each, we record latency, throughput, VRAM, and token statistics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from datasets import load_from_disk
from peft import PeftModel
from tokenizers import Tokenizer
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from metrics import peak_vram_mb, reset_vram, time_inference, token_stats


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_base_model(cfg: dict, num_labels: int):
    return AutoModelForTokenClassification.from_pretrained(
        cfg["base_model"], num_labels=num_labels
    ).eval()


def load_qlora_model(cfg: dict, num_labels: int):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
    base = AutoModelForTokenClassification.from_pretrained(
        cfg["base_model"], num_labels=num_labels, quantization_config=bnb
    )
    return PeftModel.from_pretrained(base, cfg["qlora_dir"]).eval()


def encode_batch(texts, tokenizer, hf: bool, max_length: int):
    """Build a padded (input_ids, attention_mask) batch for either tokenizer."""
    if hf:
        enc = tokenizer(
            texts, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
    tokenizer.enable_padding()
    tokenizer.enable_truncation(max_length=max_length)
    enc = tokenizer.encode_batch(texts)
    ids = torch.tensor([e.ids for e in enc])
    mask = torch.tensor([e.attention_mask for e in enc])
    return {"input_ids": ids, "attention_mask": mask}


def run_case(name, model, tokenizer, texts, hf, max_length) -> dict:
    reset_vram()
    inputs = encode_batch(texts, tokenizer, hf, max_length)
    timing = time_inference(model, inputs)
    tokens = token_stats(tokenizer, texts, hf=hf)
    return {"config": name, **timing, **tokens, "peak_vram_mb": round(peak_vram_mb(), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="benchmark/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    with open(cfg["label_map"]) as f:
        num_labels = len(json.load(f)["labels"])

    texts = list(load_from_disk(cfg["text_dataset"])["test"][cfg["text_column"]])
    texts = texts[: cfg["num_samples"]]
    max_length = cfg["max_length"]

    base_tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    custom_tok = Tokenizer.from_file(cfg["custom_tokenizer"])
    base_model = load_base_model(cfg, num_labels)
    qlora_model = load_qlora_model(cfg, num_labels)

    results = [
        run_case("base", base_model, base_tok, texts, True, max_length),
        run_case("tokenizer", base_model, custom_tok, texts, False, max_length),
        run_case("qlora", qlora_model, base_tok, texts, True, max_length),
        run_case("tokenizer+qlora", qlora_model, custom_tok, texts, False, max_length),
    ]

    Path(cfg["results_file"]).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg["results_file"], "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
