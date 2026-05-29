"""Train a custom BPE tokenizer specialized for enterprise PII text.

The custom tokenizer adds reserved special tokens for common PII shapes so
they survive tokenization as single units, reducing token count downstream.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from datasets import load_from_disk
from tokenizers import Tokenizer, models, pre_tokenizers, trainers


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def text_iterator(dataset_path: str, text_column: str):
    """Yield raw text from every split for tokenizer training."""
    ds = load_from_disk(dataset_path)
    for split in ds.values():
        for text in split[text_column]:
            yield text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="tokenizer/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

    trainer = trainers.BpeTrainer(
        vocab_size=cfg["vocab_size"],
        min_frequency=cfg["min_frequency"],
        special_tokens=["<unk>", "<pad>", "<s>", "</s>"] + cfg["special_tokens"],
    )

    tokenizer.train_from_iterator(
        text_iterator(cfg["text_dataset"], cfg["text_column"]),
        trainer=trainer,
    )

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out / "tokenizer.json"))
    print(f"Custom tokenizer saved to {out} (vocab={tokenizer.get_vocab_size()})")


if __name__ == "__main__":
    main()
