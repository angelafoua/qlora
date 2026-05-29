"""Convert PII character spans into token-level BIO labels and save the result.

We tokenize with offset mapping so each token can be matched against the
character spans of the privacy mask, yielding standard B-/I-/O tags.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import DatasetDict, load_from_disk
from transformers import AutoTokenizer

from load_dataset import load_config

IGNORE_INDEX = -100  # subword continuations / specials ignored by the loss


def build_label_list(ds, spans_column: str) -> list[str]:
    """Collect every entity type and expand it into B-/I- tags plus O."""
    types: set[str] = set()
    for split in ds.values():
        for spans in split[spans_column]:
            for sp in spans:
                types.add(sp["label"])
    labels = ["O"]
    for t in sorted(types):
        labels += [f"B-{t}", f"I-{t}"]
    return labels


def char_to_bio(spans: list[dict], offsets, label2id: dict) -> list[int]:
    """Assign a BIO id to each token given its (start, end) char offsets."""
    labels = []
    for start, end in offsets:
        if start == end:  # special token
            labels.append(IGNORE_INDEX)
            continue
        tag = "O"
        for sp in spans:
            if start >= sp["start"] and end <= sp["end"]:
                prefix = "B" if start == sp["start"] else "I"
                tag = f"{prefix}-{sp['label']}"
                break
        labels.append(label2id[tag])
    return labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="dataset/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["output_dir"])
    ds: DatasetDict = load_from_disk(str(out_dir / "raw_splits"))

    tok = AutoTokenizer.from_pretrained(cfg["base_tokenizer"])
    labels = build_label_list(ds, cfg["spans_column"])
    label2id = {l: i for i, l in enumerate(labels)}

    def encode(batch):
        enc = tok(
            batch[cfg["text_column"]],
            truncation=True,
            max_length=cfg["max_length"],
            return_offsets_mapping=True,
        )
        enc["labels"] = [
            char_to_bio(spans, offs, label2id)
            for spans, offs in zip(batch[cfg["spans_column"]], enc["offset_mapping"])
        ]
        enc.pop("offset_mapping")
        return enc

    processed = ds.map(encode, batched=True, remove_columns=ds["train"].column_names)
    processed.save_to_disk(str(out_dir))

    with open(out_dir / "label_map.json", "w") as f:
        json.dump({"labels": labels, "label2id": label2id}, f, indent=2)

    print(f"Processed dataset saved to {out_dir} ({len(labels)} BIO labels)")


if __name__ == "__main__":
    main()
