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

# Different PII datasets name the same span fields differently (e.g. ai4privacy
# uses ``privacy_mask`` with ``label/start/end``). Accept the common variants so
# the pipeline works across Nemotron-PII and similar corpora.
LABEL_KEYS = ("label", "type", "entity_type", "tag", "pii_type")
START_KEYS = ("start", "start_index", "begin", "char_start", "start_offset")
END_KEYS = ("end", "end_index", "stop", "char_end", "end_offset")


def _pick(d: dict, keys: tuple[str, ...], what: str):
    """Return the first key present in ``d`` from ``keys`` (raises if none)."""
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(
        f"Could not find a {what} field in span {d!r}; expected one of {keys}."
    )


def normalize_span(sp: dict) -> dict:
    """Coerce a raw span dict into a uniform {label, start, end} shape."""
    return {
        "label": _pick(sp, LABEL_KEYS, "label"),
        "start": int(_pick(sp, START_KEYS, "start")),
        "end": int(_pick(sp, END_KEYS, "end")),
    }


def resolve_spans_column(ds, preferred: str) -> str:
    """Return ``preferred`` if present, else auto-detect a span-like column.

    A span column holds, per example, a list of dicts each describing a PII
    entity. We sniff the first non-empty value of every column to find one.
    """
    columns = ds["train"].column_names if "train" in ds else next(iter(ds.values())).column_names
    if preferred in columns:
        return preferred

    split = ds["train"] if "train" in ds else next(iter(ds.values()))
    for col in columns:
        for value in split[col]:
            if isinstance(value, list) and value:
                first = value[0]
                if isinstance(first, dict) and any(k in first for k in LABEL_KEYS):
                    print(
                        f"Configured spans_column '{preferred}' not found; "
                        f"auto-detected '{col}' instead."
                    )
                    return col
                break  # column is a list but not of span dicts
    raise ValueError(
        f"Configured spans_column '{preferred}' doesn't exist and no span-like "
        f"column could be detected. Available columns: {columns}. "
        f"Set 'spans_column' in the config to the correct field."
    )


def build_label_list(ds, spans_column: str) -> list[str]:
    """Collect every entity type and expand it into B-/I- tags plus O."""
    types: set[str] = set()
    for split in ds.values():
        for spans in split[spans_column]:
            for sp in spans:
                types.add(normalize_span(sp)["label"])
    labels = ["O"]
    for t in sorted(types):
        labels += [f"B-{t}", f"I-{t}"]
    return labels


def char_to_bio(spans: list[dict], offsets, label2id: dict) -> list[int]:
    """Assign a BIO id to each token given its (start, end) char offsets."""
    spans = [normalize_span(sp) for sp in spans]
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

    if cfg["text_column"] not in ds["train"].column_names:
        raise ValueError(
            f"Configured text_column '{cfg['text_column']}' doesn't exist. "
            f"Available columns: {ds['train'].column_names}."
        )

    spans_column = resolve_spans_column(ds, cfg["spans_column"])

    tok = AutoTokenizer.from_pretrained(cfg["base_tokenizer"])
    labels = build_label_list(ds, spans_column)
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
            for spans, offs in zip(batch[spans_column], enc["offset_mapping"])
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
