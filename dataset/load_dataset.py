"""Load the Nemotron-PII dataset and produce reproducible train/val/test splits."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from datasets import DatasetDict, load_dataset


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_splits(cfg: dict) -> DatasetDict:
    """Load the raw dataset and carve out fixed train/val/test fractions."""
    raw = load_dataset(cfg["dataset_name"])
    # Most HF datasets expose a single "train" split; collapse to one then re-split.
    data = raw["train"] if "train" in raw else next(iter(raw.values()))

    s = cfg["splits"]
    seed = cfg["seed"]
    # First peel off train, then split the remainder into val/test.
    first = data.train_test_split(train_size=s["train"], seed=seed)
    rest_val = s["validation"] / (s["validation"] + s["test"])
    second = first["test"].train_test_split(train_size=rest_val, seed=seed)

    return DatasetDict(
        train=first["train"],
        validation=second["train"],
        test=second["test"],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="dataset/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    splits = make_splits(cfg)

    out = Path(cfg["output_dir"]) / "raw_splits"
    splits.save_to_disk(str(out))
    print(f"Saved splits to {out}")
    for name, ds in splits.items():
        print(f"  {name}: {len(ds)} examples")


if __name__ == "__main__":
    main()
