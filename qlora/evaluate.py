"""Evaluate the QLoRA PII model on the test split using seqeval (entity F1)."""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import yaml
from datasets import load_from_disk
from peft import PeftModel
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForTokenClassification,
)
from torch.utils.data import DataLoader

IGNORE_INDEX = -100


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(cfg: dict, num_labels: int):
    bnb = BitsAndBytesConfig(
        load_in_4bit=cfg["load_in_4bit"],
        bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, cfg["bnb_4bit_compute_dtype"]),
    )
    base = AutoModelForTokenClassification.from_pretrained(
        cfg["base_model"], num_labels=num_labels, quantization_config=bnb
    )
    return PeftModel.from_pretrained(base, cfg["output_dir"]).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="qlora/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    with open(cfg["label_map"]) as f:
        labels = json.load(f)["labels"]
    id2label = {i: l for i, l in enumerate(labels)}

    ds = load_from_disk(cfg["processed_dataset"])["test"]
    tok = AutoTokenizer.from_pretrained(cfg["output_dir"])
    model = load_model(cfg, num_labels=len(labels))

    loader = DataLoader(
        ds, batch_size=cfg["batch_size"],
        collate_fn=DataCollatorForTokenClassification(tok),
    )

    true_tags, pred_tags = [], []
    for batch in loader:
        gold = batch.pop("labels")
        with torch.no_grad():
            logits = model(**{k: v.to(model.device) for k, v in batch.items()}).logits
        preds = logits.argmax(-1).cpu().numpy()
        for p_row, g_row in zip(preds, gold.numpy()):
            t, p = [], []
            for pi, gi in zip(p_row, g_row):
                if gi == IGNORE_INDEX:
                    continue
                t.append(id2label[int(gi)])
                p.append(id2label[int(pi)])
            true_tags.append(t)
            pred_tags.append(p)

    metrics = {
        "precision": round(precision_score(true_tags, pred_tags), 4),
        "recall": round(recall_score(true_tags, pred_tags), 4),
        "f1": round(f1_score(true_tags, pred_tags), 4),
    }
    print(classification_report(true_tags, pred_tags))
    print(json.dumps(metrics, indent=2))

    with open(cfg["metrics_file"]) as f:
        existing = json.load(f) if cfg["metrics_file"] else []
    with open(cfg["metrics_file"], "w") as f:
        json.dump({"test": metrics, "train_log": existing}, f, indent=2)


if __name__ == "__main__":
    main()
