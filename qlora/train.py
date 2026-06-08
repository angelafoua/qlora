"""Lightweight QLoRA fine-tuning for token-level PII detection.

Loads the base model in 4-bit, attaches LoRA adapters, and runs a minimal
Trainer loop. Only the adapter weights are saved.
"""
from __future__ import annotations

import argparse
import json

import torch
import yaml
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, num_labels: int):
    bnb = BitsAndBytesConfig(
        load_in_4bit=cfg["load_in_4bit"],
        bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=cfg["bnb_4bit_use_double_quant"],
        # score is randomly initialized (not in the pretrained checkpoint) so
        # it must stay in full precision; quantizing a freshly init'd layer
        # leaves its bnb quant state uninitialized and crashes at forward time.
        llm_int8_skip_modules=["score"],
    )
    model = AutoModelForTokenClassification.from_pretrained(
        cfg["base_model"], num_labels=num_labels, quantization_config=bnb
    )
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
        task_type="TOKEN_CLS",
    )
    return get_peft_model(model, lora)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="qlora/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    with open(cfg["label_map"]) as f:
        labels = json.load(f)["labels"]

    ds = load_from_disk(cfg["processed_dataset"])
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = build_model(cfg, num_labels=len(labels))
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        bf16=cfg["bf16"],
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        seed=cfg["seed"],
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=DataCollatorForTokenClassification(tok),
    )
    trainer.train()

    model.save_pretrained(cfg["output_dir"])
    tok.save_pretrained(cfg["output_dir"])
    with open(cfg["metrics_file"], "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print(f"Adapter + tokenizer saved to {cfg['output_dir']}")


if __name__ == "__main__":
    main()
