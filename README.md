# Reducing Inference Cost for Enterprise PII Detection

A minimalist, reproducible research pipeline studying whether **tokenizer
optimization** and **lightweight QLoRA adaptation** can reduce inference
latency, token count, and memory usage while keeping reasonable PII detection
quality.

Built on a Nemotron / NeMo-compatible model and the Nemotron-PII dataset.

> This is a graduate-style research scaffold, not a production system: simple
> scripts, simple configs, functional code.

## Hypothesis

| Lever                  | Expected effect                              |
|------------------------|----------------------------------------------|
| Custom tokenizer       | fewer tokens/example → lower latency & VRAM  |
| QLoRA (4-bit) adapters | small memory footprint, fast adaptation      |
| Tokenizer + QLoRA      | combined efficiency gains                    |

## Project structure

```
dataset/      # load + BIO-preprocess the Nemotron-PII dataset
  load_dataset.py     # reproducible train/val/test splits
  preprocess.py       # char spans -> token-level BIO labels
  config.yaml
tokenizer/    # train + evaluate a PII-optimized tokenizer
  train_tokenizer.py  # custom BPE with PII special tokens
  evaluate_tokenizer.py  # base vs custom token-count reduction
  config.yaml
qlora/        # 4-bit QLoRA token-classification fine-tuning
  train.py            # minimal Trainer loop, LoRA r=8/alpha=16/dropout=0.05
  evaluate.py         # seqeval entity precision/recall/F1
  config.yaml
benchmark/    # inference-efficiency comparison across 4 configs
  benchmark.py        # latency, throughput, VRAM, token stats
  metrics.py          # timing + memory helpers
  plots.py            # comparison tables + bar charts
  config.yaml
```

Generated artifacts: `dataset/processed_dataset/`, `tokenizer/custom_tokenizer/`,
`tokenizer/tokenizer_metrics.json`, `qlora/fine_tuned_model/`, `qlora/metrics.json`,
`benchmark/benchmark_results.json`, `benchmark/latency_plots/`,
`benchmark/comparison_tables/`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your HF_TOKEN
```

A CUDA GPU is required for 4-bit QLoRA (bitsandbytes) and VRAM benchmarking.

## Experiment workflow

Run the whole pipeline:

```bash
bash run.sh
```

Or step by step:

```bash
# 1. Data
python dataset/load_dataset.py
python dataset/preprocess.py

# 2. Tokenizer
python tokenizer/train_tokenizer.py
python tokenizer/evaluate_tokenizer.py   # -> tokenizer/tokenizer_metrics.json

# 3. QLoRA
python qlora/train.py
python qlora/evaluate.py                  # -> qlora/metrics.json

# 4. Benchmark
python benchmark/benchmark.py             # -> benchmark/benchmark_results.json
python benchmark/plots.py                 # -> plots + tables
```

The benchmark compares four configurations: **base**, **custom tokenizer only**,
**QLoRA only**, and **tokenizer + QLoRA**.

## Reproducibility

- All randomness is seeded (`seed: 42` in the configs).
- Splits are deterministic and saved to disk before preprocessing.
- Every stage reads a single `config.yaml`; change one value, re-run that stage.
- Metrics are written as JSON so results are diffable across runs.

## Configuration notes

- Swap `base_model` / `dataset_name` in the configs to use a local mirror or a
  different Nemotron checkpoint.
- Adjust `lora_target_modules` to match the attention layer names of your model.
