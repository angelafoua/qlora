#!/usr/bin/env bash
# End-to-end research pipeline: data -> tokenizer -> QLoRA -> benchmark.
# Run from the repository root. Stops on the first error.
set -euo pipefail

[ -f .env ] && set -a && source .env && set +a

echo "==> [1/4] Dataset: load + preprocess"
python dataset/load_dataset.py --config dataset/config.yaml
python dataset/preprocess.py   --config dataset/config.yaml

echo "==> [2/4] Tokenizer: train + evaluate"
python tokenizer/train_tokenizer.py    --config tokenizer/config.yaml
python tokenizer/evaluate_tokenizer.py --config tokenizer/config.yaml

echo "==> [3/4] QLoRA: train + evaluate"
python qlora/train.py    --config qlora/config.yaml
python qlora/evaluate.py --config qlora/config.yaml

echo "==> [4/4] Benchmark: measure + plot"
python benchmark/benchmark.py --config benchmark/config.yaml
python benchmark/plots.py

echo "==> Done. See benchmark_results.json, latency_plots/, comparison_tables/"
