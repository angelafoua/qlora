"""Turn benchmark_results.json into comparison tables and bar charts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# (metric column, human-readable axis label)
METRICS = [
    ("latency_ms", "Latency (ms)"),
    ("tokens_per_s", "Tokens / sec"),
    ("avg_seq_len", "Avg sequence length"),
    ("peak_vram_mb", "Peak VRAM (MB)"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="benchmark/benchmark_results.json")
    ap.add_argument("--plots_dir", default="benchmark/latency_plots")
    ap.add_argument("--tables_dir", default="benchmark/comparison_tables")
    args = ap.parse_args()

    with open(args.results) as f:
        df = pd.DataFrame(json.load(f))

    plots_dir = Path(args.plots_dir)
    tables_dir = Path(args.tables_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Tables: machine-readable CSV + markdown for the paper.
    df.to_csv(tables_dir / "comparison.csv", index=False)
    (tables_dir / "comparison.md").write_text(df.to_markdown(index=False))

    # One bar chart per metric, configurations on the x-axis.
    for col, label in METRICS:
        if col not in df:
            continue
        plt.figure(figsize=(6, 4))
        plt.bar(df["config"], df[col])
        plt.ylabel(label)
        plt.title(label + " by configuration")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(plots_dir / f"{col}.png", dpi=150)
        plt.close()

    print(f"Wrote tables to {tables_dir} and plots to {plots_dir}")


if __name__ == "__main__":
    main()
