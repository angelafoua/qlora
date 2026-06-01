# Reducing Inference Cost for Enterprise PII Detection: A Tokenizer-Optimization and QLoRA Efficiency Benchmark

# Abstract

* **Problem:** PII detection at enterprise scale is constrained by inference latency, token count, and GPU memory.
* **Proposed idea:** Combine a PII-optimized custom tokenizer with lightweight 4-bit QLoRA adaptation of a Nemotron model, and measure their individual and combined efficiency effects.
* **Expected contribution:** A minimal, reproducible benchmark comparing four configurations (base, tokenizer-only, QLoRA-only, tokenizer+QLoRA) on latency, throughput, VRAM, token count, and detection quality.

# 1. Introduction

## Problem

* PII detection must run on long enterprise text streams where token count drives cost.
* Full fine-tuning of large models is memory- and compute-expensive.
* General-purpose tokenizers fragment common PII shapes (SSNs, EINs, phones), inflating token counts.
* No single, reproducible scaffold here isolates the efficiency contribution of *tokenizer* vs *adapter* changes.

## Research Questions

* Does a PII-optimized custom tokenizer reduce tokens per example versus the base tokenizer?
* Does 4-bit QLoRA adaptation reduce memory footprint while retaining detection quality?
* Do tokenizer optimization and QLoRA combine for additive efficiency gains?
* What is the latency / throughput / VRAM trade-off across the four configurations?

## Contributions

* A four-configuration inference-efficiency benchmark (base, tokenizer, QLoRA, tokenizer+QLoRA).
* A custom BPE tokenizer with reserved PII special tokens.
* A minimal QLoRA token-classification pipeline (LoRA r=8, alpha=16, dropout=0.05).
* Fully seeded, JSON-logged, diffable reproducibility scaffold.

# 2. Background and Related Work

* Closest prior work: QLoRA / parameter-efficient fine-tuning (peft) and 4-bit quantization (bitsandbytes).
* Tokenizer specialization via HuggingFace BPE with domain special tokens.
* Token-classification PII detection evaluated with seqeval entity metrics.
* Built on a Nemotron / NeMo-compatible base model (`nvidia/Nemotron-Mini-4B-Instruct`) and the `nvidia/Nemotron-PII` dataset.
* Research gap: efficiency contributions of tokenizer vs adapter are rarely isolated and benchmarked together on PII data.
* *Placeholder:* no formal citation list is present in the repository.

# 3. Core Concept / Framework

(Most important section)

* **Definition — Efficiency levers:** three independent levers are studied, per the repository hypothesis table.
  * Custom tokenizer → fewer tokens/example → lower latency & VRAM.
  * QLoRA (4-bit) adapters → small memory footprint, fast adaptation.
  * Tokenizer + QLoRA → combined efficiency gains.
* **Pipeline stages (taxonomy of components):**
  * `dataset/` — load + BIO-preprocess Nemotron-PII (char spans → token-level B-/I-/O labels).
  * `tokenizer/` — train + evaluate a PII-optimized custom BPE tokenizer.
  * `qlora/` — 4-bit QLoRA token-classification fine-tuning.
  * `benchmark/` — inference-efficiency comparison across four configurations.
* **Key concepts:**
  * PII special tokens kept as single units: `<SSN>`, `<EIN>`, `<PHONE>`, `<ADDRESS>`, `<EMAIL>`, `<ID>`.
  * Offset-mapping alignment converts character spans into BIO tags; subword continuations / specials ignored with `-100`.
  * Schema-robust span parsing (label/start/end aliases, JSON-string spans, auto-detected span column).
* Figure: pipeline diagram — Data → Tokenizer → QLoRA → Benchmark.
* Diagram: BIO alignment from character spans to token-level labels.
* Figure: four-configuration matrix (base / tokenizer / QLoRA / tokenizer+QLoRA).

# 4. Dataset / Benchmark / Experimental Design

* **Data source:** `nvidia/Nemotron-PII` (HF hub id; overridable to a local mirror).
* **Splits:** train 0.8 / validation 0.1 / test 0.1, deterministic, `seed: 42`, saved to disk before preprocessing.
* **Labels:** BIO tags built from observed entity types (O plus B-/I- per type).
* **Sequence length:** `max_length: 256`.
* **Configurations benchmarked:**
  * base — base model + base tokenizer.
  * tokenizer — base model + custom tokenizer.
  * qlora — QLoRA model + base tokenizer.
  * tokenizer+qlora — QLoRA model + custom tokenizer.
* **Benchmark sampling:** `num_samples: 64` examples drawn from the test split per configuration.

# 5. Evaluation Metrics

* **Token reduction %** — relative drop in mean tokens/example, custom vs base tokenizer.
* **Avg / max sequence length** — token counts per batch.
* **Latency (ms)** — mean wall-clock per forward pass (2 warmup, 10 timed runs).
* **Throughput (seq/s)** — sequences processed per second.
* **Tokens/sec** — token-level processing rate.
* **Peak VRAM (MB)** — peak CUDA memory since reset.
* **Precision / Recall / F1** — seqeval entity-level detection quality.

# 6. Methodology

* **Data pipeline:** load dataset → make seeded splits → resolve span column → tokenize with offsets → emit BIO labels + `label_map.json`.
* **Tokenizer:** train HuggingFace BPE (`vocab_size: 32000`, `min_frequency: 2`) with reserved PII special tokens; evaluate token reduction on the test split.
* **Model components:** `AutoModelForTokenClassification` base loaded in 4-bit (nf4, double-quant, bfloat16 compute); LoRA adapters on `q_proj`, `v_proj`.
* **Training procedure:** minimal `Trainer` loop — 3 epochs, batch size 8, lr 2e-4, weight decay 0.01, bf16, eval/save per epoch; only adapter weights saved.
* **Inference / evaluation:** seqeval precision/recall/F1 on test split; ignore `-100` positions.
* **Benchmark procedure:** encode batch per tokenizer, time forward passes, record VRAM and token stats, write JSON; plots + CSV/markdown tables generated.

# 7. Experimental Evaluation

## Models

* `nvidia/Nemotron-Mini-4B-Instruct` (base model and base tokenizer).
* Custom PII BPE tokenizer (trained in-repo).
* QLoRA-adapted variant of the base model.
* *Placeholder:* no additional model families are configured in the repository.

## Protocol

* Single `config.yaml` per stage; change one value and re-run that stage.
* Deterministic splits and `seed: 42` throughout.
* Same `max_length: 256` across configurations.
* Identical 64-sample test subset per benchmark case; 2 warmup + 10 timed runs.
* Metrics written as JSON for diffable comparison across runs.
* CUDA GPU required for 4-bit QLoRA and VRAM benchmarking.

# 8. Results

(No results produced — planned artifacts only.)

* Expected table: four-configuration comparison (`comparison.csv` / `comparison.md`) over latency, throughput, avg seq len, peak VRAM.
* Expected table: tokenizer metrics (`tokenizer_metrics.json`) — base vs custom avg tokens and reduction %.
* Expected table: QLoRA test metrics (`metrics.json`) — precision / recall / F1.
* Expected figures (`latency_plots/`): bar charts of latency, tokens/sec, avg seq len, peak VRAM by configuration.
* Planned comparisons: base vs tokenizer (token/latency effect); base vs QLoRA (memory/quality effect); base vs tokenizer+QLoRA (combined effect).

# 9. Analysis

* Ablation: isolate tokenizer-only vs QLoRA-only vs combined to attribute gains.
* Sensitivity: vary `vocab_size`, `min_frequency`, and `max_length`; observe token reduction and latency.
* Sensitivity: vary LoRA `r` / `alpha` / `dropout` and target modules for quality/memory trade-off.
* Error analysis: per-entity seqeval breakdown (`classification_report`) to find weak PII types.
* *Potential:* effect of swapping the base Nemotron checkpoint on combined efficiency.

# 10. Discussion

* Implications: tokenizer + adapter co-design can cut inference cost for enterprise PII workloads.
* Strengths: minimal, reproducible, seeded, JSON-diffable; clean separation of efficiency levers.
* Strengths: schema-robust preprocessing works across Nemotron-PII and similar corpora.
* Risks: gains may be dataset/tokenizer-specific; quality may regress under aggressive tokenization or low LoRA rank.

# 11. Limitations

* Single base model and dataset configured.
* Small benchmark sample (64 examples per configuration).
* Requires a CUDA GPU; CPU path reports zero VRAM.
* Graduate-style research scaffold, not a production system.
* *Placeholder:* no human evaluation or external baselines included.
* **Security note (repo hygiene):** `.env.example` contains what looks like a real `HF_TOKEN` value — should be a placeholder, not a live secret.

# 12. Conclusion

## Main Contribution

* A reproducible, four-configuration benchmark isolating tokenizer and QLoRA efficiency effects for PII detection.
* A PII-specialized tokenizer plus a minimal QLoRA token-classification pipeline.

## Future Work

* Add more base models / tokenizers and larger benchmark samples.
* Explore additional PII special tokens and vocabulary sizes.
* Tune LoRA configuration and target modules per architecture.
* *Potential:* extend metrics to end-to-end cost and energy.
