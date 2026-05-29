"""Small measurement helpers shared by the benchmark script.

Deliberately functional and dependency-light: time a forward pass, read peak
VRAM, and summarize token statistics.
"""
from __future__ import annotations

import time
from statistics import mean

import torch


def peak_vram_mb() -> float:
    """Peak CUDA memory in MB since the last reset (0 on CPU)."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def reset_vram() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def time_inference(model, inputs, warmup: int = 2, runs: int = 10) -> dict:
    """Return latency (ms), throughput, and tokens/sec for repeated forwards."""
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    n_tokens = int(inputs["input_ids"].numel())

    for _ in range(warmup):
        with torch.no_grad():
            model(**inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latencies = []
    for _ in range(runs):
        start = time.perf_counter()
        with torch.no_grad():
            model(**inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

    avg_ms = mean(latencies)
    return {
        "latency_ms": round(avg_ms, 2),
        "throughput_seq_per_s": round(1000 / avg_ms, 2),
        "tokens_per_s": round(n_tokens / (avg_ms / 1000), 2),
    }


def token_stats(tokenizer, texts: list[str], hf: bool = True) -> dict:
    """Average / max token count for a batch of texts."""
    if hf:
        lengths = [len(tokenizer(t)["input_ids"]) for t in texts]
    else:
        lengths = [len(tokenizer.encode(t).ids) for t in texts]
    return {
        "avg_seq_len": round(mean(lengths), 2),
        "max_seq_len": max(lengths),
        "total_tokens": sum(lengths),
    }
