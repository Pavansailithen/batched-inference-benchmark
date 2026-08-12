import csv
import os
import sys
import time
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_prompt_of_length(tokenizer: AutoTokenizer, target_length: int) -> str:
    """Generate a prompt text that tokenizes to approximately target_length tokens."""
    base_text = (
        "In modern computing systems, memory bandwidth, latency, and cache hierarchy play "
        "fundamental roles in determining the throughput and efficiency of deep learning workloads. "
        "Key-value caching reduces redundant attention computations during autoregressive decoding. "
    )
    tokens = tokenizer.encode(base_text * 15, add_special_tokens=False)
    sliced_tokens = tokens[:target_length]
    return tokenizer.decode(sliced_tokens)


def run_memory_profiling(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    batch_sizes: Optional[List[int]] = None,
    seq_lengths: Optional[List[int]] = None,
    max_new_tokens: int = 50,
    results_csv_path: str = "results/phase2_memory_profile.csv",
    plot_path: str = "results/phase2_memory_vs_batchsize.png",
    vram_budget_mb: float = 4096.0,
    safety_margin_mb: float = 500.0,
):
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8]
    if seq_lengths is None:
        seq_lengths = [64, 128, 256]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 85)
    print("PHASE 2: KV-CACHE MEMORY PROFILING & HARDWARE CEILING ANALYSIS")
    print("=" * 85)
    print(f"Device        : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Model         : {model_name} (torch.float16)")
    print(f"Batch Sizes   : {batch_sizes}")
    print(f"Seq Lengths   : {seq_lengths} tokens (Prompt)")
    print(f"Max New Tokens: {max_new_tokens} (Generated)")
    print("-" * 85)

    # 1. Load Model & Tokenizer
    print("\n[Init] Loading model and tokenizer onto device...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    # 2. Warm-up generation (absorb CUDA context init / kernel compilation)
    print("[Init] Running throwaway warm-up generation (batch=1, max_new_tokens=10)...")
    warmup_inputs = tokenizer("Warmup prompt for CUDA kernel compilation.", return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model.generate(
            input_ids=warmup_inputs.input_ids,
            attention_mask=warmup_inputs.attention_mask,
            max_new_tokens=10,
            do_sample=False,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 3. Reset peak memory stats before sweep
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Pre-generate prompts for each seq_length
    prompt_cache = {
        sl: generate_prompt_of_length(tokenizer, sl) for sl in seq_lengths
    }

    results: List[Dict] = []

    print("\n" + "=" * 85)
    print(f"{'Batch Size':<12} | {'Seq Len':<10} | {'Status':<8} | {'Peak Mem (MB)':<15} | {'Mem/Token (MB)':<16} | {'Gen Time (s)':<12}")
    print("-" * 85)

    # 4 & 5. Sweep across batch sizes and sequence lengths
    for bs in batch_sizes:
        for sl in seq_lengths:
            prompt_text = prompt_cache[sl]
            batch_prompts = [prompt_text] * bs

            # 6. Reset peak stats for isolated measurement per combination
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            batch_inputs = tokenizer(
                batch_prompts,
                padding=True,
                padding_side="left",
                return_tensors="pt",
            ).to(device)

            status = "ok"
            peak_memory_mb = 0.0
            memory_per_token_mb = 0.0
            gen_time_s = 0.0

            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                with torch.no_grad():
                    _ = model.generate(
                        input_ids=batch_inputs.input_ids,
                        attention_mask=batch_inputs.attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                gen_time_s = t1 - t0

                if torch.cuda.is_available():
                    peak_bytes = torch.cuda.max_memory_allocated()
                    peak_memory_mb = peak_bytes / (1024.0 * 1024.0)
                else:
                    peak_memory_mb = 0.0

                total_tokens = bs * (sl + max_new_tokens)
                memory_per_token_mb = peak_memory_mb / total_tokens if total_tokens > 0 else 0.0

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                status = "oom"
                peak_memory_mb = 0.0
                memory_per_token_mb = 0.0
                gen_time_s = 0.0
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"  [OOM Warning] Caught out-of-memory at batch_size={bs}, seq_length={sl}")

            row = {
                "batch_size": bs,
                "seq_length": sl,
                "max_new_tokens": max_new_tokens,
                "peak_memory_mb": round(peak_memory_mb, 2),
                "memory_per_token_mb": round(memory_per_token_mb, 6),
                "generation_time_s": round(gen_time_s, 4),
                "status": status,
            }
            results.append(row)

            print(
                f"{bs:<12} | {sl:<10} | {status:<8} | "
                f"{peak_memory_mb:<15.2f} | {memory_per_token_mb:<16.6f} | {gen_time_s:<12.4f}"
            )

    # 7. Write results to CSV
    os.makedirs(os.path.dirname(os.path.abspath(results_csv_path)), exist_ok=True)
    with open(results_csv_path, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "batch_size",
            "seq_length",
            "max_new_tokens",
            "peak_memory_mb",
            "memory_per_token_mb",
            "generation_time_s",
            "status",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("-" * 85)
    print(f"Results successfully saved to: {results_csv_path}")

    # 8. Compute projection for maximum batch size using linear regression
    successful_runs = [r for r in results if r["status"] == "ok"]
    if successful_runs:
        total_tokens_arr = np.array(
            [r["batch_size"] * (r["seq_length"] + r["max_new_tokens"]) for r in successful_runs],
            dtype=float,
        )
        peak_mem_arr = np.array([r["peak_memory_mb"] for r in successful_runs], dtype=float)

        # Fit linear regression: peak_memory_mb = fixed_overhead_mb + marginal_mb_per_token * total_tokens
        marginal_mb_per_token, fixed_overhead_mb = np.polyfit(total_tokens_arr, peak_mem_arr, 1)

        # Calculate R² (coefficient of determination)
        y_pred = marginal_mb_per_token * total_tokens_arr + fixed_overhead_mb
        ss_res = np.sum((peak_mem_arr - y_pred) ** 2)
        ss_tot = np.sum((peak_mem_arr - np.mean(peak_mem_arr)) ** 2)
        r_squared = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 1.0

        target_seq_len = 256
        tokens_per_seq = target_seq_len + max_new_tokens
        usable_vram_mb = vram_budget_mb - safety_margin_mb
        total_tokens_max = (
            (usable_vram_mb - fixed_overhead_mb) / marginal_mb_per_token
            if marginal_mb_per_token > 0
            else 0.0
        )
        est_max_batch_size = (
            int(total_tokens_max / tokens_per_seq) if tokens_per_seq > 0 else 0
        )

        print("\n" + "=" * 85)
        print("VRAM CAPACITY & MAX BATCH SIZE PROJECTION (LINEAR REGRESSION)")
        print("=" * 85)
        print(f"Fitted Fixed Overhead    : {fixed_overhead_mb:.2f} MB (model weights + CUDA context)")
        print(f"Fitted Marginal Cost     : {marginal_mb_per_token:.6f} MB/token (KV-cache + activations)")
        print(f"Goodness of Fit (R^2)    : {r_squared:.4f}")
        if r_squared < 0.90:
            print("  [Warning] Low R^2: Memory scaling may not be well-approximated by a linear model.")
        print(f"Total VRAM Ceiling       : {vram_budget_mb:.0f} MB")
        print(f"Safety Margin            : {safety_margin_mb:.0f} MB (overhead & fragmentation)")
        print(f"Usable VRAM Budget       : {usable_vram_mb:.0f} MB")
        print(f"Target Sequence Length   : {target_seq_len} prompt + {max_new_tokens} generated = {tokens_per_seq} tokens/seq")
        print(f"Max Total Tokens Budget  : {total_tokens_max:.1f} tokens")
        print(f"Estimated Max Batch Size : {est_max_batch_size} sequences")
        print("=" * 85)

    # 9. Matplotlib Plot
    plot_results(results, seq_lengths, plot_path)


def plot_results(results: List[Dict], seq_lengths: List[int], plot_path: str):
    """Generate and save peak memory vs batch size plot."""
    os.makedirs(os.path.dirname(os.path.abspath(plot_path)), exist_ok=True)
    plt.figure(figsize=(9, 6))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    markers = ["o", "s", "^", "D"]

    for idx, sl in enumerate(seq_lengths):
        matching = [r for r in results if r["seq_length"] == sl and r["status"] == "ok"]
        if not matching:
            continue
        bs_vals = [r["batch_size"] for r in matching]
        mem_vals = [r["peak_memory_mb"] for r in matching]

        plt.plot(
            bs_vals,
            mem_vals,
            label=f"Prompt Seq Len = {sl} tokens",
            color=colors[idx % len(colors)],
            marker=markers[idx % len(markers)],
            linewidth=2,
            markersize=7,
        )

    plt.axhline(
        y=4096,
        color="red",
        linestyle="--",
        alpha=0.7,
        label="4GB VRAM Hardware Limit (4096 MB)",
    )
    plt.axhline(
        y=4096 - 500,
        color="orange",
        linestyle=":",
        alpha=0.8,
        label="Usable Budget (3596 MB, 500MB Safety Margin)",
    )

    plt.title("LLM Inference Peak Memory vs Batch Size (GTX 1650, 4GB VRAM)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Batch Size", fontsize=11, labelpad=8)
    plt.ylabel("Peak GPU Memory Allocated (MB)", fontsize=11, labelpad=8)
    plt.xticks([1, 2, 4, 8])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", edgecolor="none", shadow=True, fontsize=10)
    plt.tight_layout()

    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Memory profiling chart saved to: {plot_path}")


if __name__ == "__main__":
    run_memory_profiling()
