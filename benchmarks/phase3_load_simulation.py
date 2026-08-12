import asyncio
import csv
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure the root project directory is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.batcher import BatchingEngine, InferenceResponse

# 12 diverse test prompts (short, medium, long)
TEST_PROMPTS = [
    # Short prompts (~5-10 tokens)
    "The capital of France is",
    "Define gravity in one sentence:",
    "What is the chemical formula for water?",
    "Translate 'hello world' into Spanish:",
    # Medium prompts (~15-30 tokens)
    "Explain the fundamental difference between a process and a thread in modern operating systems.",
    "Describe the key principles of functional programming and how they differ from object-oriented programming.",
    "What are the main advantages of using a transformer architecture over recurrent neural networks?",
    "Summarize the mechanism of action of mRNA vaccines in simple terms.",
    # Long prompts (~50-80 tokens)
    (
        "In computer science, cache invalidation is the process where entries in a cache "
        "are replaced or removed when the underlying data changes. It is widely regarded "
        "as one of the most challenging problems in distributed architecture. "
        "List three main strategies used to achieve cache coherence:"
    ),
    (
        "Distributed consensus algorithms like Raft and Paxos ensure fault tolerance across "
        "unreliable networks. Contrast the leader election phase in Raft with the synod protocol "
        "in Paxos, focusing on quorum requirements and term numbers:"
    ),
    (
        "Quantum computing leverages superposition and entanglement to solve certain computational "
        "problems asymptotically faster than classical computers. Explain Shor's algorithm for "
        "integer factorization and its implications on RSA cryptography:"
    ),
    (
        "Zero-knowledge proofs allow one party to prove to another that a statement is true without "
        "revealing any information beyond the statement's validity. Briefly discuss zk-SNARKs and zk-STARKs:"
    ),
]


async def run_single_repeat(
    arrival_rate: float,
    repeat_num: int,
    total_repeats: int,
    model,
    tokenizer,
    duration_sec: float = 60.0,
    seed: int = 42,
) -> Tuple[List[InferenceResponse], int, List[Tuple[float, int]], float]:
    """Runs a single Poisson load simulation repeat and drains pending requests.

    Returns:
        responses: All completed InferenceResponse objects for this repeat.
        pending_at_end: Number of requests in queue / in-flight at the end of the arrival window.
        queue_samples: List of (time_s, queue_depth) recorded every 1 second.
        repeat_wall_time: Total elapsed time from first arrival to final request completion.
    """
    # Seed generator for reproducible Poisson draws per repeat
    rng = np.random.default_rng(seed=seed)

    # Initialize isolated BatchingEngine for this repeat (reusing shared model weights)
    # Temporary log path per repeat so we can collect all responses cleanly
    repeat_log_path = f"results/temp_phase3_rate{arrival_rate:.1f}_rep{repeat_num}.csv"
    engine = BatchingEngine(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        max_batch_size=4,
        max_wait_time_ms=200.0,
        default_max_new_tokens=30,
        log_path=repeat_log_path,
        model=model,
        tokenizer=tokenizer,
    )

    futures: List[asyncio.Future] = []
    queue_samples: List[Tuple[float, int]] = []
    stop_monitor_event = asyncio.Event()

    async def _queue_monitor(t_start: float):
        """Polls engine._queue once per second and logs periodic progress."""
        last_log_time = 0.0
        while not stop_monitor_event.is_set():
            t_now = time.perf_counter() - t_start
            q_len = len(engine._queue)
            queue_samples.append((t_now, q_len))

            if t_now - last_log_time >= 5.0 or last_log_time == 0.0:
                completed_count = sum(1 for f in futures if f.done())
                print(
                    f"  [{arrival_rate:>4.1f} req/s | Rep {repeat_num}/{total_repeats} | "
                    f"{t_now:4.1f}s/{duration_sec:.0f}s] "
                    f"Enqueued: {len(futures):<4} | Completed: {completed_count:<4} | Queue Depth: {q_len:<3}"
                )
                last_log_time = t_now

            try:
                await asyncio.wait_for(stop_monitor_event.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass

    print(f"\n>>> Starting Rate={arrival_rate:.1f} req/s, Repeat {repeat_num}/{total_repeats} (Seed={seed})")

    async with engine:
        t_start = time.perf_counter()
        monitor_task = asyncio.create_task(_queue_monitor(t_start))

        prompt_idx = 0
        req_count = 0

        # Continuous Poisson arrival loop for duration_sec
        while True:
            inter_arrival = rng.exponential(1.0 / arrival_rate)
            await asyncio.sleep(inter_arrival)
            elapsed = time.perf_counter() - t_start
            if elapsed >= duration_sec:
                break

            prompt = TEST_PROMPTS[prompt_idx % len(TEST_PROMPTS)]
            prompt_idx += 1
            req_count += 1
            req_id = f"r{arrival_rate:.1f}-rep{repeat_num}-{req_count}"

            fut = await engine.add_request(prompt=prompt, request_id=req_id)
            futures.append(fut)

        # Arrival window completed: measure pending queue depth
        pending_at_end = sum(1 for f in futures if not f.done())
        print(
            f"  [{duration_sec:.0f}s Window Reached] Total Enqueued: {len(futures)} | "
            f"Pending at {duration_sec:.0f}s Mark: {pending_at_end} requests"
        )

        # Drain phase: await all pending requests to complete before exiting
        # Note: _queue_monitor stays active throughout drain so terminal shows live progress
        if pending_at_end > 0:
            print(f"  [Drain] Draining {pending_at_end} pending requests to avoid run contamination...")
            drain_t0 = time.perf_counter()
            responses: List[InferenceResponse] = await asyncio.gather(*futures)
            drain_elapsed = time.perf_counter() - drain_t0
            print(f"  [Drain Complete] Drained {pending_at_end} requests in {drain_elapsed:.2f}s")
        else:
            responses: List[InferenceResponse] = await asyncio.gather(*futures)

        # Stop monitor only AFTER all pending requests have drained
        stop_monitor_event.set()
        await monitor_task

        t_end = time.perf_counter()
        repeat_wall_time = t_end - t_start

    # Clean up temporary repeat log
    if os.path.exists(repeat_log_path):
        try:
            os.remove(repeat_log_path)
        except OSError:
            pass

    return responses, pending_at_end, queue_samples, repeat_wall_time


async def run_phase3_simulation(
    rate_config: Optional[Dict[float, Tuple[int, float]]] = None,
    seed: int = 42,
    raw_csv_path: str = "results/phase3_raw_requests.csv",
    summary_csv_path: str = "results/phase3_summary_stats.csv",
    plot_latency_path: str = "results/phase3_latency_vs_arrival_rate.png",
    plot_queue_path: str = "results/phase3_queue_growth.png",
):
    if rate_config is None:
        rate_config = {0.5: (5, 60.0), 1.0: (5, 60.0), 1.5: (3, 45.0)}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    overload_rate = max(rate_config.keys()) if rate_config else 1.5

    print("=" * 85)
    print("PHASE 3: REALISTIC LOAD SIMULATION (POISSON-PROCESS ARRIVALS)")
    print("=" * 85)
    print(f"Device               : {device} ({device_name})")
    print(f"Model                : {model_name} (torch.float16)")
    print(f"Rate Configurations  : {rate_config} (rate -> (repeats, duration_s))")
    print(f"Overload Rate Target : {overload_rate:.1f} req/s (queue growth plot)")
    print(f"Poisson Seed         : {seed} (reproducible inter-arrival draws)")
    print(f"Engine Defaults      : max_batch_size=4, max_wait_time_ms=200ms, max_new_tokens=30")
    print(f"Queue Mode           : UNBOUNDED (no admission control or request dropping)")
    print("-" * 85)

    # 1. Load model and tokenizer once
    print("\n[Init] Preloading model and tokenizer onto device...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    # Throwaway warm-up generation
    print("[Init] Running throwaway warm-up generation (batch=1, max_new_tokens=10)...")
    warmup_inputs = tokenizer("Warmup prompt for CUDA compilation.", return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model.generate(
            input_ids=warmup_inputs.input_ids,
            attention_mask=warmup_inputs.attention_mask,
            max_new_tokens=10,
            do_sample=False,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    all_raw_rows = []
    summary_stats = []
    queue_growth_by_repeat: Dict[int, List[Tuple[float, int]]] = {}

    # Initialize raw CSV header
    os.makedirs(os.path.dirname(os.path.abspath(raw_csv_path)), exist_ok=True)
    with open(raw_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "arrival_rate",
            "repeat_number",
            "request_id",
            "queueing_delay",
            "generation_time",
            "total_latency",
            "batch_size",
            "trigger_reason",
        ])

    # Initialize summary CSV header
    summary_fieldnames = [
        "arrival_rate",
        "mean_p50_latency",
        "stddev_p50_latency",
        "mean_p99_latency",
        "stddev_p99_latency",
        "mean_queueing_delay",
        "stddev_queueing_delay",
        "mean_measured_throughput_rps",
        "stddev_measured_throughput_rps",
        "mean_pending_at_60s",
        "stddev_pending_at_60s",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(summary_csv_path)), exist_ok=True)
    with open(summary_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()

    # 2. Iterate across arrival rates
    for rate, (repeats_per_rate, duration_sec) in rate_config.items():
        print("\n" + "=" * 85)
        print(f"EVALUATING ARRIVAL RATE: {rate:.1f} req/s ({repeats_per_rate} repeats x {duration_sec:.0f}s)")
        print("=" * 85)

        repeat_p50s = []
        repeat_p99s = []
        repeat_queueing_delays = []
        repeat_total_latencies = []
        repeat_throughputs = []
        repeat_pendings = []

        for rep in range(1, repeats_per_rate + 1):
            responses, pending_at_end, queue_samples, wall_time = await run_single_repeat(
                arrival_rate=rate,
                repeat_num=rep,
                total_repeats=repeats_per_rate,
                model=model,
                tokenizer=tokenizer,
                duration_sec=duration_sec,
                seed=seed,
            )

            # Store queue growth samples dynamically for the highest/overload rate
            if rate == overload_rate:
                queue_growth_by_repeat[rep] = queue_samples

            # Record raw requests
            rep_raw_rows = []
            for r in responses:
                rep_raw_rows.append([
                    rate,
                    rep,
                    r.request_id,
                    f"{r.queueing_delay:.6f}",
                    f"{r.generation_time:.6f}",
                    f"{r.total_latency:.6f}",
                    r.batch_size,
                    r.trigger_reason,
                ])
            all_raw_rows.extend(rep_raw_rows)

            # Append to raw CSV incrementally
            with open(raw_csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(rep_raw_rows)

            # Compute per-repeat metrics
            latencies = [r.total_latency for r in responses]
            queueing_delays = [r.queueing_delay for r in responses]

            p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
            p99 = float(np.percentile(latencies, 99)) if latencies else 0.0
            mean_q_delay = float(np.mean(queueing_delays)) if queueing_delays else 0.0
            mean_tot_lat = float(np.mean(latencies)) if latencies else 0.0
            measured_tput = len(responses) / wall_time if wall_time > 0 else 0.0

            repeat_p50s.append(p50)
            repeat_p99s.append(p99)
            repeat_queueing_delays.append(mean_q_delay)
            repeat_total_latencies.append(mean_tot_lat)
            repeat_throughputs.append(measured_tput)
            repeat_pendings.append(pending_at_end)

            print(
                f"  -> Rep {rep} Summary: p50={p50:.3f}s | p99={p99:.3f}s | "
                f"Mean Q Delay={mean_q_delay:.3f}s | Throughput={measured_tput:.2f} rps | "
                f"Pending@{duration_sec:.0f}s={pending_at_end}"
            )

        # Compute aggregate mean +/- stddev across repeats
        mean_p50 = float(np.mean(repeat_p50s))
        std_p50 = float(np.std(repeat_p50s, ddof=1)) if len(repeat_p50s) > 1 else 0.0

        mean_p99 = float(np.mean(repeat_p99s))
        std_p99 = float(np.std(repeat_p99s, ddof=1)) if len(repeat_p99s) > 1 else 0.0

        mean_q = float(np.mean(repeat_queueing_delays))
        std_q = float(np.std(repeat_queueing_delays, ddof=1)) if len(repeat_queueing_delays) > 1 else 0.0

        mean_tput = float(np.mean(repeat_throughputs))
        std_tput = float(np.std(repeat_throughputs, ddof=1)) if len(repeat_throughputs) > 1 else 0.0

        mean_pending = float(np.mean(repeat_pendings))
        std_pending = float(np.std(repeat_pendings, ddof=1)) if len(repeat_pendings) > 1 else 0.0

        summary_row = {
            "arrival_rate": rate,
            "mean_p50_latency": round(mean_p50, 4),
            "stddev_p50_latency": round(std_p50, 4),
            "mean_p99_latency": round(mean_p99, 4),
            "stddev_p99_latency": round(std_p99, 4),
            "mean_queueing_delay": round(mean_q, 4),
            "stddev_queueing_delay": round(std_q, 4),
            "mean_measured_throughput_rps": round(mean_tput, 4),
            "stddev_measured_throughput_rps": round(std_tput, 4),
            "mean_pending_at_60s": round(mean_pending, 2),
            "stddev_pending_at_60s": round(std_pending, 2),
        }
        summary_stats.append(summary_row)

        # Write summary stats incrementally after each arrival rate completes
        with open(summary_csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
            writer.writerow(summary_row)

        print("\n" + "-" * 85)
        print(f"AGGREGATE METRICS FOR {rate:.1f} req/s (N={repeats_per_rate} repeats, Mean +/- Stddev):")
        print(f"  p50 Latency               : {mean_p50:.3f} +/- {std_p50:.3f} s")
        print(f"  p99 Latency               : {mean_p99:.3f} +/- {std_p99:.3f} s")
        print(f"  Mean Queueing Delay       : {mean_q:.3f} +/- {std_q:.3f} s")
        print(f"  Measured Throughput       : {mean_tput:.2f} +/- {std_tput:.2f} req/s")
        print(f"  Pending Requests at {duration_sec:.0f}s   : {mean_pending:.1f} +/- {std_pending:.1f} requests")
        print("-" * 85)

    print("\n" + "=" * 85)
    print("PHASE 3 BENCHMARK COMPLETE")
    print("=" * 85)
    print(f"Raw requests log saved to   : {raw_csv_path}")
    print(f"Summary stats saved to      : {summary_csv_path}")

    # 4. Generate Matplotlib plots
    plot_latency_vs_arrival_rate(summary_stats, plot_latency_path)
    plot_queue_growth(queue_growth_by_repeat, overload_rate, plot_queue_path)


def plot_latency_vs_arrival_rate(summary_stats: List[Dict], plot_path: str):
    """Plots p50 and p99 latency with error bars (stddev) vs arrival rate."""
    os.makedirs(os.path.dirname(os.path.abspath(plot_path)), exist_ok=True)

    rates = [s["arrival_rate"] for s in summary_stats]
    p50_means = [s["mean_p50_latency"] for s in summary_stats]
    p50_stds = [s["stddev_p50_latency"] for s in summary_stats]
    p99_means = [s["mean_p99_latency"] for s in summary_stats]
    p99_stds = [s["stddev_p99_latency"] for s in summary_stats]

    plt.figure(figsize=(9, 6))

    plt.errorbar(
        rates,
        p50_means,
        yerr=p50_stds,
        fmt="-o",
        color="#1f77b4",
        capsize=5,
        capthick=1.5,
        linewidth=2,
        markersize=8,
        label="p50 Latency (Median)",
    )

    plt.errorbar(
        rates,
        p99_means,
        yerr=p99_stds,
        fmt="-s",
        color="#d62728",
        capsize=5,
        capthick=1.5,
        linewidth=2,
        markersize=8,
        label="p99 Latency (Tail)",
    )

    plt.title("Phase 3: Latency vs Arrival Rate (GTX 1650, 4GB VRAM)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Poisson Arrival Rate (requests/sec)", fontsize=11, labelpad=8)
    plt.ylabel("Latency (seconds)", fontsize=11, labelpad=8)
    plt.xticks(rates, [f"{r:.1f}" for r in rates])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", edgecolor="none", shadow=True, fontsize=10)
    plt.tight_layout()

    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Latency plot saved to       : {plot_path}")


def plot_queue_growth(
    queue_growth_by_repeat: Dict[int, List[Tuple[float, int]]],
    overload_rate: float,
    plot_path: str,
):
    """Plots pending queue depth over simulation time for all repeats at the overload rate."""
    os.makedirs(os.path.dirname(os.path.abspath(plot_path)), exist_ok=True)

    plt.figure(figsize=(9, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    has_data = False
    max_recorded_time = 0.0
    for rep, samples in sorted(queue_growth_by_repeat.items()):
        if not samples:
            continue
        times, depths = zip(*samples)
        color = colors[(rep - 1) % len(colors)]
        plt.plot(
            times,
            depths,
            label=f"Repeat {rep}",
            color=color,
            linewidth=2,
            alpha=0.85,
        )
        if times:
            max_recorded_time = max(max_recorded_time, max(times))
        has_data = True

    plt.title(f"Phase 3: Unbounded Queue Growth at {overload_rate:.1f} req/s (GTX 1650, 4GB VRAM)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Simulation Elapsed Time (seconds)", fontsize=11, labelpad=8)
    plt.ylabel("Pending Queue Depth (requests)", fontsize=11, labelpad=8)
    if max_recorded_time > 0:
        plt.xlim(0, max_recorded_time + 0.5)
    plt.grid(True, linestyle="--", alpha=0.5)
    if has_data:
        plt.legend(frameon=True, facecolor="white", edgecolor="none", shadow=True, fontsize=10)
    plt.tight_layout()

    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Queue growth plot saved to  : {plot_path}")


if __name__ == "__main__":
    asyncio.run(run_phase3_simulation())
