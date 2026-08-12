# Batched Inference Benchmark

Request-level static batching with KV-cache reuse for Qwen2.5-0.5B-Instruct, benchmarked on consumer hardware (NVIDIA GTX 1650, 4GB VRAM).

**What this is:** a from-scratch batching engine (queue + dual-trigger flush + KV-cache generation) built and benchmarked to understand LLM serving system behavior under realistic load — not a production-grade serving system, and not "continuous batching" (see Limitations).

**Hardware:** NVIDIA GeForce GTX 1650, 4GB VRAM, driver 592.82, CUDA 13.1 (PyTorch built against cu126). Windows 11, Python 3.11.9.

---

## Repo structure

```
/engine        - batching queue + KV-cache handling (engine/batcher.py)
/benchmarks    - load simulation, memory profiling, plotting scripts
/tests         - correctness harness (test_correctness.py, must pass before any benchmark is trusted)
/results       - raw CSVs and generated plots from every phase
```

## Reproducing

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

pytest tests/test_correctness.py -v -s          # Phase 0
python -m benchmarks.phase1_synthetic_feeder     # Phase 1
python -m benchmarks.phase2_memory_profile       # Phase 2
python -m benchmarks.phase3_load_simulation      # Phase 3 (~30-45 min)
```

---

## Phase 0 — Correctness harness

Before any benchmark number means anything, batched generation must produce output token-for-token identical to unbatched generation (accounting for left-padding and attention masks). This is the single test that separates a working batching implementation from one that silently produces different output under batching.

- Greedy decoding (`do_sample=False`), 3 prompts of varying length (5, 16, 50 prompt tokens), exact token-ID comparison (not decoded-text comparison, which can mask token-level misalignment).
- **Result: PASS on all 3 prompts**, exact token match up to 30 generated tokens each.
- Runs in CI-style form via `pytest tests/test_correctness.py -v -s`.

## Phase 1 — Static batching engine

An asyncio request queue that flushes and runs generation when **either** `max_batch_size=4` **or** `max_wait_time_ms=200` is reached, whichever fires first. Generation runs in a worker thread (`asyncio.to_thread`) so the event loop stays responsive to new arrivals while the GPU is busy.

Verified via a synthetic feeder that deliberately isolates each trigger path:

| Trigger | Example batch | Batch size |
|---|---|---|
| size | 4 fast-arriving requests | 4 |
| timeout | 1 isolated request, 250ms idle window | 1 |
| timeout | 3 requests, 300ms gaps | 3 |

Both trigger paths fire correctly and independently.

**Known limitation:** this engine uses `model.generate()` as a single blocking call per batch. Once a batch starts decoding, no new request can join it — completion time is set by the *slowest* sequence in the batch (head-of-line blocking). This is a structural property of static batching, not a bug, and it's the main reason vLLM-style iteration-level scheduling exists (see Phase 4 discussion below).

## Phase 2 — KV-cache memory profiling

Swept batch size × prompt sequence length ([1,2,4,8] × [64,128,256] tokens, fixed `max_new_tokens=50`), measuring `torch.cuda.max_memory_allocated()` per combination with a throwaway warm-up run to absorb CUDA context/kernel-compile cost before measuring.

**Peak memory ranged 962 MB → 1090 MB** across the entire sweep — a ~13% increase for an 8x increase in batch size and 4x increase in sequence length. For a 0.5B-parameter model on this GPU, **fixed overhead (model weights + CUDA context) dominates memory usage; KV-cache growth is comparatively small** across the tested range. This GPU is nowhere near a KV-cache-driven memory ceiling at these batch sizes — a different finding than the original hypothesis that KV-cache would be the dominant constraint, and worth stating plainly rather than forcing the data to fit the expected narrative.

A naive "memory ÷ total tokens" metric varied 19x across the sweep (8.44 → 0.445 MB/token) purely as an artifact of dividing a mostly-fixed cost by a growing denominator — this is *not* a real per-token cost and was discarded in favor of linear regression:

```
peak_memory_mb ≈ fixed_overhead_mb + marginal_mb_per_token × total_tokens
```

Fitted via `numpy.polyfit`:
- **Fixed overhead: 953.18 MB**
- **Marginal cost: 0.0518 MB/token**
- **R² = 0.9473** (linear model is a reasonable fit across the tested range)

Projected max batch size at seq_len=256, 4096MB VRAM budget minus 500MB safety margin: **~166 sequences**.

**Caveat — do not take this number at face value:** it extrapolates ~20x beyond the largest tested batch size (8). The linear fit holds across [1,8]; nothing was measured near 166, and allocator fragmentation or other overheads that are negligible at small batch sizes could behave differently at scale. Treat this as an order-of-magnitude estimate, not a guarantee.

## Phase 3 — Realistic load simulation (Poisson arrivals)

A single-request throughput probe (20 requests submitted with no pacing) measured the engine's baseline processing ceiling at **~1.44 req/sec**. Poisson-process arrival rates were chosen to straddle this ceiling rather than sweep arbitrary round numbers, so the resulting curve actually shows the approach to saturation:

| Arrival rate | Repeats | Mean p50 latency | Mean p99 latency | Mean queueing delay | Measured throughput |
|---|---|---|---|---|---|
| 0.5 req/s | 5×60s | 2.223 ± 0.099 s | 3.562 ± 0.127 s | 0.595 ± 0.034 s | 0.52 ± 0.00 rps |
| 1.0 req/s | 5×60s | 3.112 ± 0.054 s | 4.672 ± 0.246 s | 1.079 ± 0.033 s | 1.04 ± 0.00 rps |
| 1.5 req/s | 3×45s | 4.546 ± 0.376 s | 7.570 ± 0.843 s | 2.399 ± 0.348 s | 1.49 ± 0.03 rps |

p50 and p99 latency both scale superlinearly as arrival rate approaches the ~1.44 req/s ceiling — textbook queueing behavior under rising utilization. Variance also widens sharply near saturation (p99 stddev: 0.127s at 0.5 req/s → 0.843s at 1.5 req/s), itself a signature of a system operating near its capacity limit.

**Queue growth under sustained overload** — an additional single-repeat probe at 2.0 req/sec (deliberately above the measured ceiling, run separately from the main sweep):

| Arrival rate | Behavior |
|---|---|
| 1.5 req/s (~104% of ceiling) | Queue oscillates 0–9, repeatedly clears to zero |
| 2.0 req/s (~139% of ceiling) | Queue climbs past 13 and stays elevated (13–17) for the remainder of the window — never clears |

This contrast is the clearest demonstration in the project of the difference between near-critical oscillation and genuine unbounded divergence. Under sustained 2.0 req/s load, sustained throughput measured ~1.80 req/s — higher than the single-shot 1.44 req/s probe, likely because a persistently non-empty queue means more batches hit the size trigger (4 requests ready) rather than the slower timeout trigger, improving effective throughput once truly under load. Both numbers are real; they measure different things (idle-start burst vs. sustained backlog) and shouldn't be conflated.

Plots: `results/phase3_latency_vs_arrival_rate.png`, `results/phase3_queue_growth.png` (1.5 req/s), `results/phase3_queue_growth_2.0rps.png` (2.0 req/s addendum).

**Design note — unbounded queue by choice:** this simulation has no admission control or request dropping; the queue grows freely under overload by design, since the growth behavior itself is the finding. A production system would need admission control (reject/shed load past some depth) — that's out of scope here but is a known, acknowledged gap.

---

## Limitations / what this is not

- **Not continuous batching.** This is static batching: once a batch starts generating, it runs to completion before any new request can join. Real continuous batching (Orca, vLLM) inserts new requests at generation-step boundaries. The head-of-line blocking this causes (flagged in Phase 1) is a direct, measurable consequence, visible in this project's own latency variance data.
- **No admission control.** The Phase 3 queue is intentionally unbounded; a production system would need request shedding under sustained overload.
- **Phase 2's max-batch-size projection (166) is an untested extrapolation**, not a measured result — see caveat above.
- **No vLLM baseline comparison yet** (planned next — the highest-leverage remaining addition, since it gives these numbers an external reference point and motivates *why* iteration-level scheduling + paged attention outperform this approach).

## vLLM comparison — attempted, not completed

A vLLM baseline comparison was planned as the highest-leverage stretch addition
(it would give this project's numbers an external reference point and motivate
*why* iteration-level scheduling and paged attention outperform static batching).

It was not completed: vLLM does not provide official Windows wheels. `pip install
vllm` on Windows falls back to building from source, which requires a CUDA
compilation toolchain (nvcc, matching MSVC build tools) not set up on this
machine, and failed during package extraction before compilation was even
attempted. Getting vLLM running would require a Linux environment (WSL2 or
Docker) — a nontrivial additional setup step, judged out of scope for this
project's time budget on Windows-only hardware.

This is a documented scope boundary, not an oversight.

## Hardware caveat

All results are specific to a 4GB-VRAM consumer GPU (GTX 1650) running a 0.5B-parameter model. Numbers here are not comparable to results on datacenter GPUs (A100, H100) or larger models without re-running — both the fixed-overhead-dominates-KV-cache finding (Phase 2) and the ~1.44 req/s throughput ceiling (Phase 3) are direct consequences of this specific hardware/model pairing, not general properties of batched LLM serving.
