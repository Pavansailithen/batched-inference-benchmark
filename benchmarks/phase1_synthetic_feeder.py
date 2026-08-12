import asyncio
import os
import sys
import time
from typing import List

# Ensure the root project directory is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.batcher import BatchingEngine, InferenceResponse


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


async def run_feeder():
    print("=" * 80)
    print("PHASE 1: SYNTHETIC FEEDER BENCHMARK")
    print("=" * 80)
    print(f"Total test prompts: {len(TEST_PROMPTS)}")
    print("Engine configuration: max_batch_size=4, max_wait_time_ms=200ms, max_new_tokens=30")
    print("-" * 80)

    engine = BatchingEngine(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        max_batch_size=4,
        max_wait_time_ms=200.0,
        default_max_new_tokens=30,
        log_path="results/phase1_batching_log.csv",
    )

    futures = []

    async with engine:
        # Phase A: Fast Burst (4 requests with 10ms gap) -> Should trigger batch-size condition (4 items)
        print("\n[Feeder] Sending Fast Burst 1 (4 requests, 10ms interval)...")
        burst1_futs = []
        for i in range(4):
            req_id = f"req-burst1-{i+1}"
            fut = await engine.add_request(prompt=TEST_PROMPTS[i], request_id=req_id)
            burst1_futs.append(fut)
            futures.append(fut)
            await asyncio.sleep(0.01)

        # Wait for Burst 1 to finish processing so no requests are in flight
        await asyncio.gather(*burst1_futs)

        # Phase B: Isolated Slow Trickle (exactly 1 request, then wait 250ms > 200ms timeout)
        # Guarantees timeout trigger fires alone with batch_size=1
        print("\n[Feeder] Sending Isolated Slow Trickle (1 request, awaiting 250ms timeout)...")
        req_id = "req-trickle-isolated"
        fut = await engine.add_request(prompt=TEST_PROMPTS[4], request_id=req_id)
        futures.append(fut)
        await asyncio.sleep(0.25)

        # Phase C: Fast Burst 2 (4 requests with 10ms gap) -> Should trigger batch-size condition (4 items)
        print("\n[Feeder] Sending Fast Burst 2 (4 requests, 10ms interval)...")
        for i in range(5, 9):
            req_id = f"req-burst2-{i-4}"
            fut = await engine.add_request(prompt=TEST_PROMPTS[i], request_id=req_id)
            futures.append(fut)
            await asyncio.sleep(0.01)

        # Phase D: Final Slow Trickle (remaining 3 requests with 300ms gap)
        print("\n[Feeder] Sending Final Slow Trickle (3 requests, 300ms interval)...")
        for i in range(9, len(TEST_PROMPTS)):
            req_id = f"req-trickle2-{i-8}"
            fut = await engine.add_request(prompt=TEST_PROMPTS[i], request_id=req_id)
            futures.append(fut)
            await asyncio.sleep(0.30)

        # Wait for all futures to resolve
        print("\n[Feeder] Awaiting completion of all requests...")
        responses: List[InferenceResponse] = await asyncio.gather(*futures)

    # Analyze results
    print("\n" + "=" * 80)
    print("PHASE 1 EXECUTION SUMMARY")
    print("=" * 80)

    total_requests = len(responses)
    batches = {}
    for r in responses:
        if r.batch_id not in batches:
            batches[r.batch_id] = {
                "trigger_reason": r.trigger_reason,
                "batch_size": r.batch_size,
                "generation_time": r.generation_time,
                "batch_throughput": r.batch_throughput,
                "requests": [],
            }
        batches[r.batch_id]["requests"].append(r)

    total_batches = len(batches)
    size_triggered_batches = sum(1 for b in batches.values() if b["trigger_reason"] == "size")
    timeout_triggered_batches = sum(1 for b in batches.values() if b["trigger_reason"] == "timeout")

    avg_queueing_delay_ms = sum(r.queueing_delay for r in responses) / total_requests * 1000.0
    avg_generation_time_ms = sum(r.generation_time for r in responses) / total_requests * 1000.0
    avg_total_latency_ms = sum(r.total_latency for r in responses) / total_requests * 1000.0

    print(f"Total Requests Processed       : {total_requests}")
    print(f"Total Batches Formed           : {total_batches}")
    print(f"  - Size-Triggered Batches     : {size_triggered_batches}")
    print(f"  - Timeout-Triggered Batches  : {timeout_triggered_batches}")
    print(f"Average Queueing Delay         : {avg_queueing_delay_ms:.2f} ms")
    print(f"Average Generation Time        : {avg_generation_time_ms:.2f} ms")
    print(f"Average Total Latency          : {avg_total_latency_ms:.2f} ms")
    print("-" * 80)

    print("\nPer-Batch Breakdown:")
    for batch_id, info in batches.items():
        req_ids = [r.request_id for r in info["requests"]]
        print(
            f"  {batch_id:<10} | Size: {info['batch_size']} | Trigger: {info['trigger_reason']:<7} | "
            f"Gen Time: {info['generation_time']*1000.0:6.2f} ms | Throughput: {info['batch_throughput']:6.1f} tok/s | "
            f"Requests: {', '.join(req_ids)}"
        )

    print("\n" + "=" * 80)
    print(f"Log written to: {engine.log_path}")
    print("=" * 80)

    # Verification assertions
    assert size_triggered_batches >= 1, "Expected at least 1 size-triggered batch"
    assert timeout_triggered_batches >= 1, "Expected at least 1 timeout-triggered batch"
    print("\nDual-trigger condition verification: PASSED (Both size and timeout triggered)")


if __name__ == "__main__":
    asyncio.run(run_feeder())
