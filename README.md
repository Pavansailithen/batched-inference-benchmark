# Batched Inference Benchmark

A benchmark and correctness verification suite for batched vs unbatched LLM inference.

## Directory Structure

```
batched-inference-benchmark/
├── engine/              # Core inference engines and batching queue logic
├── benchmarks/          # Benchmarking scripts and performance profiling
├── tests/               # Correctness and unit tests
│   └── test_correctness.py  # Token-level batched vs unbatched correctness test
├── results/             # Benchmark outputs, logs, and profiling results
├── requirements.txt     # Pinned Python dependencies
└── README.md            # Project documentation
```

## Setup

1. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   # On Windows:
   .\.venv\Scripts\activate
   # On Linux/macOS:
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running Correctness Tests

Verify that batched LLM generation is token-for-token identical to unbatched generation:

```bash
pytest tests/test_correctness.py -v -s
```
