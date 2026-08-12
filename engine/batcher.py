import asyncio
import csv
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class InferenceResponse:
    request_id: str
    batch_id: str
    prompt: str
    generated_text: str
    generated_token_ids: List[int]
    prompt_token_count: int
    generated_token_count: int
    queueing_delay: float  # seconds
    generation_time: float  # seconds
    total_latency: float  # seconds
    batch_size: int
    batch_throughput: float  # tokens / second
    trigger_reason: str  # 'size' or 'timeout'
    arrival_timestamp: float
    completion_timestamp: float


@dataclass
class InferenceRequest:
    request_id: str
    prompt: str
    arrival_timestamp: float
    max_new_tokens: Optional[int] = None
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())


class BatchingEngine:
    """Static Batching Engine for LLM Inference.

    Manages an asyncio request queue and executes batched inference on GPU
    when either max_batch_size is reached or max_wait_time_ms has elapsed.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        max_batch_size: int = 4,
        max_wait_time_ms: float = 200.0,
        default_max_new_tokens: int = 50,
        log_path: str = "results/phase1_batching_log.csv",
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.model_name = model_name
        self.max_batch_size = max_batch_size
        self.max_wait_time_ms = max_wait_time_ms
        self.default_max_new_tokens = default_max_new_tokens
        self.log_path = log_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype = dtype

        # Tokenizer initialization with left-padding for batched generation
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Model initialization
        if model is not None:
            self.model = model
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=self.dtype if self.device != "cpu" else torch.float32,
            ).to(self.device)
        self.model.eval()

        # Queue and async coordination
        self._queue: List[InferenceRequest] = []
        self._queue_lock = asyncio.Lock()
        self._new_item_event = asyncio.Event()
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False
        self._batch_counter = 0

        # CSV Logging setup
        self._csv_lock = asyncio.Lock()
        self._init_csv_log()

    def _init_csv_log(self) -> None:
        """Initialize the CSV log file with headers if it does not already exist."""
        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
        if not os.path.exists(self.log_path) or os.path.getsize(self.log_path) == 0:
            with open(self.log_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "request_id",
                    "batch_id",
                    "prompt",
                    "generated_text",
                    "prompt_tokens",
                    "generated_tokens",
                    "queueing_delay",
                    "generation_time",
                    "total_latency",
                    "batch_size",
                    "batch_throughput",
                    "trigger_reason",
                    "arrival_timestamp",
                    "completion_timestamp",
                ])

    async def start(self) -> None:
        """Start the background batching loop worker."""
        if not self._running:
            self._running = True
            self._worker_task = asyncio.create_task(self._batching_loop())

    async def stop(self) -> None:
        """Stop the background batching loop worker and drain remaining requests."""
        if self._running:
            self._running = False
            self._new_item_event.set()
            if self._worker_task:
                await self._worker_task
                self._worker_task = None

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

    async def add_request(
        self,
        prompt: str,
        request_id: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> asyncio.Future:
        """Enqueue an inference request and return a future that resolves with InferenceResponse."""
        if request_id is None:
            request_id = str(uuid.uuid4())

        loop = asyncio.get_running_loop()
        req = InferenceRequest(
            request_id=request_id,
            prompt=prompt,
            arrival_timestamp=time.perf_counter(),
            max_new_tokens=max_new_tokens or self.default_max_new_tokens,
            future=loop.create_future(),
        )

        async with self._queue_lock:
            self._queue.append(req)
            self._new_item_event.set()

        return req.future

    async def _batching_loop(self) -> None:
        """Background loop continuously monitoring queue conditions and forming batches."""
        while self._running or len(self._queue) > 0:
            batch_to_process: List[InferenceRequest] = []
            trigger_reason: str = ""

            async with self._queue_lock:
                if not self._queue:
                    self._new_item_event.clear()
                else:
                    oldest_req = self._queue[0]
                    elapsed = time.perf_counter() - oldest_req.arrival_timestamp
                    time_remaining = (self.max_wait_time_ms / 1000.0) - elapsed

                    # Check Condition 1: max_batch_size reached
                    if len(self._queue) >= self.max_batch_size:
                        batch_size = self.max_batch_size
                        batch_to_process = self._queue[:batch_size]
                        self._queue = self._queue[batch_size:]
                        trigger_reason = "size"
                    # Check Condition 2: max_wait_time_ms elapsed for oldest request
                    elif time_remaining <= 0:
                        batch_size = min(len(self._queue), self.max_batch_size)
                        batch_to_process = self._queue[:batch_size]
                        self._queue = self._queue[batch_size:]
                        trigger_reason = "timeout"

            # If a batch was formed, process it
            if batch_to_process:
                self._batch_counter += 1
                batch_id = f"batch-{self._batch_counter}"
                await self._process_batch(batch_to_process, batch_id, trigger_reason)
                continue

            # If no batch was formed and engine is stopped with empty queue, break
            if not self._running and len(self._queue) == 0:
                break

            # If queue is empty, wait until a new item arrives
            if len(self._queue) == 0:
                try:
                    await asyncio.wait_for(self._new_item_event.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
            else:
                # Queue has items (< max_batch_size) and timeout has not expired yet
                # Wait for remaining timeout or a new arrival
                oldest_req = self._queue[0]
                elapsed = time.perf_counter() - oldest_req.arrival_timestamp
                time_remaining = max(0.0, (self.max_wait_time_ms / 1000.0) - elapsed)
                try:
                    await asyncio.wait_for(self._new_item_event.wait(), timeout=time_remaining)
                    self._new_item_event.clear()
                except asyncio.TimeoutError:
                    # Timeout reached, next loop iteration will flush batch
                    pass

    async def _process_batch(
        self,
        requests: List[InferenceRequest],
        batch_id: str,
        trigger_reason: str,
    ) -> None:
        """Tokenizes batch with left-padding, runs generation, extracts results, and logs metrics."""
        batch_formation_start = time.perf_counter()
        batch_size = len(requests)
        prompts = [req.prompt for req in requests]
        max_new_tokens = max(req.max_new_tokens for req in requests)

        # Run tokenization and GPU generation in a worker thread to keep the asyncio event loop responsive
        def _run_model_generation():
            batch_inputs = self.tokenizer(
                prompts,
                padding=True,
                padding_side="left",
                return_tensors="pt",
            ).to(self.device)

            batch_input_len = batch_inputs.input_ids.shape[1]
            gen_start_time = time.perf_counter()

            with torch.no_grad():
                batch_outputs = self.model.generate(
                    input_ids=batch_inputs.input_ids,
                    attention_mask=batch_inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            gen_end_time = time.perf_counter()
            generation_time = gen_end_time - gen_start_time

            # Reuse exact left-padding-offset extraction logic proven in Phase 0
            results = []
            total_batch_tokens = 0
            for idx in range(batch_size):
                gen_token_ids = batch_outputs[idx, batch_input_len:].tolist()
                gen_text = self.tokenizer.decode(gen_token_ids, skip_special_tokens=True)
                prompt_tokens = int(batch_inputs.attention_mask[idx].sum().item())
                gen_tokens_count = len(gen_token_ids)
                total_batch_tokens += gen_tokens_count
                results.append((gen_text, gen_token_ids, prompt_tokens, gen_tokens_count))

            return results, generation_time, total_batch_tokens

        results, generation_time, total_batch_tokens = await asyncio.to_thread(_run_model_generation)

        completion_timestamp = time.perf_counter()
        batch_throughput = total_batch_tokens / generation_time if generation_time > 0 else 0.0

        responses: List[InferenceResponse] = []
        csv_rows = []

        for idx, req in enumerate(requests):
            gen_text, gen_token_ids, prompt_tokens, gen_tokens_count = results[idx]
            queueing_delay = batch_formation_start - req.arrival_timestamp
            total_latency = completion_timestamp - req.arrival_timestamp

            response = InferenceResponse(
                request_id=req.request_id,
                batch_id=batch_id,
                prompt=req.prompt,
                generated_text=gen_text,
                generated_token_ids=gen_token_ids,
                prompt_token_count=prompt_tokens,
                generated_token_count=gen_tokens_count,
                queueing_delay=queueing_delay,
                generation_time=generation_time,
                total_latency=total_latency,
                batch_size=batch_size,
                batch_throughput=batch_throughput,
                trigger_reason=trigger_reason,
                arrival_timestamp=req.arrival_timestamp,
                completion_timestamp=completion_timestamp,
            )
            responses.append(response)

            csv_rows.append([
                response.request_id,
                response.batch_id,
                response.prompt,
                response.generated_text,
                response.prompt_token_count,
                response.generated_token_count,
                f"{response.queueing_delay:.6f}",
                f"{response.generation_time:.6f}",
                f"{response.total_latency:.6f}",
                response.batch_size,
                f"{response.batch_throughput:.2f}",
                response.trigger_reason,
                f"{response.arrival_timestamp:.6f}",
                f"{response.completion_timestamp:.6f}",
            ])

            # Resolve future
            if not req.future.done():
                req.future.set_result(response)

        # Write results incrementally to CSV
        async with self._csv_lock:
            with open(self.log_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(csv_rows)
