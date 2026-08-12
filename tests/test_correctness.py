import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_batched_matches_unbatched():
    """Verify that batched LLM generation produces identical token IDs to unbatched generation."""
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    assert torch.cuda.is_available(), "CUDA is required for this correctness test"
    device = "cuda"

    # 1. Load tokenizer and model using dtype=torch.float16 onto cuda
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
    ).to(device)
    model.eval()

    # 4. Define a fixed set of test prompts with short, medium, and long lengths
    test_prompts = [
        # Short prompt (~5 tokens)
        "The capital of France is",
        # Medium prompt (~20 tokens)
        "Explain the fundamental difference between a process and a thread in modern operating systems.",
        # Long prompt (~60 tokens)
        (
            "In computer science, cache invalidation is the process where entries in a cache "
            "are replaced or removed when the underlying data changes. It is widely regarded "
            "as one of the most challenging problems in distributed architecture. "
            "List three main strategies used to achieve cache coherence:"
        ),
    ]

    max_new_tokens = 30
    unbatched_generated_tokens = []
    unbatched_prompt_lengths = []

    print("\n" + "=" * 80)
    print("RUNNING UNBATCHED GENERATION (Batch Size = 1)")
    print("=" * 80)

    # 5. Unbatched path: run each prompt individually with seed 42
    for idx, prompt in enumerate(test_prompts):
        # 2. Set seed before generation
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs.input_ids.shape[1]
        unbatched_prompt_lengths.append(input_len)

        with torch.no_grad():
            # 3. Greedy decoding only: do_sample=False
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # Extract newly generated token IDs (strip input prompt)
        gen_tokens = outputs[0, input_len:].tolist()
        unbatched_generated_tokens.append(gen_tokens)
        print(f"Prompt {idx} (input_len={input_len}): generated {len(gen_tokens)} tokens")

    print("\n" + "=" * 80)
    print("RUNNING BATCHED GENERATION (Batch Size = 3, Left-Padded)")
    print("=" * 80)

    # 6. Batched path: tokenize all prompts together with left padding
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    batch_inputs = tokenizer(
        test_prompts,
        padding=True,
        padding_side="left",
        return_tensors="pt",
    ).to(device)

    batch_input_len = batch_inputs.input_ids.shape[1]

    with torch.no_grad():
        # Pass attention_mask explicitly to generate()
        batch_outputs = model.generate(
            input_ids=batch_inputs.input_ids,
            attention_mask=batch_inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    batched_generated_tokens = []
    # Extract each sequence's generated tokens stripping left-padding offset
    for idx in range(len(test_prompts)):
        # Since padding_side='left', all prompt tokens end at batch_input_len
        # and generated tokens start at index batch_input_len
        gen_tokens = batch_outputs[idx, batch_input_len:].tolist()
        batched_generated_tokens.append(gen_tokens)

    # 7. Assert exact token ID equality & 8. Print clear PASS/FAIL with token IDs on mismatch
    print("\n" + "=" * 80)
    print("CORRECTNESS VERIFICATION RESULTS")
    print("=" * 80)

    all_passed = True
    mismatches = []

    for idx, prompt in enumerate(test_prompts):
        unbatched_ids = unbatched_generated_tokens[idx]
        batched_ids = batched_generated_tokens[idx]
        prompt_len = unbatched_prompt_lengths[idx]

        is_match = unbatched_ids == batched_ids
        status_str = "PASS" if is_match else "FAIL"

        print(f"\n[Prompt {idx}] (Prompt Token Length: {prompt_len}) -> {status_str}")
        print(f"  Prompt snippet: {prompt[:60]}...")

        if is_match:
            print(f"  Exact Token Match ({len(unbatched_ids)} tokens): {unbatched_ids[:10]}...")
        else:
            all_passed = False
            mismatch_detail = (
                f"Prompt {idx} mismatch:\n"
                f"  Unbatched Token IDs ({len(unbatched_ids)}): {unbatched_ids}\n"
                f"  Batched Token IDs   ({len(batched_ids)}): {batched_ids}\n"
            )
            mismatches.append(mismatch_detail)
            print(f"  FAIL: Token mismatch detected!")
            print(f"    Expected (Unbatched): {unbatched_ids}")
            print(f"    Actual   (Batched):   {batched_ids}")

    print("\n" + "=" * 80)

    assert all_passed, (
        f"Batched vs Unbatched token mismatch occurred:\n" + "\n".join(mismatches)
    )


if __name__ == "__main__":
    test_batched_matches_unbatched()
