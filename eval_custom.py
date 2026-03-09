#!/usr/bin/env python3
"""Custom eval script that records per-request timing for math_500 and iquiz."""

import asyncio
import json
import re
import sys
import time
from pathlib import Path

import aiohttp

API_BASE = "http://127.0.0.1:8080/v1"
API_KEY = "EMPTY"

MATH500_ARROW = Path.home() / ".cache/evalscope/datasets/AI-ModelScope_MATH-500-26b9b771205e898900f8a5c29c52ae9f/data-00000-of-00001.arrow"
IQUIZ_IQ_ARROW = Path.home() / ".cache/evalscope/datasets/AI-ModelScope_IQuiz-0021b5c7457baf72bf85f6b53dd352e1/data-00000-of-00001.arrow"
IQUIZ_EQ_ARROW = Path.home() / ".cache/evalscope/datasets/AI-ModelScope_IQuiz-fc27b433c94891821617f3474fb4d53f/data-00000-of-00001.arrow"


def load_math500(limit=100):
    from datasets import Dataset
    ds = Dataset.from_file(str(MATH500_ARROW))
    items = []
    for i, row in enumerate(ds):
        if i >= limit:
            break
        prompt = (
            f"{row['problem']}\n"
            "Please reason step by step, and put your final answer within \\boxed{}."
        )
        items.append({
            "index": i,
            "dataset": "math_500",
            "prompt": prompt,
            "answer": row["answer"],
            "subject": row.get("subject", ""),
            "level": row.get("level", 0),
        })
    return items


def load_iquiz(limit=100):
    from datasets import Dataset
    ds_iq = Dataset.from_file(str(IQUIZ_IQ_ARROW))
    ds_eq = Dataset.from_file(str(IQUIZ_EQ_ARROW))
    items = []
    idx = 0
    for ds_name, ds in [("IQ", ds_iq), ("EQ", ds_eq)]:
        for row in ds:
            if idx >= limit:
                break
            choices_text = "\n".join(
                f"{chr(65+j)}) {c}" for j, c in enumerate(row["choices"])
            )
            prompt = (
                "回答下面的单项选择题，请选出其中的正确答案。"
                "你的回答的最后一行应该是这样的格式：\"答案：LETTER\"（不带引号），"
                "其中 LETTER 是 A,B,C,D 中的一个。请在回答前进行一步步思考。\n\n"
                f"问题：{row['question']}\n选项：\n{choices_text}\n"
            )
            items.append({
                "index": idx,
                "dataset": "iquiz",
                "subset": ds_name,
                "prompt": prompt,
                "answer": row["answer"],
                "level": row.get("level", 0),
            })
            idx += 1
    return items[:limit]


def extract_boxed(text):
    """Extract content from \\boxed{...}."""
    # Find the last \boxed{...}
    matches = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text)
    if matches:
        return matches[-1].strip()
    return None


def check_math_correct(model_output, expected_answer):
    extracted = extract_boxed(model_output)
    if extracted is None:
        return False
    # Normalize both for comparison
    def normalize(s):
        s = s.strip()
        s = s.replace(" ", "")
        s = s.replace("\\,", "")
        s = s.replace("\\;", "")
        s = s.replace("\\!", "")
        s = s.replace("\\text{", "").replace("}", "")
        s = s.replace("\\mathrm{", "")
        s = s.replace("\\left", "").replace("\\right", "")
        s = s.replace("\\dfrac", "\\frac")
        return s.lower()
    return normalize(extracted) == normalize(expected_answer)


def check_iquiz_correct(model_output, expected_answer):
    """Extract answer letter from model output."""
    # Look for 答案：X pattern
    matches = re.findall(r'答案[：:]\s*([A-D])', model_output)
    if matches:
        return matches[-1].upper() == expected_answer.upper()
    # Fallback: look for standalone letter at end
    matches = re.findall(r'\b([A-D])\b', model_output)
    if matches:
        return matches[-1].upper() == expected_answer.upper()
    return False


async def send_request(session, model_name, item, semaphore):
    """Send a single chat completion request and measure time."""
    async with semaphore:
        messages = [{"role": "user", "content": item["prompt"]}]
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 4096,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        }

        start_time = time.time()
        try:
            async with session.post(
                f"{API_BASE}/chat/completions",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                result = await resp.json()
                elapsed = time.time() - start_time

                if "error" in result:
                    print(f"  Error on item {item['index']}: {result['error']}")
                    return None

                content = result["choices"][0]["message"]["content"]
                usage = result.get("usage", {})

                if item["dataset"] == "math_500":
                    correct = check_math_correct(content, item["answer"])
                else:
                    correct = check_iquiz_correct(content, item["answer"])

                return {
                    "index": item["index"],
                    "dataset": item["dataset"],
                    "subset": item.get("subset", ""),
                    "prompt": item["prompt"],
                    "answer": item["answer"],
                    "response": content,
                    "correct": correct,
                    "elapsed_seconds": round(elapsed, 3),
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  Exception on item {item['index']}: {e}")
            return None


async def warmup(model_name, n=10):
    """Send warmup requests."""
    print(f"Warming up with {n} requests...")
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(n):
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": f"Say hello {i}"}],
                "max_tokens": 10,
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            }
            tasks.append(
                session.post(
                    f"{API_BASE}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=60),
                )
            )
        responses = await asyncio.gather(*[t.__aenter__() for t in tasks], return_exceptions=True)
        # Read all responses
        for r in responses:
            if hasattr(r, 'read'):
                await r.read()
    print("Warmup done.")


async def warmup_sequential(model_name, n=10):
    """Send warmup requests sequentially."""
    print(f"Warming up with {n} sequential requests...")
    async with aiohttp.ClientSession() as session:
        for i in range(n):
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": f"Say hello {i}"}],
                "max_tokens": 10,
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            }
            try:
                async with session.post(
                    f"{API_BASE}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    await resp.json()
                    print(f"  Warmup {i+1}/{n} done")
            except Exception as e:
                print(f"  Warmup {i+1}/{n} failed: {e}")
    print("Warmup done.")


async def run_eval(model_name, items, batch_size=100):
    """Run evaluation on all items. batch_size=1 for sequential."""
    semaphore = asyncio.Semaphore(batch_size)
    results = []

    async with aiohttp.ClientSession() as session:
        if batch_size == 1:
            # Sequential mode: send one at a time
            for i, item in enumerate(items):
                result = await send_request(session, model_name, item, semaphore)
                if result:
                    results.append(result)
                    status = "✓" if result["correct"] else "✗"
                    print(f"  [{i+1}/{len(items)}] {result['dataset']}#{result['index']} {status} ({result['elapsed_seconds']}s, {result['output_tokens']} tokens)")
                else:
                    print(f"  [{i+1}/{len(items)}] FAILED")
        else:
            tasks = [send_request(session, model_name, item, semaphore) for item in items]
            total = len(tasks)
            done = 0
            for coro in asyncio.as_completed(tasks):
                result = await coro
                done += 1
                if result:
                    results.append(result)
                    status = "✓" if result["correct"] else "✗"
                    print(f"  [{done}/{total}] {result['dataset']}#{result['index']} {status} ({result['elapsed_seconds']}s, {result['output_tokens']} tokens)")
                else:
                    print(f"  [{done}/{total}] FAILED")

    results.sort(key=lambda x: x["index"])
    return results


async def main():
    if len(sys.argv) < 3:
        print("Usage: python eval_custom.py <model_name> <output_file>")
        sys.exit(1)

    model_name = sys.argv[1]
    output_file = sys.argv[2]
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    print(f"Model: {model_name}")
    print(f"Output: {output_file}")

    # Warmup
    await warmup_sequential(model_name, n=10)

    # Load datasets
    print("Loading math_500 (100 questions)...")
    math_items = load_math500(limit=100)
    print(f"  Loaded {len(math_items)} math_500 questions")

    print("Loading iquiz (100 questions)...")
    iquiz_items = load_iquiz(limit=100)
    print(f"  Loaded {len(iquiz_items)} iquiz questions")

    # Run math_500 eval
    print(f"\n--- Evaluating math_500 ({len(math_items)} questions) ---")
    t0 = time.time()
    math_results = await run_eval(model_name, math_items, batch_size=batch_size)
    math_time = time.time() - t0
    math_correct = sum(1 for r in math_results if r["correct"])
    math_total_output_tokens = sum(r["output_tokens"] for r in math_results)
    print(f"math_500: {math_correct}/{len(math_results)} correct ({math_correct/len(math_results)*100:.1f}%)")
    print(f"  Total time: {math_time:.1f}s, Total output tokens: {math_total_output_tokens}")
    print(f"  Avg tokens/s: {math_total_output_tokens/math_time:.1f}")

    # Run iquiz eval
    print(f"\n--- Evaluating iquiz ({len(iquiz_items)} questions) ---")
    t0 = time.time()
    iquiz_results = await run_eval(model_name, iquiz_items, batch_size=batch_size)
    iquiz_time = time.time() - t0
    iquiz_correct = sum(1 for r in iquiz_results if r["correct"])
    iquiz_total_output_tokens = sum(r["output_tokens"] for r in iquiz_results)
    print(f"iquiz: {iquiz_correct}/{len(iquiz_results)} correct ({iquiz_correct/len(iquiz_results)*100:.1f}%)")
    print(f"  Total time: {iquiz_time:.1f}s, Total output tokens: {iquiz_total_output_tokens}")
    print(f"  Avg tokens/s: {iquiz_total_output_tokens/iquiz_time:.1f}")

    # Save results
    output = {
        "model": model_name,
        "math_500": {
            "results": math_results,
            "accuracy": math_correct / len(math_results) if math_results else 0,
            "total_time": round(math_time, 2),
            "total_output_tokens": math_total_output_tokens,
            "avg_tokens_per_second": round(math_total_output_tokens / math_time, 1) if math_time > 0 else 0,
        },
        "iquiz": {
            "results": iquiz_results,
            "accuracy": iquiz_correct / len(iquiz_results) if iquiz_results else 0,
            "total_time": round(iquiz_time, 2),
            "total_output_tokens": iquiz_total_output_tokens,
            "avg_tokens_per_second": round(iquiz_total_output_tokens / iquiz_time, 1) if iquiz_time > 0 else 0,
        },
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
