import asyncio
import json
import time
from pathlib import Path

from app.chat.service import process_message, clear_session


EVAL_DATASET = Path(__file__).parent / "eval_dataset.json"


async def run_evaluation() -> dict:
    with open(EVAL_DATASET) as f:
        test_cases = json.load(f)

    results: list[dict] = []
    total_latency = 0.0
    tool_accuracy_hits = 0
    context_accuracy_hits = 0

    for i, case in enumerate(test_cases):
        session_id = f"eval_{i}"
        clear_session(session_id)

        print(f"\n[{i+1}/{len(test_cases)}] {case['category']}: {case['question'][:60]}...")

        try:
            response = await process_message(
                message=case["question"],
                session_id=session_id,
            )

            tools_used_names = {t.tool_name for t in response.tools_used}
            expected_tools = set(case["expected_tools"])
            tool_match = expected_tools.issubset(tools_used_names)

            source_docs = {s.document.replace(".md", "") for s in response.sources}
            expected_context = set(case["expected_context"])
            context_match = expected_context.issubset(source_docs) if expected_context else True

            if tool_match:
                tool_accuracy_hits += 1
            if context_match:
                context_accuracy_hits += 1

            total_latency += response.latency_ms

            results.append({
                "question": case["question"],
                "category": case["category"],
                "response_length": len(response.message),
                "tools_expected": list(expected_tools),
                "tools_used": list(tools_used_names),
                "tool_match": tool_match,
                "context_match": context_match,
                "latency_ms": response.latency_ms,
                "has_response": len(response.message) > 0,
            })

            status = "PASS" if tool_match and context_match else "FAIL"
            print(f"  {status} | Tools: {tools_used_names or 'none'} | {response.latency_ms:.0f}ms")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "question": case["question"],
                "category": case["category"],
                "error": str(e),
                "tool_match": False,
                "context_match": False,
            })

        clear_session(session_id)

    total = len(test_cases)
    summary = {
        "total_cases": total,
        "tool_accuracy": round(tool_accuracy_hits / total * 100, 1),
        "context_accuracy": round(context_accuracy_hits / total * 100, 1),
        "avg_latency_ms": round(total_latency / total, 2) if total else 0,
        "pass_rate": round(
            sum(1 for r in results if r.get("tool_match") and r.get("context_match")) / total * 100, 1
        ),
        "results": results,
    }

    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total cases:      {summary['total_cases']}")
    print(f"Tool accuracy:    {summary['tool_accuracy']}%")
    print(f"Context accuracy: {summary['context_accuracy']}%")
    print(f"Pass rate:        {summary['pass_rate']}%")
    print(f"Avg latency:      {summary['avg_latency_ms']}ms")

    output_path = Path(__file__).parent / "eval_results.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

    return summary


if __name__ == "__main__":
    asyncio.run(run_evaluation())
