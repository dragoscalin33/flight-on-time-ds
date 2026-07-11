import asyncio
import sys
import time
import uuid

from langchain_core.messages import AIMessage, HumanMessage

from app.agent_graph.graph import build_graph


def print_turn(messages: list, seen: int, latency_ms: float, rag_sources: list[dict]) -> int:
    for message in messages[seen:]:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                print(f"  [tool] {call['name']}({call['args']})")

    print(f"\n{messages[-1].content}")

    sources = sorted({s["source"] for s in rag_sources})
    if sources:
        print(f"\nSources: {', '.join(sources)}")
    print(f"Latency: {latency_ms:.0f} ms")
    return len(messages)


async def run_once(graph, question: str, thread_id: str, seen: int) -> int:
    config = {"configurable": {"thread_id": thread_id}}
    start = time.time()
    result = await graph.ainvoke({"messages": [HumanMessage(content=question)]}, config)
    latency_ms = (time.time() - start) * 1000
    return print_turn(result["messages"], seen, latency_ms, result.get("rag_sources", []))


async def main() -> None:
    graph = build_graph()
    thread_id = uuid.uuid4().hex[:8]
    seen = 0

    if len(sys.argv) > 1:
        await run_once(graph, " ".join(sys.argv[1:]), thread_id, seen)
        return

    print("FlightOnTime agent (LangGraph) — type 'exit' to quit")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            seen = await run_once(graph, question, thread_id, seen)


if __name__ == "__main__":
    asyncio.run(main())
