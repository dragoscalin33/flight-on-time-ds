import time
import json
from typing import Any

from app.chat.llm_client import create_llm_client, LLMClient
from app.chat.schemas import ChatResponse, SourceReference, ToolCallInfo
from app.rag.retrieval import get_relevant_context
from app.tools import flight_predictor, airport_lookup, weather_checker
from app.config import settings

TOOL_REGISTRY: dict[str, Any] = {
    "predict_flight_delay": flight_predictor,
    "lookup_airport": airport_lookup,
    "check_weather": weather_checker,
}

TOOLS = [
    flight_predictor.TOOL_DEFINITION,
    airport_lookup.TOOL_DEFINITION,
    weather_checker.TOOL_DEFINITION,
]

SYSTEM_PROMPT = """You are the FlightOnTime AI Assistant — an intelligent flight delay prediction assistant for Brazilian domestic flights.

You help passengers understand their flight delay risk by combining:
1. A CatBoost ML model trained on 2.2M Brazilian flight records
2. Real-time weather data from OpenMeteo
3. Expert knowledge about Brazilian airports and delay patterns

## Your Capabilities
- **predict_flight_delay**: Call the ML model to predict delay probability for a specific flight
- **lookup_airport**: Search for airport information (IATA codes, names)
- **check_weather**: Get current/forecast weather at airport locations

## How to Respond
- Always use the predict_flight_delay tool when a user asks about a specific flight
- Provide context about WHY a flight might be delayed (weather, time of day, airport patterns)
- Cite your knowledge sources when giving general advice
- Be concise but informative — passengers want actionable information
- Use the risk colors: 🟢 Low Risk, 🟡 Medium Risk, 🔴 High Risk
- If the user mentions an airport by city name, resolve it to the correct airport code
- Respond in the same language the user writes in (Portuguese, English, or Spanish)
- CRITICAL: Always end with a text response to the user. Never end with just tool calls.
- CRITICAL: NEVER invent or fabricate prediction results. If a tool returns an error (e.g. "Could not connect"), you MUST tell the user the prediction service is currently offline. Do NOT make up probabilities, risk levels, or weather data. Instead, share general knowledge about the route from your context.
- If a tool returns an error, say: "The ML prediction service is currently offline, but based on historical patterns I can share some insights about your route."
- Use each tool at most ONCE per response. Do not call the same tool multiple times.

## Important Context
{rag_context}
"""

SESSION_STORE: dict[str, list[dict]] = {}
MAX_TOOL_ROUNDS = 5


async def process_message(message: str, session_id: str) -> ChatResponse:
    start_time = time.time()

    rag_results = get_relevant_context(message, n_results=3)
    rag_context = "\n\n".join(
        f"[Source: {r['source']}]\n{r['content']}" for r in rag_results
    )

    system = SYSTEM_PROMPT.format(rag_context=rag_context if rag_context else "No specific context retrieved.")

    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = []

    SESSION_STORE[session_id].append({"role": "user", "content": message})

    llm = create_llm_client()
    messages = list(SESSION_STORE[session_id])
    all_tools_used: list[ToolCallInfo] = []

    final_text = ""
    tools_with_errors = 0
    for round_num in range(MAX_TOOL_ROUNDS):
        response = await llm.chat_with_tools(
            messages=messages,
            system_prompt=system,
            tools=TOOLS if tools_with_errors < 2 else [],
        )

        text_parts = [
            block["text"]
            for block in response["content"]
            if block.get("type") == "text" and block.get("text", "").strip()
        ]

        has_tool_use = any(
            block.get("type") == "tool_use" for block in response["content"]
        )

        if not has_tool_use:
            final_text = "\n".join(text_parts)
            break

        messages.append({"role": "assistant", "content": response["content"]})

        tool_results = []
        seen_tools: set[str] = set()
        for block in response["content"]:
            if block.get("type") != "tool_use":
                continue

            tool_name = block["name"]
            tool_input = block["input"]
            tool_id = block["id"]

            if tool_name in seen_tools:
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": json.dumps({"error": "Tool already called this round, skipping duplicate"}),
                })
                continue
            seen_tools.add(tool_name)

            tool_module = TOOL_REGISTRY.get(tool_name)
            if tool_module:
                result = await tool_module.execute(**tool_input)
            else:
                result = {"error": f"Unknown tool: {tool_name}"}

            if "error" in result:
                tools_with_errors += 1

            all_tools_used.append(ToolCallInfo(
                tool_name=tool_name,
                tool_input=tool_input,
                tool_result=result,
            ))

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": json.dumps(result, ensure_ascii=False),
            })

        messages.append({"role": "user", "content": tool_results})

    if not final_text and all_tools_used:
        error_tools = [t for t in all_tools_used if "error" in t.tool_result]
        if error_tools:
            final_text = (
                "I tried to use the prediction tools but the ML service is currently offline. "
                "Based on my knowledge base, here's what I can tell you:\n\n"
                + rag_context[:500] if rag_context else
                "The prediction service is not available right now. Please try again later."
            )

    sources = [
        SourceReference(
            document=r["source"],
            chunk=r["content"][:200],
            relevance_score=r["relevance_score"],
        )
        for r in rag_results
    ]

    SESSION_STORE[session_id].append({"role": "assistant", "content": final_text})

    model_name = (
        settings.anthropic_model
        if settings.llm_provider == "anthropic"
        else settings.ollama_model
    )

    latency = (time.time() - start_time) * 1000

    return ChatResponse(
        message=final_text,
        session_id=session_id,
        sources=sources,
        tools_used=all_tools_used,
        model_used=model_name,
        latency_ms=round(latency, 2),
    )


def get_session_history(session_id: str) -> list[dict]:
    return SESSION_STORE.get(session_id, [])


def clear_session(session_id: str) -> bool:
    if session_id in SESSION_STORE:
        del SESSION_STORE[session_id]
        return True
    return False
