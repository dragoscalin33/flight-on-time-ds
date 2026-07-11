# FlightOnTime Agent — LangGraph port

The same agent as `app/chat/service.py`, rebuilt as an explicit **LangGraph state machine**. The hand-rolled version came first; this port exists to make the orchestration declarative, debuggable and resumable. Nothing in the original service was modified — both implementations share the same tools (`app/tools/`), the same RAG retrieval (`app/rag/`) and the same system prompt.

## The graph

```
START → retrieve → agent ──(tool_calls?)──> tools ──┐
                     ▲                              │
                     └──────────────────────────────┘
                     └──(no tool calls / max rounds)──> END
```

| Node | What it does |
|---|---|
| `retrieve` | Queries ChromaDB for relevant context (same `get_relevant_context` as the original), resets the tool-round counter |
| `agent` | Calls the LLM (Claude or Ollama, chosen by `LLM_PROVIDER`) with the system prompt + conversation; the model decides whether to call tools |
| `tools` | Executes the requested tools via LangGraph's `ToolNode`, appends results as `ToolMessage`s, increments the round counter |

The conditional edge `should_continue` replaces the hand-written `for round_num in range(MAX_TOOL_ROUNDS)` loop: if the last AI message contains tool calls and we're under `MAX_TOOL_ROUNDS`, go to `tools`; otherwise finish.

## Mapping: hand-rolled → LangGraph

| Hand-rolled (`app/chat/service.py`) | LangGraph (`app/agent_graph/`) |
|---|---|
| `for round_num in range(MAX_TOOL_ROUNDS)` loop | Graph edges + `should_continue` conditional; tools are only bound while `tool_rounds < MAX_TOOL_ROUNDS`, so the final LLM call is forced to answer in text and no dangling `tool_calls` are ever persisted |
| `SESSION_STORE: dict[str, list]` | `MemorySaver` checkpointer + `thread_id` |
| Manual message-format translation per provider (`llm_client.py`, 161 lines) | `ChatAnthropic` / `ChatOllama` behind one `BaseChatModel` interface |
| `TOOL_REGISTRY` + manual `tool_use`/`tool_result` block handling | `@tool` decorators + `ToolNode` |
| Error circuit breaker (2 failing tools → strip tools, force a text answer) | Same semantics via the `tool_errors` state field feeding `tools_allowed` |
| Same-round duplicate-tool dedupe | Deliberately dropped — the system prompt already forbids duplicates and `MAX_TOOL_ROUNDS` bounds the damage |
| `print` / response-object debugging | LangSmith tracing (every node, prompt, tool call and token count) |

## Run it

```bash
cd ai-assistant
source .venv/bin/activate

python -m app.agent_graph.chat "Will my GOL flight from Congonhas to Santos Dumont tomorrow at 14:00 be delayed?"

python -m app.agent_graph.chat        # interactive mode, persistent thread memory
```

Uses the same `.env` as the original service (`LLM_PROVIDER=ollama|anthropic`). The ML prediction service (`data-science`, port 8000) and backend (port 8080) should be running for the tools to return real data; the agent degrades gracefully when they're offline.

## LangSmith tracing

No code changes needed — export before running:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=<your key>       # free tier: smith.langchain.com
export LANGSMITH_PROJECT=flightontime-agent
```

Every run then shows the full trajectory in the LangSmith UI: retrieve → agent → each tool call with inputs/outputs → final answer, with latency and token breakdown per step.

## What the port buys (and what it doesn't)

Gained: checkpointed state per thread (in-process with `MemorySaver`; swap in `SqliteSaver`/Postgres to survive restarts), one-line human-in-the-loop interrupts if a tool ever becomes destructive, provider-agnostic model swapping, and step-level observability. Not gained: intelligence — the agent behaves the same. Orchestration frameworks buy you operability, not capability.
