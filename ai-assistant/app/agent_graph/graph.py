import json
from typing import Annotated, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.agent_graph.tools import AGENT_TOOLS
from app.chat.service import MAX_TOOL_ROUNDS, SYSTEM_PROMPT
from app.config import settings


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    rag_context: str
    rag_sources: list[dict]
    tool_rounds: int
    tool_errors: int


MAX_TOOL_ERRORS = 2


def create_chat_model() -> BaseChatModel:
    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            max_tokens=2048,
        )
    from langchain_ollama import ChatOllama

    return ChatOllama(model=settings.ollama_model, base_url=settings.ollama_base_url)


def retrieve(state: AgentState) -> dict:
    query = str(state["messages"][-1].content)
    try:
        from app.rag.retrieval import get_relevant_context

        results = get_relevant_context(query, n_results=3)
    except Exception:
        results = []

    context = "\n\n".join(f"[Source: {r['source']}]\n{r['content']}" for r in results)
    return {
        "rag_context": context if context else "No specific context retrieved.",
        "rag_sources": results,
        "tool_rounds": 0,
        "tool_errors": 0,
    }


def tools_allowed(state: AgentState) -> bool:
    return state["tool_rounds"] < MAX_TOOL_ROUNDS and state["tool_errors"] < MAX_TOOL_ERRORS


async def agent(state: AgentState) -> dict:
    system = SystemMessage(content=SYSTEM_PROMPT.format(rag_context=state["rag_context"]))
    model = create_chat_model()
    if tools_allowed(state):
        model = model.bind_tools(AGENT_TOOLS)
    response = await model.ainvoke([system, *state["messages"]])
    return {"messages": [response]}


_tool_node = ToolNode(AGENT_TOOLS)


def _is_error_result(message: AnyMessage) -> bool:
    try:
        payload = json.loads(str(message.content))
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(payload, dict) and "error" in payload


async def tools(state: AgentState) -> dict:
    result = await _tool_node.ainvoke({"messages": state["messages"]})
    errors = sum(1 for m in result["messages"] if _is_error_result(m))
    return {
        "messages": result["messages"],
        "tool_rounds": state["tool_rounds"] + 1,
        "tool_errors": state["tool_errors"] + errors,
    }


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


def build_graph(checkpointer: MemorySaver | None = None):
    builder = StateGraph(AgentState)
    builder.add_node("retrieve", retrieve)
    builder.add_node("agent", agent)
    builder.add_node("tools", tools)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "agent")
    builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer or MemorySaver())
