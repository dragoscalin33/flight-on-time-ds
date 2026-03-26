from abc import ABC, abstractmethod
from typing import Any

import httpx

from app.config import settings


class LLMClient(ABC):
    @abstractmethod
    async def chat_with_tools(
        self,
        messages: list[dict],
        system_prompt: str,
        tools: list[dict],
    ) -> dict:
        ...


class AnthropicClient(LLMClient):
    def __init__(self) -> None:
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        except ImportError as e:
            raise ImportError("Install anthropic: pip install anthropic") from e
        self.model = settings.anthropic_model

    async def chat_with_tools(
        self,
        messages: list[dict],
        system_prompt: str,
        tools: list[dict],
    ) -> dict:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 2048,
            "system": system_prompt,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        response = await self.client.messages.create(**kwargs)

        result: dict[str, Any] = {"content": [], "stop_reason": response.stop_reason}
        for block in response.content:
            if block.type == "text":
                result["content"].append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                result["content"].append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        return result


class OllamaClient(LLMClient):
    def __init__(self) -> None:
        self.base_url = settings.ollama_base_url
        self.model = settings.ollama_model

    async def chat_with_tools(
        self,
        messages: list[dict],
        system_prompt: str,
        tools: list[dict],
    ) -> dict:
        ollama_messages = [{"role": "system", "content": system_prompt}]

        for msg in messages:
            if isinstance(msg.get("content"), str):
                ollama_messages.append({"role": msg["role"], "content": msg["content"]})
            elif isinstance(msg.get("content"), list):
                text_parts = [
                    block["text"]
                    for block in msg["content"]
                    if block.get("type") == "text"
                ]
                if text_parts:
                    ollama_messages.append({"role": msg["role"], "content": "\n".join(text_parts)})

        ollama_tools = []
        for tool in tools:
            ollama_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            })

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
        }
        if ollama_tools:
            payload["tools"] = ollama_tools

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        message = data.get("message", {})
        tool_calls = message.get("tool_calls", [])

        result: dict[str, Any] = {"content": [], "stop_reason": "end_turn"}

        if tool_calls:
            result["stop_reason"] = "tool_use"
            for tc in tool_calls:
                func = tc.get("function", {})
                result["content"].append({
                    "type": "tool_use",
                    "id": f"ollama_{func.get('name', 'unknown')}",
                    "name": func.get("name", ""),
                    "input": func.get("arguments", {}),
                })
        elif message.get("content"):
            result["content"].append({"type": "text", "text": message["content"]})

        return result


def create_llm_client() -> LLMClient:
    if settings.llm_provider == "anthropic":
        return AnthropicClient()
    return OllamaClient()
