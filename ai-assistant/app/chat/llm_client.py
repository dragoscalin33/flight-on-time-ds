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
            content = msg.get("content")
            role = msg.get("role", "user")

            if isinstance(content, str):
                ollama_messages.append({"role": role, "content": content})
            elif isinstance(content, list):
                tool_use_blocks = [b for b in content if b.get("type") == "tool_use"]
                tool_result_blocks = [b for b in content if b.get("type") == "tool_result"]
                text_blocks = [b for b in content if b.get("type") == "text"]

                if tool_use_blocks and role == "assistant":
                    ollama_messages.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": b["name"],
                                    "arguments": b["input"],
                                }
                            }
                            for b in tool_use_blocks
                        ],
                    })
                elif tool_result_blocks:
                    for block in tool_result_blocks:
                        ollama_messages.append({
                            "role": "tool",
                            "content": block.get("content", "{}"),
                        })
                elif text_blocks:
                    text = "\n".join(b["text"] for b in text_blocks if b.get("text"))
                    if text:
                        ollama_messages.append({"role": role, "content": text})

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
            for i, tc in enumerate(tool_calls):
                func = tc.get("function", {})
                result["content"].append({
                    "type": "tool_use",
                    "id": f"ollama_{func.get('name', 'unknown')}_{i}",
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
