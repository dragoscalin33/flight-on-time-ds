from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ToolCallInfo(BaseModel):
    tool_name: str
    tool_input: dict
    tool_result: dict


class SourceReference(BaseModel):
    document: str
    chunk: str
    relevance_score: float


class ChatResponse(BaseModel):
    message: str
    session_id: str
    sources: list[SourceReference] = []
    tools_used: list[ToolCallInfo] = []
    model_used: str
    latency_ms: float
