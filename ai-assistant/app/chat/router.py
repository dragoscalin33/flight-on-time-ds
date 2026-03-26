from fastapi import APIRouter

from app.chat.schemas import ChatRequest, ChatResponse
from app.chat.service import process_message, get_session_history, clear_session

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    return await process_message(
        message=request.message,
        session_id=request.session_id,
    )


@router.get("/history/{session_id}")
async def history(session_id: str) -> list[dict]:
    return get_session_history(session_id)


@router.delete("/history/{session_id}")
async def clear_history(session_id: str) -> dict:
    cleared = clear_session(session_id)
    if cleared:
        return {"status": "cleared", "session_id": session_id}
    return {"status": "not_found", "session_id": session_id}
