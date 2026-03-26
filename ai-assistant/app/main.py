import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.chat.router import router as chat_router
from app.rag.ingestion import ingest_knowledge_base
from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Ingesting knowledge base into ChromaDB...")
    chunk_count = ingest_knowledge_base()
    logger.info(f"Knowledge base loaded: {chunk_count} chunks indexed")
    logger.info(f"LLM Provider: {settings.llm_provider}")
    if settings.llm_provider == "ollama":
        logger.info(f"Ollama model: {settings.ollama_model}")
    else:
        logger.info(f"Anthropic model: {settings.anthropic_model}")
    yield


app = FastAPI(
    title="FlightOnTime AI Assistant",
    description=(
        "Intelligent flight delay prediction assistant for Brazilian domestic flights. "
        "Combines a CatBoost ML model with RAG and LLM-powered natural language interaction."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "healthy",
        "service": "flightontime-ai-assistant",
        "llm_provider": settings.llm_provider,
    }
