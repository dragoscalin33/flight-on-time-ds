import os
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from app.config import settings

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
COLLECTION_NAME = "flightontime_knowledge"


def get_embedding_function() -> SentenceTransformerEmbeddingFunction:
    return SentenceTransformerEmbeddingFunction(model_name=settings.embedding_model)


def get_chroma_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)


def chunk_document(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    lines = text.split("\n")
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for line in lines:
        line_length = len(line)
        if current_length + line_length > chunk_size and current_chunk:
            chunks.append("\n".join(current_chunk))
            overlap_text = "\n".join(current_chunk)
            overlap_lines: list[str] = []
            overlap_length = 0
            for prev_line in reversed(current_chunk):
                if overlap_length + len(prev_line) > overlap:
                    break
                overlap_lines.insert(0, prev_line)
                overlap_length += len(prev_line)
            current_chunk = overlap_lines
            current_length = overlap_length

        current_chunk.append(line)
        current_length += line_length

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


def ingest_knowledge_base() -> int:
    client = get_chroma_client()
    embedding_fn = get_embedding_function()

    try:
        client.delete_collection(COLLECTION_NAME)
    except ValueError:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    total_chunks = 0

    for filepath in KNOWLEDGE_DIR.glob("*.md"):
        content = filepath.read_text(encoding="utf-8")
        chunks = chunk_document(content)

        ids = [f"{filepath.stem}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {"source": filepath.name, "chunk_index": i, "total_chunks": len(chunks)}
            for i in range(len(chunks))
        ]

        collection.add(documents=chunks, ids=ids, metadatas=metadatas)
        total_chunks += len(chunks)

    return total_chunks
