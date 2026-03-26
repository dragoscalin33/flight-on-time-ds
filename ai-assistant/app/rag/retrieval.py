import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from app.config import settings
from app.rag.ingestion import COLLECTION_NAME


def get_relevant_context(query: str, n_results: int = 5) -> list[dict]:
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=settings.embedding_model)

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
        )
    except ValueError:
        return []

    results = collection.query(query_texts=[query], n_results=n_results)

    contexts: list[dict] = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, distance in zip(documents, metadatas, distances):
        relevance_score = 1.0 - distance
        contexts.append({
            "content": doc,
            "source": meta.get("source", "unknown"),
            "chunk_index": meta.get("chunk_index", 0),
            "relevance_score": round(relevance_score, 4),
        })

    return contexts
