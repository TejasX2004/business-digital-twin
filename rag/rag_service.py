from pathlib import Path
from typing import Any, Dict, List, Optional

from rag.ingestion import DOCS_DIR, INDEX_DIR, ingest_documents
from rag.retriever import PolicyRetriever
from rag.vectorstore import FAISSVectorStore

# Singleton retriever instance
_RETRIEVER: Optional[PolicyRetriever] = None

NO_POLICY_RETRIEVED_MESSAGE = "No relevant business policy was retrieved."


def get_retriever() -> PolicyRetriever:
    """Returns the shared PolicyRetriever instance, initializing if necessary."""
    global _RETRIEVER
    if _RETRIEVER is None:
        index_dir = INDEX_DIR
        store = FAISSVectorStore()
        # Check if index exists on disk, if not build it
        if not (index_dir / "index.faiss").exists() or not (index_dir / "metadata.json").exists():
            ingest_documents(docs_dir=DOCS_DIR, index_dir=index_dir, vectorstore=store)
        else:
            store.load(index_dir)
        _RETRIEVER = PolicyRetriever(vectorstore=store, index_dir=index_dir)
    return _RETRIEVER


def rebuild_index(
    docs_dir: Path | str = DOCS_DIR,
    index_dir: Path | str = INDEX_DIR,
) -> int:
    """
    Rebuilds the vector index from all documents and reloads the retriever.
    Returns the number of indexed chunks.
    """
    global _RETRIEVER
    store, count = ingest_documents(docs_dir=docs_dir, index_dir=index_dir)
    _RETRIEVER = PolicyRetriever(vectorstore=store, index_dir=index_dir)
    return count


def retrieve_business_context(
    query: str,
    top_k: int = 5,
    relevance_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieves the most relevant business policy chunks for a query.

    Returns structured results:
    [
        {
            "content": "...",
            "source": "...",
            "chunk_id": "..."
        }
    ]

    If no relevant document is found above the relevance threshold,
    explicitly returns a single item indicating no relevant policy was found:
    [
        {
            "content": "No relevant business policy was retrieved.",
            "source": "none",
            "chunk_id": "none"
        }
    ]
    """
    retriever = get_retriever()
    results = retriever.retrieve(
        query=query,
        top_k=top_k,
        min_score=relevance_threshold,
    )

    if not results:
        return [
            {
                "content": NO_POLICY_RETRIEVED_MESSAGE,
                "source": "none",
                "chunk_id": "none",
            }
        ]

    # Clean to the requested schema
    output: List[Dict[str, Any]] = []
    for item in results:
        output.append({
            "content": item["content"],
            "source": item["source"],
            "chunk_id": item["chunk_id"],
        })

    return output


def format_rag_context_for_prompt(context_chunks: List[Dict[str, Any]]) -> str:
    """
    Formats retrieved business knowledge into a clean, source-attributed
    block for LLM prompt injection.
    """
    if not context_chunks:
        return NO_POLICY_RETRIEVED_MESSAGE

    if len(context_chunks) == 1 and context_chunks[0].get("source") == "none":
        return NO_POLICY_RETRIEVED_MESSAGE

    formatted_sections = []
    for idx, chunk in enumerate(context_chunks, 1):
        source = chunk.get("source", "unknown")
        chunk_id = chunk.get("chunk_id", f"chunk_{idx}")
        content = chunk.get("content", "").strip()
        formatted_sections.append(
            f"--- Policy Excerpt {idx} [Source: {source} | ID: {chunk_id}] ---\n{content}"
        )

    return "\n\n".join(formatted_sections)
