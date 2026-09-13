from pathlib import Path
from typing import Any, Dict, List, Optional

from rag.vectorstore import FAISSVectorStore

DEFAULT_INDEX_DIR = Path(__file__).resolve().parent / "index"
DEFAULT_RELEVANCE_THRESHOLD = 0.55


class PolicyRetriever:
    """
    Retriever that queries the FAISS vector store for relevant business policy chunks,
    enforcing similarity filtering so irrelevant documents are not returned.
    """

    def __init__(
        self,
        vectorstore: Optional[FAISSVectorStore] = None,
        index_dir: Optional[Path | str] = DEFAULT_INDEX_DIR,
        relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    ):
        self.relevance_threshold = relevance_threshold
        self.index_dir = Path(index_dir) if index_dir else None

        if vectorstore is not None:
            self.vectorstore = vectorstore
        else:
            self.vectorstore = FAISSVectorStore()
            if self.index_dir and self.index_dir.exists():
                self.vectorstore.load(self.index_dir)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Accepts a business question and returns the most relevant policy chunks
        with their source metadata.
        Applies similarity/relevance filtering so unrelated documents are dropped.
        """
        threshold = min_score if min_score is not None else self.relevance_threshold

        if not query or not query.strip():
            return []

        search_results = self.vectorstore.search(
            query=query.strip(),
            top_k=top_k,
            relevance_threshold=threshold,
        )

        formatted_results: List[Dict[str, Any]] = []
        for chunk, score in search_results:
            doc_name = chunk.get("document", chunk.get("document_name", ""))
            formatted_results.append({
                "content": chunk["content"],
                "source": chunk["source"],
                "document": doc_name,
                "document_name": doc_name,
                "chunk_id": chunk["chunk_id"],
                "relevance_score": round(score, 4),
                "similarity_score": round(score, 4),
            })

        return formatted_results

