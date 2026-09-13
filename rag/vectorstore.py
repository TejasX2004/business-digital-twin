import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from fastembed import TextEmbedding


class FAISSVectorStore:
    """
    Local FAISS vector store using fastembed dense embeddings.
    Embeddings are normalized so that Inner Product (IndexFlatIP)
    computes exact cosine similarity.
    """

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(
        self,
        embedding_model_name: str = DEFAULT_MODEL,
        dimension: int = 384,
    ):
        self.embedding_model_name = embedding_model_name
        self.dimension = dimension
        self._embedder: Optional[TextEmbedding] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[Dict[str, Any]] = []

    @property
    def embedder(self) -> TextEmbedding:
        if self._embedder is None:
            self._embedder = TextEmbedding(model_name=self.embedding_model_name)
        return self._embedder

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generates L2-normalized dense embeddings for a list of strings."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        raw_vectors = list(self.embedder.embed(texts))
        vectors = np.array(raw_vectors, dtype=np.float32)
        faiss.normalize_L2(vectors)
        return vectors

    def embed_query(self, query: str) -> np.ndarray:
        """Generates L2-normalized dense embedding for a single query string."""
        vectors = self.embed_texts([query])
        return vectors

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Embeds chunk contents and adds them to the FAISS index.
        Each chunk must have 'content', 'source', 'document_name', 'chunk_id'.
        """
        if not chunks:
            return

        texts = [chunk["content"] for chunk in chunks]
        vectors = self.embed_texts(texts)

        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)

        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        relevance_threshold: float = 0.45,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Performs cosine similarity search against the indexed chunks.
        Returns a list of (chunk_dict, score) tuples filtered by relevance_threshold.
        """
        if self.index is None or self.index.ntotal == 0 or not self.chunks:
            return []

        query_vec = self.embed_query(query)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vec, k)

        results: List[Tuple[Dict[str, Any], float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            cos_score = float(score)
            if cos_score >= relevance_threshold:
                results.append((self.chunks[idx], cos_score))

        return results

    def save(self, output_dir: Path | str) -> None:
        """Saves the FAISS index and chunk metadata to disk."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if self.index is not None:
            faiss.write_index(self.index, str(out_path / "index.faiss"))

        with open(out_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)

    def load(self, input_dir: Path | str) -> bool:
        """
        Loads the FAISS index and chunk metadata from disk.
        Returns True if loaded successfully, False otherwise.
        """
        in_path = Path(input_dir)
        index_file = in_path / "index.faiss"
        meta_file = in_path / "metadata.json"

        if not index_file.exists() or not meta_file.exists():
            return False

        self.index = faiss.read_index(str(index_file))
        with open(meta_file, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        return True

    def clear(self) -> None:
        """Clears all vectors and chunks in memory."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks = []
