import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rag.vectorstore import FAISSVectorStore

DOCS_DIR = Path(__file__).resolve().parent / "documents"
INDEX_DIR = Path(__file__).resolve().parent / "index"


def load_documents(docs_dir: Path | str = DOCS_DIR) -> List[Dict[str, str]]:
    """
    Reads all markdown (.md) documents from the specified directory.
    Returns a list of dicts with 'filename', 'document_name', and 'text'.
    """
    path = Path(docs_dir)
    if not path.exists():
        return []

    documents: List[Dict[str, str]] = []
    for file_path in sorted(path.glob("*.md")):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                documents.append({
                    "filename": file_path.name,
                    "document_name": file_path.stem,
                    "text": content,
                })

    return documents


def chunk_document(
    doc_text: str,
    filename: str,
    document_name: str,
    max_chunk_size: int = 600,
) -> List[Dict[str, Any]]:
    """
    Chunks a markdown document preserving header context.
    Splits primarily by '## ' sections to keep policy rules cohesive.
    Each chunk contains:
      - content
      - source
      - document_name
      - chunk_id
    """
    if not doc_text.strip():
        return []

    # Extract main title if present
    main_title_match = re.search(r"^#\s+(.+)$", doc_text, flags=re.MULTILINE)
    main_title = main_title_match.group(1).strip() if main_title_match else document_name

    # Split by markdown headers (## )
    raw_sections = [s.strip() for s in re.split(r"(?=^##\s+)", doc_text, flags=re.MULTILINE) if s.strip()]
    sections: List[str] = []
    intro_text = ""

    for s in raw_sections:
        if not s.startswith("## "):
            # Introductory text or title before first section
            lines = [l for l in s.splitlines() if not l.strip().startswith("# ")]
            rem = "\n".join(lines).strip()
            if rem:
                intro_text = rem
        else:
            if intro_text:
                sections.append(f"{intro_text}\n\n{s}")
                intro_text = ""
            else:
                sections.append(s)

    # If no '## ' sections were found, fall back to entire text
    if not sections and doc_text.strip():
        sections = [doc_text.strip()]

    chunks: List[Dict[str, Any]] = []
    chunk_idx = 0

    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue

        # If section is small/medium, keep as single chunk
        if len(sec) <= max_chunk_size:
            content = f"[{main_title}]\n{sec}"
            chunks.append({
                "content": content,
                "source": filename,
                "document": document_name,
                "document_name": document_name,
                "chunk_id": f"{filename}_chunk_{chunk_idx}",
            })
            chunk_idx += 1
        else:
            # Split longer section by paragraphs
            paragraphs = sec.split("\n\n")
            header_line = paragraphs[0] if paragraphs[0].startswith("## ") else ""
            current_sub = ""

            for p in paragraphs:
                p = p.strip()
                if not p:
                    continue

                if len(current_sub) + len(p) < max_chunk_size:
                    current_sub = f"{current_sub}\n\n{p}" if current_sub else p
                else:
                    if current_sub:
                        full_content = (
                            f"[{main_title}]\n{header_line}\n\n{current_sub}".strip()
                            if header_line and not current_sub.startswith("## ")
                            else f"[{main_title}]\n{current_sub}".strip()
                        )
                        chunks.append({
                            "content": full_content,
                            "source": filename,
                            "document": document_name,
                            "document_name": document_name,
                            "chunk_id": f"{filename}_chunk_{chunk_idx}",
                        })
                        chunk_idx += 1
                    current_sub = p

            if current_sub:
                full_content = (
                    f"[{main_title}]\n{header_line}\n\n{current_sub}".strip()
                    if header_line and not current_sub.startswith("## ")
                    else f"[{main_title}]\n{current_sub}".strip()
                )
                chunks.append({
                    "content": full_content,
                    "source": filename,
                    "document": document_name,
                    "document_name": document_name,
                    "chunk_id": f"{filename}_chunk_{chunk_idx}",
                })
                chunk_idx += 1

    return chunks


def ingest_documents(
    docs_dir: Path | str = DOCS_DIR,
    index_dir: Optional[Path | str] = INDEX_DIR,
    vectorstore: Optional[FAISSVectorStore] = None,
) -> Tuple[FAISSVectorStore, int]:
    """
    Ingests all markdown documents, chunks them with metadata,
    embeds and indexes them in FAISS, and persists to disk.
    Returns (vectorstore, total_chunks_indexed).
    """
    store = vectorstore or FAISSVectorStore()
    store.clear()

    raw_docs = load_documents(docs_dir)
    all_chunks: List[Dict[str, Any]] = []

    for doc in raw_docs:
        doc_chunks = chunk_document(
            doc_text=doc["text"],
            filename=doc["filename"],
            document_name=doc["document_name"],
        )
        all_chunks.extend(doc_chunks)

    if all_chunks:
        store.add_chunks(all_chunks)

    if index_dir:
        store.save(index_dir)

    return store, len(all_chunks)
