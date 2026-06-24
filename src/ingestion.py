"""
ingestion.py
------------
Load PDFs → chunk → embed → persist FAISS index.

Usage:
    python src/ingestion.py --docs_dir data/raw --rebuild
"""

import os
import argparse
import logging
from pathlib import Path

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import (
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    VECTOR_STORE_DIR,
    FAISS_INDEX_NAME,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. PDF EXTRACTION
# ─────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text page-by-page from a PDF.
    Returns list of {page_num, text, source} dicts.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        blocks = page.get_text("blocks")
        block_text = " ".join(b[4].strip() for b in blocks if b[4].strip())
        combined = text if len(text) >= len(block_text) else block_text
        if combined:
            pages.append({
                "page_num": page_num,
                "text": combined,
                "source": Path(pdf_path).name,
            })
    doc.close()
    log.info(f"Extracted {len(pages)} pages from '{Path(pdf_path).name}'")
    return pages


def load_all_pdfs(docs_dir: str) -> list[dict]:
    """Load all PDFs from a directory."""
    pdf_files = list(Path(docs_dir).glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in '{docs_dir}'")
    
    all_pages = []
    for pdf_path in pdf_files:
        all_pages.extend(extract_text_from_pdf(str(pdf_path)))
    
    log.info(f"Total pages loaded: {len(all_pages)} from {len(pdf_files)} file(s)")
    return all_pages


# ─────────────────────────────────────────────
# 2. CHUNKING
# ─────────────────────────────────────────────

def chunk_pages(pages: list[dict]) -> list[Document]:
    """
    Split page texts into overlapping chunks.
    Preserves source + page_num in metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    docs = []
    for page in pages:
        chunks = splitter.split_text(page["text"])
        for i, chunk in enumerate(chunks):
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "source": page["source"],
                    "page_num": page["page_num"],
                    "chunk_id": f"{page['source']}_p{page['page_num']}_c{i}",
                }
            ))

    log.info(f"Total chunks created: {len(docs)}")
    return docs


# ─────────────────────────────────────────────
# 3. EMBED + FAISS
# ─────────────────────────────────────────────

def build_faiss_index(docs: list[Document]) -> FAISS:
    """Embed chunks and build FAISS vector store."""
    log.info(f"Embedding {len(docs)} chunks with '{EMBEDDING_MODEL}'...")
    embeddings = HuggingFaceEmbeddings(
        model=EMBEDDING_MODEL,
    )
    vector_store = FAISS.from_documents(docs, embeddings)
    log.info("FAISS index built.")
    return vector_store


def save_index(vector_store: FAISS) -> str:
    """Persist FAISS index to disk."""
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    save_path = os.path.join(VECTOR_STORE_DIR, FAISS_INDEX_NAME)
    vector_store.save_local(save_path)
    log.info(f"Index saved to '{save_path}'")
    return save_path


def load_index() -> FAISS:
    """Load persisted FAISS index from disk."""
    index_path = os.path.join(VECTOR_STORE_DIR, FAISS_INDEX_NAME)
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"No index at '{index_path}'. Run ingestion first."
        )
    embeddings = HuggingFaceEmbeddings(
        model=EMBEDDING_MODEL,
    )
    vector_store = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    log.info(f"Index loaded from '{index_path}'")
    return vector_store


# ─────────────────────────────────────────────
# 4. MAIN PIPELINE
# ─────────────────────────────────────────────

def run_ingestion(docs_dir: str = "data/raw", rebuild: bool = False) -> FAISS:
    """
    Full ingestion pipeline:
    PDFs → pages → chunks → FAISS index.

    If index exists and rebuild=False, loads from disk.
    """
    index_path = os.path.join(VECTOR_STORE_DIR, FAISS_INDEX_NAME)

    if os.path.exists(index_path) and not rebuild:
        log.info("Existing index found. Loading from disk (use --rebuild to re-index).")
        return load_index()

    pages = load_all_pdfs(docs_dir)
    chunks = chunk_pages(pages)
    vector_store = build_faiss_index(chunks)
    save_index(vector_store)
    return vector_store


# ─────────────────────────────────────────────
# 5. CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pharma Doc QA — Ingestion Pipeline")
    parser.add_argument("--docs_dir", default="data/raw", help="Path to PDF directory")
    parser.add_argument("--rebuild", action="store_true", help="Force re-index even if index exists")
    args = parser.parse_args()

    vs = run_ingestion(docs_dir=args.docs_dir, rebuild=args.rebuild)
    log.info(f"Ingestion complete. Index ready.")