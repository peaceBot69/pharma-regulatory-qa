"""
retriever.py
------------
Semantic search over FAISS index.
Returns top-k chunks with scores + metadata.

Usage:
    python src/retriever.py --query "What are ICH Q1A stability testing requirements?"
"""

import argparse
import logging
from dataclasses import dataclass

from langchain_community.vectorstores import FAISS

from ingestion import load_index
from config import TOP_K, CONFIDENCE_THRESHOLD

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# RESULT SCHEMA
# ─────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    content: str
    source: str
    page_num: int
    chunk_id: str
    score: float          # cosine similarity (0–1, higher = more relevant)
    above_threshold: bool # score > CONFIDENCE_THRESHOLD


# ─────────────────────────────────────────────
# RETRIEVER CLASS
# ─────────────────────────────────────────────

class PharmaRetriever:
    def __init__(self, vector_store: FAISS = None):
        """
        Pass existing vector_store or loads from disk automatically.
        """
        self.vector_store = vector_store or load_index()
        log.info("PharmaRetriever ready.")

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[RetrievedChunk]:
        """
        Semantic search: query → top-k chunks with scores.

        Uses FAISS similarity_search_with_relevance_scores:
        returns (Document, float) pairs where float is cosine similarity.
        """
        log.info(f"Retrieving top-{top_k} chunks for query: '{query}'")

        results = self.vector_store.similarity_search_with_relevance_scores(
            query,
            k=top_k,
        )

        chunks = []
        for doc, score in results:
            chunks.append(RetrievedChunk(
                content=doc.page_content,
                source=doc.metadata.get("source", "unknown"),
                page_num=doc.metadata.get("page_num", -1),
                chunk_id=doc.metadata.get("chunk_id", ""),
                score=round(score, 4),
                above_threshold=score >= CONFIDENCE_THRESHOLD,
            ))

        self._log_results(chunks)
        return chunks

    def retrieve_above_threshold(self, query: str, top_k: int = TOP_K) -> list[RetrievedChunk]:
        """
        Same as retrieve() but filters out low-confidence chunks.
        Falls back to top-1 if nothing passes threshold.
        """
        chunks = self.retrieve(query, top_k)
        filtered = [c for c in chunks if c.above_threshold]

        if not filtered:
            log.warning(
                f"No chunks above threshold ({CONFIDENCE_THRESHOLD}). "
                f"Falling back to top-1 result (score={chunks[0].score})."
            )
            return chunks[:1]

        log.info(f"{len(filtered)}/{len(chunks)} chunks passed confidence threshold.")
        return filtered

    def format_context(self, chunks: list[RetrievedChunk]) -> str:
        """
        Format retrieved chunks into a single context string for the LLM.
        Each chunk labelled with source + page for citation.
        """
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            parts.append(
                f"[Source {i}: {chunk.source}, Page {chunk.page_num} | Score: {chunk.score}]\n"
                f"{chunk.content}"
            )
        return "\n\n---\n\n".join(parts)

    def _log_results(self, chunks: list[RetrievedChunk]):
        log.info(f"Retrieved {len(chunks)} chunks:")
        for i, c in enumerate(chunks, 1):
            status = "✓" if c.above_threshold else "✗"
            log.info(f"  [{status}] Chunk {i} | {c.source} p.{c.page_num} | score={c.score}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pharma Doc QA — Retriever")
    parser.add_argument("--query", required=True, help="Natural language query")
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--threshold_only", action="store_true",
                        help="Only return chunks above confidence threshold")
    args = parser.parse_args()

    retriever = PharmaRetriever()

    if args.threshold_only:
        results = retriever.retrieve_above_threshold(args.query, args.top_k)
    else:
        results = retriever.retrieve(args.query, args.top_k)

    print("\n" + "="*60)
    print(f"QUERY: {args.query}")
    print("="*60)
    for i, chunk in enumerate(results, 1):
        print(f"\n[Chunk {i}] {chunk.source} | Page {chunk.page_num} | Score: {chunk.score}")
        print(f"Above threshold: {chunk.above_threshold}")
        print(f"Content:\n{chunk.content[:300]}...")
