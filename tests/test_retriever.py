"""
tests/test_retriever.py
-----------------------
Unit tests for PharmaRetriever.
Mocks FAISS vector store — no disk/API calls needed.

Run:
    pytest tests/test_retriever.py -v
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from retriever import PharmaRetriever, RetrievedChunk
from config import TOP_K, CONFIDENCE_THRESHOLD


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

def make_doc(content: str, source: str = "fda_guide.pdf", page_num: int = 1, chunk_id: str = "c1") -> Document:
    return Document(
        page_content=content,
        metadata={"source": source, "page_num": page_num, "chunk_id": chunk_id},
    )


def make_retriever(scored_results: list[tuple[Document, float]]) -> PharmaRetriever:
    """Build PharmaRetriever with mocked FAISS store."""
    mock_store = MagicMock()
    mock_store.similarity_search_with_relevance_scores.return_value = scored_results
    return PharmaRetriever(vector_store=mock_store)


ABOVE = CONFIDENCE_THRESHOLD + 0.05   # 0.80 default
BELOW = CONFIDENCE_THRESHOLD - 0.05   # 0.70 default


# ─────────────────────────────────────────────
# retrieve() TESTS
# ─────────────────────────────────────────────

class TestRetrieve:

    def test_returns_list_of_retrieved_chunks(self):
        docs = [(make_doc("Stability data required for 12 months."), ABOVE)]
        r = make_retriever(docs)
        result = r.retrieve("stability requirements")
        assert isinstance(result, list)
        assert all(isinstance(c, RetrievedChunk) for c in result)

    def test_correct_chunk_count(self):
        docs = [(make_doc(f"Content {i}"), ABOVE) for i in range(4)]
        r = make_retriever(docs)
        result = r.retrieve("any query")
        assert len(result) == 4

    def test_chunk_fields_populated(self):
        doc = make_doc("ICH Q1A guidance text.", source="ich_q1a.pdf", page_num=3, chunk_id="c42")
        r = make_retriever([(doc, 0.88)])
        chunk = r.retrieve("ICH stability")[0]

        assert chunk.content == "ICH Q1A guidance text."
        assert chunk.source == "ich_q1a.pdf"
        assert chunk.page_num == 3
        assert chunk.chunk_id == "c42"
        assert chunk.score == 0.88
        assert chunk.above_threshold is True

    def test_score_rounded_to_4dp(self):
        doc = make_doc("Some content.")
        r = make_retriever([(doc, 0.876543219)])
        chunk = r.retrieve("query")[0]
        assert chunk.score == round(0.876543219, 4)

    def test_above_threshold_flag_true_when_equal(self):
        """Score exactly at threshold → above_threshold True (>=)."""
        doc = make_doc("Exact threshold content.")
        r = make_retriever([(doc, CONFIDENCE_THRESHOLD)])
        chunk = r.retrieve("query")[0]
        assert chunk.above_threshold is True

    def test_above_threshold_flag_false_below(self):
        doc = make_doc("Low confidence content.")
        r = make_retriever([(doc, BELOW)])
        chunk = r.retrieve("query")[0]
        assert chunk.above_threshold is False

    def test_missing_metadata_defaults(self):
        """Docs with no metadata → fallback defaults used."""
        doc = Document(page_content="No metadata doc.", metadata={})
        r = make_retriever([(doc, ABOVE)])
        chunk = r.retrieve("query")[0]
        assert chunk.source == "unknown"
        assert chunk.page_num == -1
        assert chunk.chunk_id == ""

    def test_passes_top_k_to_store(self):
        mock_store = MagicMock()
        mock_store.similarity_search_with_relevance_scores.return_value = []
        r = PharmaRetriever(vector_store=mock_store)
        r.retrieve("query", top_k=3)
        mock_store.similarity_search_with_relevance_scores.assert_called_once_with("query", k=3)

    def test_empty_results(self):
        r = make_retriever([])
        result = r.retrieve("query")
        assert result == []


# ─────────────────────────────────────────────
# retrieve_above_threshold() TESTS
# ─────────────────────────────────────────────

class TestRetrieveAboveThreshold:

    def test_filters_low_confidence(self):
        docs = [
            (make_doc("High confidence."), ABOVE),
            (make_doc("Low confidence."),  BELOW),
        ]
        r = make_retriever(docs)
        result = r.retrieve_above_threshold("query")
        assert len(result) == 1
        assert result[0].content == "High confidence."

    def test_all_above_threshold_returned(self):
        docs = [(make_doc(f"Good chunk {i}"), ABOVE) for i in range(3)]
        r = make_retriever(docs)
        result = r.retrieve_above_threshold("query")
        assert len(result) == 3

    def test_fallback_to_top1_when_all_below(self):
        """All chunks below threshold → returns only top-1."""
        docs = [
            (make_doc("Best but still low.", chunk_id="c1"), BELOW),
            (make_doc("Even lower.",         chunk_id="c2"), BELOW - 0.1),
        ]
        r = make_retriever(docs)
        result = r.retrieve_above_threshold("query")
        assert len(result) == 1
        assert result[0].chunk_id == "c1"

    def test_fallback_chunk_is_below_threshold(self):
        """Fallback top-1 may have above_threshold=False — that's expected."""
        docs = [(make_doc("Only chunk."), BELOW)]
        r = make_retriever(docs)
        result = r.retrieve_above_threshold("query")
        assert result[0].above_threshold is False

    def test_mixed_scores_only_above_returned(self):
        docs = [
            (make_doc("Pass 1"), ABOVE),
            (make_doc("Fail 1"), BELOW),
            (make_doc("Pass 2"), ABOVE + 0.05),
            (make_doc("Fail 2"), BELOW - 0.05),
        ]
        r = make_retriever(docs)
        result = r.retrieve_above_threshold("query")
        assert len(result) == 2
        assert all(c.above_threshold for c in result)


# ─────────────────────────────────────────────
# format_context() TESTS
# ─────────────────────────────────────────────

class TestFormatContext:

    def _make_chunk(self, content, source="doc.pdf", page_num=1, score=0.85, chunk_id="c1"):
        return RetrievedChunk(
            content=content,
            source=source,
            page_num=page_num,
            chunk_id=chunk_id,
            score=score,
            above_threshold=score >= CONFIDENCE_THRESHOLD,
        )

    def test_single_chunk_format(self):
        r = make_retriever([])
        chunk = self._make_chunk("Drug stability must be tested.", source="fda.pdf", page_num=2, score=0.91)
        output = r.format_context([chunk])

        assert "[Source 1: fda.pdf, Page 2 | Score: 0.91]" in output
        assert "Drug stability must be tested." in output

    def test_multiple_chunks_separated_by_divider(self):
        r = make_retriever([])
        chunks = [
            self._make_chunk("First chunk.", source="a.pdf", page_num=1),
            self._make_chunk("Second chunk.", source="b.pdf", page_num=5),
        ]
        output = r.format_context(chunks)

        assert "---" in output
        assert "[Source 1:" in output
        assert "[Source 2:" in output
        assert "First chunk." in output
        assert "Second chunk." in output

    def test_chunk_numbering_sequential(self):
        r = make_retriever([])
        chunks = [self._make_chunk(f"Chunk {i}") for i in range(3)]
        output = r.format_context(chunks)
        assert "[Source 1:" in output
        assert "[Source 2:" in output
        assert "[Source 3:" in output

    def test_empty_chunks_returns_empty_string(self):
        r = make_retriever([])
        output = r.format_context([])
        assert output == ""

    def test_source_and_page_in_label(self):
        r = make_retriever([])
        chunk = self._make_chunk("Content.", source="ich_q1a.pdf", page_num=12, score=0.80)
        output = r.format_context([chunk])
        assert "ich_q1a.pdf" in output
        assert "Page 12" in output

    def test_score_in_label(self):
        r = make_retriever([])
        chunk = self._make_chunk("Content.", score=0.9321)
        output = r.format_context([chunk])
        assert "0.9321" in output
