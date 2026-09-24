"""
Tests for HybridRetriever, similarity threshold cutoff, BM25 fallback, and context formatting.
Mocked Chroma collection with 0 network calls.
"""

from unittest.mock import MagicMock
from retriever import HybridRetriever, build_prompt_context


def test_similarity_threshold_filters_out_irrelevant_chunks():
    # Mock Chroma collection that returns chunks with high distance (low similarity)
    mock_collection = MagicMock()
    # Distance = 0.85 -> Similarity = 1.0 - 0.85 = 0.15 (below threshold 0.35)
    mock_collection.query.return_value = {
        "ids": [["chunk_irrelevant"]],
        "distances": [[0.85]],
        "metadatas": [[{"path": "foo.py", "start_line": 1, "end_line": 5}]]
    }

    chunks = [{
        "id": "chunk_irrelevant",
        "text": "unrelated content completely different topic",
        "metadata": {"path": "foo.py", "start_line": 1, "end_line": 5, "language": "python"}
    }]

    retriever = HybridRetriever(mock_collection, chunks)

    # Query with non-matching keywords so BM25 score is 0
    results = retriever.retrieve("completely_different_xyz_term", top_k=3, similarity_threshold=0.35)

    # Must return empty list, NOT the first 4 chunks! (Fixing Bad Fallback flaw)
    assert len(results) == 0


def test_bm25_matches_exact_keywords_even_if_vector_below_threshold():
    # If vector returns distance 0.90 (similarity 0.10 < 0.35), but BM25 matches exact keyword
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "ids": [["chunk_target"]],
        "distances": [[0.90]],
        "metadatas": [[{"path": "auth.py", "start_line": 10, "end_line": 20}]]
    }

    chunks = [{
        "id": "chunk_target",
        "text": "def authenticate_user_with_jwt_token(token): return True",
        "metadata": {"path": "auth.py", "start_line": 10, "end_line": 20, "language": "python"}
    }]

    retriever = HybridRetriever(mock_collection, chunks)
    results = retriever.retrieve("authenticate_user_with_jwt_token", top_k=3, similarity_threshold=0.35)

    # BM25 should pick up the exact keyword and return the chunk
    assert len(results) == 1
    assert results[0]["id"] == "chunk_target"
    assert results[0]["metadata"]["in_bm25"] is True


def test_prompt_context_builder_budget_cap():
    chunks = [
        {
            "id": f"chunk_{i}",
            "text": "code line " * 100,  # ~1000 chars per chunk
            "metadata": {"path": f"src/file_{i}.py", "start_line": 1, "end_line": 10}
        }
        for i in range(10)
    ]

    readme = "# Test Project\nThis is a long readme description." * 50
    file_tree = "src/\n  file_0.py\n  file_1.py\n" * 50

    context, sources = build_prompt_context(
        chunks, readme_text=readme, file_tree_text=file_tree, max_context_chars=4000
    )

    # Assert character cap respected
    assert len(context) <= 4500
    assert len(sources) >= 1
    assert "Repository Overview" in context
    assert "Repository Structure" in context


def test_bm25_stopword_filtering_prevents_false_positive_on_unrelated_queries():
    # Flaw #8 regression test:
    # A chunk has the generic word "implementation".
    # Query is "What is the Bitcoin mining algorithm implementation in this repository?".
    # Vector similarity is low (0.15 < 0.35).
    # BM25 must NOT match because generic terms ("implementation", "algorithm", "repository")
    # are filtered as BM25 stopwords.
    mock_collection = MagicMock()
    mock_collection.metadata = {"hnsw:space": "cosine"}
    mock_collection.query.return_value = {
        "ids": [["chunk_generic"]],
        "distances": [[0.85]],  # similarity 0.15
        "metadatas": [[{"path": "src/server.py", "start_line": 1, "end_line": 5}]]
    }

    chunks = [{
        "id": "chunk_generic",
        "text": "This is a standard implementation of a web server.",
        "metadata": {"path": "src/server.py", "start_line": 1, "end_line": 5, "language": "python"}
    }]

    retriever = HybridRetriever(mock_collection, chunks)
    results = retriever.retrieve(
        "What is the Bitcoin mining algorithm implementation in this repository?",
        top_k=3,
        similarity_threshold=0.35
    )

    # Must return 0 chunks (honest failure)
    assert len(results) == 0


def test_cosine_space_distance_conversion():
    # Point 2 regression test: verify similarity computation respects collection space
    mock_collection = MagicMock()
    mock_collection.metadata = {"hnsw:space": "cosine"}
    mock_collection.query.return_value = {
        "ids": [["chunk_good"]],
        "distances": [[0.20]],  # cosine similarity = 1 - 0.20 = 0.80
        "metadatas": [[{"path": "src/code.py", "start_line": 1, "end_line": 5}]]
    }

    chunks = [{
        "id": "chunk_good",
        "text": "def test_func(): return True",
        "metadata": {"path": "src/code.py", "start_line": 1, "end_line": 5, "language": "python"}
    }]

    retriever = HybridRetriever(mock_collection, chunks)
    results = retriever.retrieve("test_func", top_k=1, similarity_threshold=0.35)
    assert len(results) == 1
    assert results[0]["metadata"]["similarity"] == 0.80


def test_chroma_persist_dir_is_absolute():
    # Point 5 regression test: verify CHROMA_PERSIST_DIR is an absolute path
    import os
    from config import CHROMA_PERSIST_DIR
    assert os.path.isabs(CHROMA_PERSIST_DIR), f"CHROMA_PERSIST_DIR is not absolute: {CHROMA_PERSIST_DIR}"
    assert "chroma_db" in CHROMA_PERSIST_DIR

