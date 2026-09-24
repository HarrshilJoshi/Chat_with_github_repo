"""
Retriever module.
Combines BM25 keyword search with ChromaDB semantic vector search using
Reciprocal Rank Fusion (RRF). Filters by cosine similarity before fusion,
handles BM25 keyword candidates, and formats token-budgeted prompt contexts.
"""

import math
import re
from typing import Optional
from chromadb.api.models.Collection import Collection
from rank_bm25 import BM25Plus

from config import (
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
    MAX_CONTEXT_CHARS,
)


# Stopwords and common generic terms to filter from BM25 queries
# Prevents generic words like "algorithm" or "implementation" from matching unrelated files
BM25_STOP_WORDS = {
    # Standard English stopwords
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves",
    # Very common generic code & query terms
    "code", "function", "class", "file", "files", "method", "methods", "repo",
    "repository", "implementation", "implement", "algorithm", "project", "program",
    "show", "explain", "find", "get", "give", "tell", "describe", "example",
    "use", "used", "using", "work", "works",
    # Common query verbs
    "handle", "handles", "handling", "calculate", "calculates", "calculating",
    "call", "calls", "calling", "create", "creates", "creating",
    "make", "makes", "making", "process", "processes", "processing",
    "start", "starts", "starting"
}

# Minimum BM25 score required for a document to be considered a valid keyword match
MIN_BM25_SCORE: float = 0.8


class HybridRetriever:
    """
    Hybrid search engine combining lexical BM25 matching and semantic vector search.
    """

    def __init__(self, collection: Collection, chunks: list[dict]):
        self.collection = collection
        self.chunks = chunks
        self.chunk_by_id = {c["id"]: c for c in chunks}

        # Build in-memory BM25 index from chunk text tokens using BM25Plus
        self.corpus_ids = [c["id"] for c in chunks]
        self.tokenized_corpus = [self._tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Plus(self.tokenized_corpus) if self.tokenized_corpus else None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple lowercase alphanumeric word tokenizer for BM25."""
        return re.findall(r"\w+", text.lower())

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    ) -> list[dict]:
        """
        Execute hybrid search:
        1. Vector search in ChromaDB, converting cosine distance to similarity (1.0 - distance).
        2. Filter vector candidates strictly below similarity_threshold.
        3. BM25 keyword search on non-stopword tokens, requiring a minimum score.
        4. Reciprocal Rank Fusion (RRF) to merge and rerank results.
        """
        if not self.chunks or not query.strip():
            return []

        # --- 1. Vector Search & Similarity Filtering ---
        vector_ranked_ids: list[str] = []
        vector_sim_scores: dict[str, float] = {}

        try:
            # Query vector database for top 2 * top_k candidates
            query_results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k * 2, len(self.chunks)),
                include=["metadatas", "distances"]
            )
            ids_list = query_results.get("ids", [[]])[0]
            distances_list = query_results.get("distances", [[]])[0]

            # Verify collection distance metric: 1.0 - distance is only valid for cosine distance
            meta = getattr(self.collection, "metadata", None)
            collection_space = meta.get("hnsw:space", "cosine") if isinstance(meta, dict) else "cosine"

            for cid, dist in zip(ids_list, distances_list):
                if collection_space == "cosine":
                    sim = max(0.0, min(1.0, 1.0 - float(dist)))
                elif collection_space == "ip":
                    sim = max(0.0, min(1.0, float(dist)))
                else:  # L2 or other distance metrics
                    sim = 1.0 / (1.0 + float(dist))

                # Apply strict similarity filter BEFORE fusion
                if sim >= similarity_threshold:
                    vector_ranked_ids.append(cid)
                    vector_sim_scores[cid] = sim
        except Exception:
            # If vector search fails unexpectedly, proceed with BM25
            vector_ranked_ids = []

        # --- 2. BM25 Lexical Keyword Search ---
        bm25_ranked_ids: list[str] = []
        if self.bm25:
            raw_tokens = self._tokenize(query)
            # Remove stopwords and generic query words so filler terms do not match unrelated chunks
            content_tokens = [t for t in raw_tokens if t not in BM25_STOP_WORDS and len(t) > 1]
            distinct_terms = set(content_tokens)
            if distinct_terms:
                scores = self.bm25.get_scores(content_tokens)
                # Term match ratio: require at least min(2, ceil(n/2)) meaningful terms matched
                min_matched_terms = min(2, math.ceil(len(distinct_terms) / 2))

                scored_indices = [
                    (idx, score)
                    for idx, score in enumerate(scores)
                    if score >= MIN_BM25_SCORE and len(distinct_terms.intersection(self.tokenized_corpus[idx])) >= min_matched_terms
                ]
                scored_indices.sort(key=lambda x: x[1], reverse=True)
                bm25_ranked_ids = [self.corpus_ids[idx] for idx, _ in scored_indices[:top_k * 2]]

        # If both passes found nothing relevant, return empty (honest failure)
        if not vector_ranked_ids and not bm25_ranked_ids:
            return []

        # --- 3. Reciprocal Rank Fusion (RRF) ---
        RRF_K = 60.0
        rrf_scores: dict[str, float] = {}

        # Add vector rank contribution
        for rank, cid in enumerate(vector_ranked_ids):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (RRF_K + rank + 1))

        # Add BM25 rank contribution
        for rank, cid in enumerate(bm25_ranked_ids):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (RRF_K + rank + 1))

        # Sort all candidate IDs by fused RRF score descending
        sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Assemble output chunks with metadata and retrieval scores
        final_results: list[dict] = []
        for cid, score in sorted_candidates[:top_k]:
            chunk_data = self.chunk_by_id.get(cid)
            if chunk_data:
                meta = dict(chunk_data["metadata"])
                meta["rrf_score"] = round(score, 5)
                meta["similarity"] = round(vector_sim_scores.get(cid, 0.0), 3)
                meta["in_bm25"] = cid in bm25_ranked_ids
                meta["in_vector"] = cid in vector_ranked_ids

                final_results.append({
                    "id": cid,
                    "text": chunk_data["text"],
                    "metadata": meta
                })

        return final_results


def build_prompt_context(
    retrieved_chunks: list[dict],
    readme_text: str = "",
    file_tree_text: str = "",
    max_context_chars: int = MAX_CONTEXT_CHARS
) -> tuple[str, list[dict]]:
    """
    Format LLM prompt context strictly within the character budget (~6000 chars):
    - Trimmed README (~1200 chars) for repository purpose
    - Trimmed File Tree (~1200 chars) for structural context
    - Top Retrieved Code Chunks (~3600 chars) for factual answers
    
    Returns (context_string, list_of_sources).
    """
    sources_summary: list[dict] = []
    sections: list[str] = []

    # 1. Repository README overview (up to 1200 chars)
    if readme_text.strip():
        trimmed_readme = readme_text.strip()[:1200]
        sections.append(f"### Repository Overview (README):\n{trimmed_readme}")

    # 2. Repository File Tree structure (up to 1200 chars)
    if file_tree_text.strip():
        trimmed_tree = file_tree_text.strip()[:1200]
        sections.append(f"### Repository Structure:\n```\n{trimmed_tree}\n```")

    # 3. Retrieved Code Chunks (remaining budget, ~3600 chars)
    used_chars = sum(len(s) for s in sections)
    remaining_chars = max(1000, max_context_chars - used_chars)

    chunk_blocks: list[str] = []
    for chunk in retrieved_chunks:
        meta = chunk.get("metadata", {})
        path = meta.get("path", "unknown")
        start_l = meta.get("start_line", 1)
        end_l = meta.get("end_line", 1)
        text = chunk.get("text", "")

        header = f"--- File: {path} (Lines {start_l}-{end_l}) ---"
        block = f"{header}\n{text}\n"

        if sum(len(b) for b in chunk_blocks) + len(block) > remaining_chars:
            # If we already have at least 2 chunks, stop to preserve character budget
            if len(chunk_blocks) >= 2:
                break
            # Otherwise truncate the chunk to fit
            allowed = max(200, remaining_chars - sum(len(b) for b in chunk_blocks) - len(header) - 10)
            block = f"{header}\n{text[:allowed]}...\n"
            chunk_blocks.append(block)
            sources_summary.append({
                "path": path,
                "start_line": start_l,
                "end_line": end_l,
                "chunk_id": chunk.get("id", f"{path}#L{start_l}-{end_l}"),
                "similarity": meta.get("similarity", 0.0),
                "rrf_score": meta.get("rrf_score", 0.0),
                "snippet": text[:150]
            })
            break

        chunk_blocks.append(block)
        sources_summary.append({
            "path": path,
            "start_line": start_l,
            "end_line": end_l,
            "chunk_id": chunk.get("id", f"{path}#L{start_l}-{end_l}"),
            "similarity": meta.get("similarity", 0.0),
            "rrf_score": meta.get("rrf_score", 0.0),
            "snippet": text[:150]
        })

    if chunk_blocks:
        sections.append("### Relevant Code Snippets:\n" + "\n".join(chunk_blocks))

    full_context = "\n\n".join(sections)
    return full_context, sources_summary
