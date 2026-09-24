"""
Vector store module.
Manages persistent local ChromaDB storage, local ONNX embeddings,
collection naming with embedder tags, batch insertion, and collection metadata caching.
"""

import hashlib
import re
from typing import Optional
import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from config import (
    CHROMA_PERSIST_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDER_NAME,
)

# Global client cache to avoid re-opening SQLite database repeatedly
_CHROMA_CLIENT: Optional[chromadb.PersistentClient] = None
_DEFAULT_EMBEDDING_FUNCTION: Optional[DefaultEmbeddingFunction] = None


def get_chroma_client() -> chromadb.PersistentClient:
    """Get or initialize the persistent ChromaDB client."""
    global _CHROMA_CLIENT
    if _CHROMA_CLIENT is None:
        _CHROMA_CLIENT = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return _CHROMA_CLIENT


def get_embedding_function() -> DefaultEmbeddingFunction:
    """Get or initialize Chroma's built-in ONNX runtime embedding function (all-MiniLM-L6-v2)."""
    global _DEFAULT_EMBEDDING_FUNCTION
    if _DEFAULT_EMBEDDING_FUNCTION is None:
        _DEFAULT_EMBEDDING_FUNCTION = DefaultEmbeddingFunction()
    return _DEFAULT_EMBEDDING_FUNCTION


def build_collection_name(
    owner: str,
    repo: str,
    branch: str,
    commit_sha: str,
    embedder_name: str = EMBEDDER_NAME,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> str:
    """
    Generate a valid, deterministic ChromaDB collection name including the embedder and chunk settings.
    Chroma requirements: 3-63 characters, regex ^[a-zA-Z0-9][a-zA-Z0-9._-]{1,61}[a-zA-Z0-9]$
    """
    # Create a unique 10-char hash fingerprint of all variables to prevent collisions
    cache_key = f"{owner}/{repo}/{branch}/{commit_sha}/{embedder_name}/{chunk_size}_{chunk_overlap}"
    key_hash = hashlib.md5(cache_key.encode("utf-8")).hexdigest()[:10]

    # Create safe alphanumeric prefixes
    safe_owner = re.sub(r"[^a-zA-Z0-9]", "", owner)[:10]
    safe_repo = re.sub(r"[^a-zA-Z0-9]", "", repo)[:15]
    short_sha = re.sub(r"[^a-zA-Z0-9]", "", commit_sha)[:7]

    # Format: repo_{owner}_{repo}_{sha}_{hash}
    name = f"r_{safe_owner}_{safe_repo}_{short_sha}_{key_hash}"
    # Ensure it starts and ends with alphanumeric and is within 63 chars
    name = name[:63].strip("._-")
    return name


def check_collection_cache(
    owner: str,
    repo: str,
    branch: str,
    commit_sha: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> tuple[bool, Optional[Collection], dict]:
    """
    Check if a collection for this exact repo state and configuration already exists in ChromaDB.
    Returns (is_cached, collection, cached_meta).
    """
    client = get_chroma_client()
    collection_name = build_collection_name(
        owner, repo, branch, commit_sha, EMBEDDER_NAME, chunk_size, chunk_overlap
    )

    try:
        col = client.get_collection(
            name=collection_name,
            embedding_function=get_embedding_function()
        )
        if col.count() > 0:
            meta = col.metadata or {}
            return True, col, meta
    except Exception:
        pass

    return False, None, {}


def find_cached_collection_for_repo(owner: str, repo: str) -> Optional[Collection]:
    """Find any locally cached ChromaDB collection for the given owner/repo."""
    client = get_chroma_client()
    safe_owner = re.sub(r"[^a-zA-Z0-9]", "", owner)[:10]
    safe_repo = re.sub(r"[^a-zA-Z0-9]", "", repo)[:15]
    prefix = f"r_{safe_owner}_{safe_repo}_"
    try:
        collections = client.list_collections()
        for col in collections:
            col_name = getattr(col, "name", str(col))
            if col_name.startswith(prefix):
                full_col = client.get_collection(name=col_name, embedding_function=get_embedding_function())
                if full_col.count() > 0:
                    return full_col
    except Exception:
        pass
    return None


def store_chunks_in_chroma(
    owner: str,
    repo: str,
    branch: str,
    commit_sha: str,
    chunks: list[dict],
    readme_text: str = "",
    file_tree_text: str = "",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    force_reindex: bool = False
) -> tuple[Collection, bool]:
    """
    Persist code chunks and repository metadata into a dedicated ChromaDB collection.
    If already cached and force_reindex is False, re-uses the existing collection.
    Inserts chunks in batches to optimize memory and ONNX embedding efficiency.
    Returns (collection, was_cached).
    """
    client = get_chroma_client()
    embed_fn = get_embedding_function()
    collection_name = build_collection_name(
        owner, repo, branch, commit_sha, EMBEDDER_NAME, chunk_size, chunk_overlap
    )

    # 1. Handle re-index request
    if force_reindex:
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass

    # 2. Check existing cache
    try:
        existing_col = client.get_collection(name=collection_name, embedding_function=embed_fn)
        if existing_col.count() > 0 and not force_reindex:
            return existing_col, True
        # If empty or reindexing, delete old shell
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    # 3. Create fresh collection configured with cosine distance
    col_metadata = {
        "hnsw:space": "cosine",
        "repo": f"{owner}/{repo}",
        "branch": branch,
        "commit_sha": commit_sha,
        "embedder": EMBEDDER_NAME,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "readme": readme_text[:5000],          # Store excerpt in metadata for 0-download cache recovery
        "file_tree": file_tree_text[:5000],    # Store file tree for 0-download cache recovery
        "total_chunks": len(chunks)
    }

    collection = client.create_collection(
        name=collection_name,
        embedding_function=embed_fn,
        metadata=col_metadata
    )

    # 4. Batch insert documents into Chroma
    BATCH_SIZE = 64
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        ids = [c["id"] for c in batch]
        documents = [c["text"] for c in batch]
        metadatas = [c["metadata"] for c in batch]

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    return collection, False


def load_cached_chunks_from_chroma(collection: Collection) -> list[dict]:
    """
    Retrieve all stored chunks and metadata from ChromaDB for a cached collection.
    Allows instant rebuilding of the BM25 index in memory without re-downloading files.
    """
    result = collection.get(include=["documents", "metadatas"])
    chunks: list[dict] = []
    docs = result.get("documents", [])
    metas = result.get("metadatas", [])
    ids = result.get("ids", [])

    for i in range(len(docs)):
        chunks.append({
            "id": ids[i],
            "text": docs[i],
            "metadata": metas[i] if i < len(metas) else {}
        })

    return chunks
