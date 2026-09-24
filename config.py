"""
Central configuration module for Chat with GitHub Repository.
Loads environment variables, defines model settings, file processing constraints,
and vector database parameters.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file in the workspace
load_dotenv()

# --- API Keys & Credentials ---
# Groq is the primary LLM provider (Fast inference, required)
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip()

# Gemini is an optional fallback LLM (used only if Groq fails)
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")).strip()

# GitHub Personal Access Token (Optional: increases rate limit from 60 to 5000 req/hr)
GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "").strip()

# --- Model Selection ---
# Primary Groq model (configurable via GROQ_MODEL in .env, defaults to openai/gpt-oss-120b)
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()

# Fallback Google Gemini model (configurable via GEMINI_FALLBACK_MODEL in .env, defaults to gemini-3-flash-preview)
# Note: gemini-3-flash-preview is a Preview release model; gemini-2.0-flash has been retired by Google.
GEMINI_FALLBACK_MODEL: str = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3-flash-preview").strip()

# Embedder identifier stored in collection names and cache keys
# Uses Chroma's built-in ONNX runtime (all-MiniLM-L6-v2), 100% free and local
EMBEDDER_NAME: str = "chroma-onnx-minilm"

# --- LLM Generation Parameters ---
DEFAULT_TEMPERATURE: float = 0.2          # Temperature: 0.2 for deterministic, factual code analysis
DEFAULT_REASONING_EFFORT: str = "low"     # Low reasoning effort for openai/gpt-oss-120b to conserve tokens
DEFAULT_TOP_K: int = 5                    # Number of most relevant code chunks to supply
DEFAULT_SIMILARITY_THRESHOLD: float = 0.35 # Minimum cosine similarity (1.0 - distance) to accept a chunk
MAX_COMPLETION_TOKENS: int = 2048         # Tokens for reasoning + final answer (reasoning tokens count toward max tokens)
MAX_CONTEXT_CHARS: int = 6000             # Context budget: ~6000 chars (~1500 tokens) to prevent high cost
MAX_CHAT_HISTORY_TURNS: int = 4           # Maximum past conversation turns included in LLM context

# --- GitHub Ingestion & Chunking Limits ---
MAX_FILES_DEFAULT: int = 80               # Default maximum files to ingest from repository
MAX_FILE_SIZE_BYTES: int = 200 * 1024     # 200 KB per file cap to skip massive assets/dumps
CHUNK_SIZE: int = 1000                    # Target character count per code chunk
CHUNK_OVERLAP: int = 150                  # Character overlap between adjacent chunks to preserve context
DOWNLOAD_WORKERS: int = 8                 # Number of parallel worker threads for file downloads
REQUEST_TIMEOUT_SECONDS: int = 10         # HTTP timeout for GitHub API and raw content requests

# --- Storage Settings ---
from pathlib import Path
BASE_DIR: Path = Path(__file__).resolve().parent
CHROMA_PERSIST_DIR: str = str(BASE_DIR / "chroma_db")   # Absolute path to local ChromaDB storage


@dataclass(frozen=True)
class AppConfig:
    """Runtime configuration container allowing user overrides from Streamlit sidebar."""
    temperature: float = DEFAULT_TEMPERATURE
    top_k: int = DEFAULT_TOP_K
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    max_files: int = MAX_FILES_DEFAULT
    groq_model: str = GROQ_MODEL
    gemini_model: str = GEMINI_FALLBACK_MODEL
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
