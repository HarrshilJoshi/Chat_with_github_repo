"""
Code chunker module.
Implements Python AST-aware semantic chunking and language-agnostic boundary chunking.
Guarantees accurate line ranges, no duplicates, max chunk character limits,
and handles edge cases such as long single-line files.
"""

import ast
import hashlib
import re
from typing import Optional
from config import CHUNK_OVERLAP, CHUNK_SIZE

# Extension to language mapping
EXTENSION_LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".html": "html",
    ".css": "css",
    ".md": "markdown",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".sh": "bash",
    ".sql": "sql"
}


def get_language_from_path(path: str) -> str:
    """Determine the programming or markup language from a file path."""
    for ext, lang in EXTENSION_LANGUAGE_MAP.items():
        if path.lower().endswith(ext):
            return lang
    return "text"


def _split_long_lines_if_needed(lines: list[str], max_chunk_size: int) -> list[tuple[int, str]]:
    """
    Ensure no single line exceeds max_chunk_size (e.g. minified code or huge JSON lines).
    Returns list of (original_line_number, line_content).
    """
    numbered_lines: list[tuple[int, str]] = []
    for line_idx, line in enumerate(lines, start=1):
        if len(line) <= max_chunk_size:
            numbered_lines.append((line_idx, line))
        else:
            # Slice giant lines into pieces of max_chunk_size
            for i in range(0, len(line), max_chunk_size - 100):
                sub_line = line[i : i + max_chunk_size - 100]
                numbered_lines.append((line_idx, sub_line))
    return numbered_lines


def _slice_lines_with_overlap(
    numbered_lines: list[tuple[int, str]],
    path: str,
    language: str,
    max_chunk_size: int,
    overlap: int
) -> list[dict]:
    """
    Slice an array of (line_num, text) into overlapping chunks respecting max character limits.
    Guarantees no duplicate tail chunks.
    """
    chunks: list[dict] = []
    if not numbered_lines:
        return chunks

    current_lines: list[str] = []
    current_start_line = numbered_lines[0][0]
    current_end_line = numbered_lines[0][0]
    current_len = 0
    seen_texts: set[str] = set()

    i = 0
    while i < len(numbered_lines):
        line_num, line_str = numbered_lines[i]
        line_len = len(line_str) + 1  # count newline

        # If adding this line exceeds max chunk size and we already have content
        if current_len + line_len > max_chunk_size and current_lines:
            chunk_body = "\n".join(current_lines).strip()
            if chunk_body:
                text_hash = hashlib.md5(chunk_body.encode("utf-8")).hexdigest()
                if text_hash not in seen_texts:
                    seen_texts.add(text_hash)
                    chunk_id = f"{path}#L{current_start_line}-L{current_end_line}"
                    chunks.append({
                        "id": chunk_id,
                        "text": chunk_body,
                        "metadata": {
                            "path": path,
                            "start_line": current_start_line,
                            "end_line": current_end_line,
                            "language": language,
                            "chunk_id": chunk_id
                        }
                    })

            # Calculate overlap lines to retain for the next chunk
            overlap_lines: list[str] = []
            overlap_len = 0
            overlap_start_line = line_num

            # Walk backward from current chunk to find lines fitting in overlap size
            k = len(current_lines) - 1
            while k >= 0:
                l_str = current_lines[k]
                if overlap_len + len(l_str) <= overlap:
                    overlap_lines.insert(0, l_str)
                    overlap_len += len(l_str) + 1
                    k -= 1
                else:
                    break

            if overlap_lines:
                # Find line number corresponding to start of overlap
                match_idx = max(0, i - len(overlap_lines))
                overlap_start_line = numbered_lines[match_idx][0]
                current_lines = list(overlap_lines)
                current_len = overlap_len
                current_start_line = overlap_start_line
            else:
                current_lines = []
                current_len = 0
                current_start_line = line_num

        # Append current line
        current_lines.append(line_str)
        current_len += line_len
        current_end_line = line_num
        i += 1

    # Final residual chunk
    if current_lines:
        chunk_body = "\n".join(current_lines).strip()
        if chunk_body:
            text_hash = hashlib.md5(chunk_body.encode("utf-8")).hexdigest()
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                chunk_id = f"{path}#L{current_start_line}-L{current_end_line}"
                chunks.append({
                    "id": chunk_id,
                    "text": chunk_body,
                    "metadata": {
                        "path": path,
                        "start_line": current_start_line,
                        "end_line": current_end_line,
                        "language": language,
                        "chunk_id": chunk_id
                    }
                })

    return chunks


def chunk_python_file(
    path: str,
    code: str,
    max_chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> list[dict]:
    """
    Chunk Python source files using the Abstract Syntax Tree (AST).
    Identifies top-level functions, classes, and methods to preserve syntax structure.
    Falls back to generic chunker if the file has syntax errors.
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        # Fall back gracefully to boundary line chunker
        return chunk_generic_file(path, code, "python", max_chunk_size, overlap)

    raw_lines = code.split("\n")
    if not raw_lines or not code.strip():
        return []

    # Collect syntax boundary spans
    spans: list[tuple[int, int, str]] = []  # (start_line, end_line, type)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start_l = node.lineno
            end_l = getattr(node, "end_lineno", node.lineno)
            spans.append((start_l, end_l, type(node).__name__))

    # If file has no functions or classes, use generic chunking
    if not spans:
        return chunk_generic_file(path, code, "python", max_chunk_size, overlap)

    # Sort spans by starting line
    spans.sort(key=lambda s: s[0])

    chunks: list[dict] = []
    seen_hashes: set[str] = set()
    current_line = 1

    for start_l, end_l, _ in spans:
        # 1. Capture any top-level module code preceding this definition (imports, global vars)
        if start_l > current_line:
            pre_lines = [(ln, raw_lines[ln - 1]) for ln in range(current_line, start_l)]
            pre_chunks = _slice_lines_with_overlap(pre_lines, path, "python", max_chunk_size, overlap)
            for c in pre_chunks:
                h = hashlib.md5(c["text"].encode("utf-8")).hexdigest()
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    chunks.append(c)

        # 2. Extract the AST node block
        def_lines = [(ln, raw_lines[ln - 1]) for ln in range(start_l, min(end_l + 1, len(raw_lines) + 1))]
        block_text = "\n".join(t for _, t in def_lines).strip()
        block_len = len(block_text)

        # If the complete function/class fits within chunk limit, keep it intact
        if block_len <= max_chunk_size and block_len > 0:
            h = hashlib.md5(block_text.encode("utf-8")).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                cid = f"{path}#L{start_l}-L{end_l}"
                chunks.append({
                    "id": cid,
                    "text": block_text,
                    "metadata": {
                        "path": path,
                        "start_line": start_l,
                        "end_line": end_l,
                        "language": "python",
                        "chunk_id": cid
                    }
                })
        else:
            # Large function or class: split into sub-chunks with overlap
            sub_chunks = _slice_lines_with_overlap(def_lines, path, "python", max_chunk_size, overlap)
            for c in sub_chunks:
                h = hashlib.md5(c["text"].encode("utf-8")).hexdigest()
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    chunks.append(c)

        current_line = end_l + 1

    # 3. Capture any remaining tail lines after the last AST node
    if current_line <= len(raw_lines):
        tail_lines = [(ln, raw_lines[ln - 1]) for ln in range(current_line, len(raw_lines) + 1)]
        tail_chunks = _slice_lines_with_overlap(tail_lines, path, "python", max_chunk_size, overlap)
        for c in tail_chunks:
            h = hashlib.md5(c["text"].encode("utf-8")).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                chunks.append(c)

    return chunks


def extract_js_ts_spans(code: str) -> list[tuple[int, int, str]]:
    """
    Identify top-level functions, classes, interfaces, and export blocks in JS/TS.
    Returns list of (start_line, end_line, block_type).
    """
    lines = code.split("\n")
    spans = []

    block_start_re = re.compile(
        r"^(?:export\s+(?:default\s+)?)?(?:async\s+)?(?:function\*?|class|interface|type|enum)\b|"
        r"^(?:export\s+)?(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?(?:\([^)]*\)|[a-zA-Z0-9_$]+)\s*=>|"
        r"^(?:describe|test|it)\s*\("
    )

    in_block = False
    start_l = 0
    brace_depth = 0
    saw_open_brace = False
    block_type = ""

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()

        # If not inside a block, check if this line starts one
        if not in_block:
            leading_spaces = len(line) - len(line.lstrip())
            if leading_spaces <= 2 and block_start_re.search(stripped):
                in_block = True
                start_l = idx
                brace_depth = 0
                saw_open_brace = False
                block_type = stripped[:50]

        if in_block:
            # Strip comments and string literals for brace counting
            clean = re.sub(r'(\".*?\"|\'.*?\'|`.*?`)', '""', line)
            clean = re.sub(r'//.*$', '', clean)
            open_count = clean.count("{")
            close_count = clean.count("}")

            if open_count > 0:
                saw_open_brace = True

            brace_depth += open_count - close_count

            # If we saw an open brace and depth has returned to 0 (or below)
            if saw_open_brace and brace_depth <= 0:
                spans.append((start_l, idx, block_type))
                in_block = False
                saw_open_brace = False

    return spans


def chunk_js_ts_file(
    path: str,
    code: str,
    language: str,
    max_chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> list[dict]:
    """
    Chunk JavaScript and TypeScript files using semantic block detection (functions, classes, interfaces).
    Falls back to generic chunker if no semantic blocks are found.
    """
    spans = extract_js_ts_spans(code)
    if not spans:
        return chunk_generic_file(path, code, language, max_chunk_size, overlap)

    raw_lines = code.split("\n")
    chunks: list[dict] = []
    seen_hashes: set[str] = set()
    current_line = 1

    for start_l, end_l, _ in spans:
        # 1. Capture preceding top-level code (imports, exports, comments)
        if start_l > current_line:
            pre_lines = [(ln, raw_lines[ln - 1]) for ln in range(current_line, start_l)]
            if any(t.strip() for _, t in pre_lines):
                pre_chunks = _slice_lines_with_overlap(pre_lines, path, language, max_chunk_size, overlap)
                for c in pre_chunks:
                    h = hashlib.md5(c["text"].encode("utf-8")).hexdigest()
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        chunks.append(c)

        # 2. Extract block text
        block_lines = [(ln, raw_lines[ln - 1]) for ln in range(start_l, min(end_l + 1, len(raw_lines) + 1))]
        block_text = "\n".join(t for _, t in block_lines).strip()
        block_len = len(block_text)

        if block_len <= max_chunk_size and block_len > 0:
            h = hashlib.md5(block_text.encode("utf-8")).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                cid = f"{path}#L{start_l}-L{end_l}"
                chunks.append({
                    "id": cid,
                    "text": block_text,
                    "metadata": {
                        "path": path,
                        "start_line": start_l,
                        "end_line": end_l,
                        "language": language,
                        "chunk_id": cid
                    }
                })
        else:
            # Block larger than max_chunk_size: slice with overlap
            sub_chunks = _slice_lines_with_overlap(block_lines, path, language, max_chunk_size, overlap)
            for c in sub_chunks:
                h = hashlib.md5(c["text"].encode("utf-8")).hexdigest()
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    chunks.append(c)

        current_line = end_l + 1

    # 3. Residual lines after last block
    if current_line <= len(raw_lines):
        tail_lines = [(ln, raw_lines[ln - 1]) for ln in range(current_line, len(raw_lines) + 1)]
        if any(t.strip() for _, t in tail_lines):
            tail_chunks = _slice_lines_with_overlap(tail_lines, path, language, max_chunk_size, overlap)
            for c in tail_chunks:
                h = hashlib.md5(c["text"].encode("utf-8")).hexdigest()
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    chunks.append(c)

    return chunks


def chunk_generic_file(
    path: str,
    text: str,
    language: str,
    max_chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> list[dict]:
    """
    Chunk generic source files (JS, TS, Go, Rust, Java, Markdown, etc.).
    Splits on paragraph boundaries or logical lines while enforcing max character limits.
    """
    lines = text.split("\n")
    if not lines or not text.strip():
        return []

    numbered_lines = _split_long_lines_if_needed(lines, max_chunk_size)
    return _slice_lines_with_overlap(numbered_lines, path, language, max_chunk_size, overlap)


def chunk_files(
    files_content: list[dict],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> list[dict]:
    """
    Process a list of downloaded repository files into structured chunks.
    Automatically chooses AST parser for Python, semantic block chunker for JS/TS,
    and line boundary chunker for other languages.
    """
    all_chunks: list[dict] = []

    for file_info in files_content:
        path = file_info["path"]
        content = file_info["content"]
        language = get_language_from_path(path)

        if language == "python":
            file_chunks = chunk_python_file(path, content, max_chunk_size=chunk_size, overlap=overlap)
        elif language in ("javascript", "typescript"):
            file_chunks = chunk_js_ts_file(path, content, language, max_chunk_size=chunk_size, overlap=overlap)
        else:
            file_chunks = chunk_generic_file(path, content, language, max_chunk_size=chunk_size, overlap=overlap)

        all_chunks.extend(file_chunks)

    return all_chunks
