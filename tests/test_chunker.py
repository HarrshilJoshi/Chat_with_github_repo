"""
Tests for AST and boundary chunking, deduplication, character limits, and metadata tracking.
Pure in-memory unit tests with 0 network calls.
"""

from chunker import chunk_files, chunk_generic_file, chunk_js_ts_file, chunk_python_file


def test_ast_python_chunking_preserves_function_boundaries():
    sample_code = (
        "import os\n\n"
        "def calculate_total(items):\n"
        "    total = 0\n"
        "    for x in items:\n"
        "        total += x\n"
        "    return total\n\n"
        "class OrderProcessor:\n"
        "    def process(self, order):\n"
        "        return True\n"
    )

    chunks = chunk_python_file("orders.py", sample_code, max_chunk_size=500, overlap=50)

    assert len(chunks) >= 2
    # Ensure metadata exists on every chunk
    for c in chunks:
        assert "path" in c["metadata"]
        assert "start_line" in c["metadata"]
        assert "end_line" in c["metadata"]
        assert c["metadata"]["language"] == "python"
        assert c["metadata"]["path"] == "orders.py"
        assert c["metadata"]["start_line"] <= c["metadata"]["end_line"]

    # Verify functions are not split in half arbitrarily
    calc_chunk = next((c for c in chunks if "def calculate_total" in c["text"]), None)
    assert calc_chunk is not None
    assert "return total" in calc_chunk["text"]


def test_no_duplicate_chunks_from_residual_lines():
    # Test repetitive lines that could cause tail overlap duplication in buggy chunkers
    lines = [f"line_{i} = {i}" for i in range(40)]
    code = "\n".join(lines)

    chunks = chunk_generic_file("data.py", code, "python", max_chunk_size=150, overlap=30)
    seen_texts = set()

    for c in chunks:
        assert c["text"] not in seen_texts, f"Duplicate chunk found: {c['text']}"
        seen_texts.add(c["text"])


def test_max_chunk_size_enforced_even_on_huge_lines():
    # Single very long line (e.g. 3000 chars)
    giant_line = "const payload = {" + '"key": "val",' * 200 + "};"
    chunks = chunk_generic_file("bundle.js", giant_line, "javascript", max_chunk_size=400, overlap=50)

    assert len(chunks) > 1
    for c in chunks:
        assert len(c["text"]) <= 450  # Enforces reasonable chunk bounds
        assert c["metadata"]["language"] == "javascript"


def test_chunk_files_pipeline():
    mock_files = [
        {
            "path": "main.py",
            "content": "def run():\n    print('Hello World')\n"
        },
        {
            "path": "config.json",
            "content": '{\n  "version": "1.0.0"\n}\n'
        }
    ]

    all_chunks = chunk_files(mock_files, chunk_size=300, overlap=50)
    assert len(all_chunks) >= 2

    paths = {c["metadata"]["path"] for c in all_chunks}
    assert "main.py" in paths
    assert "config.json" in paths


def test_js_ts_semantic_chunking_preserves_function_boundaries():
    sample_js = (
        "const cookie = require('cookie');\n\n"
        "function parseCookies(str, options) {\n"
        "    if (!str) return {};\n"
        "    return cookie.parse(str, options);\n"
        "}\n\n"
        "const formatCookie = (name, val) => {\n"
        "    return `${name}=${val}`;\n"
        "};\n"
    )

    chunks = chunk_js_ts_file("cookie.js", sample_js, "javascript", max_chunk_size=300, overlap=50)

    assert len(chunks) >= 2
    for c in chunks:
        assert c["metadata"]["path"] == "cookie.js"
        assert c["metadata"]["language"] == "javascript"
        assert c["metadata"]["start_line"] <= c["metadata"]["end_line"]

    parse_chunk = next((c for c in chunks if "function parseCookies" in c["text"]), None)
    assert parse_chunk is not None
    assert "return cookie.parse" in parse_chunk["text"]

    format_chunk = next((c for c in chunks if "formatCookie" in c["text"]), None)
    assert format_chunk is not None
    assert "name}=${val}" in format_chunk["text"]
