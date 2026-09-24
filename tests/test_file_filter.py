"""
Tests for file prioritization, exclusion rules, and size caps.
Pure in-memory unit tests with 0 network calls.
"""

from github_loader import prioritize_and_filter_files


def test_readme_and_configs_prioritized_over_generic_files():
    mock_tree = [
        {"path": "tests/test_misc.py", "type": "blob", "size": 1000},
        {"path": "README.md", "type": "blob", "size": 1200},
        {"path": "requirements.txt", "type": "blob", "size": 500},
        {"path": "src/utils.py", "type": "blob", "size": 800},
        {"path": "app.py", "type": "blob", "size": 900},
    ]

    selected, skipped = prioritize_and_filter_files(mock_tree, max_files=3)
    selected_paths = [f["path"] for f in selected]

    # Tier 1 (README.md, app.py, requirements.txt) must occupy the top 3 slots
    assert "README.md" in selected_paths
    assert "requirements.txt" in selected_paths
    assert "app.py" in selected_paths
    assert "tests/test_misc.py" not in selected_paths
    assert skipped == 2


def test_lockfiles_and_minified_files_excluded():
    mock_tree = [
        {"path": "package-lock.json", "type": "blob", "size": 50000},
        {"path": "yarn.lock", "type": "blob", "size": 30000},
        {"path": "poetry.lock", "type": "blob", "size": 25000},
        {"path": "dist/bundle.min.js", "type": "blob", "size": 80000},
        {"path": "src/index.js", "type": "blob", "size": 2000},
    ]

    selected, skipped = prioritize_and_filter_files(mock_tree, max_files=10)
    selected_paths = [f["path"] for f in selected]

    assert selected_paths == ["src/index.js"]
    assert skipped == 0


def test_files_exceeding_size_limit_skipped():
    mock_tree = [
        {"path": "small.py", "type": "blob", "size": 5000},
        {"path": "huge_data.json", "type": "blob", "size": 500 * 1024},  # 500 KB > 200 KB limit
    ]

    selected, _ = prioritize_and_filter_files(mock_tree, max_files=10)
    selected_paths = [f["path"] for f in selected]

    assert "small.py" in selected_paths
    assert "huge_data.json" not in selected_paths


def test_ignored_directories_skipped():
    mock_tree = [
        {"path": "node_modules/express/index.js", "type": "blob", "size": 2000},
        {"path": ".git/config", "type": "blob", "size": 500},
        {"path": "__pycache__/module.cpython-310.pyc", "type": "blob", "size": 1200},
        {"path": "src/main.py", "type": "blob", "size": 1500},
    ]

    selected, _ = prioritize_and_filter_files(mock_tree, max_files=10)
    selected_paths = [f["path"] for f in selected]

    assert selected_paths == ["src/main.py"]
