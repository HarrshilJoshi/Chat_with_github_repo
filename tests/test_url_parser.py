"""
Tests for GitHub URL and identifier parsing.
All edge cases tested with 0 network calls.
"""

import pytest
from github_loader import parse_github_url


def test_standard_https_url():
    owner, repo, branch = parse_github_url("https://github.com/streamlit/streamlit")
    assert owner == "streamlit"
    assert repo == "streamlit"
    assert branch is None


def test_url_with_git_extension():
    owner, repo, branch = parse_github_url("https://github.com/torvalds/linux.git")
    assert owner == "torvalds"
    assert repo == "linux"
    assert branch is None


def test_url_with_http_and_www():
    owner, repo, branch = parse_github_url("http://www.github.com/psf/requests")
    assert owner == "psf"
    assert repo == "requests"
    assert branch is None


def test_url_with_branch_tree():
    owner, repo, branch = parse_github_url("https://github.com/facebook/react/tree/dev")
    assert owner == "facebook"
    assert repo == "react"
    assert branch == "dev"


def test_url_with_branch_and_subpath():
    owner, repo, branch = parse_github_url("https://github.com/pallets/flask/tree/main/src/flask")
    assert owner == "pallets"
    assert repo == "flask"
    assert branch == "main"


def test_bare_owner_repo():
    owner, repo, branch = parse_github_url("anthonysandesh/Chat-with-GitHub")
    assert owner == "anthonysandesh"
    assert repo == "Chat-with-GitHub"
    assert branch is None


def test_whitespace_and_trailing_slash():
    owner, repo, branch = parse_github_url("   https://github.com/fastapi/fastapi/   ")
    assert owner == "fastapi"
    assert repo == "fastapi"
    assert branch is None


def test_invalid_urls():
    with pytest.raises(ValueError):
        parse_github_url("")

    with pytest.raises(ValueError):
        parse_github_url("invalid_just_a_word")

    with pytest.raises(ValueError):
        parse_github_url("https://gitlab.com/owner/repo")
