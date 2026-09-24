"""
Tests for follow-up detection rules in llm.py.
Ensures repository-referencing queries (e.g. 'in this repository')
do not trigger query rewriting as follow-ups.
"""

from llm import is_follow_up_query


def test_repo_determiner_not_follow_up():
    """Verify 'this repo/repository/project/codebase' does not trigger follow-up rewriting."""
    history_len = 3

    # Unrelated or standalone queries that mention the repo
    assert not is_follow_up_query("What is the Bitcoin mining algorithm implementation in this repository?", history_len)
    assert not is_follow_up_query("What does this repository do?", history_len)
    assert not is_follow_up_query("How do I install dependencies and run this project locally?", history_len)
    assert not is_follow_up_query("Explain the architecture of this codebase.", history_len)
    assert not is_follow_up_query("What files are in this project?", history_len)
    assert not is_follow_up_query("How does request routing work in this repo?", history_len)


def test_legitimate_follow_ups_detected():
    """Verify true conversational follow-ups are correctly detected."""
    history_len = 2

    assert is_follow_up_query("Explain that in more detail.", history_len)
    assert is_follow_up_query("Explain that code", history_len)
    assert is_follow_up_query("Show those files", history_len)
    assert is_follow_up_query("Show me the exact function where that happens.", history_len)
    assert is_follow_up_query("How does it handle errors?", history_len)
    assert is_follow_up_query("Why?", history_len)
    assert is_follow_up_query("Tell me more about it.", history_len)
    assert is_follow_up_query("What about logging?", history_len)
    assert is_follow_up_query("Elaborate on that please.", history_len)


def test_no_history_never_follow_up():
    """When history is empty, no query is a follow-up."""
    assert not is_follow_up_query("Explain that in more detail.", history_len=0)
    assert not is_follow_up_query("How does it work?", history_len=0)
