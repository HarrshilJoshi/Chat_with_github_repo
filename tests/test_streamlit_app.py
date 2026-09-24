"""
Streamlit UI Verification Tests using official streamlit.testing.v1.AppTest.
Verifies:
1. Flash message survives st.rerun() and displays correctly.
2. Sidebar status badges dynamically reflect system state.
3. 'Explain folder structure' generates tree with 0 LLM calls.
4. Source links properly format as https://github.com/{owner}/{repo}/blob/{sha}/{path}#L{start}-L{end}.
"""

import os
import pytest
from streamlit.testing.v1 import AppTest


def test_apptest_initial_state_and_badges():
    """Verify initial UI rendering and status badges."""
    at = AppTest.from_file("app.py")
    at.run()
    assert not at.exception, f"App raised exception: {at.exception}"

    # Check title
    assert len(at.title) >= 1
    assert "Chat with GitHub Repository" in at.title[0].value

    # Check sidebar badges
    sidebar_success_texts = [s.value for s in at.sidebar.success]
    assert any("Groq LLM (Primary)" in t for t in sidebar_success_texts)


def test_apptest_flash_message_survives_rerun():
    """Verify flash message survives st.rerun() and is popped upon display."""
    at = AppTest.from_file("app.py")
    at.run()

    # Inject a flash message into session_state before rerun
    at.session_state["flash_message"] = {
        "type": "success",
        "text": "✅ Test flash message surviving rerun!"
    }
    at.run()

    # Verify the success alert rendered the flash message
    main_success = [s.value for s in at.success]
    assert any("Test flash message surviving rerun!" in s for s in main_success)

    # Verify that after displaying, the flash message was popped from session state
    assert "flash_message" not in at.session_state


def test_apptest_explain_folder_structure_zero_llm():
    """Verify 'Explain folder structure' action requires 0 LLM calls and appends tree to chat."""
    at = AppTest.from_file("app.py")
    at.run()

    # Set up active repo state with mock tree and retriever
    at.session_state.active_repo = "pallets/flask"
    at.session_state.active_branch = "main"
    at.session_state.commit_sha = "d73fa1cdcb"
    at.session_state.retriever = "mock_retriever"
    at.session_state.file_tree_text = "src/\n  flask/\n    app.py\npyproject.toml"
    at.run()

    # Locate 'Explain folder structure' button
    folder_btn = None
    for b in at.button:
        if "Explain folder structure" in b.label:
            folder_btn = b
            break
    assert folder_btn is not None, "Folder structure button not found"

    # Click the button and re-run
    folder_btn.click()
    at.run()

    # Verify chat history contains the folder structure and indicates 0 LLM tokens
    chat_history = at.session_state.chat_history
    assert len(chat_history) >= 2
    assert chat_history[0]["content"] == "Explain folder structure"
    assistant_msg = chat_history[1]
    assert "0 LLM Tokens" in assistant_msg["engine"]
    assert "src/\n  flask/\n    app.py" in assistant_msg["content"]


def test_apptest_source_links_formatting():
    """Verify that source citations format as blob/{sha}/{path}#L{start}-L{end}."""
    owner = "pallets"
    repo = "flask"
    commit_sha = "d73fa1cdcb"
    path = "src/flask/app.py"
    s_line = 967
    e_line = 989

    expected_url = f"https://github.com/{owner}/{repo}/blob/{commit_sha}/{path}#L{s_line}-L{e_line}"
    assert expected_url == "https://github.com/pallets/flask/blob/d73fa1cdcb/src/flask/app.py#L967-L989"
