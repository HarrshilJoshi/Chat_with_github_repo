"""
Streamlit Web Application for Chat with GitHub Repository.
Provides repository loading, branch selection, persistent ChromaDB caching,
hybrid search retrieval, real-time streaming LLM answers, and clickable GitHub source citations.
"""

import os
import streamlit as st

from config import (
    CHROMA_PERSIST_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    EMBEDDER_NAME,
    GEMINI_API_KEY,
    GEMINI_FALLBACK_MODEL,
    GITHUB_TOKEN,
    GROQ_API_KEY,
    GROQ_MODEL,
    MAX_FILES_DEFAULT,
)
from github_loader import (
    build_ascii_file_tree,
    fetch_branch_commit_sha,
    fetch_files_parallel,
    fetch_repo_branches,
    fetch_repo_details,
    get_repo_tree,
    parse_github_url,
    prioritize_and_filter_files,
)
from chunker import chunk_files, get_language_from_path
from vector_store import (
    check_collection_cache,
    find_cached_collection_for_repo,
    load_cached_chunks_from_chroma,
    store_chunks_in_chroma,
)
from retriever import HybridRetriever, build_prompt_context
from llm import generate_answer_stream, rewrite_query_if_needed, is_follow_up_query, is_same_topic


def _render_sidebar(retriever_ready: bool, chunk_count: int, repo_identifier: str, commit_sha: str) -> dict:
    """Render the sidebar with real status badges, settings, and token metrics."""
    with st.sidebar:
        st.header("⚙️ System Status")

        # 1. Groq Status Badge
        if GROQ_API_KEY:
            st.success(f"⚡ **Groq LLM (Primary)**: Connected\n`{GROQ_MODEL}`")
        else:
            st.error("⚡ **Groq LLM**: Missing `GROQ_API_KEY` in `.env`")

        # 2. Gemini Buffer Badge
        if GEMINI_API_KEY:
            st.success(f"🛡️ **Gemini (Fallback)**: Ready\n`{GEMINI_FALLBACK_MODEL}`")
        else:
            st.caption("🛡️ **Gemini Buffer**: Optional fallback (not configured)")

        # 3. Vector Embedder Badge
        st.info(f"🧠 **Embeddings**: Local ONNX MiniLM\n`{EMBEDDER_NAME}` (100% Free & Local)")

        # 4. ChromaDB Badge
        if retriever_ready:
            st.success(f"🗄️ **ChromaDB**: Active ({chunk_count} chunks indexed)")
        else:
            st.caption(f"🗄️ **ChromaDB**: Persistent at `{CHROMA_PERSIST_DIR}`")

        # 5. GitHub API Token Badge
        if GITHUB_TOKEN:
            st.success("🔑 **GitHub Token**: Authenticated (5,000 req/hr)")
        else:
            st.caption("🔑 **GitHub Token**: Public API mode (60 req/hr)")

        # 6. Runtime Configuration Section
        st.markdown("---")
        with st.expander("🛠️ Advanced Settings", expanded=False):
            temperature = st.slider(
                "Temperature", min_value=0.0, max_value=1.0, value=DEFAULT_TEMPERATURE, step=0.05,
                help="Lower temperature is more factual and deterministic for code."
            )
            top_k = st.slider(
                "Top Chunks (k)", min_value=2, max_value=10, value=DEFAULT_TOP_K, step=1,
                help="Number of code chunks to retrieve per question."
            )
            similarity_threshold = st.slider(
                "Similarity Threshold", min_value=0.10, max_value=0.70, value=DEFAULT_SIMILARITY_THRESHOLD, step=0.05,
                help="Cosine similarity cutoff to reject irrelevant code matches."
            )
            max_files = st.number_input(
                "Max Files to Ingest", min_value=10, max_value=200, value=MAX_FILES_DEFAULT, step=10,
                help="Limits repository file count to keep processing fast and focused."
            )

        # 7. Token Usage Transparency
        last_chars = st.session_state.get("last_context_chars", 0)
        if last_chars > 0:
            st.markdown("---")
            st.caption(f"📊 **Last Query Context**: ~{last_chars:,} chars (~{last_chars // 4:,} tokens)")

        # 8. Active Repository Information
        if repo_identifier:
            st.markdown("---")
            st.subheader("📁 Active Repository")
            st.code(f"{repo_identifier}\nSHA: {commit_sha[:8]}", language="text")
            if st.button("🔄 Clear / Switch Repository", use_container_width=True):
                st.session_state.clear()
                st.rerun()

    return {
        "temperature": temperature,
        "top_k": top_k,
        "similarity_threshold": similarity_threshold,
        "max_files": max_files
    }


def _index_repository(
    owner: str,
    repo: str,
    branch: str,
    max_files: int,
    force_reindex: bool = False
) -> None:
    """Fetch, chunk, and index repository files into ChromaDB with caching support."""
    repo_identifier = f"{owner}/{repo}"
    
    with st.status(f"Ingesting repository `{repo_identifier}`...", expanded=True) as status_box:
        # Step 1: Discover Commit SHA for exact version caching
        st.write("🔍 Inspecting repository and commit hash...")
        commit_sha = fetch_branch_commit_sha(owner, repo, branch, GITHUB_TOKEN)

        # Step 2: Check persistent cache in ChromaDB
        is_cached, cached_col, cached_meta = check_collection_cache(
            owner, repo, branch, commit_sha, CHUNK_SIZE, CHUNK_OVERLAP
        )

        if is_cached and not force_reindex:
            st.write(f"⚡ Found cached index in ChromaDB (Commit: `{commit_sha[:8]}`). Loading without re-downloading...")
            stored_chunks = load_cached_chunks_from_chroma(cached_col)
            readme_text = cached_meta.get("readme", "")
            file_tree_text = cached_meta.get("file_tree", "")

            # Rebuild in-memory BM25 index and retriever
            retriever = HybridRetriever(cached_col, stored_chunks)

            # Store in session state
            st.session_state.active_repo = repo_identifier
            st.session_state.active_branch = branch
            st.session_state.commit_sha = commit_sha
            st.session_state.retriever = retriever
            st.session_state.readme_text = readme_text
            st.session_state.file_tree_text = file_tree_text
            st.session_state.chunk_count = len(stored_chunks)
            st.session_state.chat_history = []
            st.session_state.flash_message = {
                "type": "success",
                "text": f"✅ Loaded **{len(stored_chunks)} chunks** for `{repo_identifier}` from local ChromaDB cache!"
            }
            status_box.update(label="Repository loaded from cache!", state="complete", expanded=False)
            return

        # Step 3: Fetch recursive repository tree
        st.write("🌳 Fetching recursive repository file tree...")
        tree_items, is_truncated = get_repo_tree(owner, repo, commit_sha, GITHUB_TOKEN)
        if is_truncated:
            st.warning("⚠️ GitHub file tree response was truncated due to repository size. Key files prioritized.")

        # Step 4: Prioritize and filter files
        st.write("🎯 Prioritizing key files (README, configs, entry points, src/)...")
        selected_items, skipped_count = prioritize_and_filter_files(tree_items, max_files=max_files)
        if not selected_items:
            raise ValueError("No valid code or documentation files found in this repository.")

        # Step 5: Parallel download files with progress bar
        progress_bar = st.progress(0.0, text="Starting parallel download...")
        def progress_cb(completed: int, total: int, path: str):
            progress_bar.progress(completed / total, text=f"Downloading ({completed}/{total}): `{path}`")

        downloaded_files = fetch_files_parallel(
            owner, repo, commit_sha, selected_items, GITHUB_TOKEN, progress_callback=progress_cb
        )
        progress_bar.empty()

        # Step 6: Extract README and build compact ASCII file tree
        readme_text = ""
        for f in downloaded_files:
            if f["path"].lower().split("/")[-1].startswith("readme"):
                readme_text = f["content"]
                break

        indexed_paths = [f["path"] for f in downloaded_files]
        file_tree_text = build_ascii_file_tree(indexed_paths)

        # Step 7: AST & boundary chunking
        st.write("🧩 Chunking files using AST syntax analysis...")
        chunks = chunk_files(downloaded_files, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        if not chunks:
            raise ValueError("Failed to create code chunks from downloaded files.")

        # Step 8: Store in persistent ChromaDB
        st.write(f"🧠 Generating local ONNX embeddings for {len(chunks)} chunks and saving to ChromaDB...")
        collection, _ = store_chunks_in_chroma(
            owner, repo, branch, commit_sha, chunks,
            readme_text=readme_text,
            file_tree_text=file_tree_text,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            force_reindex=force_reindex
        )

        # Step 9: Initialize Hybrid Retriever
        retriever = HybridRetriever(collection, chunks)

        # Update session state
        st.session_state.active_repo = repo_identifier
        st.session_state.active_branch = branch
        st.session_state.commit_sha = commit_sha
        st.session_state.retriever = retriever
        st.session_state.readme_text = readme_text
        st.session_state.file_tree_text = file_tree_text
        st.session_state.chunk_count = len(chunks)
        st.session_state.chat_history = []
        
        skip_msg = f" (skipped {skipped_count} lower-priority files)" if skipped_count > 0 else ""
        st.session_state.flash_message = {
            "type": "success",
            "text": f"✅ Indexed **{len(downloaded_files)} files** ({len(chunks)} chunks) from `{repo_identifier}` into ChromaDB!{skip_msg}"
        }
        status_box.update(label="Indexing complete!", state="complete", expanded=False)


def _handle_assistant_answer(
    user_query: str,
    config_dict: dict
) -> None:
    """Execute hybrid retrieval and stream response with clickable GitHub sources."""
    retriever: HybridRetriever = st.session_state.retriever
    owner, repo = st.session_state.active_repo.split("/", 1)
    commit_sha = st.session_state.commit_sha

    # 1. Determine if this is a follow-up query
    is_follow_up = is_follow_up_query(user_query, len(st.session_state.chat_history))

    # 2. Query Rewriting (only if follow-up and history exists)
    standalone_query = rewrite_query_if_needed(
        user_query, st.session_state.chat_history, GROQ_API_KEY
    )

    # 3. Hybrid Retrieval with Cosine Similarity Pre-filtering
    retrieved_chunks = retriever.retrieve(
        standalone_query,
        top_k=config_dict["top_k"],
        similarity_threshold=config_dict["similarity_threshold"]
    )

    # 4. Context Grounding for Follow-up Queries:
    # If the user asks a follow-up on the SAME topic ("explain that in more detail", "where does that happen?"),
    # pull forward the referenced source chunks from the immediately preceding assistant answer
    # (limited to at most 2 chunks) so that the exact code lines discussed remain in context.
    # If the user switches topics (e.g. "What about the CLI?"), do NOT carry forward old routing chunks.
    if is_follow_up and st.session_state.chat_history:
        prev_user_q = ""
        prev_sources = []
        for msg in reversed(st.session_state.chat_history):
            if not prev_sources and msg.get("role") == "assistant" and msg.get("sources"):
                prev_sources = msg["sources"]
            if not prev_user_q and msg.get("role") == "user":
                prev_user_q = msg.get("content", "")
            if prev_sources and prev_user_q:
                break

        same_topic = is_same_topic(user_query, prev_user_q)
        if same_topic and prev_sources:
            existing_ids = {c["id"] for c in retrieved_chunks}
            carried_chunks = []
            for s in prev_sources:
                cid = s.get("chunk_id")
                if cid and cid in retriever.chunk_by_id and cid not in existing_ids:
                    c_data = retriever.chunk_by_id[cid]
                    carried_chunks.append({
                        "id": cid,
                        "text": c_data["text"],
                        "metadata": dict(c_data["metadata"])
                    })
            if carried_chunks:
                # Limit carried-forward chunks to at most 2 to prevent crowding out new relevant chunks
                carried_chunks = carried_chunks[:2]
                retrieved_chunks = carried_chunks + retrieved_chunks
                retrieved_chunks = retrieved_chunks[:config_dict["top_k"] + len(carried_chunks)]

    # 5. Honest failure if no content meets relevance standards
    if not retrieved_chunks:
        answer_text = (
            "⚠️ **No relevant code or documentation was found** in the indexed files of this repository "
            f"matching your question: *\"{user_query}\"*\n\n"
            "Try rephrasing your question with specific file names, functions, or concepts."
        )
        st.markdown(answer_text)
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer_text,
            "engine": "System Guard",
            "sources": []
        })
        return

    # 4. Assemble bounded prompt context (~6000 chars)
    prompt_context, sources = build_prompt_context(
        retrieved_chunks,
        readme_text=st.session_state.readme_text,
        file_tree_text=st.session_state.file_tree_text
    )

    # 5. Stream LLM answer
    try:
        stream_gen, engine_name, approx_chars = generate_answer_stream(
            query=user_query,
            context=prompt_context,
            chat_history=st.session_state.chat_history,
            repo_identifier=st.session_state.active_repo,
            groq_api_key=GROQ_API_KEY,
            gemini_api_key=GEMINI_API_KEY,
            temperature=config_dict["temperature"]
        )

        st.session_state.last_context_chars = approx_chars
        full_answer = st.write_stream(stream_gen)
        st.caption(f"⚡ Generated via {engine_name}")

        # 6. Render clickable source links
        if sources:
            with st.expander(f"🔍 Referenced Sources ({len(sources)} chunks)", expanded=False):
                for src in sources:
                    path = src["path"]
                    s_line = src["start_line"]
                    e_line = src["end_line"]
                    github_url = f"https://github.com/{owner}/{repo}/blob/{commit_sha}/{path}#L{s_line}-L{e_line}"
                    st.markdown(
                        f"• [`{path}` (Lines {s_line}-{e_line})]({github_url}) "
                        f"*(Similarity: {src.get('similarity', 0.0)} | RRF: {src.get('rrf_score', 0.0)})*"
                    )
                    st.code(src.get("snippet", ""), language=get_language_from_path(path))

        # Save assistant message to chat history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": full_answer,
            "engine": engine_name,
            "sources": sources
        })
    except Exception as e:
        st.error(f"Error generating answer: {e}")


def main() -> None:
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="Chat with GitHub Repository",
        page_icon="💬",
        layout="wide"
    )

    # Display persistent flash messages that survive st.rerun()
    if "flash_message" in st.session_state:
        msg = st.session_state.pop("flash_message")
        if msg["type"] == "success":
            st.success(msg["text"])
        elif msg["type"] == "error":
            st.error(msg["text"])

    # Initialize session state variables
    if "active_repo" not in st.session_state:
        st.session_state.active_repo = None
    if "active_branch" not in st.session_state:
        st.session_state.active_branch = "main"
    if "commit_sha" not in st.session_state:
        st.session_state.commit_sha = ""
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "chunk_count" not in st.session_state:
        st.session_state.chunk_count = 0
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_context_chars" not in st.session_state:
        st.session_state.last_context_chars = 0
    if "readme_text" not in st.session_state:
        st.session_state.readme_text = ""
    if "file_tree_text" not in st.session_state:
        st.session_state.file_tree_text = ""

    # Render sidebar with system health and settings
    config_dict = _render_sidebar(
        retriever_ready=(st.session_state.retriever is not None),
        chunk_count=st.session_state.chunk_count,
        repo_identifier=st.session_state.active_repo or "",
        commit_sha=st.session_state.commit_sha
    )

    st.title("Chat with GitHub Repository 💬")
    st.caption("Ask questions about any GitHub codebase using hybrid semantic search and high-speed LLM inference.")

    # Guard: Require Groq API key
    if not GROQ_API_KEY:
        st.error(
            "⚠️ **Required Configuration Missing:** `GROQ_API_KEY` is not set in `.env`.\n\n"
            "Add your free Groq API key to `.env` in the project root: `GROQ_API_KEY=your_key_here`"
        )
        st.stop()

    # --- Repository Loader Section ---
    col1, col2, col3 = st.columns([3, 1.2, 1.2])
    with col1:
        repo_input = st.text_input(
            "GitHub Repository URL or 'owner/repo':",
            value=st.session_state.active_repo if st.session_state.active_repo else "",
            placeholder="e.g. streamlit/streamlit or https://github.com/fastapi/fastapi",
            help="Enter any public repository or a private repository accessible with your GITHUB_TOKEN."
        )
    with col2:
        branch_input = st.text_input(
            "Branch:",
            value=st.session_state.active_branch,
            help="Branch name to index. Defaults to 'main' or the repo's default branch."
        )
    with col3:
        st.write("")
        st.write("")
        btn_col_a, btn_col_b = st.columns(2)
        with btn_col_a:
            load_btn = st.button("🚀 Load", use_container_width=True)
        with btn_col_b:
            reindex_btn = st.button("🔄 Re-index", use_container_width=True, help="Force refresh and re-embed.")

    if (load_btn or reindex_btn) and repo_input:
        try:
            parsed_owner, parsed_repo, url_branch = parse_github_url(repo_input)
            chosen_branch = url_branch or branch_input.strip() or "main"

            try:
                # Verify repository access and default branch if needed
                repo_details = fetch_repo_details(parsed_owner, parsed_repo, GITHUB_TOKEN)
                if not url_branch and not branch_input.strip():
                    chosen_branch = repo_details.get("default_branch", "main")
            except PermissionError:
                # If rate limited, proceed with user-chosen branch or fallback
                pass

            _index_repository(
                parsed_owner,
                parsed_repo,
                chosen_branch,
                max_files=int(config_dict["max_files"]),
                force_reindex=bool(reindex_btn)
            )
            st.rerun()
        except PermissionError as pe:
            # Fall back to any locally cached collection in ChromaDB
            cached_col = find_cached_collection_for_repo(parsed_owner, parsed_repo)
            if cached_col and not reindex_btn:
                meta = cached_col.metadata or {}
                stored_chunks = load_cached_chunks_from_chroma(cached_col)
                retriever = HybridRetriever(cached_col, stored_chunks)
                repo_id = f"{parsed_owner}/{parsed_repo}"
                st.session_state.active_repo = repo_id
                st.session_state.active_branch = meta.get("branch", "main")
                st.session_state.commit_sha = meta.get("commit_sha", "")
                st.session_state.retriever = retriever
                st.session_state.readme_text = meta.get("readme", "")
                st.session_state.file_tree_text = meta.get("file_tree", "")
                st.session_state.chunk_count = len(stored_chunks)
                st.session_state.chat_history = []
                st.session_state.flash_message = {
                    "type": "success",
                    "text": f"⚡ Loaded **{len(stored_chunks)} chunks** for `{repo_id}` from local ChromaDB cache (offline / rate-limit recovery)!"
                }
                st.rerun()
            else:
                st.error(f"Error loading repository: {pe}")
        except Exception as e:
            st.error(f"Error loading repository: {e}")

    # --- Active Chat Interface ---
    if st.session_state.active_repo and st.session_state.retriever:
        st.markdown(f"### 💬 Codebase Chat: `{st.session_state.active_repo}` (`{st.session_state.active_branch}`)")

        # Quick Action Buttons
        st.markdown("**Quick Actions:**")
        qcol1, qcol2, qcol3 = st.columns(3)
        trigger_query: str = ""

        with qcol1:
            if st.button("📝 Summarize this repo", use_container_width=True):
                trigger_query = "Provide a comprehensive summary of this repository, its core purpose, main architecture, and key components."
        with qcol2:
            if st.button("🚀 How do I run this project?", use_container_width=True):
                trigger_query = "How do I install dependencies, configure, and run this project? Provide step-by-step setup instructions based on the codebase."
        with qcol3:
            if st.button("📂 Explain folder structure", use_container_width=True):
                # COST RULE: Build folder structure directly from the stored file tree with ZERO LLM calls!
                tree_text = st.session_state.file_tree_text or "(No file tree available)"
                structure_content = (
                    f"### 📂 Repository Folder Structure for `{st.session_state.active_repo}`\n\n"
                    "Here is the directory layout of the indexed files in this repository:\n\n"
                    f"```text\n{tree_text}\n```\n\n"
                    "*Generated directly from the indexed repository tree without calling the LLM.*"
                )
                st.session_state.chat_history.append({"role": "user", "content": "Explain folder structure"})
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": structure_content,
                    "engine": "Local Tree Generator (0 LLM Tokens)",
                    "sources": []
                })
                st.rerun()

        # Render Conversation History
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "engine" in msg:
                    st.caption(f"⚡ Generated via {msg['engine']}")
                # Render previous sources
                sources = msg.get("sources", [])
                if sources:
                    with st.expander(f"🔍 Referenced Sources ({len(sources)} chunks)", expanded=False):
                        owner, repo = st.session_state.active_repo.split("/", 1)
                        for src in sources:
                            path = src["path"]
                            s_line = src["start_line"]
                            e_line = src["end_line"]
                            github_url = f"https://github.com/{owner}/{repo}/blob/{st.session_state.commit_sha}/{path}#L{s_line}-L{e_line}"
                            st.markdown(f"• [`{path}` (Lines {s_line}-{e_line})]({github_url})")
                            st.code(src.get("snippet", ""), language=get_language_from_path(path))

        # Accept User Input
        user_input = st.chat_input("Ask a question about this repository's code, structure, or setup...")
        active_query = trigger_query or user_input

        if active_query:
            # 1. Render User Message
            st.session_state.chat_history.append({"role": "user", "content": active_query})
            with st.chat_message("user"):
                st.markdown(active_query)

            # 2. Render Assistant Streaming Response
            with st.chat_message("assistant"):
                _handle_assistant_answer(active_query, config_dict)
    else:
        st.info("👆 Enter a GitHub repository above and click **Load** to index its codebase and start chatting!")


if __name__ == "__main__":
    main()
