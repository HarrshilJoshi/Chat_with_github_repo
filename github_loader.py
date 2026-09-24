"""
GitHub Loader module.
Handles URL parsing, branch discovery, recursive repository tree fetching,
intelligent file prioritization, and parallel downloading with retry backoff.
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from config import (
    DOWNLOAD_WORKERS,
    MAX_FILE_SIZE_BYTES,
    MAX_FILES_DEFAULT,
    REQUEST_TIMEOUT_SECONDS,
)

# Priority filenames: project entry points, configuration, and manifests
PRIORITY_EXACT_NAMES = {
    "readme.md", "readme.rst", "readme.txt", "readme",
    "requirements.txt", "package.json", "pyproject.toml", "setup.py", "setup.cfg",
    "cargo.toml", "go.mod", "pom.xml", "build.gradle", "gemfile", "composer.json",
    "dockerfile", "docker-compose.yml", "docker-compose.yaml", "makefile",
    ".env.example", "tsconfig.json", "cmakelists.txt",
    "main.py", "app.py", "index.js", "index.ts", "server.js", "server.ts",
    "manage.py", "run.py", "cli.py", "wsgi.py", "asgi.py"
}

# Key application source directories
PRIORITY_DIRS = ("src/", "app/", "lib/", "core/", "pkg/", "internal/")

# Allowed code and documentation extensions
VALID_EXTENSIONS = (
    ".py", ".js", ".ts", ".jsx", ".tsx", ".md", ".txt", ".json",
    ".html", ".css", ".go", ".rs", ".java", ".c", ".cpp", ".h",
    ".hpp", ".cs", ".yaml", ".yml", ".toml", ".sh", ".sql", ".rb",
    ".php", ".kt", ".swift"
)

# Files and directories to unconditionally exclude
EXCLUDED_NAMES = {
    "package-lock.json", "yarn.lock", "poetry.lock", "pipfile.lock",
    "pnpm-lock.yaml", "cargo.lock", "composer.lock"
}

EXCLUDED_DIR_PATTERNS = (
    "node_modules/", ".git/", "dist/", "build/", "vendor/",
    "__pycache__/", ".next/", ".nuxt/", ".venv/", "venv/", "env/",
    ".idea/", ".vscode/", ".pytest_cache/", "coverage/", ".turbo/"
)


def get_http_session(token: str = "") -> requests.Session:
    """Create a requests session configured with retries and authentication headers."""
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update({"Accept": "application/vnd.github.v3+json"})
    if token:
        session.headers["Authorization"] = f"token {token}"
    return session


def parse_github_url(url: str) -> tuple[str, str, Optional[str]]:
    """
    Parse a GitHub repository identifier or URL into (owner, repo, branch).
    
    Handles:
    - https://github.com/owner/repo.git
    - http://github.com/owner/repo
    - www.github.com/owner/repo
    - github.com/owner/repo/tree/feature-branch
    - github.com/owner/repo/tree/main/src/subfolder
    - owner/repo
    - trailing slashes and spaces
    """
    from urllib.parse import urlparse

    cleaned = url.strip()
    if not cleaned:
        raise ValueError("Repository input cannot be empty.")

    if "://" in cleaned:
        parsed = urlparse(cleaned)
        host = parsed.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        if host != "github.com":
            raise ValueError(f"Invalid host: '{parsed.netloc}'. Only 'github.com' repositories are supported.")
        path = parsed.path.strip("/")
    else:
        cleaned = cleaned.rstrip("/")
        if cleaned.lower().startswith("www."):
            cleaned = cleaned[4:]
        if cleaned.lower().startswith("github.com/"):
            cleaned = cleaned[len("github.com/"):]
        elif "/" in cleaned and any(cleaned.lower().startswith(d) for d in ("gitlab.com", "bitbucket.org", "gitea.com")):
            raise ValueError("Only 'github.com' repositories are supported.")
        path = cleaned

    parts = [p for p in path.split("/") if p]
    if len(parts) < 2:
        raise ValueError(
            f"Invalid GitHub repository format: '{url}'. "
            "Please provide 'owner/repo' or a valid GitHub repository URL."
        )

    owner = parts[0]
    repo = parts[1]
    if repo.endswith(".git"):
        repo = repo[:-4]

    branch = None
    if len(parts) >= 4 and parts[2] == "tree":
        branch = parts[3]

    return owner, repo, branch


def fetch_repo_details(owner: str, repo: str, token: str = "") -> dict:
    """Fetch repository metadata including default branch."""
    session = get_http_session(token)
    url = f"https://api.github.com/repos/{owner}/{repo}"
    resp = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)

    if resp.status_code == 404:
        raise ValueError(
            f"Repository '{owner}/{repo}' not found. Check the name, or if it is private, "
            "ensure a valid `GITHUB_TOKEN` is set in `.env`."
        )
    if resp.status_code in (403, 429):
        raise PermissionError(
            "GitHub API rate limit exceeded. Add a `GITHUB_TOKEN` in `.env` to increase limits to 5,000 req/hr."
        )
    if resp.status_code != 200:
        raise RuntimeError(f"GitHub API error ({resp.status_code}): {resp.text}")

    return resp.json()


def fetch_repo_branches(owner: str, repo: str, default_branch: str, token: str = "") -> list[str]:
    """Fetch available branch names for the repository with default branch listed first."""
    session = get_http_session(token)
    url = f"https://api.github.com/repos/{owner}/{repo}/branches?per_page=100"
    branches = [default_branch]

    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        if resp.status_code == 200:
            for b in resp.json():
                name = b.get("name")
                if name and name not in branches:
                    branches.append(name)
    except Exception:
        # If branch listing fails (e.g. rate limit), fall back gracefully to just the default branch
        pass

    return branches


def fetch_branch_commit_sha(owner: str, repo: str, branch: str, token: str = "") -> str:
    """Fetch the latest commit SHA for a branch to support content caching."""
    session = get_http_session(token)
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{branch}"
    resp = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)

    if resp.status_code != 200:
        raise RuntimeError(f"Failed to fetch commit SHA for branch '{branch}' ({resp.status_code})")

    data = resp.json()
    return data.get("sha", "")


def get_repo_tree(owner: str, repo: str, commit_sha: str, token: str = "") -> tuple[list[dict], bool]:
    """
    Fetch the recursive file tree for the repository at the given commit SHA.
    Returns (tree_items, is_truncated).
    """
    session = get_http_session(token)
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{commit_sha}?recursive=1"
    resp = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)

    if resp.status_code != 200:
        raise RuntimeError(f"Failed to fetch repository tree ({resp.status_code}): {resp.text}")

    data = resp.json()
    tree = data.get("tree", [])
    is_truncated = bool(data.get("truncated", False))
    return tree, is_truncated


def prioritize_and_filter_files(tree_items: list[dict], max_files: int = MAX_FILES_DEFAULT) -> tuple[list[dict], int]:
    """
    Sort and filter repository files by priority:
    1. README, package manifests, configs, and entry files
    2. Core source directories (src/, app/, lib/)
    3. Other valid code/documentation files
    
    Excludes lockfiles, minified assets, binaries, test caches, and files exceeding size limit.
    Returns (selected_items, skipped_count).
    """
    tier_1 = []
    tier_2 = []
    tier_3 = []
    total_valid_candidates = 0

    for item in tree_items:
        if item.get("type") != "blob":
            continue

        path = item.get("path", "")
        size = item.get("size", 0)
        path_lower = path.lower()
        filename_lower = path_lower.split("/")[-1]

        # 1. Skip files exceeding maximum size limit (200 KB)
        if size and size > MAX_FILE_SIZE_BYTES:
            continue

        # 2. Skip excluded directories
        if any(ign in path_lower for ign in EXCLUDED_DIR_PATTERNS):
            continue

        # 3. Skip lockfiles and minified files
        if filename_lower in EXCLUDED_NAMES:
            continue
        if filename_lower.endswith((".min.js", ".min.css", ".bundle.js", ".map")):
            continue

        # 4. Filter for valid code and doc extensions
        if not filename_lower.endswith(VALID_EXTENSIONS) and filename_lower not in PRIORITY_EXACT_NAMES:
            continue

        total_valid_candidates += 1

        # Priority Tier 1: Configs, manifests, READMEs, entry points
        if filename_lower in PRIORITY_EXACT_NAMES:
            tier_1.append(item)
        # Priority Tier 2: Core source directory code
        elif any(path_lower.startswith(pdir) or f"/{pdir}" in path_lower for pdir in PRIORITY_DIRS):
            tier_2.append(item)
        # Priority Tier 3: Other valid code and documentation
        else:
            tier_3.append(item)

    # Sort each tier stably by path
    tier_1.sort(key=lambda x: x["path"])
    tier_2.sort(key=lambda x: x["path"])
    tier_3.sort(key=lambda x: x["path"])

    combined = tier_1 + tier_2 + tier_3
    selected = combined[:max_files]
    skipped_count = max(0, total_valid_candidates - len(selected))

    return selected, skipped_count


def _download_single_file(session: requests.Session, raw_url: str, path: str) -> Optional[dict]:
    """Helper worker to download a single file's text content with retry logic."""
    try:
        resp = session.get(raw_url, timeout=REQUEST_TIMEOUT_SECONDS)
        if resp.status_code == 200:
            text = resp.text
            if text and text.strip():
                return {"path": path, "content": text, "size": len(text)}
    except Exception:
        pass
    return None


def fetch_files_parallel(
    owner: str,
    repo: str,
    commit_sha: str,
    file_items: list[dict],
    token: str = "",
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> list[dict]:
    """
    Download repository files in parallel using a ThreadPoolExecutor.
    Executes safely in main thread by monitoring futures, avoiding st.* calls inside worker threads.
    """
    session = get_http_session(token)
    downloaded_files = []
    total = len(file_items)
    completed = 0

    # Build tasks map: future -> path
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        future_to_item = {}
        for item in file_items:
            path = item["path"]
            raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{commit_sha}/{path}"
            future = executor.submit(_download_single_file, session, raw_url, path)
            future_to_item[future] = path

        for future in as_completed(future_to_item):
            completed += 1
            path = future_to_item[future]
            try:
                res = future.result()
                if res:
                    downloaded_files.append(res)
            except Exception:
                pass

            if progress_callback:
                progress_callback(completed, total, path)

    # Keep deterministic order matching input items
    path_order = {item["path"]: i for i, item in enumerate(file_items)}
    downloaded_files.sort(key=lambda x: path_order.get(x["path"], 999999))

    return downloaded_files


def build_ascii_file_tree(paths: list[str], max_lines: int = 60) -> str:
    """
    Build an indented ASCII file tree for prompt context and repository inspection.
    Truncates gracefully if line count exceeds budget.
    """
    if not paths:
        return "(No files indexed)"

    # Build nested dictionary tree
    tree: dict = {}
    for p in sorted(paths):
        parts = p.split("/")
        curr = tree
        for part in parts:
            curr = curr.setdefault(part, {})

    lines: list[str] = []

    def _render(node: dict, prefix: str = "") -> None:
        if len(lines) >= max_lines:
            return
        keys = sorted(node.keys())
        for i, key in enumerate(keys):
            if len(lines) >= max_lines:
                return
            is_last = (i == len(keys) - 1)
            connector = "└── " if is_last else "├── "
            child_prefix = "    " if is_last else "│   "
            lines.append(f"{prefix}{connector}{key}")
            if node[key]:  # Non-empty dict means directory
                _render(node[key], prefix + child_prefix)

    _render(tree)
    if len(lines) >= max_lines and len(paths) > max_lines:
        lines.append(f"... ({len(paths) - max_lines} more files)")

    return "\n".join(lines)
