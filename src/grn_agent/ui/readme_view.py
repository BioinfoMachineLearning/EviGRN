"""First-page overview content: figure path, GitHub remote, and README excerpt."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

_GITHUB_REMOTE = re.compile(
    r"^(?:https?://github\.com/|git@github\.com:)(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+?)(?:\.git)?/?$"
)
_README_GITHUB = re.compile(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")

_OVERVIEW_CANDIDATES = (
    Path("docs/figures_and_tables/figures/fig00_overview_evigrn.png"),
    Path("docs/Figure1_Overview (2).png"),
)


def normalize_github_remote(url: str) -> str | None:
    """Turn a GitHub remote URL into an https repository link."""
    text = str(url).strip()
    match = _GITHUB_REMOTE.match(text)
    if match is None:
        return None
    return f"https://github.com/{match.group('owner')}/{match.group('repo')}"


def project_github_url(repo_root: str | Path) -> str | None:
    """Read origin, then the README, without inventing a repository URL."""
    root = Path(repo_root)
    try:
        raw = subprocess.check_output(
            ["git", "-C", str(root), "remote", "get-url", "origin"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        raw = ""
    normalized = normalize_github_remote(raw)
    if normalized is not None:
        return normalized
    readme = root / "README.md"
    if not readme.is_file():
        return None
    for match in _README_GITHUB.findall(readme.read_text(encoding="utf-8")):
        normalized = normalize_github_remote(match)
        if normalized is not None:
            return normalized
    return None


def overview_figure_path(repo_root: str | Path) -> Path | None:
    root = Path(repo_root)
    for relative in _OVERVIEW_CANDIDATES:
        path = root / relative
        if path.is_file():
            return path
    return None


def how_evigrn_works(readme_text: str) -> str:
    """README text that explains the method, stopping before install instructions."""
    lines: list[str] = []
    started = False
    for line in readme_text.splitlines():
        if line.startswith("# "):
            continue
        if line.startswith("!["):
            continue
        stripped = line.strip()
        if not started:
            if not stripped:
                continue
            if stripped.startswith("[") and "github.com" in stripped:
                continue
            started = True
        if line.startswith("## Install"):
            break
        lines.append(line)
    return "\n".join(lines).strip()
