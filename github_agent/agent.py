"""
github_agent/agent.py

Analyzes a GitHub repository to produce a milestone verification
signal score (0-10000).

Signals analyzed:
  1. Commit frequency and recency
  2. Contributor diversity
  3. Burst pattern detection (suspicious last-minute commits)
  4. Meaningful diff ratio (real code vs padding)
  5. Code complexity growth

FastAPI endpoint:
  POST /analyze
  Body: { "repo_url": "...", "milestone_description": "...", "days": 30 }
  Returns: { "score": 7840, "confidence": 8200, "signals": {...}, "flags": [...] }
"""

import math
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="LaunchVault GitHub Agent")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}

# ── Request / Response models ─────────────────────────────────────


class AnalyzeRequest(BaseModel):
    repo_url: str
    milestone_description: str
    days: int = 30


class SignalScores(BaseModel):
    commit_frequency: int  # 0-10000
    recency: int  # 0-10000
    contributor_diversity: int  # 0-10000
    burst_pattern: int  # 0-10000 (10000 = no burst detected)
    meaningful_diff: int  # 0-10000
    overall: int  # 0-10000 weighted average


class AnalyzeResponse(BaseModel):
    score: int  # 0-10000 final score
    confidence: int  # 0-10000 how confident we are in the score
    signals: SignalScores
    flags: list[str]  # human readable warnings
    commit_count: int
    contributor_count: int
    repo: str


# ── GitHub API helpers ────────────────────────────────────────────


def parse_repo(repo_url: str) -> tuple[str, str]:
    """
    Extract owner and repo name from GitHub URL.
    Handles:
      https://github.com/owner/repo
      https://github.com/owner/repo.git
      owner/repo
    """
    repo_url = repo_url.strip().rstrip("/")

    # Handle full URL
    match = re.search(r"github\.com[/:]([^/]+)/([^/\.]+)", repo_url)
    if match:
        return match.group(1), match.group(2)

    # Handle owner/repo shorthand
    parts = repo_url.split("/")
    if len(parts) == 2:
        return parts[0], parts[1]

    raise ValueError(f"Cannot parse repo URL: {repo_url}")


def github_get(url: str, params: dict = None) -> dict | list:
    """Make a GitHub API request with error handling."""
    response = requests.get(url, headers=HEADERS, params=params, timeout=15)

    if response.status_code == 401:
        raise HTTPException(status_code=401, detail="GitHub token invalid or missing")
    if response.status_code == 403:
        raise HTTPException(status_code=429, detail="GitHub API rate limit reached")
    if response.status_code == 404:
        raise HTTPException(status_code=404, detail="Repository not found or private")
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"GitHub API error: {response.text[:200]}",
        )

    return response.json()


def get_commits(owner: str, repo: str, since_days: int) -> list[dict]:
    """Fetch all commits from the last `since_days` days."""
    since = (datetime.now(timezone.utc) - timedelta(days=since_days)).isoformat()

    commits = []
    page = 1

    while True:
        url = f"https://api.github.com/repos/{owner}/{repo}/commits"
        data = github_get(url, params={"since": since, "per_page": 100, "page": page})

        if not data:
            break

        commits.extend(data)

        if len(data) < 100:
            break

        page += 1

    return commits


def get_commit_detail(owner: str, repo: str, sha: str) -> dict:
    """Fetch detailed info for a single commit including diff stats."""
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
    return github_get(url)


# ── Signal analyzers ──────────────────────────────────────────────


def analyze_commit_frequency(commits: list[dict], days: int) -> tuple[int, list[str]]:
    """
    Score based on how consistently commits were made over the period.

    Scoring:
      0 commits         → 0
      1-5 commits       → 2000-4000
      6-15 commits      → 4000-7000
      16-30 commits     → 7000-8500
      31+ commits       → 8500-10000
    """
    flags = []
    count = len(commits)

    if count == 0:
        flags.append("No commits found in the evaluation window")
        return 0, flags

    if count < 3:
        flags.append(f"Very few commits ({count}) — low development activity")

    # Score based on commit count relative to days
    # Ideal: ~1 commit per day
    ratio = count / days

    if ratio >= 1.0:
        score = 9000
    elif ratio >= 0.5:
        score = 7500
    elif ratio >= 0.2:
        score = 6000
    elif ratio >= 0.1:
        score = 4000
    elif ratio >= 0.05:
        score = 2500
    else:
        score = 1000

    return score, flags


def analyze_recency(commits: list[dict]) -> tuple[int, list[str]]:
    """
    Score based on how recently the last commit was made.

    Last commit < 3 days ago  → 9000-10000
    Last commit < 7 days ago  → 7000-9000
    Last commit < 14 days ago → 5000-7000
    Last commit < 30 days ago → 2000-5000
    Older                     → 0-2000
    """
    flags = []

    if not commits:
        return 0, flags

    # Get the most recent commit date
    latest_date_str = commits[0]["commit"]["author"]["date"]
    latest_date = datetime.fromisoformat(latest_date_str.replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    days_ago = (now - latest_date).days

    if days_ago <= 3:
        score = 9500
    elif days_ago <= 7:
        score = 8000
    elif days_ago <= 14:
        score = 6000
    elif days_ago <= 21:
        score = 4000
    elif days_ago <= 30:
        score = 2500
    else:
        score = 1000
        flags.append(f"Last commit was {days_ago} days ago — stale repository")

    return score, flags


def analyze_contributor_diversity(commits: list[dict]) -> tuple[int, list[str]]:
    """
    Score based on number of unique contributors.

    A startup claiming a team but with only 1 contributor is suspicious.

    1 contributor  → 4000 (solo founder is normal but lower confidence)
    2 contributors → 6500
    3 contributors → 8000
    4+             → 9000-10000
    """
    flags = []

    if not commits:
        return 0, flags

    authors = set()
    for commit in commits:
        author = commit.get("author")
        if author and author.get("login"):
            authors.add(author["login"])
        else:
            # Fall back to commit author name
            name = commit["commit"]["author"].get("name", "unknown")
            authors.add(name)

    count = len(authors)

    if count == 1:
        score = 4000
        flags.append("Only 1 contributor — if startup claims a team this is a concern")
    elif count == 2:
        score = 6500
    elif count == 3:
        score = 8000
    elif count >= 4:
        score = 9500
    else:
        score = 2000

    return score, flags


def analyze_burst_pattern(commits: list[dict], days: int) -> tuple[int, list[str]]:
    """
    Detect suspicious burst activity — many commits in a very short
    window right before the milestone deadline.

    A genuine project has commits spread over time.
    A founder trying to fake activity might commit 50 times in 2 days.

    Method:
      - Split the window into thirds (early, middle, late)
      - If >70% of commits are in the final third → suspicious burst
    """
    flags = []

    if len(commits) < 5:
        # Not enough data to detect burst
        return 7000, flags

    now = datetime.now(timezone.utc)
    window_start = now - timedelta(days=days)
    window_third = days / 3

    early_count = 0
    middle_count = 0
    late_count = 0

    for commit in commits:
        date_str = commit["commit"]["author"]["date"]
        date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        days_into_window = (date - window_start).days

        if days_into_window < window_third:
            early_count += 1
        elif days_into_window < window_third * 2:
            middle_count += 1
        else:
            late_count += 1

    total = len(commits)
    late_ratio = late_count / total if total > 0 else 0

    if late_ratio > 0.80:
        score = 1500
        flags.append(
            f"Burst pattern detected: {late_ratio:.0%} of commits in "
            f"final third of window — possible last-minute padding"
        )
    elif late_ratio > 0.65:
        score = 4000
        flags.append(
            f"Moderate burst pattern: {late_ratio:.0%} of commits "
            f"concentrated toward end of window"
        )
    elif late_ratio > 0.50:
        score = 6500
    else:
        score = 9000

    return score, flags


def analyze_meaningful_diffs(
    owner: str, repo: str, commits: list[dict], sample_size: int = 10
) -> tuple[int, list[str]]:
    """
    Sample up to `sample_size` commits and check whether the diffs
    represent meaningful work vs trivial changes.

    Meaningful: large additions, diverse file types, source code files
    Trivial: single-line changes, README-only edits, whitespace

    We sample rather than fetch all commits to avoid API rate limits.
    """
    flags = []

    if not commits:
        return 5000, flags

    # Sample evenly across the commit list
    step = max(1, len(commits) // sample_size)
    sample = commits[::step][:sample_size]

    total_additions = 0
    total_deletions = 0
    total_files = 0
    trivial_count = 0
    meaningful_count = 0

    for commit in sample:
        try:
            detail = get_commit_detail(owner, repo, commit["sha"])
            stats = detail.get("stats", {})
            files = detail.get("files", [])

            additions = stats.get("additions", 0)
            deletions = stats.get("deletions", 0)
            total_changes = additions + deletions

            total_additions += additions
            total_deletions += deletions
            total_files += len(files)

            # Classify this commit
            if total_changes < 5:
                trivial_count += 1
            elif total_changes > 50:
                meaningful_count += 1
            else:
                # Medium — check file types
                source_extensions = {
                    ".sol",
                    ".py",
                    ".ts",
                    ".js",
                    ".rs",
                    ".go",
                    ".java",
                    ".cpp",
                    ".c",
                    ".swift",
                    ".kt",
                }
                has_source = any(
                    any(f["filename"].endswith(ext) for ext in source_extensions)
                    for f in files
                )
                if has_source:
                    meaningful_count += 1
                else:
                    trivial_count += 1

        except Exception:
            # Skip commits we can't fetch (rate limit, etc.)
            continue

    sampled = trivial_count + meaningful_count
    if sampled == 0:
        return 5000, ["Could not sample commit diffs"]

    meaningful_ratio = meaningful_count / sampled

    if meaningful_ratio >= 0.8:
        score = 9000
    elif meaningful_ratio >= 0.6:
        score = 7000
    elif meaningful_ratio >= 0.4:
        score = 5000
    elif meaningful_ratio >= 0.2:
        score = 3000
        flags.append(
            f"Low meaningful diff ratio ({meaningful_ratio:.0%}) — many trivial commits"
        )
    else:
        score = 1500
        flags.append(
            f"Very low meaningful diff ratio ({meaningful_ratio:.0%}) — "
            f"commits appear to be padding"
        )

    return score, flags


# ── Main analysis function ─────────────────────────────────────────


def analyze_repository(
    repo_url: str, milestone_description: str, days: int = 30
) -> dict:
    """
    Full repository analysis pipeline.
    Returns structured score dict ready for synthesis agent.
    """
    owner, repo = parse_repo(repo_url)
    all_flags = []

    # Fetch commits
    commits = get_commits(owner, repo, days)

    # Run all signal analyzers
    freq_score, freq_flags = analyze_commit_frequency(commits, days)
    rec_score, rec_flags = analyze_recency(commits)
    div_score, div_flags = analyze_contributor_diversity(commits)
    burst_score, burst_flags = analyze_burst_pattern(commits, days)
    diff_score, diff_flags = analyze_meaningful_diffs(owner, repo, commits)

    all_flags = freq_flags + rec_flags + div_flags + burst_flags + diff_flags

    # Weighted average
    # Meaningful diffs and burst pattern weighted higher —
    # these are the hardest to fake
    weights = {
        "freq": 0.15,
        "rec": 0.15,
        "div": 0.10,
        "burst": 0.30,
        "diff": 0.30,
    }

    overall = int(
        freq_score * weights["freq"]
        + rec_score * weights["rec"]
        + div_score * weights["div"]
        + burst_score * weights["burst"]
        + diff_score * weights["diff"]
    )

    # Confidence: how much data did we have to work with?
    # More commits = more confident in the score
    commit_count = len(commits)
    if commit_count >= 20:
        confidence = 9000
    elif commit_count >= 10:
        confidence = 7500
    elif commit_count >= 5:
        confidence = 6000
    elif commit_count >= 1:
        confidence = 4000
    else:
        confidence = 1000

    # Count unique contributors
    authors = set()
    for c in commits:
        author = c.get("author")
        if author and author.get("login"):
            authors.add(author["login"])
        else:
            authors.add(c["commit"]["author"].get("name", "unknown"))

    return {
        "score": overall,
        "confidence": confidence,
        "signals": {
            "commit_frequency": freq_score,
            "recency": rec_score,
            "contributor_diversity": div_score,
            "burst_pattern": burst_score,
            "meaningful_diff": diff_score,
            "overall": overall,
        },
        "flags": all_flags,
        "commit_count": commit_count,
        "contributor_count": len(authors),
        "repo": f"{owner}/{repo}",
    }


# ── FastAPI routes ────────────────────────────────────────────────


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    """Analyze a GitHub repository for milestone verification."""
    try:
        result = analyze_repository(
            repo_url=req.repo_url,
            milestone_description=req.milestone_description,
            days=req.days,
        )
        return result
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/health")
async def health():
    return {"status": "ok", "agent": "github"}


# ── Direct test ───────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python agent.py <github_repo_url> [days]")
        print("Example: python agent.py https://github.com/owner/repo 30")
        sys.exit(1)

    repo_url = sys.argv[1]
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    print(f"Analyzing {repo_url} over last {days} days...")
    result = analyze_repository(repo_url, "milestone", days)
    print(json.dumps(result, indent=2))
