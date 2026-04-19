"""
server.py — LaunchVault Agent Orchestration Server

Single FastAPI server the frontend calls to run the full verification pipeline:
  POST /verify   — accepts files + github URL, runs all agents, writes score on-chain
  GET  /health   — liveness check

Usage:
  cd learnchain-agents
  python server.py           (default port 8000)
  python server.py --port 8001
"""

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv()

app = FastAPI(title="LaunchVault Agent Server", version="1.0.0")

# Allow frontend origins — localhost for dev, Vercel for production
_EXTRA_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        *_EXTRA_ORIGINS,
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Module loader helpers ─────────────────────────────────────────

_BASE = Path(__file__).parent


def _load(subpath: str):
    """Dynamically load a module from a relative path."""
    path = _BASE / subpath
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Routes ────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok", "server": "launchvault-agents"}


@app.post("/verify")
async def verify(
    campaign_address: str = Form(...),
    milestone_index: int = Form(...),
    milestone_description: str = Form(...),
    github_url: str = Form(default=""),
    company_name: str = Form(default=""),
    company_website: str = Form(default=""),
    team_members: str = Form(default=""),  # comma-separated names
    files: list[UploadFile] = File(default=[]),
):
    """
    Full verification pipeline:
      1. OSINT entity verification (if company_name provided)
      2. Document forensics (if files uploaded — PDF/JPG/PNG/WEBP)
      3. GitHub analysis (if github_url provided)
      4. Claude synthesis of all signals
      5. Oracle writes final score on-chain
      6. Returns full report
    """

    # ── 0. Parse team members ────────────────────────────────────
    team_list = (
        [t.strip() for t in team_members.split(",") if t.strip()]
        if team_members.strip() else []
    )

    # ── 1. OSINT analysis ─────────────────────────────────────────
    osint_result = None
    if company_name.strip():
        try:
            osint_mod = _load("osint_agent/agent.py")
            osint_result = osint_mod.analyze_entity(
                company_name=company_name.strip(),
                milestone_description=milestone_description,
                website=company_website.strip() or None,
                github_url=github_url.strip() or None,
                team_members=team_list or None,
            )
        except Exception as e:
            osint_result = {
                "score": 5000,
                "confidence": 1000,
                "verdict": "PARTIAL",
                "flags": [f"OSINT analysis error: {str(e)}"],
                "verified_facts": [],
                "signals": {},
                "consistency_assessment": "",
                "strongest_positive": "",
                "strongest_concern": "",
                "recommendation": "",
                "company_name": company_name,
            }

    # ── 2. Document analysis ──────────────────────────────────────
    document_result = None
    if files:
        allowed_ext = {".pdf", ".jpg", ".jpeg", ".png", ".webp"}
        file_data = []
        for upload in files:
            ext = Path(upload.filename or "").suffix.lower()
            if ext not in allowed_ext:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {upload.filename}. Allowed: PDF, JPG, PNG, WEBP",
                )
            content = await upload.read()
            file_data.append((upload.filename, content))

        try:
            doc_mod = _load("document_agent/agent.py")
            document_result = doc_mod.run_full_analysis(file_data, milestone_description)
        except Exception as e:
            # Document analysis failure is non-fatal — flag it and continue
            document_result = {
                "score": 5000,
                "confidence": 1000,
                "doc_count": len(file_data),
                "flags": [f"Document analysis error: {str(e)}"],
                "signals": {"document_forensics": 5000, "llm_consistency": 5000},
                "llm_reasoning": "",
            }

    # ── 2. GitHub analysis ────────────────────────────────────────
    github_result = None
    if github_url and github_url.strip():
        try:
            gh_mod = _load("github_agent/agent.py")
            github_result = gh_mod.analyze_repository(
                repo_url=github_url.strip(),
                milestone_description=milestone_description,
                days=30,
            )
        except Exception as e:
            github_result = {
                "score": 5000,
                "confidence": 1000,
                "flags": [f"GitHub analysis error: {str(e)}"],
                "signals": {},
                "commit_count": 0,
                "contributor_count": 0,
                "repo": github_url,
            }

    # ── 3. Synthesis + on-chain write ─────────────────────────────
    if not document_result and not github_result and not osint_result:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of: company_name (OSINT), files (documents), or github_url",
        )

    try:
        synth_mod = _load("synthesis_agent/agent.py")
        report = synth_mod.synthesize(
            campaign_address=campaign_address,
            milestone_index=milestone_index,
            milestone_description=milestone_description,
            github_result=github_result,
            document_result=document_result,
            osint_result=osint_result,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")

    blockchain = report.get("blockchain", {})
    return {
        "final_score": report.get("final_score", 5000),
        "verdict": report.get("verdict", "INCONCLUSIVE"),
        "confidence": report.get("confidence", 5000),
        "reasoning": report.get("reasoning", ""),
        "recommendation": report.get("recommendation", ""),
        "key_positive_signals": report.get("key_positive_signals", []),
        "key_negative_signals": report.get("key_negative_signals", []),
        "tx_hash": blockchain.get("tx_hash"),
        "on_chain": blockchain.get("written", False),
        "blockchain_error": blockchain.get("error"),
        "github": github_result,
        "documents": document_result,
        "osint": osint_result,
    }


# ── Entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LaunchVault Agent Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    args = parser.parse_args()

    print(f"\nLaunchVault Agent Server")
    print(f"  http://{args.host}:{args.port}/health")
    print(f"  http://{args.host}:{args.port}/verify  (POST)\n")

    uvicorn.run(app, host=args.host, port=args.port)
