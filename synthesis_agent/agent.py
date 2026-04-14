"""
synthesis_agent/agent.py

The final agent in the pipeline. Takes outputs from the GitHub
agent and document agent, synthesizes them into a single score,
and writes it to the blockchain via the oracle writer.

Flow:
  1. Receive signals from GitHub agent + document agent
  2. Call Claude to reason holistically over all signals
  3. Produce final score 0-10000
  4. Write score to blockchain
  5. Return full reasoning report
"""

import json
import os
import sys
from datetime import datetime, timezone

import anthropic
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oracle_writer.oracle_writer import OracleWriter

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def synthesize(
    campaign_address: str,
    milestone_index: int,
    milestone_description: str,
    github_result: dict | None,
    document_result: dict | None,
    osint_result: dict | None = None,
) -> dict:
    """
    Synthesize all agent signals into a final score and write to chain.

    Args:
        campaign_address:     Deployed CampaignV2 contract address
        milestone_index:      Which milestone (usually currentMilestoneIndex)
        milestone_description: What the milestone claims
        github_result:        Output from github_agent (or None if no repo)
        document_result:      Output from document_agent (or None if no docs)

    Returns:
        Full synthesis report including final score, reasoning, and TX hash.
    """

    # ── Step 1 — Build context for Claude ────────────────────────

    sections = []
    sections.append(f'MILESTONE CLAIM: "{milestone_description}"')

    if github_result:
        sections.append(f"""
GITHUB ANALYSIS:
  Repository: {github_result.get("repo", "unknown")}
  Overall Score: {github_result.get("score", 0)} / 10000
  Confidence: {github_result.get("confidence", 0)} / 10000
  Commit Count (last 30 days): {github_result.get("commit_count", 0)}
  Contributors: {github_result.get("contributor_count", 0)}
  Signal Breakdown:
    - Commit Frequency:      {github_result.get("signals", {}).get("commit_frequency", 0)}
    - Recency:               {github_result.get("signals", {}).get("recency", 0)}
    - Contributor Diversity: {github_result.get("signals", {}).get("contributor_diversity", 0)}
    - Burst Pattern:         {github_result.get("signals", {}).get("burst_pattern", 0)}
    - Meaningful Diffs:      {github_result.get("signals", {}).get("meaningful_diff", 0)}
  Flags: {json.dumps(github_result.get("flags", []), indent=4)}
""")
    else:
        sections.append("GITHUB ANALYSIS: Not provided")

    if document_result:
        sections.append(f"""
DOCUMENT ANALYSIS:
  Documents Submitted: {document_result.get("doc_count", 0)}
  Overall Score: {document_result.get("score", 0)} / 10000
  Confidence: {document_result.get("confidence", 0)} / 10000
  Signal Breakdown:
    - Document Forensics: {document_result.get("signals", {}).get("document_forensics", 0)}
    - LLM Consistency:    {document_result.get("signals", {}).get("llm_consistency", 0)}
  LLM Reasoning: {document_result.get("llm_reasoning", "none")}
  Flags: {json.dumps(document_result.get("flags", []), indent=4)}
""")
    else:
        sections.append("DOCUMENT ANALYSIS: Not provided")

    if osint_result:
        sections.append(f"""
OSINT / ENTITY VERIFICATION:
  Company: {osint_result.get("company_name", "unknown")}
  Overall Score: {osint_result.get("score", 0)} / 10000
  Verdict: {osint_result.get("verdict", "PARTIAL")}
  Confidence: {osint_result.get("confidence", 0)} / 10000
  Signal Breakdown:
    - Domain Intelligence:   {osint_result.get("signals", {}).get("domain_intelligence", 0)}
    - Company Registration:  {osint_result.get("signals", {}).get("company_registration", 0)}
    - Web Presence:          {osint_result.get("signals", {}).get("web_presence", 0)}
    - News Coverage:         {osint_result.get("signals", {}).get("news_coverage", 0)}
    - Team Verification:     {osint_result.get("signals", {}).get("team_verification", 0)}
  Consistency Assessment: {osint_result.get("consistency_assessment", "none")}
  Strongest Positive: {osint_result.get("strongest_positive", "none")}
  Strongest Concern: {osint_result.get("strongest_concern", "none")}
  Verified Facts: {json.dumps(osint_result.get("verified_facts", [])[:5], indent=4)}
  Flags: {json.dumps(osint_result.get("flags", [])[:5], indent=4)}
""")
    else:
        sections.append("OSINT / ENTITY VERIFICATION: Not performed")

    context = "\n".join(sections)

    # ── Step 2 — Claude synthesis ─────────────────────────────────

    prompt = f"""You are an autonomous AI due diligence agent for a blockchain crowdfunding platform.
Investors are using your score to decide whether to approve fund release to a startup founder.
Your score will be written directly to a smart contract. There is no human review after you.

This is not a creative task. Be precise, skeptical, and calibrated.

{context}

Based on all signals above, provide your final assessment as a JSON object:
{{
  "final_score": <0-10000, your confidence the milestone was genuinely achieved>,
  "verdict": "<PASS|FAIL|INCONCLUSIVE>",
  "confidence": <0-10000, how confident you are in this score>,
  "key_positive_signals": [<list the strongest evidence FOR milestone completion>],
  "key_negative_signals": [<list the strongest evidence AGAINST milestone completion>],
  "reasoning": "<3-4 sentences explaining your final verdict>",
  "recommendation_to_investors": "<one sentence plain English advice>"
}}

Signal weighting guidance:
  - OSINT is foundational: if the entity itself is unverified or suspicious,
    discount all other signals heavily regardless of how good they look.
  - GitHub and documents verify the MILESTONE; OSINT verifies the ENTITY.
    Both must be credible for a high score.
  - A legitimate company with weak GitHub signal = 50-65 score.
  - A suspicious entity with perfect GitHub signal = 20-45 score (could be faked).

Scoring guide:
  8000-10000: Strong evidence milestone achieved by a verified, legitimate entity
  6000-7999:  Likely achieved, entity appears legitimate but some gaps
  4000-5999:  Uncertain — mixed evidence or insufficient verification
  2000-3999:  Significant concerns about milestone OR entity legitimacy
  0-1999:     Strong evidence of non-completion, fraud, or non-existent entity

Respond ONLY with the JSON object."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = response.content[0].text.strip()
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1])

        synthesis = json.loads(response_text)
        final_score = int(synthesis.get("final_score", 5000))
        final_score = max(0, min(10000, final_score))

    except Exception as e:
        print(f"Claude synthesis failed: {e}")
        # Fallback: weighted average of available scores
        scores = []
        if github_result:
            scores.append(github_result.get("score", 5000))
        if document_result:
            scores.append(document_result.get("score", 5000))
        final_score = int(sum(scores) / len(scores)) if scores else 5000
        synthesis = {
            "final_score": final_score,
            "verdict": "INCONCLUSIVE",
            "confidence": 3000,
            "reasoning": f"Claude synthesis failed ({e}). Used fallback average.",
            "recommendation_to_investors": "Manual review recommended.",
        }

    # ── Step 3 — Write to blockchain ──────────────────────────────

    tx_hash = None
    blockchain_error = None

    try:
        writer = OracleWriter()
        can_submit, reason = writer.can_submit_score(campaign_address, milestone_index)

        if can_submit:
            tx_hash = writer.submit_score(
                campaign_address, milestone_index, final_score
            )
            print(f"Score written to blockchain: {final_score}/10000")
            print(f"TX hash: {tx_hash}")
        else:
            blockchain_error = f"Cannot submit: {reason}"
            print(f"Blockchain submission skipped: {reason}")

    except Exception as e:
        blockchain_error = str(e)
        print(f"Blockchain write failed: {e}")

    # ── Step 4 — Build full report ────────────────────────────────

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "campaign_address": campaign_address,
        "milestone_index": milestone_index,
        "milestone_description": milestone_description,
        "final_score": final_score,
        "verdict": synthesis.get("verdict", "INCONCLUSIVE"),
        "confidence": synthesis.get("confidence", 5000),
        "reasoning": synthesis.get("reasoning", ""),
        "recommendation": synthesis.get("recommendation_to_investors", ""),
        "key_positive_signals": synthesis.get("key_positive_signals", []),
        "key_negative_signals": synthesis.get("key_negative_signals", []),
        "signals": {
            "github": github_result.get("score") if github_result else None,
            "document": document_result.get("score") if document_result else None,
            "osint": osint_result.get("score") if osint_result else None,
            "final": final_score,
        },
        "blockchain": {
            "tx_hash": tx_hash,
            "error": blockchain_error,
            "written": tx_hash is not None,
        },
    }

    return report


# ── Full pipeline runner ──────────────────────────────────────────


def run_full_pipeline(
    campaign_address: str,
    milestone_index: int,
    milestone_description: str,
    repo_url: str | None = None,
    document_paths: list[str] | None = None,
) -> dict:
    """
    Run the complete due diligence pipeline:
      1. GitHub analysis (if repo_url provided)
      2. Document analysis (if documents provided)
      3. Synthesis + blockchain write
    """
    github_result = None
    document_result = None

    # GitHub analysis
    if repo_url:
        print(f"\nAnalyzing GitHub repository: {repo_url}")
        try:
            import importlib.util

            gh_agent_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "github_agent",
                "agent.py",
            )
            spec = importlib.util.spec_from_file_location("github_agent", gh_agent_path)
            gh_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(gh_module)
            github_result = gh_module.analyze_repository(
                repo_url, milestone_description, days=30
            )
            print(f"GitHub score: {github_result['score']}/10000")
        except Exception as e:
            print(f"GitHub analysis failed: {e}")

    # Document analysis
    if document_paths:
        print(f"\nAnalyzing {len(document_paths)} document(s)...")
        try:
            import importlib.util

            doc_agent_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "document_agent",
                "agent.py",
            )
            spec = importlib.util.spec_from_file_location(
                "document_agent", doc_agent_path
            )
            doc_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(doc_module)

            files = []
            for path in document_paths:
                with open(path, "rb") as f:
                    files.append((os.path.basename(path), f.read()))
            document_result = doc_module.run_full_analysis(files, milestone_description)
            print(f"Document score: {document_result['score']}/10000")
        except Exception as e:
            print(f"Document analysis failed: {e}")

    # Synthesis
    print(f"\nSynthesizing signals...")
    report = synthesize(
        campaign_address=campaign_address,
        milestone_index=milestone_index,
        milestone_description=milestone_description,
        github_result=github_result,
        document_result=document_result,
    )

    return report


# ── CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="LaunchVault Synthesis Agent — full due diligence pipeline"
    )
    parser.add_argument("campaign", help="CampaignV2 contract address")
    parser.add_argument("milestone", help="Milestone index (integer)", type=int)
    parser.add_argument("claim", help="Milestone description/claim")
    parser.add_argument("--repo", help="GitHub repo URL", default=None)
    parser.add_argument(
        "--docs", help="Document file paths (comma-separated)", default=None
    )

    args = parser.parse_args()

    doc_paths = args.docs.split(",") if args.docs else None

    report = run_full_pipeline(
        campaign_address=args.campaign,
        milestone_index=args.milestone,
        milestone_description=args.claim,
        repo_url=args.repo,
        document_paths=doc_paths,
    )

    print("\n" + "=" * 60)
    print("FINAL SYNTHESIS REPORT")
    print("=" * 60)
    print(json.dumps(report, indent=2))
