"""
document_agent/agent.py

Analyzes documents submitted by founders for milestone verification.
Detects forgeries, inconsistencies, and suspicious metadata.

Three layers of analysis:
  1. PDF metadata forensics — creation dates, software, modification history
  2. Image forensics — Error Level Analysis (ELA) for manipulation detection
  3. LLM consistency check — Claude analyzes text for internal inconsistencies

FastAPI endpoint:
  POST /analyze
  Body: multipart/form-data with files[] and milestone_description
  Returns: { "score": 7500, "confidence": 8000, "documents": [...], "flags": [...] }
"""

import base64
import io
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import anthropic
import fitz  # PyMuPDF
import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageChops, ImageEnhance

load_dotenv()

app = FastAPI(title="LaunchVault Document Agent")

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# ─────────────────────────────────────────────────────────────────
# LAYER 1 — PDF METADATA FORENSICS
# ─────────────────────────────────────────────────────────────────


def analyze_pdf_metadata(file_bytes: bytes, filename: str) -> dict:
    """
    Extract and analyze PDF metadata for forensic signals.

    Red flags:
      - Creation date much newer than claimed document date
      - Created with Canva, online PDF tools, or image editors
      - Modified after creation (possible tampering)
      - Missing metadata entirely (suspicious for official docs)
      - Author name inconsistent with claimed organization

    Returns dict with score (0-10000), flags, and raw metadata.
    """
    flags = []
    score = 8000  # Start optimistic, deduct for red flags

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        metadata = doc.metadata
        doc.close()
    except Exception as e:
        return {
            "score": 3000,
            "flags": [f"Could not parse PDF metadata: {str(e)}"],
            "metadata": {},
        }

    raw_meta = {
        "title": metadata.get("title", ""),
        "author": metadata.get("author", ""),
        "creator": metadata.get("creator", ""),
        "producer": metadata.get("producer", ""),
        "created": metadata.get("creationDate", ""),
        "modified": metadata.get("modDate", ""),
    }

    # Check for suspicious creator software
    suspicious_creators = [
        "canva",
        "photoshop",
        "gimp",
        "paint",
        "inkscape",
        "word",
        "google docs",
        "libreoffice draw",
        "ilovepdf",
        "smallpdf",
        "pdf24",
        "sejda",
    ]

    creator_lower = (raw_meta["creator"] + raw_meta["producer"]).lower()

    for tool in suspicious_creators:
        if tool in creator_lower:
            flags.append(
                f"Document created with '{tool}' — "
                f"unexpected for official business documents"
            )
            score -= 2000
            break

    # Check if document was recently created
    created_str = raw_meta["created"]
    if created_str:
        try:
            # PyMuPDF returns dates like "D:20240315120000"
            if created_str.startswith("D:"):
                created_str = created_str[2:]
            created_date = datetime.strptime(created_str[:8], "%Y%m%d").replace(
                tzinfo=timezone.utc
            )
            days_old = (datetime.now(timezone.utc) - created_date).days

            if days_old < 3:
                flags.append(
                    f"PDF was created {days_old} day(s) ago — "
                    f"suspiciously recent for a milestone document"
                )
                score -= 3000
            elif days_old < 7:
                flags.append(f"PDF was created only {days_old} days ago")
                score -= 1000

        except ValueError:
            pass  # Date parsing failed — not a red flag itself

    # Check if modified after creation (possible post-hoc editing)
    mod_str = raw_meta["modified"]
    if created_str and mod_str and mod_str != created_str:
        flags.append(
            "Document was modified after initial creation — "
            "possible post-submission editing"
        )
        score -= 1500

    # Missing author is suspicious for official docs
    if not raw_meta["author"] and not raw_meta["creator"]:
        flags.append("No author or creator metadata — unusual for official documents")
        score -= 500

    score = max(0, min(10000, score))

    return {"score": score, "flags": flags, "metadata": raw_meta}


def extract_pdf_text(file_bytes: bytes) -> str:
    """Extract all text from a PDF for LLM analysis."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────
# LAYER 2 — IMAGE FORENSICS (ERROR LEVEL ANALYSIS)
# ─────────────────────────────────────────────────────────────────


def error_level_analysis(image_bytes: bytes) -> dict:
    """
    Perform Error Level Analysis (ELA) on an image.

    How it works:
      JPEG compression introduces uniform quantization errors.
      When an image is saved, every region gets the same error level.
      If someone pastes content into an image and saves it,
      the pasted region has DIFFERENT error levels than the original.
      ELA amplifies these differences, making manipulations visible.

    We analyze the ELA result statistically:
      - High variance in error levels → possible manipulation
      - Regions with anomalously high errors → likely pasted content
      - Uniform error distribution → likely authentic

    Returns score (0-10000) where 10000 = no manipulation detected.
    """
    flags = []

    try:
        original = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return {
            "score": 5000,
            "flags": [f"Could not open image: {str(e)}"],
            "ela_stats": {},
        }

    # Save at known quality to establish baseline compression
    buffer = io.BytesIO()
    original.save(buffer, "JPEG", quality=90)
    buffer.seek(0)
    resaved = Image.open(buffer).convert("RGB")

    # Compute difference — manipulated regions show higher differences
    ela_image = ImageChops.difference(original, resaved)
    ela_array = np.array(ela_image, dtype=np.float32)

    # Statistical analysis of error levels
    mean_error = float(np.mean(ela_array))
    std_error = float(np.std(ela_array))
    max_error = float(np.max(ela_array))
    high_error_pct = float(np.mean(ela_array > (mean_error + 2 * std_error)))

    ela_stats = {
        "mean_error": round(mean_error, 3),
        "std_error": round(std_error, 3),
        "max_error": round(max_error, 3),
        "high_error_pct": round(high_error_pct, 4),
    }

    # Score based on statistical profile
    # High std relative to mean = inconsistent compression = manipulation
    score = 9000

    if std_error > mean_error * 2.5:
        flags.append(
            f"High ELA variance (std={std_error:.1f}, mean={mean_error:.1f}) "
            f"— inconsistent compression suggests image manipulation"
        )
        score -= 4000

    elif std_error > mean_error * 1.5:
        flags.append(f"Moderate ELA variance — possible minor image editing")
        score -= 2000

    # High error percentage means many anomalous regions
    if high_error_pct > 0.15:
        flags.append(
            f"{high_error_pct:.1%} of image has anomalously high error levels "
            f"— significant regions may have been added or replaced"
        )
        score -= 3000

    elif high_error_pct > 0.05:
        flags.append(f"{high_error_pct:.1%} of image has elevated error levels")
        score -= 1000

    score = max(0, min(10000, score))

    return {"score": score, "flags": flags, "ela_stats": ela_stats}


def analyze_image_metadata(image_bytes: bytes, filename: str) -> dict:
    """
    Extract EXIF metadata from an image.
    Checks for GPS coordinates, camera model, and timestamp consistency.
    """
    flags = []
    score = 8000

    try:
        img = Image.open(io.BytesIO(image_bytes))
        exif_data = img._getexif() if hasattr(img, "_getexif") else None

        if exif_data is None:
            # No EXIF is actually fine — many images strip it
            return {"score": 7000, "flags": [], "exif": {}}

        # Tag IDs for common EXIF fields
        DATETIME_TAG = 306
        GPS_TAG = 34853
        MAKE_TAG = 271
        MODEL_TAG = 272
        SOFTWARE_TAG = 305

        exif_summary = {}

        if DATETIME_TAG in exif_data:
            exif_summary["datetime"] = exif_data[DATETIME_TAG]

        if MAKE_TAG in exif_data:
            exif_summary["camera_make"] = exif_data[MAKE_TAG]

        if MODEL_TAG in exif_data:
            exif_summary["camera_model"] = exif_data[MODEL_TAG]

        if SOFTWARE_TAG in exif_data:
            software = str(exif_data[SOFTWARE_TAG]).lower()
            exif_summary["software"] = software

            suspicious = ["photoshop", "gimp", "canva", "paint"]
            for s in suspicious:
                if s in software:
                    flags.append(
                        f"Image processed with '{s}' — "
                        f"verify this is an authentic photo"
                    )
                    score -= 2000

        if GPS_TAG in exif_data:
            exif_summary["has_gps"] = True
            # GPS data present — could verify against claimed location
            # For now just note it's available
            flags.append(
                "GPS metadata present — location data available for verification"
            )

        return {"score": score, "flags": flags, "exif": exif_summary}

    except Exception:
        return {"score": 6000, "flags": [], "exif": {}}


# ─────────────────────────────────────────────────────────────────
# LAYER 3 — LLM CONSISTENCY ANALYSIS
# ─────────────────────────────────────────────────────────────────


def analyze_with_llm(documents: list[dict], milestone_description: str) -> dict:
    """
    Use Claude to analyze document text for:
      1. Internal inconsistencies within each document
      2. Cross-document inconsistencies (names, dates, amounts)
      3. Whether content plausibly supports the milestone claim
      4. Language patterns suggesting AI generation or copy-paste

    Returns score (0-10000) and detailed reasoning.
    """
    if not documents:
        return {
            "score": 5000,
            "flags": ["No document text available for LLM analysis"],
            "reasoning": "",
        }

    # Build context for Claude
    doc_summaries = []
    for i, doc in enumerate(documents):
        if doc.get("text"):
            # Truncate to avoid context overflow
            text = doc["text"][:2000]
            doc_summaries.append(f"Document {i + 1} ({doc['filename']}):\n{text}")

    if not doc_summaries:
        return {
            "score": 5000,
            "flags": ["Documents contain no extractable text"],
            "reasoning": "",
        }

    combined_text = "\n\n---\n\n".join(doc_summaries)

    prompt = f"""You are a document forensics expert analyzing startup milestone documents for a blockchain crowdfunding platform. Investors are relying on your assessment.

MILESTONE CLAIM: "{milestone_description}"

SUBMITTED DOCUMENTS:
{combined_text}

Analyze these documents carefully and respond with a JSON object containing:
{{
  "authenticity_score": <0-100, where 100 = completely authentic>,
  "milestone_support_score": <0-100, how well docs support the milestone claim>,
  "flags": [<list of specific concerns found>],
  "reasoning": "<2-3 sentence summary of your assessment>",
  "cross_document_issues": [<inconsistencies between documents>],
  "internal_issues": [<inconsistencies within individual documents>]
}}

Look specifically for:
- Dates that contradict each other or the milestone timeline
- Company names or registration numbers that seem fabricated
- Financial figures that are internally inconsistent
- Language that seems copy-pasted from templates
- Claims that are vague or unverifiable
- Anything that doesn't match what a legitimate business document would contain

Respond ONLY with the JSON object, no preamble."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = response.content[0].text.strip()

        # Strip markdown code fences if present
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1])

        result = json.loads(response_text)

        # Convert 0-100 scores to 0-10000
        auth_score = int(result.get("authenticity_score", 50) * 100)
        milestone_score = int(result.get("milestone_support_score", 50) * 100)

        # Weighted combination
        combined_score = int(auth_score * 0.6 + milestone_score * 0.4)

        all_flags = (
            result.get("flags", [])
            + result.get("cross_document_issues", [])
            + result.get("internal_issues", [])
        )

        return {
            "score": combined_score,
            "flags": all_flags,
            "reasoning": result.get("reasoning", ""),
        }

    except json.JSONDecodeError:
        return {
            "score": 5000,
            "flags": ["LLM response could not be parsed"],
            "reasoning": "",
        }
    except Exception as e:
        return {
            "score": 5000,
            "flags": [f"LLM analysis failed: {str(e)}"],
            "reasoning": "",
        }


# ─────────────────────────────────────────────────────────────────
# MAIN ANALYSIS PIPELINE
# ─────────────────────────────────────────────────────────────────


def analyze_document(
    file_bytes: bytes, filename: str, milestone_description: str
) -> dict:
    """
    Run all three analysis layers on a single document.
    Returns per-document result.
    """
    extension = Path(filename).suffix.lower()
    results = {}
    all_flags = []

    # ── PDF analysis ─────────────────────────────────────────────
    if extension == ".pdf":
        meta_result = analyze_pdf_metadata(file_bytes, filename)
        results["metadata"] = meta_result
        all_flags.extend(meta_result["flags"])

        text = extract_pdf_text(file_bytes)
        results["text"] = text

    # ── Image analysis ────────────────────────────────────────────
    elif extension in {".jpg", ".jpeg", ".png", ".webp"}:
        ela_result = error_level_analysis(file_bytes)
        exif_result = analyze_image_metadata(file_bytes, filename)

        results["ela"] = ela_result
        results["exif"] = exif_result

        all_flags.extend(ela_result["flags"])
        all_flags.extend(exif_result["flags"])

        results["text"] = ""  # Images have no text

    else:
        results["text"] = ""
        all_flags.append(f"Unsupported file type: {extension}")

    # Calculate document-level score
    scores = []
    if "metadata" in results:
        scores.append(results["metadata"]["score"])
    if "ela" in results:
        scores.append(results["ela"]["score"])
    if "exif" in results:
        scores.append(results["exif"]["score"])

    doc_score = int(sum(scores) / len(scores)) if scores else 5000

    return {
        "filename": filename,
        "type": extension,
        "score": doc_score,
        "flags": all_flags,
        "text": results.get("text", ""),
        "details": results,
    }


def run_full_analysis(
    files: list[tuple[str, bytes]], milestone_description: str
) -> dict:
    """
    Analyze all submitted documents and produce a combined score.

    Pipeline:
      1. Per-document forensic analysis (metadata + ELA)
      2. LLM cross-document consistency analysis
      3. Weighted combination of all signals
    """
    all_flags = []
    doc_results = []
    doc_texts = []

    # Per-document analysis
    for filename, file_bytes in files:
        result = analyze_document(file_bytes, filename, milestone_description)
        doc_results.append(result)
        all_flags.extend(result["flags"])

        if result["text"]:
            doc_texts.append({"filename": filename, "text": result["text"]})

    # LLM cross-document analysis
    llm_result = analyze_with_llm(doc_texts, milestone_description)
    all_flags.extend(llm_result["flags"])

    # Combine scores
    # Per-document scores: 50% weight
    # LLM analysis: 50% weight
    if doc_results:
        avg_doc_score = int(sum(d["score"] for d in doc_results) / len(doc_results))
    else:
        avg_doc_score = 5000

    llm_score = llm_result["score"]
    final_score = int(avg_doc_score * 0.5 + llm_score * 0.5)

    # Confidence: based on how many documents were submitted
    # and whether text was extractable
    doc_count = len(files)
    has_text = len(doc_texts) > 0

    if doc_count >= 3 and has_text:
        confidence = 8500
    elif doc_count >= 2 and has_text:
        confidence = 7000
    elif doc_count >= 1 and has_text:
        confidence = 5500
    elif doc_count >= 1:
        confidence = 3500
    else:
        confidence = 1000
        all_flags.append("No documents submitted for analysis")

    return {
        "score": final_score,
        "confidence": confidence,
        "documents": doc_results,
        "llm_reasoning": llm_result.get("reasoning", ""),
        "flags": list(set(all_flags)),  # deduplicate
        "doc_count": doc_count,
        "signals": {
            "document_forensics": avg_doc_score,
            "llm_consistency": llm_score,
            "overall": final_score,
        },
    }


# ─────────────────────────────────────────────────────────────────
# FASTAPI ROUTES
# ─────────────────────────────────────────────────────────────────


@app.post("/analyze")
async def analyze(
    milestone_description: str = Form(...), files: list[UploadFile] = File(...)
):
    """Analyze submitted documents for milestone verification."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    file_data = []
    for upload in files:
        content = await upload.read()
        file_data.append((upload.filename, content))

    try:
        result = run_full_analysis(file_data, milestone_description)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "agent": "document"}


# ─────────────────────────────────────────────────────────────────
# DIRECT TEST
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python agent.py <file_path> <milestone_description>")
        print('Example: python agent.py report.pdf "Prototype complete"')
        sys.exit(1)

    file_path = sys.argv[1]
    milestone = sys.argv[2]

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    filename = Path(file_path).name
    print(f"Analyzing {filename}...")

    result = run_full_analysis([(filename, file_bytes)], milestone)
    print(json.dumps(result, indent=2, default=str))
