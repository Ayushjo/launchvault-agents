"""
osint_agent/agent.py

Open-Source Intelligence (OSINT) agent for startup verification.

Analyzes a company's real-world presence across six independent signal
layers to determine whether the entity is genuine before milestone
funds are released to founders.

Six signal layers (each scored 0-10000):
  1. Domain Intelligence  — WHOIS age, DNS health, SSL cert, website activity
  2. Company Registration — Companies House (UK), OpenCorporates (global), SEC EDGAR (US)
  3. Web Presence         — LinkedIn, Twitter/X, App Store, Google Play, Product Hunt
  4. News Coverage        — Google News RSS: mentions, dates, milestone alignment
  5. Team Verification    — GitHub profile existence for named contributors
  6. Claude Consistency   — AI synthesis of all OSINT signals vs milestone claim

Scoring philosophy:
  - Absence of data is NOT the same as negative evidence.
    A pre-seed startup may have no press coverage and no Companies House
    filing. Confidence is lowered but score stays neutral.
  - Presence of contradictions IS negative evidence.
    A domain registered 4 days ago for a company claiming 2 years of
    operation is a hard red flag.
  - Timeline consistency is weighted heavily.
    Everything should predate the campaign by a plausible margin.

FastAPI endpoint:
  POST /analyze
  Body: JSON { company_name, website?, milestone_description,
                github_url?, team_members? }
  Returns: { score, confidence, signals, verified_facts, flags, ... }
"""

import json
import os
import re
import socket
import ssl
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import anthropic
import dns.resolver
import requests
import whois
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="LaunchVault OSINT Agent")
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# ── Constants ─────────────────────────────────────────────────────────────────

REQ_TIMEOUT = 8  # seconds per HTTP request
REQ_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

PARKED_INDICATORS = [
    "domain for sale", "this domain is for sale", "buy this domain",
    "parked by", "domain parking", "coming soon", "under construction",
    "sedo.com", "godaddy.com/domains", "namecheap.com",
    "this web page is parked", "placeholder page",
]

WEBSITE_BUILDER_INDICATORS = [
    "wix.com", "squarespace.com", "webflow.io", "weebly.com",
    "wordpress.com", "carrd.co", "strikingly.com",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_domain(website: str) -> str | None:
    """Extract bare domain from any URL or domain string."""
    if not website:
        return None
    website = website.strip().lower()
    if not website.startswith("http"):
        website = "https://" + website
    try:
        parsed = urlparse(website)
        domain = parsed.netloc or parsed.path
        # Strip www.
        domain = re.sub(r"^www\.", "", domain)
        # Strip port if present
        domain = domain.split(":")[0]
        return domain if "." in domain else None
    except Exception:
        return None


def slug_from_name(name: str) -> str:
    """Convert company name to URL-slug format."""
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug


def safe_get(url: str, timeout: int = REQ_TIMEOUT, **kwargs) -> requests.Response | None:
    """GET request with full error suppression."""
    try:
        resp = requests.get(
            url, headers=REQ_HEADERS, timeout=timeout,
            allow_redirects=True, **kwargs
        )
        return resp
    except Exception:
        return None


def safe_head(url: str, timeout: int = REQ_TIMEOUT) -> int | None:
    """HEAD request — returns status code or None."""
    try:
        resp = requests.head(
            url, headers=REQ_HEADERS, timeout=timeout,
            allow_redirects=True
        )
        return resp.status_code
    except Exception:
        return None


def days_since(dt: datetime) -> int | None:
    """Days elapsed since a datetime. Returns None if input is None."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max(0, (datetime.now(timezone.utc) - dt).days)


# ── Layer 1: Domain Intelligence ──────────────────────────────────────────────

def analyze_domain(domain: str) -> dict:
    """
    WHOIS + DNS + HTTP + SSL analysis.

    Scoring philosophy:
      - A domain registered years ago is a strong positive signal.
      - A domain registered < 90 days ago is suspicious for an entity
        claiming operational history.
      - Active DNS records (MX = real email, A = real hosting) add confidence.
      - A live, content-rich website is evidence of real operations.
    """
    flags: list[str] = []
    verified_facts: list[str] = []
    score = 5000  # neutral start
    domain_age_days: int | None = None
    whois_data: dict = {}
    dns_data: dict = {}
    http_data: dict = {}

    # ── WHOIS ─────────────────────────────────────────────────────
    try:
        w = whois.whois(domain)
        creation = w.creation_date
        if isinstance(creation, list):
            creation = creation[0]

        domain_age_days = days_since(creation)
        registrar = str(w.registrar or "").strip()
        privacy_protected = any(
            p in str(w.org or "").lower()
            for p in ["privacy", "whoisguard", "domains by proxy", "perfect privacy", "withheld"]
        )

        whois_data = {
            "creation_date": creation.isoformat() if creation else None,
            "domain_age_days": domain_age_days,
            "registrar": registrar,
            "privacy_protected": privacy_protected,
            "country": str(w.country or "").strip() or None,
            "registrant_org": str(w.org or "").strip() or None,
        }

        if domain_age_days is not None:
            if domain_age_days < 7:
                score = 1500
                flags.append(
                    f"Domain '{domain}' was registered {domain_age_days} day(s) ago — "
                    "extremely suspicious for an entity claiming operational history"
                )
            elif domain_age_days < 30:
                score = 2500
                flags.append(f"Domain registered only {domain_age_days} days ago")
            elif domain_age_days < 90:
                score = 3500
                flags.append(f"Domain is less than 3 months old ({domain_age_days} days)")
            elif domain_age_days < 180:
                score = 5000
            elif domain_age_days < 365:
                score = 6500
                verified_facts.append(f"Domain registered {domain_age_days} days ago (6-12 months)")
            elif domain_age_days < 730:
                score = 7500
                verified_facts.append(f"Domain registered {domain_age_days} days ago (1-2 years old)")
            else:
                score = 8500
                years = domain_age_days // 365
                verified_facts.append(f"Domain has been registered for {years}+ year(s) — established presence")

        if privacy_protected:
            flags.append("WHOIS privacy protection enabled — registrant identity hidden")
        elif whois_data["registrant_org"]:
            verified_facts.append(f"Registrant organization: {whois_data['registrant_org']}")

    except Exception as e:
        flags.append(f"WHOIS lookup failed for '{domain}': {str(e)[:80]}")
        whois_data["error"] = str(e)[:80]

    # ── DNS ───────────────────────────────────────────────────────
    has_mx = False
    has_a = False
    has_txt = False

    try:
        dns.resolver.resolve(domain, "MX")
        has_mx = True
        score = min(10000, score + 400)
        verified_facts.append("Domain has MX records — real email infrastructure configured")
    except Exception:
        flags.append("No MX records found — domain may not have email infrastructure")

    try:
        answers = dns.resolver.resolve(domain, "A")
        has_a = True
        ips = [str(r) for r in answers]
        score = min(10000, score + 200)
        verified_facts.append(f"Domain resolves to IP: {ips[0]}")
    except Exception:
        flags.append("Domain has no A record — no active web hosting found")

    try:
        dns.resolver.resolve(domain, "TXT")
        has_txt = True
        # TXT records often contain SPF/DKIM/verification tokens — real business indicator
        score = min(10000, score + 200)
        verified_facts.append("TXT records present (SPF/DKIM/verification tokens found)")
    except Exception:
        pass

    dns_data = {
        "has_mx": has_mx,
        "has_a": has_a,
        "has_txt": has_txt,
    }

    # ── HTTP ──────────────────────────────────────────────────────
    website_url = f"https://{domain}"
    resp = safe_get(website_url)
    if resp is None:
        resp = safe_get(f"http://{domain}")

    if resp is not None and resp.status_code == 200:
        content = resp.text.lower()
        content_len = len(resp.text)

        is_parked = any(ind in content for ind in PARKED_INDICATORS)
        is_website_builder = any(ind in content for ind in WEBSITE_BUILDER_INDICATORS)

        if is_parked:
            score = max(0, score - 1500)
            flags.append("Website appears to be a parked domain or placeholder page")
        elif is_website_builder:
            flags.append("Website built on a drag-and-drop builder (Wix/Squarespace/etc.) — low engineering credibility")
        elif content_len > 20000:
            score = min(10000, score + 300)
            verified_facts.append(f"Active website with substantial content ({content_len // 1000}KB)")
        elif content_len > 5000:
            score = min(10000, score + 150)
            verified_facts.append("Active website found")
        else:
            flags.append("Website exists but has very little content")

        # Check SSL
        try:
            ctx = ssl.create_default_context()
            with ctx.wrap_socket(socket.create_connection((domain, 443), timeout=5), server_hostname=domain) as ssock:
                cert = ssock.getpeercert()
                not_after_str = cert.get("notAfter", "")
                if not_after_str:
                    not_after = datetime.strptime(not_after_str, "%b %d %H:%M:%S %Y %Z").replace(tzinfo=timezone.utc)
                    days_left = (not_after - datetime.now(timezone.utc)).days
                    if days_left > 0:
                        score = min(10000, score + 200)
                        verified_facts.append(f"Valid SSL certificate (expires in {days_left} days)")
                    else:
                        flags.append("SSL certificate has expired")
        except Exception:
            pass

        http_data = {
            "status_code": resp.status_code,
            "content_length": content_len,
            "is_parked": is_parked,
            "is_website_builder": is_website_builder,
        }
    elif resp is not None:
        flags.append(f"Website returned HTTP {resp.status_code}")
        http_data = {"status_code": resp.status_code}
    else:
        if has_a:
            flags.append("Domain has an A record but website is not reachable")
        http_data = {"status_code": None}

    score = max(0, min(10000, score))
    confidence = 8000 if domain_age_days is not None else 4000

    return {
        "score": score,
        "confidence": confidence,
        "domain": domain,
        "whois": whois_data,
        "dns": dns_data,
        "http": http_data,
        "flags": flags,
        "verified_facts": verified_facts,
    }


# ── Layer 2: Company Registration ─────────────────────────────────────────────

def check_company_registration(company_name: str) -> dict:
    """
    Search official company registries:
      - Companies House (UK): free API, no auth
      - OpenCorporates (global): free tier, no auth
      - SEC EDGAR (US): free EFTS search

    Scoring:
      Found + active + incorporated > 1 year:  8500
      Found + active + incorporated < 3 months: 4000 (suspicious)
      Not found anywhere:                       4500 (neutral — many legit startups)
      Found but dissolved/struck off:           1500 (major red flag)
    """
    flags: list[str] = []
    verified_facts: list[str] = []
    score = 4500  # neutral default (not finding ≠ fraud for early startups)
    found_registrations: list[dict] = []
    confidence = 5000

    # ── Companies House (UK) ───────────────────────────────────────
    try:
        url = f"https://api.company-information.service.gov.uk/search/companies?q={requests.utils.quote(company_name)}&items_per_page=5"
        resp = requests.get(url, timeout=REQ_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            items = data.get("items", [])
            for item in items[:3]:
                name_match = company_name.lower() in item.get("title", "").lower()
                if name_match:
                    inc_date_str = item.get("date_of_creation", "")
                    status = item.get("company_status", "").lower()
                    company_type = item.get("company_type", "")

                    entry = {
                        "registry": "Companies House (UK)",
                        "name": item.get("title"),
                        "number": item.get("company_number"),
                        "status": status,
                        "type": company_type,
                        "incorporated": inc_date_str,
                    }
                    found_registrations.append(entry)

                    if inc_date_str:
                        try:
                            inc_date = datetime.strptime(inc_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                            age_days = days_since(inc_date)
                            entry["age_days"] = age_days

                            if status in ("active", "registered"):
                                if age_days and age_days > 365:
                                    score = 8500
                                    years = age_days // 365
                                    verified_facts.append(
                                        f"Registered at Companies House (UK): {item.get('title')}, "
                                        f"incorporated {years}+ year(s) ago, status: active"
                                    )
                                elif age_days and age_days < 90:
                                    score = 4000
                                    flags.append(
                                        f"Company found at Companies House but only incorporated "
                                        f"{age_days} days ago — very recently formed"
                                    )
                                else:
                                    score = 7000
                                    verified_facts.append(
                                        f"Registered at Companies House: {item.get('title')}, status: active"
                                    )
                            elif status in ("dissolved", "liquidation", "receivership", "struck-off"):
                                score = 1500
                                flags.append(
                                    f"Company found at Companies House but status is '{status}' — "
                                    "company may be defunct or struck off"
                                )
                            else:
                                score = 6000
                        except ValueError:
                            pass
                    confidence = 8500
                    break
    except Exception as e:
        flags.append(f"Companies House search failed: {str(e)[:60]}")

    # ── OpenCorporates (global) ────────────────────────────────────
    if not found_registrations:
        try:
            url = (
                f"https://api.opencorporates.com/v0.4/companies/search"
                f"?q={requests.utils.quote(company_name)}&order=score&per_page=5"
            )
            resp = requests.get(url, timeout=REQ_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                companies = data.get("results", {}).get("companies", [])
                for item in companies[:3]:
                    c = item.get("company", {})
                    name_match = company_name.lower() in c.get("name", "").lower()
                    if name_match:
                        inc_date_str = c.get("incorporation_date", "")
                        status = c.get("current_status", "").lower()
                        jurisdiction = c.get("jurisdiction_code", "")

                        entry = {
                            "registry": f"OpenCorporates ({jurisdiction.upper()})",
                            "name": c.get("name"),
                            "number": c.get("company_number"),
                            "status": status,
                            "jurisdiction": jurisdiction,
                            "incorporated": inc_date_str,
                        }
                        found_registrations.append(entry)

                        if inc_date_str:
                            try:
                                inc_date = datetime.strptime(inc_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                                age_days = days_since(inc_date)
                                entry["age_days"] = age_days

                                if status and "active" in status:
                                    if age_days and age_days > 365:
                                        score = 8000
                                        verified_facts.append(
                                            f"Company found in OpenCorporates ({jurisdiction.upper()}): "
                                            f"{c.get('name')}, incorporated {age_days // 365}+ year(s) ago"
                                        )
                                    else:
                                        score = 6500
                                        verified_facts.append(
                                            f"Company found in OpenCorporates ({jurisdiction.upper()}): {c.get('name')}"
                                        )
                                elif status and any(s in status for s in ["dissolved", "inactive", "struck"]):
                                    score = 2000
                                    flags.append(
                                        f"Company found in OpenCorporates but has inactive status: '{status}'"
                                    )
                                else:
                                    score = max(score, 6000)
                                    verified_facts.append(
                                        f"Company registered in OpenCorporates ({jurisdiction.upper()})"
                                    )
                            except ValueError:
                                score = max(score, 5500)
                                verified_facts.append(
                                    f"Company registered in OpenCorporates ({jurisdiction.upper()})"
                                )
                        confidence = 7500
                        break
        except Exception as e:
            flags.append(f"OpenCorporates search failed: {str(e)[:60]}")

    # ── SEC EDGAR (US) ────────────────────────────────────────────
    if not found_registrations:
        try:
            url = (
                f"https://efts.sec.gov/LATEST/search-index?q=%22{requests.utils.quote(company_name)}%22"
                f"&dateRange=custom&startdt=2015-01-01&forms=S-1,10-K"
            )
            resp = requests.get(url, timeout=REQ_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                hits = data.get("hits", {}).get("hits", [])
                if hits:
                    entity = hits[0].get("_source", {}).get("entity_name", "")
                    if company_name.lower() in entity.lower():
                        score = max(score, 7500)
                        verified_facts.append(
                            f"Company found in SEC EDGAR filings: {entity} — publicly registered in the US"
                        )
                        confidence = 8000
        except Exception:
            pass  # SEC EDGAR failure is non-critical

    if not found_registrations and score == 4500:
        flags.append(
            "Company not found in any official registry (Companies House, OpenCorporates, SEC EDGAR) — "
            "this is normal for very early-stage startups but reduces verification confidence"
        )

    return {
        "score": score,
        "confidence": confidence,
        "registrations_found": found_registrations,
        "flags": flags,
        "verified_facts": verified_facts,
    }


# ── Layer 3: Web Presence ─────────────────────────────────────────────────────

def check_web_presence(
    company_name: str,
    domain: str | None,
    milestone_description: str,
) -> dict:
    """
    Check existence across key platforms:
      LinkedIn company page, Twitter/X, App Store (if mobile claim),
      Google Play (if Android claim), Product Hunt.

    Each found platform adds to the presence score. The breadth
    of presence signals how established the entity is.
    """
    flags: list[str] = []
    verified_facts: list[str] = []
    platforms_found: list[str] = []
    platform_details: dict = {}

    slug = slug_from_name(company_name)
    desc_lower = milestone_description.lower()
    claims_mobile = any(
        w in desc_lower for w in ["android", "ios", "mobile", "app store", "play store", "iphone"]
    )

    # ── LinkedIn ──────────────────────────────────────────────────
    li_url = f"https://www.linkedin.com/company/{slug}/"
    li_status = safe_head(li_url, timeout=6)
    # LinkedIn returns 999 for bots but the profile exists; 404 = not found
    if li_status == 200:
        platforms_found.append("LinkedIn")
        platform_details["linkedin"] = {"url": li_url, "status": "found"}
        verified_facts.append(f"LinkedIn company page found: linkedin.com/company/{slug}")
    elif li_status == 999:
        # 999 = "request blocked" but page exists — LinkedIn anti-scraping
        platforms_found.append("LinkedIn")
        platform_details["linkedin"] = {"url": li_url, "status": "exists_blocked"}
        verified_facts.append(f"LinkedIn company page exists (access blocked by LinkedIn): /company/{slug}")
    elif li_status == 404:
        flags.append(f"No LinkedIn company page found at /company/{slug} — tried common slug format")
        platform_details["linkedin"] = {"url": li_url, "status": "not_found"}
    else:
        platform_details["linkedin"] = {"url": li_url, "status": f"unknown_{li_status}"}

    # ── Twitter / X ───────────────────────────────────────────────
    for tw_url in [
        f"https://x.com/{slug}",
        f"https://twitter.com/{slug}",
    ]:
        tw_status = safe_head(tw_url, timeout=6)
        if tw_status in (200, 301, 302):
            platforms_found.append("Twitter/X")
            platform_details["twitter"] = {"url": tw_url, "status": "found"}
            verified_facts.append(f"Twitter/X account found: @{slug}")
            break
        elif tw_status == 404:
            platform_details["twitter"] = {"url": tw_url, "status": "not_found"}
            break
    else:
        if "twitter" not in platform_details:
            flags.append(f"No Twitter/X account found for @{slug}")

    # ── Product Hunt ──────────────────────────────────────────────
    ph_url = f"https://www.producthunt.com/search?q={requests.utils.quote(company_name)}"
    ph_resp = safe_get(ph_url, timeout=6)
    if ph_resp and ph_resp.status_code == 200:
        if company_name.lower() in ph_resp.text.lower():
            platforms_found.append("Product Hunt")
            platform_details["product_hunt"] = {"status": "found"}
            verified_facts.append(f"'{company_name}' appears in Product Hunt search results")

    # ── App Store (if mobile product claimed) ─────────────────────
    if claims_mobile:
        try:
            itunes_url = (
                f"https://itunes.apple.com/search?term={requests.utils.quote(company_name)}"
                f"&entity=software&limit=5"
            )
            itunes_resp = safe_get(itunes_url, timeout=6)
            if itunes_resp and itunes_resp.status_code == 200:
                results = itunes_resp.json().get("results", [])
                # Check if any result name-matches
                matches = [
                    r for r in results
                    if company_name.lower() in r.get("trackName", "").lower()
                    or company_name.lower() in r.get("sellerName", "").lower()
                ]
                if matches:
                    app = matches[0]
                    platforms_found.append("App Store")
                    platform_details["app_store"] = {
                        "status": "found",
                        "app_name": app.get("trackName"),
                        "rating": app.get("averageUserRating"),
                        "ratings_count": app.get("userRatingCount"),
                    }
                    verified_facts.append(
                        f"App Store listing found: '{app.get('trackName')}' "
                        f"({app.get('userRatingCount', 0):,} ratings)"
                    )
                else:
                    flags.append(
                        "Milestone claims a mobile app but no matching App Store listing found"
                    )
                    platform_details["app_store"] = {"status": "not_found"}
        except Exception:
            pass

    # ── Score from platform breadth ───────────────────────────────
    n = len(platforms_found)
    if n == 0:
        score = 2500
        flags.append("No web presence found on any major platform (LinkedIn, Twitter, Product Hunt)")
    elif n == 1:
        score = 4500
    elif n == 2:
        score = 6500
        verified_facts.append(f"Found on {n} platforms: {', '.join(platforms_found)}")
    elif n == 3:
        score = 7500
        verified_facts.append(f"Found on {n} platforms: {', '.join(platforms_found)}")
    else:
        score = 8500
        verified_facts.append(f"Strong web presence across {n} platforms: {', '.join(platforms_found)}")

    # Bonus: if claiming mobile app AND it's found
    if "App Store" in platforms_found:
        score = min(10000, score + 1000)

    confidence = 6000 + min(2000, n * 500)

    return {
        "score": score,
        "confidence": confidence,
        "platforms_found": platforms_found,
        "platform_details": platform_details,
        "flags": flags,
        "verified_facts": verified_facts,
    }


# ── Layer 4: News Coverage ────────────────────────────────────────────────────

def check_news_coverage(
    company_name: str,
    milestone_description: str,
    domain: str | None = None,
) -> dict:
    """
    Fetch Google News RSS to find mentions of the company.

    Key insight: Presence of news articles that PREDATE the campaign
    is strong evidence the company existed before the fundraise.
    A first mention that coincides exactly with campaign launch
    is suspicious (PR stunt vs. genuine operations).
    """
    flags: list[str] = []
    verified_facts: list[str] = []
    articles: list[dict] = []
    score = 4500  # neutral — most early startups have no press

    queries = [company_name]
    if domain:
        queries.append(domain.split(".")[0])  # domain root as alternate search

    all_articles: list[dict] = []

    for query in queries[:2]:
        try:
            rss_url = (
                f"https://news.google.com/rss/search"
                f"?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
            )
            resp = safe_get(rss_url, timeout=8)
            if not resp or resp.status_code != 200:
                continue

            soup = BeautifulSoup(resp.text, "xml")
            items = soup.find_all("item")

            for item in items[:20]:
                title = item.find("title")
                pub_date = item.find("pubDate")
                link = item.find("link")
                source = item.find("source")

                title_text = title.get_text().strip() if title else ""
                pub_date_text = pub_date.get_text().strip() if pub_date else ""
                link_text = link.get_text().strip() if link else ""
                source_text = source.get_text().strip() if source else ""

                # Only count articles where company name actually appears in title
                if company_name.lower() not in title_text.lower():
                    continue

                article = {
                    "title": title_text,
                    "date": pub_date_text,
                    "source": source_text,
                    "url": link_text,
                }

                # Parse date
                try:
                    pub_dt = datetime.strptime(pub_date_text[:25], "%a, %d %b %Y %H:%M")
                    pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                    article["parsed_date"] = pub_dt.isoformat()
                    article["days_ago"] = days_since(pub_dt)
                except Exception:
                    article["parsed_date"] = None
                    article["days_ago"] = None

                all_articles.append(article)

        except Exception as e:
            flags.append(f"News search failed: {str(e)[:60]}")

    # Deduplicate by title
    seen_titles: set[str] = set()
    for a in all_articles:
        t = a["title"].lower()[:60]
        if t not in seen_titles:
            seen_titles.add(t)
            articles.append(a)

    count = len(articles)

    if count == 0:
        flags.append(
            f"No news coverage found for '{company_name}' — "
            "normal for early-stage startups but reduces verification confidence"
        )
    elif count >= 10:
        score = 8500
        verified_facts.append(f"Strong media coverage: {count} news articles found about {company_name}")
    elif count >= 4:
        score = 7000
        verified_facts.append(f"Media coverage found: {count} articles mentioning {company_name}")
    elif count >= 1:
        score = 5500
        verified_facts.append(f"{count} news article(s) found mentioning {company_name}")

    # Timeline analysis — are articles old enough to predate the campaign?
    dated_articles = [a for a in articles if a.get("days_ago") is not None]
    if dated_articles:
        oldest = max(dated_articles, key=lambda a: a["days_ago"])
        oldest_days = oldest["days_ago"]

        if oldest_days > 365:
            score = min(10000, score + 1000)
            verified_facts.append(
                f"Earliest press coverage is {oldest_days} days old ({oldest_days // 30} months) — "
                "company has demonstrable media history predating this campaign"
            )
        elif oldest_days > 90:
            score = min(10000, score + 400)
            verified_facts.append(f"News coverage exists from {oldest_days} days ago")
        elif oldest_days < 14 and count > 0:
            flags.append(
                f"All {count} news article(s) are very recent (within 2 weeks) — "
                "possible PR campaign coinciding with fundraise, not evidence of prior operations"
            )

        # Check if milestone-related keywords appear in any headline
        desc_words = set(re.findall(r"\b\w{4,}\b", milestone_description.lower()))
        for article in articles:
            title_words = set(re.findall(r"\b\w{4,}\b", article["title"].lower()))
            overlap = desc_words & title_words - {"that", "this", "with", "from", "have", "been", "will"}
            if len(overlap) >= 2:
                verified_facts.append(
                    f"News article relates to milestone claim: \"{article['title'][:80]}...\""
                )
                score = min(10000, score + 500)
                break

    confidence = 6000 if count > 0 else 4000

    return {
        "score": score,
        "confidence": confidence,
        "article_count": count,
        "articles": articles[:10],  # cap at 10 for report
        "flags": flags,
        "verified_facts": verified_facts,
    }


# ── Layer 5: Team Verification ────────────────────────────────────────────────

def verify_team_members(
    team_members: list[str],
    github_url: str | None = None,
) -> dict:
    """
    Verify named team members exist and have credible profiles:
      - GitHub profile (via API — reuses existing token)
      - LinkedIn profile (guessed slug)

    For each member, check:
      1. GitHub username guess (first.last, firstlast, first-last)
      2. GitHub profile has real activity (repos, followers)

    A startup claiming a team of engineers but having no
    GitHub profiles for any of them is a red flag.
    """
    if not team_members:
        return {
            "score": 5000,
            "confidence": 2000,
            "members_verified": 0,
            "members_total": 0,
            "profiles": [],
            "flags": ["No team member names provided — team verification skipped"],
            "verified_facts": [],
        }

    flags: list[str] = []
    verified_facts: list[str] = []
    profiles: list[dict] = []

    gh_headers = {}
    if GITHUB_TOKEN:
        gh_headers["Authorization"] = f"token {GITHUB_TOKEN}"
        gh_headers["Accept"] = "application/vnd.github.v3+json"

    # Extract GitHub org name from URL if provided
    gh_org = None
    if github_url:
        match = re.search(r"github\.com/([^/]+)", github_url)
        if match:
            gh_org = match.group(1)

    members_verified = 0

    for full_name in team_members[:6]:  # cap at 6 to avoid rate limits
        full_name = full_name.strip()
        if not full_name:
            continue

        parts = full_name.lower().split()
        member_profile: dict = {"name": full_name, "github": None, "linkedin": None}

        # Generate username candidates
        candidates = []
        if len(parts) >= 2:
            candidates = [
                f"{parts[0]}{parts[-1]}",        # johnsmith
                f"{parts[0]}-{parts[-1]}",        # john-smith
                f"{parts[0]}.{parts[-1]}",        # john.smith
                parts[0],                          # john
                f"{parts[0][0]}{parts[-1]}",       # jsmith
            ]
        elif len(parts) == 1:
            candidates = [parts[0]]

        # Check GitHub
        found_gh = False
        for username in candidates:
            try:
                resp = requests.get(
                    f"https://api.github.com/users/{username}",
                    headers=gh_headers, timeout=5
                )
                if resp.status_code == 200:
                    user = resp.json()
                    repos = user.get("public_repos", 0)
                    followers = user.get("followers", 0)

                    member_profile["github"] = {
                        "username": username,
                        "repos": repos,
                        "followers": followers,
                        "name": user.get("name", ""),
                        "company": user.get("company", ""),
                    }

                    # Verify name match
                    gh_name = (user.get("name") or "").lower()
                    name_parts_match = any(p in gh_name for p in parts if len(p) > 2)

                    if name_parts_match and repos > 0:
                        members_verified += 1
                        verified_facts.append(
                            f"GitHub profile verified: {full_name} → @{username} "
                            f"({repos} repos, {followers} followers)"
                        )
                        found_gh = True
                    elif repos > 5:
                        # Username matches, active profile
                        members_verified += 1
                        verified_facts.append(f"Likely GitHub profile: @{username} ({repos} repos)")
                        found_gh = True

                    break
            except Exception:
                pass

        if not found_gh:
            flags.append(f"No GitHub profile found for '{full_name}' using common username patterns")

        # Check LinkedIn (guessed slug)
        li_slug = "-".join(parts[:2]) if len(parts) >= 2 else parts[0]
        li_url = f"https://www.linkedin.com/in/{li_slug}/"
        li_status = safe_head(li_url, timeout=5)
        if li_status in (200, 999):
            member_profile["linkedin"] = {"url": li_url, "status": "found"}
            verified_facts.append(f"LinkedIn profile likely exists for {full_name}: /in/{li_slug}")

        profiles.append(member_profile)

    total = len([m for m in team_members if m.strip()])

    if total == 0:
        score = 5000
        confidence = 2000
    elif members_verified == 0:
        score = 3000
        flags.append(
            f"None of the {total} named team member(s) could be verified on GitHub"
        )
        confidence = 5000
    elif members_verified >= total:
        score = 8500
        confidence = 7500
        verified_facts.append(f"All {total} team member(s) verified on GitHub")
    elif members_verified >= total * 0.6:
        score = 7000
        confidence = 6500
        verified_facts.append(f"{members_verified}/{total} team member(s) verified")
    else:
        score = 5000
        confidence = 5000
        flags.append(f"Only {members_verified}/{total} team member(s) verified on GitHub")

    return {
        "score": score,
        "confidence": confidence,
        "members_verified": members_verified,
        "members_total": total,
        "profiles": profiles,
        "flags": flags,
        "verified_facts": verified_facts,
    }


# ── Layer 6: Claude Consistency Analysis ──────────────────────────────────────

def analyze_consistency_with_claude(
    company_name: str,
    milestone_description: str,
    domain_result: dict,
    registration_result: dict,
    presence_result: dict,
    news_result: dict,
    team_result: dict,
) -> dict:
    """
    Use Claude to reason about timeline consistency and cross-signal
    contradictions that raw scoring cannot detect:

    Examples:
      - Domain registered 2 weeks ago but company claims 3 years of operation
      - App Store listing found with 50k ratings (strong positive) but company
        not in any registry (weak negative) — Claude weighs these correctly
      - News articles from 2 years ago (strong positive) but team members
        have no GitHub history (weak negative) — nuanced assessment
    """
    domain_age = domain_result.get("whois", {}).get("domain_age_days")
    domain_age_str = f"{domain_age} days old" if domain_age else "unknown"

    registrations = registration_result.get("registrations_found", [])
    reg_summary = (
        ", ".join(f"{r['registry']}: {r['name']} (status: {r['status']})" for r in registrations)
        if registrations
        else "Not found in any checked registry"
    )

    platforms = presence_result.get("platforms_found", [])
    articles = news_result.get("article_count", 0)

    all_verified = (
        domain_result.get("verified_facts", [])
        + registration_result.get("verified_facts", [])
        + presence_result.get("verified_facts", [])
        + news_result.get("verified_facts", [])
        + team_result.get("verified_facts", [])
    )
    all_flags = (
        domain_result.get("flags", [])
        + registration_result.get("flags", [])
        + presence_result.get("flags", [])
        + news_result.get("flags", [])
        + team_result.get("flags", [])
    )

    prompt = f"""You are an expert OSINT analyst reviewing a startup founder's milestone completion claim for a blockchain crowdfunding platform. Investors will use your assessment to decide whether to release funds.

COMPANY: {company_name}
MILESTONE CLAIM: "{milestone_description}"

OSINT FINDINGS:

Domain Intelligence (score {domain_result.get('score', 0)}/10000):
  Domain age: {domain_age_str}
  Has MX records: {domain_result.get('dns', {}).get('has_mx', False)}
  Website active: {domain_result.get('http', {}).get('status_code') == 200}
  Key facts: {'; '.join(domain_result.get('verified_facts', ['none'])[:3])}
  Flags: {'; '.join(domain_result.get('flags', ['none'])[:3])}

Company Registration (score {registration_result.get('score', 0)}/10000):
  {reg_summary}

Web Presence (score {presence_result.get('score', 0)}/10000):
  Platforms found: {platforms if platforms else 'None'}

News Coverage (score {news_result.get('score', 0)}/10000):
  Article count: {articles}
  Key findings: {'; '.join(news_result.get('verified_facts', ['none'])[:2])}

Team Verification (score {team_result.get('score', 0)}/10000):
  Members verified: {team_result.get('members_verified', 0)}/{team_result.get('members_total', 0)}

TOP VERIFIED FACTS:
{chr(10).join(f'  + {f}' for f in all_verified[:8])}

TOP FLAGS / CONCERNS:
{chr(10).join(f'  ! {f}' for f in all_flags[:8])}

Your task: Synthesize these OSINT signals into a calibrated entity verification assessment.

Focus especially on:
1. Timeline consistency — does the entity's history plausibly predate this fundraise?
2. Signal coherence — do the signals tell a consistent story, or are there contradictions?
3. Red flag weight — how much should each concern lower the score?
4. Absence of evidence — not finding a company in a registry is NOT fraud for a seed-stage startup

Respond with ONLY a JSON object:
{{
  "osint_score": <0-10000>,
  "verdict": "<VERIFIED|PARTIAL|UNVERIFIED|SUSPICIOUS>",
  "confidence": <0-10000>,
  "consistency_assessment": "<2-3 sentences on whether signals tell a consistent story>",
  "strongest_positive": "<single most compelling evidence of legitimacy>",
  "strongest_concern": "<single most important red flag, or 'none' if clean>",
  "score_adjustments": [
    {{"signal": "<name>", "direction": "<up|down>", "reason": "<why>"}}
  ],
  "recommendation": "<one sentence for investors>"
}}

Scoring guide:
  8000-10000: Entity clearly exists and has verifiable history — legitimate
  6000-7999:  Likely legitimate with some gaps in evidence
  4000-5999:  Insufficient evidence to verify, neutral
  2000-3999:  Multiple concerns, hard to verify
  0-1999:     Strong indicators of fraudulent or non-existent entity"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:-1])

        result = json.loads(text)
        return {
            "osint_score": max(0, min(10000, int(result.get("osint_score", 5000)))),
            "verdict": result.get("verdict", "PARTIAL"),
            "confidence": result.get("confidence", 5000),
            "consistency_assessment": result.get("consistency_assessment", ""),
            "strongest_positive": result.get("strongest_positive", ""),
            "strongest_concern": result.get("strongest_concern", "none"),
            "score_adjustments": result.get("score_adjustments", []),
            "recommendation": result.get("recommendation", ""),
            "error": None,
        }
    except Exception as e:
        # Fallback: weighted average of raw scores
        raw_scores = [
            domain_result.get("score", 5000),
            registration_result.get("score", 4500),
            presence_result.get("score", 5000),
            news_result.get("score", 4500),
            team_result.get("score", 5000),
        ]
        fallback_score = int(sum(raw_scores) / len(raw_scores))
        return {
            "osint_score": fallback_score,
            "verdict": "PARTIAL",
            "confidence": 3000,
            "consistency_assessment": f"Claude analysis failed ({e}). Used raw signal average.",
            "strongest_positive": "",
            "strongest_concern": "",
            "score_adjustments": [],
            "recommendation": "Manual review recommended — AI synthesis unavailable.",
            "error": str(e),
        }


# ── Main Analysis Pipeline ────────────────────────────────────────────────────

def analyze_entity(
    company_name: str,
    milestone_description: str,
    website: str | None = None,
    github_url: str | None = None,
    team_members: list[str] | None = None,
) -> dict:
    """
    Run all OSINT layers in parallel then synthesize with Claude.

    Returns a structured report with:
      - score (0-10000): overall entity legitimacy score
      - confidence (0-10000): how confident we are in the score
      - per-layer signal breakdown
      - verified_facts: concrete evidence found
      - flags: concerns and red flags
      - Claude consistency verdict
    """
    domain = extract_domain(website) if website else None
    team_members = team_members or []

    results: dict = {}
    errors: dict = {}

    # ── Run all layers concurrently (except Claude synthesis) ────
    def run_domain():
        if domain:
            return "domain", analyze_domain(domain)
        return "domain", {
            "score": 4000, "confidence": 1000,
            "flags": ["No website provided — domain intelligence skipped"],
            "verified_facts": [], "whois": {}, "dns": {}, "http": {},
        }

    def run_registration():
        return "registration", check_company_registration(company_name)

    def run_presence():
        return "presence", check_web_presence(company_name, domain, milestone_description)

    def run_news():
        return "news", check_news_coverage(company_name, milestone_description, domain)

    def run_team():
        return "team", verify_team_members(team_members, github_url)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(run_domain): "domain",
            executor.submit(run_registration): "registration",
            executor.submit(run_presence): "presence",
            executor.submit(run_news): "news",
            executor.submit(run_team): "team",
        }
        for future in as_completed(futures, timeout=45):
            try:
                key, result = future.result(timeout=45)
                results[key] = result
            except Exception as e:
                layer = futures[future]
                errors[layer] = str(e)
                results[layer] = {
                    "score": 5000, "confidence": 1000,
                    "flags": [f"Layer '{layer}' failed: {str(e)[:80]}"],
                    "verified_facts": [],
                }

    domain_r     = results.get("domain", {})
    reg_r        = results.get("registration", {})
    presence_r   = results.get("presence", {})
    news_r       = results.get("news", {})
    team_r       = results.get("team", {})

    # ── Claude consistency synthesis ─────────────────────────────
    claude_r = analyze_consistency_with_claude(
        company_name, milestone_description,
        domain_r, reg_r, presence_r, news_r, team_r,
    )

    # ── Final score: weighted average of Claude's adjusted score + raw layers ──
    # Claude's synthesis is the primary score, raw signals add confidence bounds
    raw_weighted = (
        domain_r.get("score", 5000)     * 0.25 +
        reg_r.get("score", 4500)        * 0.25 +
        presence_r.get("score", 5000)   * 0.20 +
        news_r.get("score", 4500)       * 0.15 +
        team_r.get("score", 5000)       * 0.15
    )
    claude_score = claude_r.get("osint_score", raw_weighted)

    # Blend: 60% Claude, 40% raw weighted
    final_score = int(claude_score * 0.6 + raw_weighted * 0.4)
    final_score = max(0, min(10000, final_score))

    # Confidence: average of per-layer confidences
    layer_confidences = [
        domain_r.get("confidence", 5000),
        reg_r.get("confidence", 5000),
        presence_r.get("confidence", 5000),
        news_r.get("confidence", 5000),
        team_r.get("confidence", 5000),
        claude_r.get("confidence", 5000),
    ]
    final_confidence = int(sum(layer_confidences) / len(layer_confidences))

    # Aggregate facts and flags
    all_verified = []
    all_flags = []
    for r in [domain_r, reg_r, presence_r, news_r, team_r]:
        all_verified.extend(r.get("verified_facts", []))
        all_flags.extend(r.get("flags", []))

    return {
        "score": final_score,
        "confidence": final_confidence,
        "verdict": claude_r.get("verdict", "PARTIAL"),
        "consistency_assessment": claude_r.get("consistency_assessment", ""),
        "strongest_positive": claude_r.get("strongest_positive", ""),
        "strongest_concern": claude_r.get("strongest_concern", ""),
        "recommendation": claude_r.get("recommendation", ""),
        "company_name": company_name,
        "domain_analyzed": domain,
        "signals": {
            "domain_intelligence":   domain_r.get("score", 5000),
            "company_registration":  reg_r.get("score", 4500),
            "web_presence":          presence_r.get("score", 5000),
            "news_coverage":         news_r.get("score", 4500),
            "team_verification":     team_r.get("score", 5000),
            "claude_consistency":    claude_score,
        },
        "verified_facts": list(dict.fromkeys(all_verified)),   # deduplicated, ordered
        "flags": list(dict.fromkeys(all_flags)),
        "details": {
            "domain":       domain_r,
            "registration": reg_r,
            "presence":     presence_r,
            "news":         news_r,
            "team":         team_r,
        },
        "errors": errors,
    }


# ── FastAPI ───────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    company_name: str
    milestone_description: str
    website: Optional[str] = None
    github_url: Optional[str] = None
    team_members: Optional[list[str]] = None


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    if not req.company_name.strip():
        raise HTTPException(status_code=400, detail="company_name is required")
    try:
        return analyze_entity(
            company_name=req.company_name.strip(),
            milestone_description=req.milestone_description,
            website=req.website,
            github_url=req.github_url,
            team_members=req.team_members,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OSINT analysis failed: {str(e)}")


@app.get("/health")
async def health():
    return {"status": "ok", "agent": "osint"}


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LaunchVault OSINT Agent")
    parser.add_argument("company", help="Company name to verify")
    parser.add_argument("claim", help="Milestone claim / description")
    parser.add_argument("--website", default=None)
    parser.add_argument("--github", default=None)
    parser.add_argument("--team", help="Comma-separated team member names", default=None)
    args = parser.parse_args()

    team = [t.strip() for t in args.team.split(",")] if args.team else []

    print(f"\nRunning OSINT analysis for: {args.company}")
    result = analyze_entity(
        company_name=args.company,
        milestone_description=args.claim,
        website=args.website,
        github_url=args.github,
        team_members=team,
    )
    print(json.dumps(result, indent=2, default=str))
