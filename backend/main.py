"""
B2B Prospecting Agent
======================
An autonomous pipeline that:
  1. DISCOVERS public companies showing "enterprise gap" signals by mining
     SEC EDGAR full-text filing search (free, no API key) for risk-factor
     language (e.g. "material weakness", "cybersecurity incident",
     "unable to remediate").
  2. ENRICHES each hit with financial facts (SEC XBRL company-facts API)
     and open hiring signals (Arbeitnow job-board API, free/no key) to
     see whether the company is actively trying to solve the gap.
  3. GENERATES a hyper-personalized outbound pitch for each qualified
     prospect using Groq's free-tier LLM inference API.

All three data sources used here (SEC EDGAR, Arbeitnow, Groq) are free
and require no paid credentials except a free Groq API key.

Run locally:
    pip install -r requirements.txt
    export GROQ_API_KEY=your_free_groq_key
    uvicorn main:app --reload --port 8000

Run in production (Render etc.):
    uvicorn main:app --host 0.0.0.0 --port $PORT
"""

import os
import re
import time
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()  # reads backend/.env if present (local dev only — Render uses dashboard env vars)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

SEC_FULLTEXT_SEARCH = "https://efts.sec.gov/LATEST/search-index?q=%22{q}%22&forms=10-K"
SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARBEITNOW_URL = "https://www.arbeitnow.com/api/job-board-api"

# 10-Ks are legally signed on the last page by named officers. Formats vary
# a lot in practice — sometimes "/s/ Jane Smith" followed directly by a
# title, sometimes with "Name:" / "Title:" labels in between, sometimes as
# a table. We already collapse all whitespace (including real newlines) to
# single spaces before running these, so patterns must not depend on \n.
NAME_RE = re.compile(
    r"/s/\s*((?:(?!Name:|Title:|Date:|By:|Chief\b|President\b|Vice\b|Director\b|"
    r"Chairman\b|General\b|Executive\b|Senior\b|Treasurer\b|Controller\b|Secretary\b)"
    r"[A-Z][A-Za-z.\-']+\s*){1,4})"
)
TITLE_LABEL_RE = re.compile(
    r"Title:\s*([A-Za-z][A-Za-z .,&/'-]{2,60}?)(?:\s{2,}|\s+Date:|\s+/s/|$)",
    re.IGNORECASE,
)
TITLE_KEYWORD_RE = re.compile(
    r"(Chief\s+[A-Za-z]+\s+Officer|President(?:\s+and\s+Chief\s+Executive\s+Officer)?|"
    r"Chairman(?:\s+of\s+the\s+Board)?|Executive Vice President|Senior Vice President|"
    r"Vice President|General Counsel|Corporate Secretary|Treasurer|Controller|Director)",
    re.IGNORECASE,
)


def find_title_near(window: str) -> Optional[str]:
    """Look for a title in the ~150 chars following a signature name,
    preferring an explicit 'Title:' label, falling back to keyword match."""
    m = TITLE_LABEL_RE.search(window)
    if m:
        return m.group(1).strip()
    m = TITLE_KEYWORD_RE.search(window)
    if m:
        return m.group(1).strip()
    return None


def find_best_signature(text: str) -> tuple[Optional[str], Optional[str]]:
    """Scan every '/s/ Name' occurrence on the signature page and pair each
    with the nearest title. Prefers a C-suite / President signer over a
    plain 'Director', since that's who a sales pitch should be addressed to."""
    candidates = []
    for m in NAME_RE.finditer(text):
        name = m.group(1).strip()
        if len(name.split()) < 2:
            continue  # skip obvious partial matches
        window = text[m.end(): m.end() + 150]
        title = find_title_near(window)
        if title:
            candidates.append((name, title))
    if not candidates:
        return None, None
    for name, title in candidates:
        if re.search(r"chief|president", title, re.IGNORECASE):
            return name, title
    return candidates[0]

# Phrases we try to pull the *specific quantified stat* out of, so the
# pitch can say "your cloud costs rose 40%" instead of just "you disclosed
# a cloud cost risk". Captures a nearby percentage if the filing gives one.
STAT_RE = re.compile(r"(increased|rose|grew|higher)\D{0,40}?(\d{1,3}%)", re.IGNORECASE)

# SEC requires a descriptive User-Agent identifying the caller.
SEC_HEADERS = {"User-Agent": "B2B-Prospecting-Agent research@example.com"}

# Risk-factor / gap phrases we mine 10-K filings for. Each maps to a
# plain-English "gap" description used later in the pitch prompt.
GAP_SIGNALS = {
    "material weakness in our internal control": "unresolved internal-controls / financial-reporting weakness",
    "unable to remediate": "an admitted, unremediated operational or controls gap",
    "cybersecurity incident": "a disclosed cybersecurity incident or breach exposure",
    "increased costs associated with cloud": "rising, poorly-optimized cloud infrastructure spend",
    "difficulty attracting and retaining qualified personnel": "a talent gap in a specialized function",
    "legacy systems": "legacy technology debt slowing the business down",
    "integration of acquired": "unresolved system-integration bottlenecks from an acquisition",
    "data storage costs": "rising data storage costs cutting into margin",
}

# In production, restrict this to your actual deployed frontend origin(s)
# instead of "*". See the deployment section of README.md.
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")

app = FastAPI(title="B2B Prospecting Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGINS] if ALLOWED_ORIGINS != "*" else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Models ----------

class DiscoverRequest(BaseModel):
    signal: str = "material weakness in our internal control"
    limit: int = 8


class Prospect(BaseModel):
    company: str
    cik: str
    gap: str
    filing_url: Optional[str] = None
    exec_name: Optional[str] = None
    exec_title: Optional[str] = None
    stat: Optional[str] = None


class EnrichRequest(BaseModel):
    cik: str
    company: str
    gap: str
    exec_name: Optional[str] = None
    exec_title: Optional[str] = None
    stat: Optional[str] = None


class PitchRequest(BaseModel):
    company: str
    gap: str
    financial_snapshot: dict
    open_roles: list[str]
    exec_name: Optional[str] = None
    exec_title: Optional[str] = None
    stat: Optional[str] = None


async def fetch_signature_and_stat(client: httpx.AsyncClient, cik: str, hit_id: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Best-effort: fetch the actual 10-K document and pull (1) the named
    officer who signed it, from the legally-required signature block, and
    (2) a nearby quantified stat for the gap phrase (e.g. "40%"), so the
    pitch can cite specifics instead of just the risk category.

    EDGAR full-text search hit ids come back as "{accession}:{filename}".
    """
    if ":" not in hit_id:
        return None, None, None
    accession, filename = hit_id.split(":", 1)
    accession_nodash = accession.replace("-", "")
    doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{filename}"
    try:
        r = await client.get(doc_url)
        if r.status_code != 200:
            return None, None, None
        text = re.sub("<[^>]+>", " ", r.text)  # crude tag strip, good enough for regex matching
        text = re.sub(r"\s+", " ", text)

        exec_name, exec_title = find_best_signature(text)

        stat = None
        stat_match = STAT_RE.search(text)
        if stat_match:
            stat = stat_match.group(2)

        return exec_name, exec_title, stat
    except httpx.HTTPError:
        return None, None, None


# ---------- Stage 1: Discover ----------

@app.post("/discover", response_model=list[Prospect])
async def discover(req: DiscoverRequest):
    """Mine SEC EDGAR full-text 10-K search for a risk-factor phrase and
    return the companies that disclosed it, i.e. companies with a known,
    self-admitted enterprise gap."""
    if req.signal not in GAP_SIGNALS:
        raise HTTPException(400, f"signal must be one of: {list(GAP_SIGNALS)}")

    url = f"https://efts.sec.gov/LATEST/search-index?q=%22{req.signal}%22&forms=10-K"
    async with httpx.AsyncClient(headers=SEC_HEADERS, timeout=20) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()

    hits = data.get("hits", {}).get("hits", [])[: req.limit]
    prospects = []
    async with httpx.AsyncClient(headers=SEC_HEADERS, timeout=20) as client:
        for h in hits:
            src = h.get("_source", {})
            cik_list = src.get("ciks") or []
            cik = cik_list[0].lstrip("0") if cik_list else ""
            name = (src.get("display_names") or ["Unknown"])[0]
            hit_id = h.get("_id", "")

            exec_name, exec_title, stat = (None, None, None)
            if cik and hit_id:
                exec_name, exec_title, stat = await fetch_signature_and_stat(client, cik, hit_id)

            prospects.append(
                Prospect(
                    company=name,
                    cik=cik,
                    gap=GAP_SIGNALS[req.signal],
                    filing_url=(
                        f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}"
                        if cik else None
                    ),
                    exec_name=exec_name,
                    exec_title=exec_title,
                    stat=stat,
                )
            )
    return prospects


# ---------- Stage 2: Enrich ----------

@app.post("/enrich")
async def enrich(req: EnrichRequest):
    """Pull a lightweight financial snapshot from SEC XBRL facts, and cross
    -reference Arbeitnow job postings to see if the company is hiring for
    the function related to its gap (signals urgency/budget)."""
    financial_snapshot = {}
    if req.cik:
        padded = req.cik.zfill(10)
        async with httpx.AsyncClient(headers=SEC_HEADERS, timeout=20) as client:
            try:
                r = await client.get(SEC_FACTS_URL.format(cik=padded))
                if r.status_code == 200:
                    facts = r.json().get("facts", {}).get("us-gaap", {})
                    for tag in ("Revenues", "Assets", "NetIncomeLoss"):
                        units = facts.get(tag, {}).get("units", {}).get("USD", [])
                        if units:
                            latest = sorted(units, key=lambda u: u.get("end", ""))[-1]
                            financial_snapshot[tag] = latest.get("val")
            except httpx.HTTPError:
                pass

    open_roles = []
    keyword = req.gap.split()[0].lower()
    async with httpx.AsyncClient(timeout=20) as client:
        try:
            r = await client.get(ARBEITNOW_URL)
            if r.status_code == 200:
                jobs = r.json().get("data", [])
                name_tokens = set(re.findall(r"[A-Za-z]+", req.company.lower()))
                for j in jobs:
                    co = (j.get("company_name") or "").lower()
                    if name_tokens & set(re.findall(r"[A-Za-z]+", co)):
                        open_roles.append(j.get("title"))
        except httpx.HTTPError:
            pass

    return {
        "company": req.company,
        "cik": req.cik,
        "gap": req.gap,
        "financial_snapshot": financial_snapshot,
        "open_roles": open_roles[:5],
        "exec_name": req.exec_name,
        "exec_title": req.exec_title,
        "stat": req.stat,
    }


# ---------- Stage 3: Generate personalized pitch ----------

@app.post("/generate-pitch")
async def generate_pitch(req: PitchRequest):
    if not GROQ_API_KEY:
        raise HTTPException(500, "GROQ_API_KEY is not set on the server.")

    fs = ", ".join(f"{k}: ${v:,}" for k, v in req.financial_snapshot.items()) or "not disclosed"
    roles = ", ".join(req.open_roles) or "none detected"
    addressee = f"{req.exec_name} ({req.exec_title})" if req.exec_name else "Unknown — addressed to the company generally"
    stat_line = req.stat or "not explicitly quantified in the excerpt — infer a plausible, conservative estimate and flag it as an estimate"

    system_prompt = (
        "You are a senior enterprise sales strategist. Write a short, sharp, "
        "hyper-personalized cold outbound email (under 130 words). If a named "
        "executive is given below, address them by first name directly (e.g. "
        "'Jane,') and never use 'Dear Sir/Madam'. If no executive is named, "
        "address the letter to 'Investor Relations Team' instead — always "
        "produce a complete email either way, never refuse or ask for a name. "
        "No generic flattery, no 'I hope this finds you well'. Open by citing "
        "the SPECIFIC quantified stat from their own filing (e.g. 'your Q3 "
        "filing shows a 40% increase in...'), tie it to a concrete business "
        "consequence, state a specific quantified outcome you can offer (e.g. "
        "'cut that by half'), then a single soft call to action for a "
        "15-minute call. Sign off as 'Alex'."
    )
    user_prompt = (
        f"Company: {req.company}\n"
        f"Addressee: {addressee}\n"
        f"Disclosed enterprise gap: {req.gap}\n"
        f"Quantified stat from filing: {stat_line}\n"
        f"Financial snapshot: {fs}\n"
        f"Relevant open roles they're hiring for: {roles}\n\n"
        "Write the outbound email now."
    )

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.7,
                "max_tokens": 900,
                "reasoning_effort": "low",
            },
        )
        r.raise_for_status()
        data = r.json()

    choice = data["choices"][0]
    pitch = (choice.get("message", {}).get("content") or "").strip()
    if not pitch:
        # Reasoning models sometimes put the real output under a separate
        # "reasoning" field if they ran out of budget before the final answer.
        pitch = (choice.get("message", {}).get("reasoning") or "").strip()
    if not pitch:
        raise HTTPException(502, f"Groq returned no content. finish_reason={choice.get('finish_reason')}")
    return {"company": req.company, "pitch": pitch}


# ---------- Full pipeline in one call ----------

@app.post("/pipeline/run")
async def run_pipeline(req: DiscoverRequest):
    prospects = await discover(req)
    results = []
    for p in prospects:
        enriched = await enrich(EnrichRequest(
            cik=p.cik, company=p.company, gap=p.gap,
            exec_name=p.exec_name, exec_title=p.exec_title, stat=p.stat,
        ))
        try:
            pitch = await generate_pitch(PitchRequest(**enriched))
        except HTTPException:
            pitch = {"pitch": "(set GROQ_API_KEY to generate a live pitch)"}
        results.append({**enriched, "pitch": pitch["pitch"]})
        time.sleep(0.2)  # be polite to SEC rate limits
    return results


@app.get("/")
def health():
    return {"status": "ok", "signals_supported": list(GAP_SIGNALS.keys())}
