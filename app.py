"""
Fact-Check Agent: a "Truth Layer" for marketing PDFs.

Pipeline: PDF -> text -> claim extraction (LLM) -> live web verification
(LLM + web search tool, run in parallel) -> verdict report.
"""
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import fitz  # PyMuPDF
import pandas as pd
import requests
import streamlit as st
from google import genai
from google.genai import types

MAX_CLAIMS = 12
MAX_PDF_CHARS = 60_000
MAX_WORKERS = 3  # keep low: free tier is rate limited

st.set_page_config(page_title="Fact-Check Agent", page_icon="🔎", layout="wide")


# ---------- config ----------
def get_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, os.environ.get(name, default))
    except Exception:
        return os.environ.get(name, default)


API_KEY = get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY")
MODEL = get_secret("MODEL", "gemini-3.8-flash")


TAVILY_KEY = get_secret("TAVILY_API_KEY")


@st.cache_resource
def _make_client(key: str) -> genai.Client:
    return genai.Client(api_key=key)


def client() -> genai.Client:
    return _make_client(API_KEY)


def resp_text(resp) -> str:
    """Concatenate only the text parts (avoids thought_signature warnings)."""
    try:
        parts = resp.candidates[0].content.parts or []
        return "".join(p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False))
    except Exception:
        return resp.text or ""


def call_with_retry(fn, tries: int = 5):
    """Retry on rate limits / transient errors with exponential backoff."""
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            transient = any(k in msg for k in ("429", "RESOURCE_EXHAUSTED", "500", "503", "UNAVAILABLE"))
            if transient and i < tries - 1:
                time.sleep(min(4 * 2**i, 40))
                continue
            raise


# ---------- helpers ----------
def extract_pdf_text(data: bytes) -> str:
    doc = fitz.open(stream=data, filetype="pdf")
    pages = []
    for i, page in enumerate(doc, start=1):
        pages.append(f"[Page {i}]\n{page.get_text()}")
    return "\n".join(pages)


def parse_json(text: str):
    """Pull the first JSON object/array out of an LLM response."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    # try whichever bracket opens first, so {"a": [1]} is parsed as the object
    pairs = [("{", "}"), ("[", "]")]
    pairs.sort(key=lambda p: text.find(p[0]) if text.find(p[0]) != -1 else 10**9)
    for opener, closer in pairs:
        s, e = text.find(opener), text.rfind(closer)
        if s != -1 and e > s:
            try:
                return json.loads(text[s : e + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("No valid JSON in model response")


# ---------- step 1: extract claims ----------
EXTRACT_SYSTEM = """You extract verifiable factual claims from documents for a fact-checking pipeline.
Return ONLY a JSON array, no prose. Each item:
{"claim": "<the claim, self-contained, quoting the document's figure exactly>",
 "category": "statistic|date|financial|technical",
 "page": <int or null>,
 "search_query": "<a good web search query to check it>"}
Rules:
- Only include concrete, checkable claims: numbers, percentages, dates, rankings, prices, revenues, market sizes, product specs, benchmarks, "first/largest" statements.
- Skip opinions, vague marketing language and anything not checkable.
- Make each claim self-contained (include the entity and the time frame it refers to).
- Prefer the most specific and most suspicious claims. Max %d claims."""


def extract_claims(text: str) -> list[dict]:
    resp = call_with_retry(
        lambda: client().models.generate_content(
            model=MODEL,
            contents=f"Document:\n\n{text[:MAX_PDF_CHARS]}",
            config=types.GenerateContentConfig(
                system_instruction=EXTRACT_SYSTEM % MAX_CLAIMS,
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
    )
    claims = parse_json(resp_text(resp))
    return claims[:MAX_CLAIMS]


# ---------- step 2: verify against the live web ----------
VERIFY_SYSTEM = """You are a rigorous fact-checker. Today's date is {today}.
You are given ONE claim from a document. You MUST use Google Search for every claim to find authoritative, CURRENT sources
(official sites, regulators, company filings, reputable news, standards bodies, peer-reviewed work).

Decide a verdict:
- "Verified": the claim matches current data from reliable sources.
- "Inaccurate": the claim was once true or is partly right but the numbers/dates/details are outdated or off.
  Example: an old statistic that has since been updated, or a figure that is close but wrong.
- "False": reliable sources contradict the claim outright, or the claimed event/figure/entity does not exist.
- "Unverifiable": you searched properly but found neither support nor contradiction. Do NOT use this
  when sources contradict the claim.

Critical rules:
- Always look for the LATEST figure. A claim matching an old report is Inaccurate if newer data exists.
- For Inaccurate/False you MUST state the correct, current fact with the figure and its date.
- Never invent sources. Only cite URLs you actually found via search.
- Be strict about numbers: 15% vs 12% is not "Verified".

Return ONLY a JSON object, no prose:
{{"verdict": "Verified|Inaccurate|False|Unverifiable",
  "confidence": "high|medium|low",
  "correct_fact": "<the real, current fact; for Verified restate the confirmed fact>",
  "explanation": "<1-3 sentences: what you found and why you chose this verdict>",
  "sources": [{{"title": "...", "url": "..."}}]}}"""


TAVILY_SYSTEM = """You are a rigorous fact-checker. Today's date is {today}.
You get ONE claim and numbered web search results (title, url, date if known, content).
Judge the claim using the evidence. Prefer the most recent and most authoritative sources.

Verdicts:
- "Verified": evidence shows the claim matches current data.
- "Inaccurate": was once true / partly right, but numbers, dates or details are outdated or off.
- "False": evidence contradicts the claim outright, or the claimed thing does not exist.
- "Unverifiable": the evidence is insufficient either way. Do NOT use this when evidence contradicts the claim.

Rules:
- Always look for the LATEST figure; a claim matching an old report is Inaccurate if newer data exists.
- For Inaccurate/False you MUST state the correct current fact, with the figure and its date.
- Be strict about numbers: 15% vs 12% is not Verified.
- Use only the numbered results for sources; never invent URLs.

Return ONLY JSON:
{{"verdict": "Verified|Inaccurate|False|Unverifiable",
  "confidence": "high|medium|low",
  "correct_fact": "<the real, current fact>",
  "explanation": "<1-3 sentences>",
  "source_ids": [<numbers of the results you relied on>]}}"""


def tavily_search(query: str, max_results: int = 5) -> list[dict]:
    r = requests.post(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {TAVILY_KEY}"},
        json={"query": query, "search_depth": "basic", "max_results": max_results},
        timeout=30,
    )
    r.raise_for_status()
    return r.json().get("results", [])


def verify_claim_tavily(claim: dict) -> dict:
    queries = [claim.get("search_query") or claim["claim"], f"{claim['claim']} latest data {date.today().year}"]
    seen, results = set(), []
    for q in queries:
        for r in tavily_search(q):
            if r.get("url") and r["url"] not in seen:
                seen.add(r["url"])
                results.append(r)
    results = results[:8]
    if not results:
        return {**claim, "verdict": "Unverifiable", "confidence": "low", "correct_fact": "",
                "explanation": "Web search returned no results.", "sources": []}

    evidence = "\n\n".join(
        f"[{i}] {r.get('title','')} ({r.get('url')})"
        + (f" - {r['published_date']}" if r.get("published_date") else "")
        + f"\n{(r.get('content') or '')[:900]}"
        for i, r in enumerate(results, start=1)
    )
    resp = call_with_retry(
        lambda: client().models.generate_content(
            model=MODEL,
            contents=f"Claim: {claim['claim']}\n\nSearch results:\n{evidence}",
            config=types.GenerateContentConfig(
                system_instruction=TAVILY_SYSTEM.format(today=date.today().strftime("%B %d, %Y")),
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
    )
    try:
        out = parse_json(resp_text(resp))
    except Exception:
        out = {"verdict": "Unverifiable", "confidence": "low", "correct_fact": "",
               "explanation": "Model response could not be parsed.", "source_ids": []}
    if out.get("verdict") not in ("Verified", "Inaccurate", "False", "Unverifiable"):
        out["verdict"] = "Unverifiable"
    ids = [i for i in out.pop("source_ids", []) if isinstance(i, int) and 1 <= i <= len(results)]
    chosen = [results[i - 1] for i in ids] or results[:3]
    out["sources"] = [{"title": r.get("title") or r["url"], "url": r["url"]} for r in chosen[:4]]
    return {**claim, **out}


def verify_claim_gemini(claim: dict) -> dict:
    prompt = (
        f"Claim: {claim['claim']}\nCategory: {claim.get('category','')}\n"
        f"Suggested search: {claim.get('search_query','')}\n\n"
        "Search the web for this claim, then answer with the JSON object only."
    )
    system = VERIFY_SYSTEM.format(today=date.today().strftime("%B %d, %Y"))

    resp = call_with_retry(
        lambda: client().models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.1,
            ),
        )
    )

    # Sources the search tool actually used (fallback if the model lists none)
    found = []
    try:
        gm = resp.candidates[0].grounding_metadata
        for ch in (gm.grounding_chunks or []):
            if ch.web and ch.web.uri:
                found.append({"title": ch.web.title or ch.web.uri, "url": ch.web.uri})
    except Exception:
        pass

    try:
        out = parse_json(resp_text(resp))
    except Exception:
        out = {
            "verdict": "Unverifiable",
            "confidence": "low",
            "correct_fact": "",
            "explanation": "Model response could not be parsed.",
            "sources": [],
        }

    if out.get("verdict") not in ("Verified", "Inaccurate", "False", "Unverifiable"):
        out["verdict"] = "Unverifiable"
    # prefer grounded URLs (real search results) over model-typed ones
    out["sources"] = found[:4] or out.get("sources", [])
    return {**claim, **out}


def verify_claim(claim: dict) -> dict:
    return verify_claim_tavily(claim) if TAVILY_KEY else verify_claim_gemini(claim)


def verify_all(claims: list[dict], progress) -> list[dict]:
    results = [None] * len(claims)
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(verify_claim, cl): i for i, cl in enumerate(claims)}
        for f in as_completed(futures):
            i = futures[f]
            try:
                results[i] = f.result()
            except Exception as e:  # keep going if one claim fails
                results[i] = {
                    **claims[i],
                    "verdict": "Unverifiable",
                    "confidence": "low",
                    "correct_fact": "",
                    "explanation": f"Verification error: {e}",
                    "sources": [],
                }
            done += 1
            progress.progress(done / len(claims), text=f"Verified {done}/{len(claims)} claims")
    return results


# ---------- UI ----------
def esc(t) -> str:
    """Escape $ so Streamlit does not render dollar amounts as LaTeX."""
    return str(t or "").replace("$", "\\$")


BADGE = {
    "Verified": ("✅", "green"),
    "Inaccurate": ("⚠️", "orange"),
    "False": ("❌", "red"),
    "Unverifiable": ("❔", "gray"),
}

st.title("🔎 Fact-Check Agent")
st.caption(
    "Upload a PDF. The agent extracts factual claims, checks them against the live web, "
    "and flags what is Verified, Inaccurate (outdated) or False, with the correct facts."
)

if not API_KEY:
    st.error("GEMINI_API_KEY is not set. Add it in Streamlit secrets or as an environment variable.")
    st.stop()

file = st.file_uploader("Upload a PDF", type=["pdf"])

if file and st.button("Run fact-check", type="primary"):
    try:
        with st.status("Reading PDF...", expanded=False) as status:
            text = extract_pdf_text(file.getvalue())
            if len(text.strip()) < 50:
                st.error("No readable text found (scanned PDF?). Try a text-based PDF.")
                st.stop()
            status.update(label="Extracting claims...")
            claims = extract_claims(text)
            status.update(label=f"Found {len(claims)} claims. Verifying on the live web...", state="running")

        if not claims:
            st.warning("No checkable claims found in this document.")
            st.stop()

        bar = st.progress(0.0, text="Starting verification...")
        results = verify_all(claims, bar)
        bar.empty()
        errs = [r["explanation"] for r in results if r["explanation"].startswith("Verification error")]
        if len(errs) == len(results):
            st.error(f"All verifications failed. First error: {errs[0][:400]}")
        st.session_state["results"] = results
    except Exception as e:
        msg = str(e)
        if "API key" in msg or "API_KEY_INVALID" in msg or "403" in msg:
            st.error("Invalid or restricted Gemini API key.")
        elif "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            st.error("Gemini rate limit / daily quota hit. Wait a minute and retry.")
        else:
            st.error(f"Something went wrong: {e}")

results = st.session_state.get("results")
if results:
    counts = pd.Series([r["verdict"] for r in results]).value_counts()
    cols = st.columns(4)
    for col, v in zip(cols, ["Verified", "Inaccurate", "False", "Unverifiable"]):
        col.metric(f"{BADGE[v][0]} {v}", int(counts.get(v, 0)))

    order = {"False": 0, "Inaccurate": 1, "Unverifiable": 2, "Verified": 3}
    for r in sorted(results, key=lambda x: order[x["verdict"]]):
        icon, _ = BADGE[r["verdict"]]
        with st.expander(f"{icon} {r['verdict']}  |  {esc(r['claim'])}", expanded=r["verdict"] in ("False", "Inaccurate")):
            st.markdown(f"**Claim:** {esc(r['claim'])}")
            if r.get("page"):
                st.caption(f"Page {r['page']}  ·  {r.get('category','')}  ·  confidence: {r.get('confidence','')}")
            if r["verdict"] in ("False", "Inaccurate"):
                st.markdown(f"**Correct fact:** {esc(r.get('correct_fact',''))}")
            elif r["verdict"] == "Verified":
                st.markdown(f"**Confirmed:** {esc(r.get('correct_fact',''))}")
            st.markdown(f"**Why:** {esc(r.get('explanation',''))}")
            for s in r.get("sources", [])[:4]:
                st.markdown(f"- [{s.get('title') or s.get('url')}]({s.get('url')})")

    df = pd.DataFrame(
        [
            {
                "verdict": r["verdict"],
                "claim": r["claim"],
                "correct_fact": r.get("correct_fact", ""),
                "explanation": r.get("explanation", ""),
                "sources": " | ".join(s.get("url", "") for s in r.get("sources", [])),
            }
            for r in results
        ]
    )
    st.download_button("Download report (CSV)", df.to_csv(index=False), "fact_check_report.csv", "text/csv")