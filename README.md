# Fact-Check Agent

A "Truth Layer" web app: upload a PDF, and it extracts factual claims, verifies each one against the
live web, and flags it as **Verified**, **Inaccurate** (outdated or off) or **False**, with the correct
current fact and source links.

## How it works
1. **Extract**: PyMuPDF reads the PDF; Gemini pulls out concrete, checkable claims (stats, dates,
   financial and technical figures) as structured JSON.
2. **Verify**: each claim is checked in parallel by Gemini with Google Search grounding. The prompt
   includes today's date and forces a search for the *latest* figure, so outdated stats are caught.
3. **Report**: verdict, correct fact, explanation and sources per claim. False and Inaccurate claims
   come first. The report is downloadable as CSV.

A fourth internal verdict, **Unverifiable**, is used when searching finds neither support nor
contradiction, so the app never claims "False" without evidence.

## Run locally
```bash
pip install -r requirements.txt
export GEMINI_API_KEY=AIza...
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
1. Push this repo to GitHub.
2. On share.streamlit.io, create a new app pointing at `app.py`.
3. In **Advanced settings > Secrets**, add:
   ```toml
   GEMINI_API_KEY = "AIza..."
   TAVILY_API_KEY = "tvly-..."
   ```
4. Deploy, then test the live URL with a fresh PDF.

## Search provider
- If `TAVILY_API_KEY` is set, claims are verified with **Tavily web search + Gemini reasoning** (works on free keys).
- Otherwise it falls back to **Gemini Google Search grounding**, which needs a Google project with grounding quota.

## Config
- `MODEL` (secret or env var): defaults to `gemini-3.8-flash`.
- Limits in `app.py`: `MAX_CLAIMS` (12), `MAX_WORKERS` (3). Rate-limit errors are retried with backoff.

## Limitations
- Scanned/image-only PDFs are not supported (no OCR).
- Web search quality bounds verification quality; always review flagged claims.
