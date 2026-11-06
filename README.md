# AI Daily Intelligence Agent

An AI-powered automation system that collects AI news from configured feeds, researches related coverage, and creates one combined daily news programme with one PDF.

The current local workflow uses **Groq through its official OpenAI-compatible
Chat Completions API** as its only configured LLM provider. No Gemini, Qwen, or
DeepSeek fallback is configured. n8n Cloud orchestration, notifications, and
video rendering are future integrations.

## Project Overview

The goal of this project is to build an automated AI news and research assistant that keeps users updated on important developments in the AI ecosystem.

Every day, the system aims to:

1. Discover new articles through configured RSS and API sources.
2. Deduplicate articles, respect collection history, and extract accessible original article content where opted in.
3. Group related coverage and research selected groups together in one AI request.
4. Validate story-level citations and generate one combined 1-minute update and one combined 3-minute bulletin.
5. Save one daily programme JSON file and create one dated, source-linked PDF.

## Feature Status

### Implemented

- **Automated Data Collection:** Collect updates from trusted AI news websites, research blogs, and paper repositories.
- **Ethical Article Extraction:** Check `robots.txt`, apply a per-site request delay, extract permitted article content and metadata, and use feed summaries when extraction is unavailable.
- **Processing and Deduplication:** Normalize URLs, remove duplicate articles, and track previously reported updates.
- **Multi-source Research:** Group related articles and retain source-by-source text, metadata differences, missing metadata, and URLs without claiming unsupported corroboration.
- **Daily Programme:** Select important grouped stories based on source evidence, retain single-/multi-source status, validate claims and citations, and generate two combined anchor scripts. Groq is the sole configured provider and uses the existing local schema validation.
- **PDF Reporting:** Generate one dated report containing the daily overview, researched stories, sources, and both combined scripts.
- **Testing:** Use mocked network and provider responses; tests do not make live API requests.

### Pending

- **Workflow Automation:** Connect the modular stages to n8n Cloud.
- **Notifications:** Deliver reports through Email, Telegram, and WhatsApp.
- **Video Rendering:** Turn generated scripts and scene specifications into video; no video-generation services are configured.

## Configured Data Sources

| Source | Collector | Crawl4AI |
|---|---|---|
| OpenAI | RSS | No |
| Google DeepMind | RSS | Opt-in fallback after HTTP extraction |
| Anthropic News | Sitemap | No |
| Hugging Face Papers | Official API | No |
| Hugging Face Blog | RSS | No |
| Inc42 | RSS | No |
| Latent Space | RSS | No |
| TLDR AI | RSS | No |
| Lenny's Newsletter | RSS | No |
| Interconnects AI | RSS | No |
| AI Engineer | RSS | No |
| Hugging Face Spaces | Official API | No |
| X / Twitter | Official API (optional token) | No |
| Product Hunt | Official API (optional token) | No |

Additional sources may be integrated in future versions.

**Collection policy:** Prefer official APIs and RSS feeds. Respect website terms of service and `robots.txt`. Do not assume that a website permits scraping.

Article extraction is separately opted into per RSS source in
`config/sources.yaml`. Only Google DeepMind is currently enabled; verify a
publisher's terms before opting another source in. Extraction fails closed if
`robots.txt` cannot be checked or disallows access. It does not bypass
paywalls, authentication, or anti-bot controls. JavaScript-only pages and
inaccessible pages use the RSS/source-summary fallback.

Research grouping labels each topic as single-source or multi-source and
retains every source URL. AI-generated claims are checked against the
provided URLs before saving. The daily package contains two combined scripts,
their story/source references, and content-quality metadata. Video rendering
is not implemented; scene production remains a future stage.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Data Collection | Requests, Feedparser, BeautifulSoup |
| AI Summarization | Groq OpenAI-compatible Chat Completions API |
| Data Processing | Python |
| PDF Generation | ReportLab |
| Video Generation | FFmpeg, Text-to-Speech |
| Workflow Automation | n8n |
| API Layer | FastAPI |
| Notifications | Email, Telegram Bot API, WhatsApp Business Cloud API |
| Database / Storage | JSON initially, Supabase later |
| Testing | pytest |

## Project Architecture

```text
AI Sources
    |
    v
Data Collection
    |
    v
Cleaning and Deduplication
    |
    v
Article Extraction and Source Grouping
    |
    v
Groq Daily Research
    |
    +------------------+
    |                  |
    v                  v
PDF Generation     Future Video Rendering
    |                  |
    +--------+---------+
             |
             v
      Notifications
             |
     Email / Telegram

       n8n Automation
     Coordinates Pipeline
```

## Project Structure

```text
ai-daily-intelligence-agent/
│
├── config/
│   └── sources.yaml
│
├── src/
│   ├── collectors/
│   │   ├── rss_collector.py
│   │   ├── huggingface_collector.py
│   │   └── article_extractor.py
│   │
│   ├── processing/
│   │   ├── normalize.py
│   │   ├── deduplicate.py
│   │   └── history.py
│   │
│   ├── research/
│   │   ├── grouping.py
│   │   ├── daily_programme.py
│   │   ├── briefs.py
│   │   └── schemas.py
│   │
│   ├── summarization/
│   │   ├── gemini_client.py
│   │   └── cache.py
│   │
│   ├── reporting/
│   │   ├── sections.py
│   │   ├── research_sections.py
│   │   └── pdf_generator.py
│   │
│   ├── video/
│   │   ├── planner.py
│   │   ├── voiceover.py
│   │   └── renderer.py
│   │
│   ├── collect.py
│   ├── generate_research.py
│   ├── generate_report.py
│   ├── generate_video.py
│   ├── run_research_report.py
│   └── run_daily_programme.py
│
├── n8n/
│   └── workflows/
│
├── tests/
│
├── output/
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

The modules are callable independently for workflow automation; video
rendering and notification modules are future stages.

## Development Roadmap

- [x] **Phase 1 — Data Collection:** Collect updates from the four configured sources, report per-source outcomes, and save the latest JSON output.
- [x] **Phase 2 — Data Processing:** Normalize content and URLs, detect duplicates, and skip previously reported updates.
- [x] **Phase 3 — AI Summarization:** Use the configured Groq API to summarize and categorize collected updates.
- [x] **Phase 4 — PDF Reporting:** Create edition-numbered daily PDFs with summaries, key points, importance, and clickable source links.
- [x] **Phase 5 — Testing:** Add pytest coverage for collection, processing, summarization, and PDF reporting.
- [x] **Phase 6 — Crawling and Research Grouping:** Extract permitted article content and prepare source-linked research groups.
- [x] **Phase 7 — Daily Research and Programmes:** Generate one combined daily research package and the 1-minute and 3-minute scripts.
- [x] **Phase 8 — Daily PDF Reporting:** Include the successful stories, source citations, and both combined scripts in one PDF.
- [ ] **Phase 9 — n8n Integration:** Connect collection, research, report, and future video stages to n8n Cloud.
- [ ] **Phase 10 — Notifications:** Integrate Email and Telegram notifications.
- [x] **Phase 11 — Video Integration:** Plan scenes from exact daily narration, synthesize voiceovers, build SRT subtitles, and generate HD visual frames and video packages.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ajitkumar696/ai-daily-intelligence-agent.git
cd ai-daily-intelligence-agent
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file using `.env.example` as a reference.

Add API keys and credentials only to `.env`. Never commit secrets to GitHub.

### 5. Run tests

```bash
python -m pytest
```

### 6. Run data collection

```powershell
python src\collect.py
```

### 7. Run the complete daily news programme

```powershell
python -m src.run_daily_programme --limit 3
```

This command discovers news, removes duplicate and previously reported
articles, researches up to three grouped topics in one AI request, saves
one `output/daily_programme.json`, and creates one dated PDF under
`output/reports/`. The example caps research at three groups; use `--limit 0`
or omit the option to include all groups. The limit applies to grouped topics
sent to the configured AI provider, not to article collection. No JSON/PDF
report is created when there is no validated research.
Daily AI output is parsed and validated against the existing schemas, then
source-validated. Groq receives strict JSON Schema output requests, and
application validation enforces the existing schemas before output is saved.
The default model is `openai/gpt-oss-20b` with an 8,192-token output cap.
Provider diagnostics record Groq and the configured model; API keys are not
logged. HTTP 429 and server failures receive bounded retries with backoff.
Groq is the only configured provider; there is no automatic fallback. Both
research commands exit cleanly without creating a new programme or PDF on
failure.

For an initial small end-to-end run use `--limit 1`. It still collects the
configured feeds but sends at most one group to the configured AI provider.
The full pipeline calls the configured AI provider on a cache miss.
Collection makes network requests to the configured RSS/API sources but does
not call an AI provider.

The stages can also be run separately. Collection alone:

```powershell
python -m src.collect
```

Research the saved `output/updates.json` without running collection or
producing a PDF:

```powershell
python -m src.generate_research --limit 1
```

The research-only command writes `output/daily_programme.json`; it calls the
configured AI provider only on a cache miss. Legacy per-article summarization
commands (`src.summarize` and `src.run_daily_report`) also use the configured
Groq provider.

### Outputs and validation

- `output/updates.json`: latest newly collected article records.
- `output/processed_history.json`: collection history used across runs.
- `output/daily_programme.json`: validated daily stories, both scripts, source links, and content-quality metadata.
- `output/summary_cache.json`: cached model results.
- `output/reports/AI_Daily_Intelligence_<date>.pdf`: the one combined daily PDF (reruns get edition-numbered filenames).

Run the complete mocked test suite and static checks:

```powershell
python -m pytest -q
python -m compileall -q src tests
git diff --check
```

Tests mock network and provider responses and never make live API requests.

## Important Development Principles

- Work in small, testable steps.
- Preserve the original source URL for every collected update.
- Generate summaries only from the available source content.
- Never invent facts or unsupported claims.
- Mark information that cannot be verified as unverified.
- Prefer official APIs and RSS feeds over unauthorized scraping.
- Keep failures isolated so one failed component does not restart the entire pipeline.
- Use AI only where it adds value; use Python for deterministic processing.

## Cost Considerations

| Service | Cost |
|---|---|
| Python and core libraries | Free |
| Groq API | Free plan available; model limits apply |
| ReportLab | Free |
| FFmpeg | Free |
| Telegram Bot API | Free |
| Email | Depends on provider and usage |
| n8n Cloud | Paid after trial; self-hosting is an alternative |
| WhatsApp Business Cloud API | Usage-based charges may apply |
| pytest | Free |

## Current Status

**Implemented:** RSS and Hugging Face collection, opt-in ethical extraction,
deduplication, multi-source grouping, one combined Groq daily research
request, combined 1- and 3-minute scripts, and one source-linked PDF.

**Pending:** n8n automation, Email/Telegram notifications, and video rendering.

Article extraction is opt-in per source, currently configured only for Google DeepMind RSS, and is skipped when `robots.txt` cannot be checked or disallows the URL. A source's website terms still need to be reviewed before relying on extraction in production; an allow rule in `robots.txt` is not a legal permissions check.

### Summarization and PDF report

The Google Gen AI SDK and ReportLab dependencies are pinned in `requirements.txt`.
To install the PDF dependency manually:

```powershell
python -m pip install reportlab==5.1.0
```

To install all project dependencies instead:

```powershell
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env`, then set `GROQ_API_KEY` and optionally
`GROQ_BASE_URL=https://api.groq.com/openai/v1` and
`GROQ_MODEL=openai/gpt-oss-20b`. The default base URL and model are shown in
`.env.example`. The selected model supports Groq strict JSON Schema output and
is listed for the Free plan in Groq's current documentation. No fallback LLM
provider is configured. The complete pipeline defaults to all collected
groups; specify `--limit` to cap research volume.

Check the selected provider's local key, model, and base URL without making
an API request:

```powershell
python -m src.summarize --check-config
```

Summarize only three articles:

```powershell
python -m src.summarize --limit 3
```

Generate a PDF from the saved summaries:

```powershell
python -m src.generate_report
```

Or run summarization and PDF generation together:

```powershell
python -m src.run_daily_report --limit 3
```

Use `--limit 0` to process all input articles. Summaries are cached in
`output/summary_cache.json`; structured summaries are written to
`output/summarized_updates.json`; dated, edition-numbered PDFs are saved under
`output/reports/`.

The selected Groq API and PDF integrations require the dependencies and API
keys to be configured manually. Automated tests use mocked responses and do
not make live API calls.

### Optional Crawl4AI browser extraction

Crawl4AI remains optional. Only Google DeepMind RSS entries are currently
configured to use it after the existing HTTP article extractor fails. Install
the optional dependency and its browser with:

```powershell
python -m pip install -r requirements-crawl4ai.txt
crawl4ai-setup
```

The adapter checks `robots.txt`, restricts URLs and redirects to the configured
host allowlist, deduplicates URLs, and waits between pages. It does not attempt
to bypass paywalls, authentication, or anti-bot controls. Respect each site's
terms before enabling extraction for another source.

## Future Vision

Build a reliable AI-powered daily intelligence platform that makes it easier to follow developments in Artificial Intelligence, Machine Learning, Generative AI, research papers, and the broader AI ecosystem.

---

**Built with Python and Groq; n8n Cloud is a future orchestrator.**
