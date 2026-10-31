# AI Daily Intelligence Agent

An AI-powered automation system that collects the latest Artificial Intelligence updates from trusted sources, removes duplicate information, generates source-grounded summaries, and delivers a daily intelligence report.

The system uses **Python, Claude API, and n8n** to automate research, reporting, and notifications.

## Project Overview

The goal of this project is to build an automated AI news and research assistant that keeps users updated on important developments in the AI ecosystem.

Every day, the system aims to:

1. Collect new AI updates from selected websites and research sources.
2. Remove duplicate articles and previously reported updates.
3. Generate concise summaries using collected source content.
4. Preserve the original source link for every update.
5. Generate a consolidated daily PDF report.
6. Create a short video covering important updates.
7. Deliver notifications through Email, Telegram, and WhatsApp.
8. Run the entire pipeline automatically using n8n.

## Key Features

- **Automated Data Collection:** Collect updates from trusted AI news websites, research blogs, and paper repositories.
- **Duplicate Detection:** Identify duplicate articles and avoid reporting the same update repeatedly.
- **AI-Powered Summarization:** Generate concise summaries using Claude API.
- **Source-Grounded Reporting:** Keep summaries tied to collected content and preserve original source links.
- **Daily PDF Generation:** Combine updates into a structured PDF report.
- **Automated Video Creation:** Convert verified updates into short news videos.
- **Multi-Channel Notifications:** Deliver reports through Email, Telegram, and WhatsApp.
- **Workflow Automation:** Use n8n to schedule and coordinate the pipeline.
- **Testing:** Use pytest to validate individual modules.

## Initial Data Sources

| Source | Website |
|---|---|
| OpenAI News | https://openai.com/news/ |
| Anthropic News | https://www.anthropic.com/news |
| Google DeepMind | https://deepmind.google/discover/blog/ |
| Hugging Face Papers | https://huggingface.co/papers |
| Latent Space | https://www.latent.space/podcast |

Additional sources may be integrated in future versions.

**Collection policy:** Prefer official APIs and RSS feeds. Respect website terms of service and `robots.txt`. Do not assume that a website permits scraping.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Data Collection | Requests, Feedparser, BeautifulSoup |
| AI Summarization | Claude API |
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
AI Summarization and Verification
    |
    v
Daily Report Generation
    |
    +------------------+
    |                  |
    v                  v
PDF Generation     Video Generation
    |                  |
    +--------+---------+
             |
             v
      Notifications
             |
     Email / Telegram /
         WhatsApp

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
│   │   ├── rss.py
│   │   ├── api.py
│   │   └── html_links.py
│   │
│   ├── processing/
│   │   ├── clean.py
│   │   └── dedupe.py
│   │
│   ├── ai/
│   │   ├── summarize.py
│   │   ├── verify.py
│   │   └── prompts/
│   │
│   ├── reports/
│   │   └── pdf_builder.py
│   │
│   ├── video/
│   │   ├── script.py
│   │   ├── tts.py
│   │   └── render.py
│   │
│   ├── notify/
│   │   ├── email.py
│   │   ├── telegram.py
│   │   └── whatsapp.py
│   │
│   ├── storage/
│   │
│   └── api.py
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

*The structure will evolve as development progresses. The initial implementation will remain simple and modular.*

## Development Roadmap

- [ ] **Phase 1 — Data Collection:** Collect updates from the five initial sources and save them in JSON format.
- [ ] **Phase 2 — Data Processing:** Clean data, normalize URLs, detect duplicates, and skip previously reported updates.
- [ ] **Phase 3 — AI Summarization:** Use Claude API to summarize and categorize collected updates.
- [ ] **Phase 4 — PDF Generation:** Create a consolidated daily PDF with clickable source links.
- [ ] **Phase 5 — Testing:** Add pytest tests for collection, processing, summarization, and PDF generation.
- [ ] **Phase 6 — n8n Integration:** Connect the Python pipeline to n8n through a FastAPI service.
- [ ] **Phase 7 — Notifications:** Integrate Email, Telegram, and WhatsApp.
- [ ] **Phase 8 — Video Generation:** Generate short news videos using verified summaries, narration, subtitles, and FFmpeg.

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
pytest
```

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
| Claude API | Paid, usage-based |
| ReportLab | Free |
| FFmpeg | Free |
| Telegram Bot API | Free |
| Email | Depends on provider and usage |
| n8n Cloud | Paid after trial; self-hosting is an alternative |
| WhatsApp Business Cloud API | Usage-based charges may apply |
| pytest | Free |

## Current Status

**Project stage:** Initial development

The first milestone is to validate data collection from the five initial sources, fix any incorrect feed endpoints, and save the collected updates in `updates.json`.

Further modules will be implemented and tested incrementally.

## Future Vision

Build a reliable AI-powered daily intelligence platform that makes it easier to follow developments in Artificial Intelligence, Machine Learning, Generative AI, research papers, and the broader AI ecosystem.

---

**Built with Python, Claude API, and n8n.**
