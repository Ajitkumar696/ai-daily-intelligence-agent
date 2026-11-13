# AI Daily Intelligence Agent

An AI-powered automation platform that collects AI news and research papers, analyzes important developments, and generates structured intelligence reports.

The project uses Python, Groq LLMs, FastAPI, and n8n to automate research and content generation.

## Project Overview

The goal is to build an automated AI intelligence assistant that helps users follow developments in Artificial Intelligence, Machine Learning, Deep Learning, and Generative AI.

The platform has two main components:

1. **AI News Intelligence:** Collects and researches AI news from configured sources and generates daily reports.
2. **Research Paper Novelty Generator (Planned):** Analyzes research papers to explain the problem they address, limitations of existing approaches, proposed solutions, and their key innovations.

---

## 1. AI News Intelligence Pipeline

### Implemented Features

- **Automated Data Collection:** Collect AI news through RSS feeds and official APIs.
- **Article Extraction:** Extract accessible article content while respecting robots.txt, configured permissions, and website restrictions.
- **Data Cleaning and Deduplication:** Normalize URLs, remove duplicate articles, and track previously processed updates.
- **Multi-source Research:** Group related articles and retain their original source links.
- **AI Research and Summarization:** Use Groq to generate structured research summaries with source validation.
- **Daily Programme Generation:** Create combined 1-minute and 3-minute news scripts.
- **PDF Reporting:** Generate dated PDF reports with researched stories and clickable source links.
- **Video Generation:** Generate narrated MP4 news videos with subtitles and visual scenes.
- **FastAPI Backend:** Provide API endpoints for triggering pipeline execution, checking job status, and accessing generated files.
- **n8n Integration:** Include a workflow and Python runner for daily automation.

### News Pipeline

```text
AI News Sources
      |
      v
Data Collection
      |
      v
Cleaning and Deduplication
      |
      v
Article Extraction
      |
      v
Multi-source Research
      |
      v
Groq LLM Analysis
      |
      v
Daily Programme JSON
      |
      +------------------+
      |                  |
      v                  v
   PDF Report       Video Generation
      |                  |
      +--------+---------+
               |
               v
        n8n Automation
```

## 2. Research Paper Novelty Generator (Planned)

The next development phase is a research paper analysis module inspired by the paper-discovery and paper-explanation experience of platforms such as Papers with Code.

Its primary objective is to explain:

**What problem did a research paper solve, and what is novel about its proposed approach?**

### Planned Features

1. **Paper Discovery:** Discover AI/ML research papers from arXiv and Hugging Face Papers.
2. **Paper Extraction:** Extract abstracts, methodology, experiments, results, and conclusions.
3. **Problem Identification:** Identify the specific research problem addressed by the authors.
4. **Existing Limitations:** Explain the limitations of previous approaches discussed in the paper.
5. **Proposed Solution:** Explain the method or architecture introduced by the researchers.
6. **Novelty Identification:** Identify the key innovations and contributions claimed by the authors.
7. **Results Analysis:** Explain what improvements were demonstrated in the experiments.
8. **Evidence Validation:** Connect generated explanations to the actual paper content.
9. **Code Discovery:** Identify official GitHub repositories when their association with the paper can be verified.

### Research Paper Pipeline

```text
arXiv + Hugging Face Papers
            |
            v
      Paper Discovery
            |
            v
    Filtering and Deduplication
            |
            v
       PDF Extraction
            |
            v
    Methodology and Results
            |
            v
       Groq LLM Analysis
            |
            v
      Novelty Validation
            |
            v
       Structured JSON
            |
            v
    Research Paper Profiles
```

### Generated Research Paper Profile

Each paper will have a structured profile containing:

| Field | Description |
|---|---|
| Paper Title | Research paper name |
| Authors | Paper authors |
| Problem | Problem addressed by the researchers |
| Existing Limitations | Limitations of previous approaches |
| Proposed Solution | Method introduced by the authors |
| Novelty | Key new contributions |
| Results | Experimentally demonstrated improvements |
| Evidence | Supporting paper sections |
| Paper URL | Original research paper |
| GitHub URL | Verified associated code repository, if available |

**Research integrity:** The system will distinguish between authors' claimed contributions and independently verified novelty. It will not claim that an approach is the first of its kind without sufficient evidence.

## 3. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Data Collection | Requests, Feedparser, BeautifulSoup |
| Research Paper Sources | arXiv API, Hugging Face Papers |
| AI Analysis | Groq API |
| Data Processing | Python |
| PDF Extraction | PyMuPDF (planned for paper module) |
| PDF Generation | ReportLab |
| Video Generation | FFmpeg, Text-to-Speech |
| API Development | FastAPI |
| Workflow Automation | n8n |
| Data Storage | JSON |
| Testing | pytest |

## 4. Project Structure

```text
ai-daily-intelligence-agent/
│
├── config/
│   └── sources.yaml
│
├── src/
│   ├── collectors/
│   ├── processing/
│   ├── research/
│   ├── summarization/
│   ├── reporting/
│   ├── video/
│   │
│   ├── papers/                 # Planned module
│   │   ├── discovery.py
│   │   ├── collectors.py
│   │   ├── extractor.py
│   │   ├── cleaner.py
│   │   ├── analyzer.py
│   │   ├── novelty.py
│   │   ├── validator.py
│   │   └── pipeline.py
│   │
│   ├── collect.py
│   ├── generate_research.py
│   ├── generate_report.py
│   ├── generate_video.py
│   └── run_daily_programme.py
│
├── tests/
├── output/
├── run_daily_pipeline.py
├── n8n_daily_workflow.json
├── n8n_api.py
├── files_api.py
├── requirements.txt
├── .env.example
└── README.md
```

*The `papers/` module above represents the proposed structure; its files should be added as implementation progresses.*

## 5. Installation

### Clone the repository

```bash
git clone https://github.com/Ajitkumar696/ai-daily-intelligence-agent.git
cd ai-daily-intelligence-agent
```

### Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
pip install -r requirements.txt
```

### Configure environment variables

Create `.env` using `.env.example` as a reference.

Configure:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL=openai/gpt-oss-20b
```

Never commit API keys or secrets to GitHub.

## 6. Running the Project

### Run data collection

```powershell
python -m src.collect
```

### Generate the daily research programme

```powershell
python -m src.run_daily_programme --limit 1
```

### Generate a PDF report

```powershell
python -m src.generate_report
```

### Generate a video

```powershell
python -m src.generate_video --type three_minute
```

### Run the complete daily pipeline

```powershell
python run_daily_pipeline.py
```

### Run tests

```powershell
python -m pytest -q
```

## 7. Development Roadmap

- [x] AI news collection from configured sources.
- [x] Article extraction and deduplication.
- [x] Multi-source research and source validation.
- [x] Groq-based structured AI research.
- [x] Combined daily news programme.
- [x] PDF report generation.
- [x] 1-minute and 3-minute video generation.
- [x] FastAPI backend and file-serving endpoints.
- [x] n8n workflow and daily pipeline runner.
- [ ] Activate and configure automated n8n delivery.
- [ ] Configure Telegram and email notifications.
- [ ] Build research paper discovery module.
- [ ] Extract full research paper content.
- [ ] Implement problem and limitation identification.
- [ ] Implement novelty and contribution analysis.
- [ ] Add evidence-based validation.
- [ ] Generate structured research paper profiles.
- [ ] Integrate paper reports into the existing platform.

## 8. Important Development Principles

- Preserve original source URLs and paper links.
- Generate explanations only from available evidence.
- Never invent research findings, statistics, or contributions.
- Distinguish author claims from independently verified findings.
- Respect source terms, robots.txt, and API restrictions.
- Keep research, news, and video modules independently testable.
- Use Python for deterministic operations and LLMs for language understanding.
- Keep API credentials outside version control.

## 9. Future Vision

Build a unified AI intelligence platform that does more than report AI news.

It will help users understand:

- What is happening in the AI ecosystem?
- What new research papers are being published?
- What problems are researchers trying to solve?
- What innovations are they introducing?
- What improvements do their experiments demonstrate?

The long-term goal is to transform AI news and research papers into understandable, evidence-backed knowledge.

---

**Built with Python, Groq, FastAPI, FFmpeg, and n8n.**
