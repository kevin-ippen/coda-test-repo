# QSR Research Agent

An agentic research system built on the QSR Research Library corpus
(`serverless_stable_h7wanf_catalog`). Designed for Databricks / Mosaic AI.

## Architecture

```
Persona → JTBD Flow → Agent Loop → Tools → Databricks (SQL + Vector Search)
```

### 5 flows (jobs to be done)

| Flow | When to use | Tools activated |
|---|---|---|
| `meeting_prep_brief` | 2 hours before a C-suite meeting | All 7 |
| `explain_the_miss` | Post-earnings or anomaly investigation | Financials, Macro, Corpus, Signals |
| `trade_area_score` | Territory evaluation, expansion discussion | Metro Profile, Corpus, Macro |
| `morning_signal_brief` | Daily digest for account prep | Signals, Macro, Regulatory, Corpus |
| `competitive_deep_dive` | Pre-meeting competitive research on a topic | Comp Profile, Financials, Corpus, Signals |

### 7 tools

| Tool | Data source | Layer |
|---|---|---|
| `search_corpus` | 4,959 vector chunks (GTE-Large-EN) | Gold VS |
| `get_competitor_financials` | SEC XBRL — DPZ, MCD, YUM, CMG, PZZA, WEN, DIN, FAT | Bronze |
| `get_metro_profile` | 413 metro profiles | Gold |
| `get_macro_indicators` | FRED with MoM/YoY deltas | Silver |
| `get_regulatory_docs` | 585 Federal Register docs (QSR tagged) | Silver |
| `get_trade_signals` | PMQ, NRN, QSR Mag, Restnt Biz, Pizza Today, news | Bronze |
| `get_competitor_profile` | Unified profiles + Wikipedia | Silver |

## Setup

### Environment variables (cluster-level or `.env`)

```bash
DATABRICKS_HOST=https://fevm-serverless-stable-h7wanf.cloud.databricks.com
DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/<your_warehouse_id>
DATABRICKS_TOKEN=<your_pat>
ANTHROPIC_API_KEY=<your_key>
```

### Dependencies

```bash
pip install anthropic databricks-vectorsearch databricks-sql-connector
```

## Usage

```python
from flows import meeting_prep_brief, explain_the_miss, trade_area_score

# Meeting prep
brief = meeting_prep_brief(
    company="Casey's General Stores",
    attendees=["Darren Rebelez, CEO", "Brian Johnson, CIO"],
    meeting_context="Databricks unified data + AI platform pitch",
    competitor_tickers=["MCD", "WEN"],
)
print(brief)

# Explain a result
analysis = explain_the_miss(
    company="Domino's Pizza",
    metric="US same-store sales",
    period="Q3 2024",
    actual="-2.9%",
    company_ticker="DPZ",
    peer_tickers=["PZZA", "MCD"],
)
print(analysis)

# Trade area
report = trade_area_score(city="Columbus, Ohio", brand="Domino's")
print(report)
```

## Build sequence (recommended)

1. **Verify connectivity** — run `tools.get_macro_indicators()` directly and confirm rows return
2. **Test vector search** — run `tools.search_corpus("digital ordering consumer behavior", n_results=3)`
3. **Run morning brief** — simplest flow, only 4 tools, good smoke test
4. **Run trade area score** — validates metro profiles + VS
5. **Run explain the miss** — validates XBRL + FRED + corpus together
6. **Run meeting prep brief** — full 7-tool integration test

## P1 gaps to fill (improve grounding further)

| Gap | Impact | Effort |
|---|---|---|
| USDA FoodData Central (nutrition API) | Menu/nutrition grounding | Low (free API key) |
| XBRL → `gold_qsr_financials` time-series pivot | Enable direct trend queries without RAG | Medium |
| Scheduled RSS backfill (weekly) | Deeper trade media archive | Low |

## Mosaic AI Workflow (daily morning brief)

Schedule `demo_notebook.py` as a Databricks Job:
- Trigger: Daily at 6:00 AM CT
- Task: Notebook (`morning_signal_brief()`)
- Output: Write to `rl_gold.agent_outputs` Delta table + email/Slack notification
