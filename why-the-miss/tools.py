"""
QSR Research Agent — Tool Definitions & Implementations

Each tool has two parts:
  1. A Python function that hits Databricks (SQL warehouse or Vector Search)
  2. An Anthropic-format schema dict used when calling client.messages.create()

The TOOL_REGISTRY maps schema names → functions so the agent loop can dispatch
without a big if/elif chain.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from config import config

# ── Databricks connection helpers ─────────────────────────────────────────────

def _sql(query: str) -> list[dict]:
    """Execute SQL against the warehouse; return list-of-dicts."""
    from databricks import sql as dbsql  # imported lazily so tests don't require it
    with dbsql.connect(
        server_hostname=config.hostname(),
        http_path=config.http_path,
        access_token=config.access_token,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def _vs(query: str, n: int = 8, filters: Optional[dict] = None) -> list[dict]:
    """Run a similarity search against gold_corpus_chunks_index."""
    from databricks.vector_search.client import VectorSearchClient
    vsc = VectorSearchClient(
        workspace_url=config.workspace_host,
        personal_access_token=config.access_token,
        disable_notice=True,
    )
    idx = vsc.get_index(config.vs_endpoint, config.vs_index_name)
    resp = idx.similarity_search(
        query_text=query,
        columns=[
            "chunk_text", "source_type", "document_title",
            "source_url", "corpus_folder", "published_date",
        ],
        num_results=n,
        filters=filters or {},
    )
    rows  = resp.get("result", {}).get("data_array", [])
    cols  = [c["name"] for c in resp.get("manifest", {}).get("columns", [])]
    return [dict(zip(cols, row)) for row in rows]


# ── Tool implementations ──────────────────────────────────────────────────────

def search_corpus(
    query: str,
    n_results: int = 8,
    source_types: Optional[list[str]] = None,
    corpus_folder: Optional[str] = None,
) -> dict:
    """
    Semantic search across all 3,582 corpus documents.
    Covers SEC filings, Pew Research, Federal Register, trade media, Wikipedia, and more.
    """
    filters: dict = {}
    if source_types:
        filters["source_type"] = source_types
    if corpus_folder:
        filters["corpus_folder"] = corpus_folder

    results = _vs(query, n=min(n_results, 20), filters=filters)
    return {"query": query, "n_results": len(results), "results": results}


def get_competitor_financials(
    tickers: list[str],
    metrics: Optional[list[str]] = None,
    periods_back: int = 8,
) -> dict:
    """
    Pull structured XBRL quarterly financials for one or more QSR public companies.
    Available: DPZ, MCD, YUM, CMG, PZZA, WEN, DIN, FAT.
    """
    tickers_sql = ", ".join(f"'{t.upper()}'" for t in tickers)
    metrics_clause = (
        "AND concept IN ({})".format(", ".join(f"'{m}'" for m in metrics))
        if metrics else ""
    )

    q = f"""
    SELECT ticker, period_of_report, concept, value, unit_of_measure
    FROM {config.fqn(config.bronze_xbrl)}
    WHERE ticker IN ({tickers_sql})
      {metrics_clause}
      AND period_of_report >= add_months(current_date(), -{periods_back * 3})
    ORDER BY ticker, period_of_report DESC, concept
    LIMIT 300
    """
    rows = _sql(q)
    return {
        "tickers": tickers,
        "periods_back_quarters": periods_back,
        "row_count": len(rows),
        "data": rows,
    }


def get_metro_profile(city: str) -> dict:
    """
    Pull comprehensive trade area profile from gold_metro_profiles.
    Includes demographics, income, commute, restaurant density, weather metro, geo spine.
    """
    q = f"""
    SELECT *
    FROM {config.fqn(config.gold_metro)}
    WHERE lower(city_name)  LIKE lower('%{city}%')
       OR lower(metro_name) LIKE lower('%{city}%')
    LIMIT 1
    """
    rows = _sql(q)
    if not rows:
        return {"error": f"No metro profile found for '{city}'. Try a larger city or metro name."}
    return {"city": city, "profile": rows[0]}


def get_macro_indicators(
    series_ids: Optional[list[str]] = None,
    months_back: int = 24,
) -> dict:
    """
    Pull FRED macro indicators with MoM and YoY deltas from the silver layer.
    Default series: CPI, Food Away From Home CPI, Unemployment, PCE, Consumer Sentiment.
    """
    default = [
        "CPIAUCSL",   # CPI All Items
        "CPIFABSL",   # CPI Food Away from Home — most relevant for QSR pricing narrative
        "UNRATE",     # Unemployment Rate
        "PCE",        # Personal Consumption Expenditures
        "MICH",       # U Mich Consumer Sentiment
        "RSXFS",      # Retail Sales ex. Food Services
    ]
    series = series_ids or default
    series_sql = ", ".join(f"'{s}'" for s in series)

    q = f"""
    SELECT series_id, series_name, date, value, mom_change, yoy_change
    FROM {config.fqn(config.silver_fred)}
    WHERE series_id IN ({series_sql})
      AND date >= add_months(current_date(), -{months_back})
    ORDER BY series_id, date DESC
    """
    rows = _sql(q)
    return {
        "series_requested": series,
        "months_back": months_back,
        "row_count": len(rows),
        "data": rows,
    }


def get_regulatory_docs(
    topics: Optional[list[str]] = None,
    days_back: int = 180,
) -> dict:
    """
    Pull Federal Register documents tagged by QSR topic from the silver layer.
    Topics: labor, food_safety, franchise, delivery, wage, nutrition_labeling.
    """
    topic_clause = ""
    if topics:
        or_parts = " OR ".join(f"topic_tags LIKE '%{t}%'" for t in topics)
        topic_clause = f"AND ({or_parts})"

    q = f"""
    SELECT title, published_date, topic_tags, summary, document_number, html_url
    FROM {config.fqn(config.silver_regulatory)}
    WHERE published_date >= current_date() - {days_back}
      {topic_clause}
    ORDER BY published_date DESC
    LIMIT 30
    """
    rows = _sql(q)
    return {
        "topics_filtered": topics,
        "days_back": days_back,
        "count": len(rows),
        "documents": rows,
    }


def get_trade_signals(days_back: int = 14) -> dict:
    """
    Pull recent QSR trade media (PMQ, NRN, QSR Mag, Restaurant Business, Pizza Today, Food Safety News)
    and general news. Best for 'what is the industry talking about right now'.
    """
    q = f"""
    SELECT title, source, published_at, summary, url, 'trade_media' AS signal_type
    FROM {config.fqn(config.bronze_trade)}
    WHERE published_at >= current_date() - {days_back}
    UNION ALL
    SELECT title, source, published_at, description AS summary, url, 'news' AS signal_type
    FROM {config.fqn(config.bronze_news)}
    WHERE published_at >= current_date() - {days_back}
    ORDER BY published_at DESC
    LIMIT 60
    """
    rows = _sql(q)
    return {
        "days_back": days_back,
        "count": len(rows),
        "signals": rows,
    }


def get_competitor_profile(company: str) -> dict:
    """
    Pull unified competitor profile (silver) with cuisine segment, franchise model,
    store count, key markets, and Wikipedia summary.
    Works with company name or ticker: 'Papa Johns', 'PZZA', 'Dominos', 'DPZ', etc.
    """
    q = f"""
    SELECT cp.*, wq.body AS wiki_summary
    FROM {config.fqn(config.silver_competitors)} cp
    LEFT JOIN {config.fqn(config.bronze_wiki_qsr)} wq
           ON lower(wq.title) LIKE lower('%{company}%')
    WHERE lower(cp.company_name) LIKE lower('%{company}%')
       OR upper(cp.ticker) = upper('{company}')
    LIMIT 1
    """
    rows = _sql(q)
    if not rows:
        return {"error": f"No competitor profile found for '{company}'."}
    return {"company": company, "profile": rows[0]}


# ── Tool schemas (Anthropic API format) ───────────────────────────────────────

TOOL_SCHEMAS: list[dict] = [
    {
        "name": "search_corpus",
        "description": (
            "Semantic search across all 3,582 corpus documents including SEC filings, "
            "Pew Research studies, Federal Register regulations, trade media (PMQ, NRN, QSR Mag), "
            "Wikipedia QSR profiles, and more. Use for qualitative evidence, narrative grounding, "
            "and any question that benefits from retrieved text passages."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query"
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of chunks to retrieve (default 8, max 20)",
                    "default": 8,
                },
                "source_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Filter by source type. Options: regulatory_filing, research_article, "
                        "news, encyclopedia, recipe, government_dataset"
                    ),
                },
                "corpus_folder": {
                    "type": "string",
                    "description": (
                        "Filter by folder. Options: regulatory_filings, competitor_profiles, "
                        "industry_news, food_psychology, audience_demographics, quarterly_reports, "
                        "digital_ordering_app_adoption, current_events, audience_targeting"
                    ),
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_competitor_financials",
        "description": (
            "Pull structured quarterly financials from SEC XBRL filings. "
            "Covers revenue, net income, EPS, store counts, and other GAAP metrics "
            "for 8 public QSR companies: DPZ (Domino's), MCD (McDonald's), YUM (Yum! Brands), "
            "CMG (Chipotle), PZZA (Papa John's), WEN (Wendy's), DIN (Dine Brands), FAT (FAT Brands)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Ticker symbols, e.g. ['DPZ', 'PZZA', 'MCD']",
                },
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "XBRL concept names to filter. Common: Revenues, NetIncomeLoss, "
                        "EarningsPerShareBasic, NumberOfRestaurants. Omit for all metrics."
                    ),
                },
                "periods_back": {
                    "type": "integer",
                    "description": "Quarters of history to retrieve (default 8)",
                    "default": 8,
                },
            },
            "required": ["tickers"],
        },
    },
    {
        "name": "get_metro_profile",
        "description": (
            "Pull a comprehensive trade area profile from the gold layer for a city or metro. "
            "Includes demographics (income, age, household size), commute patterns, "
            "restaurant establishment density, weather metro, and geographic spine. "
            "413 metros covered."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City or metro name, e.g. 'Columbus OH', 'Dallas', 'Oklahoma City'",
                },
            },
            "required": ["city"],
        },
    },
    {
        "name": "get_macro_indicators",
        "description": (
            "Pull FRED macro-economic indicators with month-over-month and year-over-year deltas "
            "from the pre-computed silver layer. Key QSR-relevant series: "
            "CPIAUCSL (headline CPI), CPIFABSL (Food Away From Home CPI — most relevant for menu pricing), "
            "UNRATE (unemployment — labor cost proxy), PCE (consumer spending), MICH (consumer sentiment). "
            "Use to ground 'macro headwinds/tailwinds' explanations with actual data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "series_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "FRED series IDs. Omit to retrieve the default QSR-relevant set.",
                },
                "months_back": {
                    "type": "integer",
                    "description": "Months of history to retrieve (default 24)",
                    "default": 24,
                },
            },
        },
    },
    {
        "name": "get_regulatory_docs",
        "description": (
            "Pull Federal Register documents tagged by QSR topic from the silver layer. "
            "All 585 documents are pre-tagged. Topics: labor, food_safety, franchise, "
            "delivery, wage, nutrition_labeling. Use for regulatory risk context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Topic tags to filter, e.g. ['labor', 'franchise']",
                },
                "days_back": {
                    "type": "integer",
                    "description": "Days of regulatory history to retrieve (default 180)",
                    "default": 180,
                },
            },
        },
    },
    {
        "name": "get_trade_signals",
        "description": (
            "Pull recent QSR trade media and news. Sources include PMQ, NRN, QSR Magazine, "
            "Restaurant Business, Pizza Today, Food Safety News, and general news APIs. "
            "Best for 'what is the industry talking about right now' questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "days_back": {
                    "type": "integer",
                    "description": "Days to look back (default 14)",
                    "default": 14,
                },
            },
        },
    },
    {
        "name": "get_competitor_profile",
        "description": (
            "Pull a unified competitor profile from the silver layer. "
            "Includes cuisine segment, franchise model, store count, key markets, "
            "founding history, and Wikipedia summary. "
            "Accepts company name (e.g. 'Papa Johns', 'Dominos') or ticker (e.g. 'PZZA', 'DPZ')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "company": {
                    "type": "string",
                    "description": "Company name or ticker",
                },
            },
            "required": ["company"],
        },
    },
]


# ── Tool dispatcher ───────────────────────────────────────────────────────────

TOOL_REGISTRY: dict[str, Any] = {
    "search_corpus":            search_corpus,
    "get_competitor_financials": get_competitor_financials,
    "get_metro_profile":        get_metro_profile,
    "get_macro_indicators":     get_macro_indicators,
    "get_regulatory_docs":      get_regulatory_docs,
    "get_trade_signals":        get_trade_signals,
    "get_competitor_profile":   get_competitor_profile,
}


def execute_tool(name: str, inputs: dict) -> Any:
    """Dispatch a single tool call. Returns the result or an error dict."""
    if name not in TOOL_REGISTRY:
        return {"error": f"Unknown tool: '{name}'", "available": list(TOOL_REGISTRY)}
    try:
        return TOOL_REGISTRY[name](**inputs)
    except Exception as exc:
        return {"error": str(exc), "tool": name, "inputs": inputs}
