"""
QSR Research Agent — JTBD Flows

Five purpose-built flows, each with:
  - A tailored system prompt that specifies the output structure
  - A clean Python function with typed args and docstring
  - Selective tool exposure (not every flow needs all 7 tools)

Import and call from a Databricks notebook or Claude Code:

    from flows import (
        meeting_prep_brief,
        explain_the_miss,
        trade_area_score,
        morning_signal_brief,
        competitive_deep_dive,
    )
"""
from __future__ import annotations

import textwrap
from typing import Optional

from agent import run_agent, BASE_SYSTEM
from tools import TOOL_SCHEMAS


# ── Helper: select tool schemas by name ──────────────────────────────────────

def _tools(*names: str) -> list[dict]:
    return [s for s in TOOL_SCHEMAS if s["name"] in names]


ALL_TOOLS = [s["name"] for s in TOOL_SCHEMAS]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. MEETING PREP BRIEF
# ═══════════════════════════════════════════════════════════════════════════════

_MEETING_PREP_SYSTEM = BASE_SYSTEM + textwrap.dedent("""
    You are preparing a concise intelligence brief for a C-suite meeting.
    Be specific and data-backed. Every claim should trace to a corpus source.
    No filler. If the corpus doesn't have data on something, say so.

    Output structure (markdown):

    ## [Company] — Meeting Brief
    **Date**: [today] | **Attendees**: [list]

    ### Company snapshot
    2-3 sentences: current financial state, trajectory, and the headline strategic story.

    ### Competitive position
    How does this company compare to key peers on revenue, store count, and margins?
    Is their performance company-specific or industry-wide?

    ### Macro context
    Which macro factors are most relevant right now? Pull actual FRED data points with dates.

    ### What each attendee cares about
    For each named attendee, 1-2 sentences on their likely priorities and what will resonate.

    ### Regulatory watch
    Any pending regulations that directly affect their model (franchise, labor, delivery).

    ### Recommended talking points
    3-5 specific, grounded angles. Avoid generic sales language.

    ### Anticipated objections
    2-3 likely pushbacks and how to address each with corpus evidence.
""")


def meeting_prep_brief(
    company: str,
    attendees: list[str],
    meeting_context: str,
    competitor_tickers: Optional[list[str]] = None,
    verbose: bool = False,
) -> str:
    """
    Generate a grounded C-suite meeting brief.

    Args:
        company:             Company name, e.g. "Domino's Pizza" or "Casey's General Stores"
        attendees:           Attendee names/titles, e.g. ["Darren Rebelez, CEO", "Russell Keene, CFO"]
        meeting_context:     What the meeting is about and your objective
        competitor_tickers:  Optional tickers for benchmarking, e.g. ["DPZ", "PZZA"]
        verbose:             Print agent trace

    Returns:
        Formatted markdown brief as a string

    Example:
        brief = meeting_prep_brief(
            company="Casey's General Stores",
            attendees=["Darren Rebelez, CEO", "Brian Johnson, CIO", "Megan Elfers, CMO"],
            meeting_context="Introducing Databricks for unified data + AI platform",
            competitor_tickers=["MCD", "WEN"],
        )
    """
    comp_tickers_str = (
        f"Benchmark competitors: {', '.join(competitor_tickers)}"
        if competitor_tickers else ""
    )
    user_msg = textwrap.dedent(f"""
        Prepare a meeting brief for the following engagement:

        **Company**: {company}
        **Attendees**: {', '.join(attendees)}
        **Meeting context**: {meeting_context}
        {comp_tickers_str}

        Use all available tools. Prioritize recency — start with trade signals and macro
        context, then pull financials for XBRL-covered companies. Use corpus search for
        qualitative evidence on strategy, competitive positioning, and consumer trends.
    """).strip()

    return run_agent(
        messages=[{"role": "user", "content": user_msg}],
        system_prompt=_MEETING_PREP_SYSTEM,
        tools=TOOL_SCHEMAS,  # all 7 tools
        verbose=verbose,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. EXPLAIN THE MISS (OR WIN)
# ═══════════════════════════════════════════════════════════════════════════════

_EXPLAIN_MISS_SYSTEM = BASE_SYSTEM + textwrap.dedent("""
    You are a financial analyst building a grounded causal narrative for a business result.
    Your output needs to hold up to scrutiny — every cause should be backed by data.
    Distinguish macro/industry headwinds (external) from execution factors (internal).
    Be honest when the corpus doesn't have enough data to support a specific attribution.

    Output structure (markdown):

    ## [Company] — [Metric] [Period]

    ### The result
    What happened. Actual vs. expected if known.

    ### Macro headwinds / tailwinds
    Specific FRED data points with periods and direction of change. What was the consumer
    environment like in this period?

    ### Industry-wide vs. company-specific
    How did peers perform in the same period? Pull XBRL comparisons where possible.
    Is this a rising-tide / falling-tide story, or did this company diverge from peers?

    ### What the company said
    Relevant language from SEC filings or corpus documents. Quote with source + date.

    ### Industry signals from trade media
    What was trade press discussing in this period? Does it corroborate the result?

    ### Attribution summary
    A 3-5 sentence executive paragraph attributing the result to specific, grounded causes.
    This is the paragraph someone would use in a board presentation.
""")


def explain_the_miss(
    company: str,
    metric: str,
    period: str,
    actual: Optional[str] = None,
    expected: Optional[str] = None,
    company_ticker: Optional[str] = None,
    peer_tickers: Optional[list[str]] = None,
    verbose: bool = False,
) -> str:
    """
    Generate a grounded explanation for a business result.

    Args:
        company:        Company name
        metric:         Metric in question, e.g. "same-store sales", "revenue", "margins"
        period:         Time period, e.g. "Q3 2024", "FY 2024", "H1 2024"
        actual:         What actually happened, e.g. "-2.9% comp sales"
        expected:       What was expected, e.g. "+1.2% consensus"
        company_ticker: For XBRL pull, e.g. "DPZ"
        peer_tickers:   Peer tickers for comparison, e.g. ["PZZA", "MCD"]
        verbose:        Print agent trace

    Example:
        analysis = explain_the_miss(
            company="Domino's Pizza",
            metric="US same-store sales",
            period="Q3 2024",
            actual="-2.9%",
            expected="+0.5%",
            company_ticker="DPZ",
            peer_tickers=["PZZA", "MCD", "YUM"],
        )
    """
    actual_str   = f"Actual:   {actual}"   if actual   else ""
    expected_str = f"Expected: {expected}" if expected else ""
    peers_str    = (
        f"Benchmark peers: {', '.join(peer_tickers)}"
        if peer_tickers else ""
    )
    company_str = f"{company} ({company_ticker})" if company_ticker else company

    user_msg = textwrap.dedent(f"""
        Explain the following business result:

        **Company**: {company_str}
        **Metric**: {metric}
        **Period**: {period}
        {actual_str}
        {expected_str}
        {peers_str}

        Pull macro indicators for the relevant period. Retrieve competitor XBRL for peers
        in the same period if tickers are provided. Search the corpus for filing language,
        trade media coverage, and consumer research relevant to this time period and metric.
        Build a thorough, evidence-backed causal narrative.
    """).strip()

    return run_agent(
        messages=[{"role": "user", "content": user_msg}],
        system_prompt=_EXPLAIN_MISS_SYSTEM,
        tools=_tools(
            "search_corpus",
            "get_competitor_financials",
            "get_macro_indicators",
            "get_trade_signals",
        ),
        verbose=verbose,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TRADE AREA SCORE
# ═══════════════════════════════════════════════════════════════════════════════

_TRADE_AREA_SYSTEM = BASE_SYSTEM + textwrap.dedent("""
    You are a site selection and trade area analyst producing a scored market assessment.
    Scores must reflect actual data — do not assign scores without a supporting data point.
    Where data is missing from the corpus, say so and mark the dimension as unscored.

    Output structure (markdown):

    ## [City/Metro] — Trade Area Intelligence

    ### Market snapshot
    3-4 sentences: who lives here, how they eat, what the competitive environment looks like.

    ### Scored dimensions
    | Dimension | Score | Key data point |
    |---|---|---|
    | Consumer spending power | X/10 | [specific income/PCE stat + source] |
    | Labor cost pressure | X/10 | [BLS QCEW wage data or metro profile] |
    | Health trend headwinds | X/10 | [CDC obesity/diabetes rate if available] |
    | Competitive density | X/10 | [establishment count per capita if available] |
    | Population growth trajectory | X/10 | [Census ACS data] |
    | Food-away-from-home affinity | X/10 | [USDA food expenditure proxy] |

    **Overall market attractiveness**: X/10

    ### Key risks
    2-3 specific, data-backed risks for QSR operators in this market.

    ### Key opportunities
    2-3 specific, data-backed opportunities.

    ### Comparable markets
    Which other metros in the corpus have a similar demographic and economic profile?
    (Pull 1-2 from metro profiles.)
""")


def trade_area_score(
    city: str,
    brand: Optional[str] = None,
    verbose: bool = False,
) -> str:
    """
    Generate a scored trade area intelligence report.

    Args:
        city:    City or metro name, e.g. "Oklahoma City", "Columbus OH"
        brand:   Optional brand context for relevance tuning, e.g. "Domino's pizza delivery"
        verbose: Print agent trace

    Example:
        report = trade_area_score(
            city="Columbus, Ohio",
            brand="Domino's pizza delivery",
        )
    """
    brand_context = f"\nBrand context: evaluating specifically for {brand}." if brand else ""

    user_msg = textwrap.dedent(f"""
        Generate a trade area intelligence report for:

        **Market**: {city}
        {brand_context}

        Pull the metro profile first. Then search the corpus for any CDC health data,
        USDA food environment context, and labor market signals for this area.
        Score each dimension based on actual data retrieved.
    """).strip()

    return run_agent(
        messages=[{"role": "user", "content": user_msg}],
        system_prompt=_TRADE_AREA_SYSTEM,
        tools=_tools(
            "get_metro_profile",
            "search_corpus",
            "get_macro_indicators",
        ),
        verbose=verbose,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MORNING SIGNAL BRIEF
# ═══════════════════════════════════════════════════════════════════════════════

_MORNING_BRIEF_SYSTEM = BASE_SYSTEM + textwrap.dedent("""
    You are producing a concise daily intelligence brief for a QSR account executive.
    Target read time: under 3 minutes. Cut anything that isn't actionable or materially new.
    Prioritize trade media over general news. Cite sources inline (not in footnotes).

    Output structure (markdown):

    ## QSR Intelligence Brief — [Today's Date]

    ### Top signals
    3-5 bullets. Format: **[Source, Date]** What happened. Why it matters for QSR operators.

    ### Macro pulse
    2-3 key indicators. What direction are they moving? Has anything changed in the last month?
    Focus on CPIFABSL (food away from home pricing pressure) and MICH (consumer sentiment).

    ### Regulatory watch
    Anything new or notable in the last 14 days. Skip if nothing material.

    ### What to watch (30-60 day horizon)
    1-2 emerging themes worth monitoring. Specific and grounded, not generic trends.

    ---
    If there is genuinely nothing material on a topic, omit that section rather than padding.
""")


def morning_signal_brief(verbose: bool = False) -> str:
    """
    Generate a daily QSR intelligence brief.
    Pulls last 7 days of trade signals, latest macro indicator movement, and recent regulatory.

    Example:
        brief = morning_signal_brief()
        print(brief)
    """
    user_msg = textwrap.dedent("""
        Generate today's QSR intelligence brief.

        Pull trade signals from the last 7 days. Check the latest 3 months of macro indicators
        for anything that has moved meaningfully. Scan regulatory docs from the last 14 days.
        Synthesize into the tightest, most signal-dense brief possible.
    """).strip()

    return run_agent(
        messages=[{"role": "user", "content": user_msg}],
        system_prompt=_MORNING_BRIEF_SYSTEM,
        tools=_tools(
            "get_trade_signals",
            "get_macro_indicators",
            "get_regulatory_docs",
            "search_corpus",
        ),
        verbose=verbose,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. COMPETITIVE DEEP DIVE
# ═══════════════════════════════════════════════════════════════════════════════

_COMPETITIVE_DIVE_SYSTEM = BASE_SYSTEM + textwrap.dedent("""
    You are a competitive intelligence analyst. Your output should be specific enough
    to inform a sales or product strategy decision. Avoid generic competitive framing.
    Every section needs at least one corpus-sourced data point or passage.

    Output structure (markdown):

    ## [Company] — Competitive Intelligence: [Topic]

    ### Current position
    Where does this company stand on [topic] today? Be specific — metrics, not adjectives.

    ### Financial trajectory
    What do the XBRL numbers reveal about their investment in / performance on this topic?
    Use YoY comparisons where possible.

    ### What they're saying publicly
    Relevant language from SEC filings, earnings commentary, or Wikipedia. Quote with source.

    ### Peer comparison
    How are the named competitors approaching this topic? Who is ahead? Who is behind?

    ### Consumer angle
    What does the Pew / consumer research corpus say about consumer attitudes on this topic?
    Does consumer sentiment support or challenge this company's approach?

    ### Implications for engagement
    2-3 sentences: given all of the above, how should you position in conversations with this company?
""")


def competitive_deep_dive(
    company: str,
    topic: str,
    company_ticker: Optional[str] = None,
    peer_tickers: Optional[list[str]] = None,
    verbose: bool = False,
) -> str:
    """
    Generate a competitive intelligence deep dive on a specific company + topic.

    Args:
        company:        Primary company to analyze, e.g. "Domino's Pizza"
        topic:          Specific topic, e.g. "digital ordering adoption", "delivery economics",
                        "loyalty program ROI", "AI / data infrastructure investment"
        company_ticker: For XBRL financials, e.g. "DPZ"
        peer_tickers:   Peer tickers to benchmark against, e.g. ["PZZA", "MCD", "CMG"]
        verbose:        Print agent trace

    Example:
        intel = competitive_deep_dive(
            company="Domino's Pizza",
            topic="digital ordering and loyalty program",
            company_ticker="DPZ",
            peer_tickers=["PZZA", "MCD", "CMG"],
        )
    """
    tickers = [t for t in ([company_ticker] + (peer_tickers or [])) if t]
    tickers_str = f"Pull XBRL for: {', '.join(tickers)}" if tickers else ""
    company_str = f"{company} ({company_ticker})" if company_ticker else company

    user_msg = textwrap.dedent(f"""
        Generate a competitive intelligence deep dive:

        **Company**: {company_str}
        **Topic**: {topic}
        {tickers_str}

        First pull the competitor profile for {company}. Then retrieve XBRL financials
        for all tickers. Search the corpus for relevant evidence on {topic} — especially
        SEC filing language, trade media coverage, and Pew consumer research.
        Build a specific, evidence-backed competitive assessment.
    """).strip()

    return run_agent(
        messages=[{"role": "user", "content": user_msg}],
        system_prompt=_COMPETITIVE_DIVE_SYSTEM,
        tools=_tools(
            "get_competitor_profile",
            "get_competitor_financials",
            "search_corpus",
            "get_trade_signals",
        ),
        verbose=verbose,
    )
