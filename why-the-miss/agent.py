"""
QSR Research Agent — Core Agent Loop

run_agent() is the main entry point. It handles:
  - The Anthropic tool-use loop (tool_use → execute → tool_result → repeat)
  - Verbose tracing for debugging
  - Graceful handling of unexpected stop reasons

Usage:
    from agent import run_agent
    from tools import TOOL_SCHEMAS

    result = run_agent(
        messages=[{"role": "user", "content": "..."}],
        system_prompt="...",
        verbose=True
    )
    print(result)
"""
from __future__ import annotations

import json
import textwrap
from typing import Optional

import anthropic

from config import config
from tools import TOOL_SCHEMAS, execute_tool

# ── Shared Anthropic client ───────────────────────────────────────────────────

_client: Optional[anthropic.Anthropic] = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=config.anthropic_api_key)
    return _client


# ── Base system prompt (shared across all flows) ──────────────────────────────

BASE_SYSTEM = textwrap.dedent("""\
    You are a senior research analyst with access to a comprehensive QSR (Quick Service
    Restaurant) intelligence corpus covering publicly traded chains, consumer research,
    macro-economic data, and regulatory filings.

    Corpus capabilities:
    - SEC XBRL financials for 8 public QSR companies (DPZ, MCD, YUM, CMG, PZZA, WEN, DIN, FAT)
    - 71 full-text Pew Research studies on consumer behavior and digital attitudes
    - 585 Federal Register documents tagged by QSR topic (labor, food safety, franchise, delivery)
    - FRED macro indicators with MoM/YoY deltas (CPI, CPIFABSL, UNRATE, PCE, MICH)
    - QSR trade media: PMQ, NRN, QSR Magazine, Restaurant Business, Pizza Today
    - 413 metro profiles with demographics, wages, and restaurant density
    - 958K county-level USDA Food Atlas rows and 551K CDC PLACES health indicator rows
    - 4,959 searchable chunks across 3,582 documents via GTE-Large-EN vector search

    Core operating principles:
    1. Cite specific data sources and dates. Never assert a number without a source.
    2. Distinguish structured data (XBRL, FRED numbers) from qualitative evidence (Pew text, trade media).
    3. Be direct and specific — avoid generic statements that apply to any company or industry.
    4. When corpus data is stale or missing, say so explicitly rather than filling gaps with inference.
    5. Use multiple tool calls when needed. A well-grounded response is better than a fast one.
""")


# ── Agent loop ────────────────────────────────────────────────────────────────

def run_agent(
    messages: list[dict],
    system_prompt: str,
    tools: Optional[list[dict]] = None,
    max_iterations: int = None,
    verbose: bool = False,
) -> str:
    """
    Run the tool-use agent loop until stop_reason == 'end_turn' or max_iterations.

    Args:
        messages:       Initial messages in Anthropic format.
        system_prompt:  System prompt for this specific flow.
        tools:          Tool schemas to make available. Defaults to all 7.
        max_iterations: Safety ceiling on tool loops. Defaults to config value.
        verbose:        Print iteration-level trace to stdout.

    Returns:
        Final text response as a string.
    """
    client     = _get_client()
    tools_use  = tools or TOOL_SCHEMAS
    max_iters  = max_iterations or config.max_agent_iterations
    conv       = list(messages)

    for iteration in range(max_iters):
        if verbose:
            print(f"\n[agent] iteration={iteration + 1}")

        response = client.messages.create(
            model=config.model,
            max_tokens=config.max_tokens,
            system=system_prompt,
            tools=tools_use,
            messages=conv,
        )

        if verbose:
            print(f"[agent] stop_reason={response.stop_reason}  "
                  f"input_tokens={response.usage.input_tokens}  "
                  f"output_tokens={response.usage.output_tokens}")

        # ── End turn: return the text ─────────────────────────────────────────
        if response.stop_reason == "end_turn":
            return _extract_text(response)

        # ── Unexpected stop ───────────────────────────────────────────────────
        if response.stop_reason != "tool_use":
            if verbose:
                print(f"[agent] unexpected stop_reason={response.stop_reason}, returning text")
            return _extract_text(response)

        # ── Tool use: collect calls, execute, append results ──────────────────
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            if verbose:
                args_preview = json.dumps(block.input)[:160]
                print(f"[agent] tool_call  name={block.name}  input={args_preview}")

            result = execute_tool(block.name, block.input)

            if verbose:
                result_preview = json.dumps(result, default=str)[:200]
                print(f"[agent] tool_result {result_preview}")

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result, default=str),
            })

        # Extend the conversation with the assistant turn + all tool results
        conv.append({"role": "assistant", "content": response.content})
        conv.append({"role": "user",      "content": tool_results})

    return "[max_iterations reached] Partial result — increase max_iterations or simplify the query."


def _extract_text(response) -> str:
    """Pull the first text block from a response, or empty string."""
    for block in response.content:
        if block.type == "text":
            return block.text
    return ""
