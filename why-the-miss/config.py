"""
QSR Research Agent — Configuration
Reads from env vars or Databricks secrets; falls back to defaults for the RL workspace.
"""
import os
from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    # ── Databricks ────────────────────────────────────────────────────────────
    workspace_host: str = field(
        default_factory=lambda: os.getenv(
            "DATABRICKS_HOST",
            "https://fevm-serverless-stable-h7wanf.cloud.databricks.com"
        )
    )
    http_path: str = field(
        default_factory=lambda: os.getenv("DATABRICKS_HTTP_PATH", "")
    )
    access_token: str = field(
        default_factory=lambda: os.getenv("DATABRICKS_TOKEN", "")
    )

    # ── Catalog / schema references ───────────────────────────────────────────
    catalog: str = "serverless_stable_h7wanf_catalog"

    # bronze tables
    bronze_xbrl:     str = "rl_bronze.bronze_xbrl_financials"
    bronze_trade:    str = "rl_bronze.bronze_trade_media_rss"
    bronze_news:     str = "rl_bronze.bronze_news_articles"
    bronze_wiki_qsr: str = "rl_bronze.bronze_wikipedia_qsr"

    # silver tables
    silver_fred:         str = "rl_silver.silver_fred_monthly"
    silver_competitors:  str = "rl_silver.silver_competitor_profiles"
    silver_regulatory:   str = "rl_silver.silver_regulatory_tagged"

    # gold tables
    gold_metro:   str = "rl_gold.gold_metro_profiles"
    gold_docs:    str = "rl_gold.gold_corpus_docs"

    # ── Vector Search ─────────────────────────────────────────────────────────
    vs_endpoint:   str = "rl_corpus_vs_endpoint"
    vs_index_name: str = "serverless_stable_h7wanf_catalog.rl_gold.gold_corpus_chunks_index"

    # ── Anthropic ─────────────────────────────────────────────────────────────
    anthropic_api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )
    model:      str = "claude-opus-4-5"
    max_tokens: int = 4096

    # ── Behaviour ─────────────────────────────────────────────────────────────
    max_agent_iterations: int = 14
    default_tool_timeout:  int = 30   # seconds

    def fqn(self, schema_table: str) -> str:
        """Return a fully qualified table name: catalog.schema.table"""
        return f"{self.catalog}.{schema_table}"

    def hostname(self) -> str:
        """Strip https:// for the SQL connector."""
        return self.workspace_host.replace("https://", "").rstrip("/")


# Module-level singleton — import this everywhere
config = AgentConfig()
