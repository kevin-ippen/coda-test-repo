# Databricks notebook source
# MAGIC %md
# MAGIC # QSR Research Agent — Quickstart Notebook
# MAGIC
# MAGIC **Prerequisites**:
# MAGIC - Databricks Runtime 14.x+ (ML recommended)
# MAGIC - Cluster has access to `rl_corpus_vs_endpoint` Vector Search endpoint
# MAGIC - `ANTHROPIC_API_KEY` set in cluster environment variables or Databricks secrets
# MAGIC - `qsr_research_agent/` package in the repo root or installed via `%pip install -e .`
# MAGIC
# MAGIC **What this notebook demonstrates**:
# MAGIC 1. Meeting prep brief (Casey's CEO meeting)
# MAGIC 2. Explain the Miss (Domino's Q3 2024 comp sales)
# MAGIC 3. Trade Area Score (Columbus, OH for Domino's)
# MAGIC 4. Morning Signal Brief (daily digest)
# MAGIC 5. Competitive Deep Dive (Domino's digital ordering)

# COMMAND ----------

# MAGIC %pip install anthropic databricks-vectorsearch databricks-sql-connector --quiet

# COMMAND ----------

import sys
sys.path.insert(0, "/Workspace/Users/<YOUR_USER>/qsr_research_agent")  # adjust path

from flows import (
    meeting_prep_brief,
    explain_the_miss,
    trade_area_score,
    morning_signal_brief,
    competitive_deep_dive,
)

print("QSR Research Agent loaded.")

# COMMAND ----------

# MAGIC %md ## 1. Meeting prep brief — Casey's General Stores CEO meeting

caseys_brief = meeting_prep_brief(
    company="Casey's General Stores",
    attendees=[
        "Darren Rebelez, President & CEO",
        "Brian Johnson, SVP & CIO",
        "Megan Elfers, SVP & CMO",
    ],
    meeting_context=(
        "Introducing Databricks as the unified data and AI platform. "
        "Focus on loyalty analytics, personalization at scale, and "
        "reducing dependency on disparate point solutions."
    ),
    competitor_tickers=["MCD", "WEN"],  # for benchmarking context
    verbose=True,
)

displayHTML(caseys_brief.replace("\n", "<br>"))

# COMMAND ----------

# MAGIC %md ## 2. Explain the miss — Domino's Q3 2024 comp sales

miss_analysis = explain_the_miss(
    company="Domino's Pizza",
    metric="US same-store sales",
    period="Q3 2024",
    actual="-2.9%",
    expected="+0.5% (consensus estimate)",
    company_ticker="DPZ",
    peer_tickers=["PZZA", "MCD", "YUM"],
    verbose=True,
)

displayHTML(miss_analysis.replace("\n", "<br>"))

# COMMAND ----------

# MAGIC %md ## 3. Trade area score — Columbus, OH

columbus_report = trade_area_score(
    city="Columbus, Ohio",
    brand="Domino's pizza delivery",
    verbose=True,
)

displayHTML(columbus_report.replace("\n", "<br>"))

# COMMAND ----------

# MAGIC %md ## 4. Morning signal brief

daily_brief = morning_signal_brief(verbose=True)

displayHTML(daily_brief.replace("\n", "<br>"))

# COMMAND ----------

# MAGIC %md ## 5. Competitive deep dive — Domino's digital ordering

digital_intel = competitive_deep_dive(
    company="Domino's Pizza",
    topic="digital ordering adoption, loyalty program, and first-party data strategy",
    company_ticker="DPZ",
    peer_tickers=["PZZA", "MCD", "CMG"],
    verbose=True,
)

displayHTML(digital_intel.replace("\n", "<br>"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Running as a Mosaic AI Workflow (scheduled)
# MAGIC
# MAGIC To run the morning brief on a daily schedule:
# MAGIC
# MAGIC 1. Create a Databricks Workflow (Jobs > Create Job)
# MAGIC 2. Add a notebook task pointing to this notebook
# MAGIC 3. Under "Parameters", you can pass `flow=morning_brief` if you parameterize
# MAGIC 4. Set a daily schedule (e.g. 6:00 AM CT)
# MAGIC 5. Set up email/Slack notifications on the job
# MAGIC
# MAGIC To persist outputs to Delta:
# MAGIC
# MAGIC ```python
# MAGIC from datetime import date
# MAGIC from pyspark.sql import Row
# MAGIC
# MAGIC row = Row(
# MAGIC     run_date=str(date.today()),
# MAGIC     flow="morning_brief",
# MAGIC     output=daily_brief,
# MAGIC )
# MAGIC spark.createDataFrame([row]).write.mode("append").saveAsTable(
# MAGIC     "serverless_stable_h7wanf_catalog.rl_gold.agent_outputs"
# MAGIC )
# MAGIC ```
