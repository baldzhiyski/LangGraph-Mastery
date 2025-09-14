# LangGraph Deep Research Agent (with Bright Data)

> A multi-step, production-grade research agent that plans, searches, crawls, ranks, and synthesizes insights from the open web (Google, Bing, Reddit, and more) using [LangGraph]. Network resilience and anti-blocking powered by [Bright Data] proxies.

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB.svg" />
  <img alt="LangGraph" src="https://img.shields.io/badge/LangGraph-Agents%20&%20State-blue" />
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green" />
</p>

---

## ✨ Highlights

* **Multi-step research pipeline**: Planner → Search/Fetch → Parse → Rank → Synthesize → Report
* **Multi-source search**: Google, Bing, Reddit (easily add more: HN, ArXiv, news, etc.)
* **Bright Data integration**: rotating proxies, geo-targeting, retry logic, and rate limiting
* **LangGraph orchestration**: deterministic graphs with guarded transitions and typed state
* **Citations-first output**: every claim paired with source URLs & snippets
* **Config-driven**: enable/disable tools, tweak limits, prompts, and models via `.env` or YAML
* **Observability**: optional tracing via LangSmith/OpenTelemetry; structured logs with `rich`
* **Extensible**: add nodes/tools without changing the rest of the graph
---

## 🧠 Architecture

**Stateful graph (LangGraph)** coordinating tools & LLM reasoning:

```
[Input]
  ↓
(Planner) ──▶ (Search Fanout)
                ├─ Google
                ├─ Bing
                └─ Reddit
                     ↓
                 (Fetcher)
                     ↓
                 (Parser)
                     ↓
              (Rank & Deduplicate)
                     ↓
               (Synthesizer – LLM)
                     ↓
               (Reporter / Outputs)
                     ↓
                  [Result]
```

* **Planner** creates sub-queries and acceptance criteria
* **Search Fanout** queries multiple engines concurrently
* **Fetcher** retrieves pages via Bright Data with retries/backoff & geo options
* **Parser** extracts readable text, title, publish date, and metadata
* **Rank/Dedupe** normalizes & scores sources; removes near-duplicates
* **Synthesizer** produces a grounded report with inline citations
* **Reporter** renders JSON/Markdown (and optionally PDF)

---


