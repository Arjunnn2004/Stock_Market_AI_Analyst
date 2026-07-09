# Stock AI Analyst

Stock AI Analyst is a multi-agent stock research app built with LangChain and LangGraph.

The current codebase no longer depends on Gemini. The workflow is driven by a LangGraph supervisor and LangChain-style synthesis helpers.

It turns a natural-language question such as "Should I invest in Nvidia?" or "Should I invest in Nykaa?" into a structured research report with:

- financial data from Yahoo Finance
- recent headlines and sentiment
- technical indicators
- local RAG over filings and transcripts
- risk synthesis
- relationship extraction
- a short-horizon forecast
- a final analyst-style summary

The financial agent normalizes conversational research questions before resolving the ticker, so prompts like "Should I invest in PAYTM?" and "Should I invest in Nykaa?" can still map to the correct Yahoo Finance symbol.

## Stack

- FastAPI backend
- Streamlit frontend
- LangGraph for orchestration
- LangChain for prompt-style synthesis helpers and analysis heuristics
- yfinance for market data
- local `.txt` documents in `backend/data/filings` for RAG

## How It Works

1. The API receives a research question.
2. LangGraph routes the query through the agent workflow.
3. The financial agent normalizes the research question, resolves the ticker, and fetches company metrics.
4. The news agent fetches recent Yahoo Finance headlines.
5. The sentiment, technical, risk, graph, and forecast agents each build one slice of the analysis.
6. The final node combines all outputs into a readable report.

The FastAPI app exposes:

- `GET /health` for a quick status check
- `GET /analyze` for the single financial summary view
- `GET /multi-agent/analyze` for the full LangGraph workflow used by the frontend

## Project Layout

```text
stock-ai-analyst/
  backend/
    agents/
      financial_agent.py
      forecast_agent.py
      graph_agent.py
      news_agent.py
      rag_agent.py
      risk_agent.py
      sentiment_agent.py
      supervisor_agent.py
      technical_agent.py
    api/
      main.py
    core/
      config.py
    services/
      langchain_service.py
    workflows/
      stock_graph.py
    data/
      filings/
  frontend/
    streamlit_app.py
  requirements.txt
  README.md
```

## Run It

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the backend:

```bash
uvicorn backend.api.main:app --reload
```

Start the frontend:

```bash
streamlit run frontend/streamlit_app.py
```

## Notes

- The old Gemini service has been removed.
- The current workflow is deterministic and explainable, so it is easier to follow and debug.
- The frontend calls the multi-agent endpoint directly and displays each agent output in tabs.
- Drop `.txt` filings or transcripts into `backend/data/filings` to activate the local RAG step.
- If no local filings are present, the RAG section will show an empty state instead of failing.
