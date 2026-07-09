from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from backend.agents.financial_agent import FinancialAgent
from backend.agents.supervisor_agent import SupervisorAgent


app = FastAPI(
    title="Stock AI Analyst",
    description="Multi-agent stock research assistant powered by yfinance, LangChain, and LangGraph.",
    version="0.2.0",
)

financial_agent = FinancialAgent()
supervisor_agent = SupervisorAgent()


class AnalyzeResponse(BaseModel):
    query: str
    ticker: str
    company_name: str
    metrics: dict[str, object]
    summary: str


class MultiAgentResponse(BaseModel):
    query: str
    ticker: str
    company_name: str
    final_analysis: str
    agents: dict[str, object]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/analyze", response_model=AnalyzeResponse)
def analyze(query: str = Query(..., min_length=1)) -> AnalyzeResponse:
    try:
        result = financial_agent.analyze(query)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Financial data lookup failed: {exc}") from exc

    return AnalyzeResponse(
        query=result.query,
        ticker=result.ticker,
        company_name=result.company_name,
        metrics=result.metrics,
        summary=result.summary,
    )


@app.get("/multi-agent/analyze", response_model=MultiAgentResponse)
def multi_agent_analyze(
    query: str = Query(..., min_length=1)
) -> MultiAgentResponse:
    try:
        result = supervisor_agent.analyze(query)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Multi-agent analysis failed: {exc}") from exc

    return MultiAgentResponse(**result)
