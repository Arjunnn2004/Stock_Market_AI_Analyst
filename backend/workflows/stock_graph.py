from __future__ import annotations

from dataclasses import asdict
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from backend.agents.financial_agent import FinancialAgent
from backend.agents.forecast_agent import ForecastAgent
from backend.agents.graph_agent import GraphAgent
from backend.agents.news_agent import NewsAgent
from backend.agents.rag_agent import RagAgent
from backend.agents.risk_agent import RiskAgent
from backend.agents.sentiment_agent import SentimentAgent
from backend.agents.technical_agent import TechnicalAgent
from backend.services.langchain_service import LangChainService


class StockAnalysisState(TypedDict, total=False):
    query: str
    ticker: str
    company_name: str
    financial: dict[str, Any]
    news: dict[str, Any]
    sentiment: dict[str, Any]
    technical: dict[str, Any]
    rag: dict[str, Any]
    risk: dict[str, Any]
    graph: dict[str, Any]
    forecast: dict[str, Any]
    agents: dict[str, Any]
    final_analysis: str


class StockAnalysisGraph:
    def __init__(self) -> None:
        self.llm = LangChainService()
        self.financial_agent = FinancialAgent(self.llm)
        self.news_agent = NewsAgent(self.llm)
        self.sentiment_agent = SentimentAgent(self.llm)
        self.technical_agent = TechnicalAgent(self.llm)
        self.rag_agent = RagAgent()
        self.risk_agent = RiskAgent(self.llm)
        self.graph_agent = GraphAgent(self.llm)
        self.forecast_agent = ForecastAgent(self.llm)
        self.graph = self._build_graph().compile()

    def analyze(self, query: str) -> dict[str, Any]:
        result = self.graph.invoke({"query": query})
        return {
            "query": result.get("query", query),
            "ticker": result.get("ticker", ""),
            "company_name": result.get("company_name", ""),
            "final_analysis": result.get("final_analysis", ""),
            "agents": result.get("agents", {}),
        }

    def _build_graph(self) -> StateGraph[StockAnalysisState]:
        graph = StateGraph(StockAnalysisState)
        graph.add_node("financial", self._financial_node)
        graph.add_node("news", self._news_node)
        graph.add_node("sentiment", self._sentiment_node)
        graph.add_node("technical", self._technical_node)
        graph.add_node("rag", self._rag_node)
        graph.add_node("risk", self._risk_node)
        graph.add_node("graph", self._graph_node)
        graph.add_node("forecast", self._forecast_node)
        graph.add_node("final", self._final_node)

        graph.add_edge(START, "financial")
        graph.add_edge("financial", "news")
        graph.add_edge("news", "sentiment")
        graph.add_edge("sentiment", "technical")
        graph.add_edge("technical", "rag")
        graph.add_edge("rag", "risk")
        graph.add_edge("risk", "graph")
        graph.add_edge("graph", "forecast")
        graph.add_edge("forecast", "final")
        graph.add_edge("final", END)
        return graph

    def _financial_node(self, state: StockAnalysisState) -> dict[str, Any]:
        analysis = self.financial_agent.analyze(state["query"])
        financial = asdict(analysis)
        return {
            "financial": financial,
            "ticker": financial["ticker"],
            "company_name": financial["company_name"],
        }

    def _news_node(self, state: StockAnalysisState) -> dict[str, Any]:
        news = self.news_agent.analyze(state["ticker"], state["company_name"])
        return {"news": news}

    def _sentiment_node(self, state: StockAnalysisState) -> dict[str, Any]:
        sentiment = self.sentiment_agent.analyze(state.get("news", {}), state.get("company_name"))
        return {"sentiment": sentiment}

    def _technical_node(self, state: StockAnalysisState) -> dict[str, Any]:
        technical = self.technical_agent.analyze(state["ticker"])
        return {"technical": technical}

    def _rag_node(self, state: StockAnalysisState) -> dict[str, Any]:
        rag = self.rag_agent.analyze(state["ticker"], state["query"])
        return {"rag": rag}

    def _risk_node(self, state: StockAnalysisState) -> dict[str, Any]:
        risk = self.risk_agent.analyze(state.get("financial", {}), state.get("news", {}), state.get("rag", {}), state.get("technical", {}))
        return {"risk": risk}

    def _graph_node(self, state: StockAnalysisState) -> dict[str, Any]:
        graph = self.graph_agent.analyze(state["ticker"], state["company_name"], state.get("rag", {}))
        return {"graph": graph}

    def _forecast_node(self, state: StockAnalysisState) -> dict[str, Any]:
        forecast = self.forecast_agent.analyze(state["ticker"])
        return {"forecast": forecast}

    def _final_node(self, state: StockAnalysisState) -> dict[str, Any]:
        agents = {
            "financial": state.get("financial", {}),
            "news": state.get("news", {}),
            "sentiment": state.get("sentiment", {}),
            "technical": state.get("technical", {}),
            "rag": state.get("rag", {}),
            "risk": state.get("risk", {}),
            "graph": state.get("graph", {}),
            "forecast": state.get("forecast", {}),
        }
        final_analysis = self.llm.summarize_multi_agent(state["query"], agents)
        return {"agents": agents, "final_analysis": final_analysis}