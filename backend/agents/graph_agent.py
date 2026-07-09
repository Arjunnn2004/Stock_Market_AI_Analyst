from __future__ import annotations

from typing import Any

from backend.services.langchain_service import LangChainService


class GraphAgent:
    def __init__(self, llm: LangChainService | None = None) -> None:
        self.llm = llm or LangChainService()

    def analyze(self, ticker: str, company_name: str, rag: dict[str, Any] | None = None) -> dict[str, Any]:
        relationship_data = self.llm.extract_relationships(company_name, ticker, rag)
        return {
            "agent": "graph",
            "company": company_name,
            "relationships": relationship_data.get("relationships", []),
            "note": relationship_data.get("note", "LLM-derived relationship extraction from available documents."),
        }

