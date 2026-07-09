from __future__ import annotations

from typing import Any

from backend.services.langchain_service import LangChainService
from backend.core.config import get_settings


class RiskAgent:
    def __init__(self, llm: LangChainService | None = None) -> None:
        self.llm = llm or LangChainService()
        self.settings = get_settings()

    def analyze(
        self,
        financial: dict[str, Any],
        news: dict[str, Any],
        rag: dict[str, Any] | None = None,
        technical: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        risk_output = self.llm.summarize_risk(
            financial.get("company_name", "the company"),
            financial,
            news,
            rag,
            technical,
        )
        risks = risk_output.get("risks", []) if isinstance(risk_output, dict) else []
        if isinstance(risks, list):
            risks = [str(item) for item in risks if str(item).strip()]
        else:
            risks = []

        if not risks:
            risks = ["LLM risk synthesis returned no actionable risks."]

        return {
            "agent": "risk",
            "risks": list(dict.fromkeys(risks))[: self.settings.max_risk_items],
            "note": risk_output.get("note", "") if isinstance(risk_output, dict) else "",
        }

