from __future__ import annotations

from typing import Any

from backend.workflows.stock_graph import StockAnalysisGraph


class SupervisorAgent:
    def __init__(self) -> None:
        self.workflow = StockAnalysisGraph()

    def analyze(self, query: str) -> dict[str, Any]:
        return self.workflow.analyze(query)

