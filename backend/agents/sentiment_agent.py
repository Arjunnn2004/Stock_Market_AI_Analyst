from __future__ import annotations

from typing import Any

from backend.services.langchain_service import LangChainService


class SentimentAgent:
    def __init__(self, llm: LangChainService | None = None) -> None:
        self.llm = llm or LangChainService()

    def analyze(self, news: dict[str, Any], company_name: str | None = None) -> dict[str, Any]:
        headlines = news.get("headlines", [])
        if not headlines:
            return {
                "agent": "sentiment",
                "score": 0.0,
                "label": "Neutral",
                "positive_hits": 0,
                "negative_hits": 0,
                "rationale": "No recent headlines found.",
            }

        sentiment = self.llm.classify_sentiment(company_name or "the company", headlines)
        label = sentiment.get("label", "Neutral") if isinstance(sentiment, dict) else "Neutral"
        rationale = sentiment.get("rationale", "") if isinstance(sentiment, dict) else ""
        score = sentiment.get("score", 0.0) if isinstance(sentiment, dict) else 0.0

        return {
            "agent": "sentiment",
            "score": score if isinstance(score, int | float) else 0.0,
            "label": label,
            "positive_hits": 0,
            "negative_hits": 0,
            "rationale": rationale,
        }

