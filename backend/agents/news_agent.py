from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import yfinance as yf

from backend.services.langchain_service import LangChainService
from backend.core.config import get_settings


class NewsAgent:
    def __init__(self, llm: LangChainService | None = None) -> None:
        self.llm = llm or LangChainService()
        self.settings = get_settings()

    def analyze(self, ticker: str, company_name: str) -> dict[str, Any]:
        headlines = self._fetch_yfinance_news(ticker)
        sentiment = self.llm.classify_sentiment(company_name, headlines)

        return {
            "agent": "news",
            "headlines": headlines,
            "sentiment": sentiment,
        }

    def _fetch_yfinance_news(self, ticker: str) -> list[dict[str, Any]]:
        try:
            items = yf.Ticker(ticker).news or []
        except Exception:
            return []

        headlines: list[dict[str, Any]] = []
        for item in items[: self.settings.max_headlines_considered]:
            content = item.get("content", item)
            published = content.get("pubDate") or item.get("providerPublishTime")
            if isinstance(published, int | float):
                published = datetime.fromtimestamp(published, tz=timezone.utc).isoformat()

            headlines.append(
                {
                    "title": content.get("title") or item.get("title"),
                    "publisher": content.get("provider", {}).get("displayName")
                    if isinstance(content.get("provider"), dict)
                    else item.get("publisher"),
                    "url": content.get("canonicalUrl", {}).get("url")
                    if isinstance(content.get("canonicalUrl"), dict)
                    else item.get("link"),
                    "published": published,
                }
            )

        return [headline for headline in headlines if headline.get("title")]

