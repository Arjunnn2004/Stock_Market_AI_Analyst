from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import yfinance as yf

from backend.services.langchain_service import LangChainService


@dataclass(frozen=True)
class FinancialAnalysis:
    query: str
    ticker: str
    company_name: str
    metrics: dict[str, Any]
    summary: str


class FinancialAgent:
    COMPANY_ALIASES = {
        "nykaa": "FSN.NS",
        "fsn": "FSN.NS",
        "fsn ecommerce": "FSN.NS",
        "fsn e-commerce": "FSN.NS",
        "reliance": "RELIANCE.NS",
        "tcs": "TCS.NS",
        "infosys": "INFY.NS",
        "infy": "INFY.NS",
        "hdfcbank": "HDFCBANK.NS",
        "paytm": "PAYTM.NS",
    }

    def __init__(self, llm: LangChainService | None = None) -> None:
        self.llm = llm or LangChainService()

    def analyze(self, query: str) -> FinancialAnalysis:
        ticker = self._resolve_ticker(query)
        stock = yf.Ticker(ticker)
        info = stock.info

        if not _is_valid_equity_info(info):
            raise ValueError(f"Could not find financial data for '{query}'.")

        company_name = info.get("longName") or info.get("shortName") or ticker
        metrics = self._build_metrics(info)
        summary = self.llm.summarize_stock(company_name, metrics)

        return FinancialAnalysis(
            query=query,
            ticker=ticker,
            company_name=company_name,
            metrics=metrics,
            summary=summary,
        )

    def _resolve_ticker(self, query: str) -> str:
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Query cannot be empty.")

        normalized_query = self._normalize_research_question(cleaned_query)
        for alias, ticker in self.COMPANY_ALIASES.items():
            alias_pattern = rf"\b{re.escape(alias)}\b"
            if re.search(alias_pattern, normalized_query):
                resolved = self._first_valid_ticker([ticker])
                if resolved:
                    return resolved

        ticker_match = re.fullmatch(r"[A-Za-z]{1,8}(?:\.[A-Za-z]{1,4})?", cleaned_query)
        if ticker_match:
            resolved = self._first_valid_ticker([ticker_match.group(0).upper()])
            if resolved:
                return resolved

        searched = self._search_yahoo_symbol(cleaned_query)
        if searched:
            return searched

        raise ValueError(f"Could not resolve '{query}' to a supported Yahoo Finance ticker.")

    @staticmethod
    def _normalize_research_question(query: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", " ", query.lower()).strip()
        prefixes = (
            r"^should i invest in ",
            r"^should i buy ",
            r"^should i sell ",
            r"^analyze ",
            r"^analyse ",
            r"^research ",
            r"^tell me about ",
            r"^what about ",
            r"^is ",
            r"^how is ",
            r"^would you buy ",
            r"^invest in ",
        )

        for pattern in prefixes:
            normalized = re.sub(pattern, "", normalized).strip()

        return normalized

    def _search_yahoo_symbol(self, query: str) -> str | None:
        try:
            quotes = yf.Search(query, max_results=10).quotes
        except Exception:
            return None

        equity_quotes = [quote for quote in quotes if quote.get("quoteType") == "EQUITY"]
        if not equity_quotes:
            return None

        symbols = [quote.get("symbol") for quote in equity_quotes if quote.get("symbol")]
        return self._first_valid_ticker(symbols)

    @staticmethod
    def _first_valid_ticker(candidates: list[str]) -> str | None:
        for candidate in candidates:
            try:
                info = yf.Ticker(candidate).info
            except Exception:
                continue
            if _is_valid_equity_info(info):
                return candidate
        return None

    @staticmethod
    def _build_metrics(info: dict[str, Any]) -> dict[str, Any]:
        return {
            "current_price": info.get("currentPrice"),
            "revenue_growth_percent": _to_percent(info.get("revenueGrowth")),
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "market_cap": info.get("marketCap"),
            "profit_margin_percent": _to_percent(info.get("profitMargins")),
            "gross_margin_percent": _to_percent(info.get("grossMargins")),
            "total_revenue": info.get("totalRevenue"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "currency": info.get("currency"),
        }


def _to_percent(value: Any) -> float | None:
    if isinstance(value, int | float):
        return value * 100
    return None


def _is_valid_equity_info(info: dict[str, Any]) -> bool:
    if not info or info.get("quoteType") in {None, "NONE"}:
        return False
    return bool(info.get("symbol") or info.get("shortName") or info.get("longName"))
