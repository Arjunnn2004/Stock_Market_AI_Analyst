from __future__ import annotations

from typing import Any

import yfinance as yf

from backend.core.config import get_settings
from backend.services.langchain_service import LangChainService


class ForecastAgent:
    def __init__(self, llm: LangChainService | None = None) -> None:
        self.llm = llm or LangChainService()
        self.settings = get_settings()

    def analyze(self, ticker: str) -> dict[str, Any]:
        history = yf.Ticker(ticker).history(
            period=self.settings.forecast_history_period,
            interval=self.settings.forecast_history_interval,
            auto_adjust=True,
        )
        if history.empty or len(history) < self.settings.forecast_min_history_points:
            return {"agent": "forecast", "error": "Not enough price history for forecast."}

        close = history["Close"].dropna()
        last_price = float(close.iloc[-1])
        return_5d = float(close.pct_change(self.settings.forecast_return_5d_window).iloc[-1])
        return_30d = float(close.pct_change(self.settings.forecast_return_30d_window).iloc[-1])
        volatility = float(close.pct_change().dropna().std() * (252**0.5))
        moving_average_20 = float(close.rolling(self.settings.forecast_short_ma_window).mean().iloc[-1])
        moving_average_50 = (
            float(close.rolling(self.settings.forecast_long_ma_window).mean().iloc[-1])
            if len(close) >= self.settings.forecast_long_ma_window
            else None
        )

        summary = {
            "last_price": round(last_price, 4),
            "return_5d_percent": round(return_5d * 100, 4),
            "return_30d_percent": round(return_30d * 100, 4),
            "annualized_volatility": round(volatility, 4),
            "moving_average_20": round(moving_average_20, 4),
            "moving_average_50": round(moving_average_50, 4) if moving_average_50 is not None else None,
            "recent_high": round(float(close.tail(self.settings.forecast_recent_high_low_window).max()), 4),
            "recent_low": round(float(close.tail(self.settings.forecast_recent_high_low_window).min()), 4),
        }

        outlook = self.llm.forecast_outlook(ticker, summary)
        expected_price = outlook.get("expected_price")
        confidence = outlook.get("confidence")

        return {
            "agent": "forecast",
            "horizon_days": self.settings.forecast_return_30d_window,
            "direction": outlook.get("direction", "Neutral"),
            "expected_return_percent": outlook.get("expected_return_percent", round(return_30d * 100, 2)),
            "expected_price": expected_price if isinstance(expected_price, int | float) else round(last_price, 2),
            "confidence": confidence if isinstance(confidence, int | float) else None,
            "summary": summary,
            "rationale": outlook.get("rationale", ""),
            "method": "LangChain heuristics over recent price statistics.",
        }

