from __future__ import annotations

from typing import Any

import pandas as pd
import yfinance as yf

from backend.core.config import get_settings
from backend.services.langchain_service import LangChainService


class TechnicalAgent:
    def __init__(self, llm: LangChainService | None = None) -> None:
        self.llm = llm or LangChainService()
        self.settings = get_settings()

    def analyze(self, ticker: str) -> dict[str, Any]:
        history = yf.Ticker(ticker).history(
            period=self.settings.technical_history_period,
            interval=self.settings.technical_history_interval,
            auto_adjust=True,
        )
        if history.empty:
            return {"agent": "technical", "error": "No price history available."}

        close = history["Close"].dropna()
        indicators = {
            f"rsi_{self.settings.technical_rsi_period}": _rsi(close, self.settings.technical_rsi_period),
            f"sma_{self.settings.technical_sma_short_window}": _last(close.rolling(self.settings.technical_sma_short_window).mean()),
            f"sma_{self.settings.technical_sma_long_window}": _last(close.rolling(self.settings.technical_sma_long_window).mean()),
            f"ema_{self.settings.technical_ema_window}": _last(close.ewm(span=self.settings.technical_ema_window, adjust=False).mean()),
        }

        macd_line = close.ewm(span=self.settings.technical_macd_fast_window, adjust=False).mean() - close.ewm(span=self.settings.technical_macd_slow_window, adjust=False).mean()
        signal_line = macd_line.ewm(span=self.settings.technical_macd_signal_window, adjust=False).mean()
        indicators["macd"] = _last(macd_line)
        indicators["macd_signal"] = _last(signal_line)

        rolling_20 = close.rolling(self.settings.technical_bollinger_window)
        middle = rolling_20.mean()
        std = rolling_20.std()
        indicators["bollinger_upper"] = _last(middle + self.settings.technical_bollinger_stddev * std)
        indicators["bollinger_lower"] = _last(middle - self.settings.technical_bollinger_stddev * std)
        indicators["last_close"] = _last(close)

        interpretation = self.llm.interpret_technical(ticker, indicators)
        return {
            "agent": "technical",
            "indicators": indicators,
            "trend": interpretation.get("trend", "Neutral"),
            "rationale": interpretation.get("rationale", ""),
            "interpretation": interpretation,
        }


def _rsi(close: pd.Series, period: int = 14) -> float | None:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return _last(100 - (100 / (1 + rs)))


def _last(series: pd.Series) -> float | None:
    clean = series.dropna()
    if clean.empty:
        return None
    return round(float(clean.iloc[-1]), 4)


