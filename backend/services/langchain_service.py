from __future__ import annotations

import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from backend.core.config import get_settings


class LangChainService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.final_report_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a concise equity research assistant. Use only the supplied data and avoid personalized investment advice.",
                ),
                (
                    "human",
                    "Query: {query}\n\nCompose a report with sections for Executive view, Financial picture, News and sentiment, Technical trend, Key risks, 30-day forecast, and Bottom line.",
                ),
            ]
        )

    def summarize_stock(self, company: str, metrics: dict[str, object]) -> str:
        revenue_growth = metrics.get("revenue_growth_percent")
        trailing_pe = metrics.get("trailing_pe")
        market_cap = metrics.get("market_cap")
        currency = metrics.get("currency") or "USD"

        parts = [f"{company} shows"]
        if isinstance(revenue_growth, int | float):
            parts.append(f"revenue growth of {revenue_growth:.2f}%")
        else:
            parts.append("limited available revenue growth data")

        if isinstance(trailing_pe, int | float):
            parts.append(f"and trades at a trailing PE of {trailing_pe:.2f}")

        if isinstance(market_cap, int | float):
            parts.append(f"with a market cap of {currency} {market_cap:,.0f}")

        return " ".join(parts) + ". Review recent filings and news before drawing an investment conclusion."

    def classify_sentiment(self, company: str, headlines: list[dict[str, object]]) -> dict[str, object]:
        if not headlines:
            return {"label": "Neutral", "score": 0.0, "rationale": "No recent headlines found."}

        positive_words = (
            "beat",
            "bullish",
            "growth",
            "gain",
            "record",
            "rally",
            "rise",
            "strong",
            "surge",
            "upgrade",
            "profit",
            "outperform",
        )
        negative_words = (
            "bearish",
            "downgrade",
            "drop",
            "fall",
            "lawsuit",
            "loss",
            "probe",
            "risk",
            "slump",
            "weak",
            "warning",
            "miss",
        )

        text = " ".join(str(item.get("title", "")) for item in headlines[: self.settings.max_headlines_considered]).lower()
        positive_hits = sum(text.count(word) for word in positive_words)
        negative_hits = sum(text.count(word) for word in negative_words)
        raw_score = positive_hits - negative_hits
        score = max(-1.0, min(1.0, raw_score / 4 if raw_score else 0.0))

        if score > 0.15:
            label = "Positive"
        elif score < -0.15:
            label = "Negative"
        else:
            label = "Neutral"

        rationale = self._headline_rationale(company, positive_hits, negative_hits, headlines)
        return {"label": label, "score": score, "rationale": rationale}

    def interpret_technical(self, ticker: str, indicators: dict[str, object]) -> dict[str, object]:
        signals: list[str] = []
        score = 0

        rsi = indicators.get(f"rsi_{self.settings.technical_rsi_period}")
        sma_50 = indicators.get(f"sma_{self.settings.technical_sma_short_window}")
        sma_200 = indicators.get(f"sma_{self.settings.technical_sma_long_window}")
        ema_20 = indicators.get(f"ema_{self.settings.technical_ema_window}")
        macd = indicators.get("macd")
        macd_signal = indicators.get("macd_signal")
        last_close = indicators.get("last_close")
        bollinger_upper = indicators.get("bollinger_upper")
        bollinger_lower = indicators.get("bollinger_lower")

        if isinstance(rsi, int | float):
            if rsi >= self.settings.technical_rsi_overbought:
                signals.append(f"RSI is elevated at {rsi:.2f}, suggesting overbought conditions.")
                score -= 1
            elif rsi <= self.settings.technical_rsi_oversold:
                signals.append(f"RSI is depressed at {rsi:.2f}, suggesting oversold conditions.")
                score += 1

        if isinstance(sma_50, int | float) and isinstance(sma_200, int | float):
            if sma_50 > sma_200:
                signals.append("The 50-day average is above the 200-day average, which is constructive.")
                score += 1
            else:
                signals.append("The 50-day average is below the 200-day average, which is weak.")
                score -= 1

        if isinstance(macd, int | float) and isinstance(macd_signal, int | float):
            if macd > macd_signal:
                signals.append("MACD is above its signal line, indicating positive momentum.")
                score += 1
            else:
                signals.append("MACD is below its signal line, indicating negative momentum.")
                score -= 1

        if isinstance(last_close, int | float) and isinstance(bollinger_upper, int | float) and last_close >= bollinger_upper:
            signals.append("Price is near the upper Bollinger band, so upside may be stretched.")
            score -= 1
        elif isinstance(last_close, int | float) and isinstance(bollinger_lower, int | float) and last_close <= bollinger_lower:
            signals.append("Price is near the lower Bollinger band, so downside may be extended.")
            score += 1

        if isinstance(ema_20, int | float) and isinstance(last_close, int | float):
            if last_close > ema_20:
                signals.append("The latest close is above the 20-day EMA.")
                score += 1
            else:
                signals.append("The latest close is below the 20-day EMA.")
                score -= 1

        if score > 1:
            trend = "Bullish"
        elif score < -1:
            trend = "Bearish"
        else:
            trend = "Neutral"

        rationale = " ".join(signals) if signals else f"Not enough momentum signals were available for {ticker}."
        return {"trend": trend, "rationale": rationale, "confidence": min(1.0, abs(score) / 4 if score else 0.25)}

    def summarize_risk(
        self,
        company: str,
        financial: dict[str, object],
        news: dict[str, object],
        rag: dict[str, object] | None = None,
        technical: dict[str, object] | None = None,
    ) -> dict[str, object]:
        risks: list[str] = []

        metrics = financial.get("metrics", {}) if isinstance(financial, dict) else {}
        trailing_pe = metrics.get("trailing_pe")
        revenue_growth = metrics.get("revenue_growth_percent")
        profit_margin = metrics.get("profit_margin_percent")

        if isinstance(trailing_pe, int | float) and trailing_pe > self.settings.risk_trailing_pe_threshold:
            risks.append(f"Valuation is elevated with a trailing PE near {trailing_pe:.2f}.")
        if isinstance(revenue_growth, int | float) and revenue_growth < 0:
            risks.append(f"Revenue growth is negative at {revenue_growth:.2f}%.")
        if isinstance(profit_margin, int | float) and profit_margin < 0:
            risks.append(f"Profit margin is negative at {profit_margin:.2f}%.")

        sentiment = news.get("sentiment", {}) if isinstance(news, dict) else {}
        if isinstance(sentiment, dict) and sentiment.get("label") == "Negative":
            risks.append("Recent headlines skew negative.")

        if isinstance(technical, dict):
            trend = technical.get("trend")
            if trend == "Bearish":
                risks.append("Technical trend is bearish.")
            elif trend == "Neutral":
                risks.append("Technical momentum is mixed.")

        for excerpt in (rag or {}).get("risk_mentions", [])[:3] if isinstance(rag, dict) else []:
            risks.append(str(excerpt))

        if not risks:
            risks = [f"No major structured risks were detected for {company} in the available data."]

        return {
            "risks": list(dict.fromkeys(risks))[: self.settings.max_risk_items],
            "note": "LangChain heuristics synthesized these risks from financial, news, technical, and RAG inputs.",
        }

    def forecast_outlook(self, ticker: str, summary: dict[str, object]) -> dict[str, object]:
        return_5d = summary.get("return_5d_percent")
        return_30d = summary.get("return_30d_percent")
        volatility = summary.get("annualized_volatility")
        moving_average_20 = summary.get("moving_average_20")
        moving_average_50 = summary.get("moving_average_50")
        last_price = summary.get("last_price")

        score = 0.0
        rationale_parts: list[str] = []

        if isinstance(return_5d, int | float):
            score += return_5d / 10
            rationale_parts.append(f"Five-day performance is {return_5d:.2f}%.")

        if isinstance(return_30d, int | float):
            score += return_30d / 15
            rationale_parts.append(f"Thirty-day performance is {return_30d:.2f}%.")

        if isinstance(moving_average_20, int | float) and isinstance(moving_average_50, int | float):
            if moving_average_20 > moving_average_50:
                score += 0.75
                rationale_parts.append("The 20-day average is above the 50-day average.")
            else:
                score -= 0.75
                rationale_parts.append("The 20-day average is below the 50-day average.")

        if isinstance(volatility, int | float):
            if volatility > self.settings.forecast_volatility_threshold:
                score -= 0.5
                rationale_parts.append("Volatility is high, which lowers confidence.")
            else:
                score += 0.25

        if score > 0.75:
            direction = "Bullish"
        elif score < -0.75:
            direction = "Bearish"
        else:
            direction = "Neutral"

        expected_return = round((return_30d or 0.0) * 0.7 + score * 3, 2) if isinstance(return_30d, int | float) else round(score * 3, 2)
        confidence = max(0.1, min(1.0, 0.35 + abs(score) / 3))
        expected_price = None
        if isinstance(last_price, int | float):
            expected_price = round(last_price * (1 + expected_return / 100), 2)

        return {
            "direction": direction,
            "expected_return_percent": expected_return,
            "expected_price": expected_price,
            "confidence": confidence,
            "rationale": " ".join(rationale_parts) or f"No strong short-horizon signal was found for {ticker}.",
        }

    def extract_relationships(self, company: str, ticker: str, rag: dict[str, object] | None = None) -> dict[str, object]:
        sources = (rag or {}).get("sources", []) if isinstance(rag, dict) else []
        if not sources:
            return {"relationships": [], "note": "No filing or transcript excerpts were available for relationship extraction."}

        relationship_keywords = {
            "partner": "partnership",
            "partnership": "partnership",
            "supplier": "supplier",
            "customer": "customer",
            "competitor": "competitor",
            "subsidiary": "subsidiary",
            "joint venture": "joint venture",
            "collaboration": "collaboration",
            "acquire": "acquisition",
            "distribution": "distribution",
        }

        relationships: list[dict[str, object]] = []
        for source in sources[: self.settings.max_relationship_sources]:
            if not isinstance(source, dict):
                continue
            text = str(source.get("text", ""))
            lower_text = text.lower()
            for keyword, relation in relationship_keywords.items():
                if keyword in lower_text:
                    relationships.append(
                        {
                            "relation": relation,
                            "target": self._extract_target_company(text, company),
                            "evidence": text[:240],
                            "source": source.get("source"),
                        }
                    )
                    break

        return {
            "relationships": relationships[: self.settings.max_relationships],
            "note": "LangChain heuristics extracted explicit relationship mentions from the available documents.",
        }

    def summarize_multi_agent(self, query: str, agent_outputs: dict[str, object]) -> str:
        self.final_report_prompt.invoke({"query": query}).to_string()

        financial = agent_outputs.get("financial", {})
        news = agent_outputs.get("news", {})
        sentiment = agent_outputs.get("sentiment", {})
        technical = agent_outputs.get("technical", {})
        risk = agent_outputs.get("risk", {})
        forecast = agent_outputs.get("forecast", {})
        rag = agent_outputs.get("rag", {})

        company = financial.get("company_name", "The company") if isinstance(financial, dict) else "The company"
        ticker = financial.get("ticker", "N/A") if isinstance(financial, dict) else "N/A"
        metrics = financial.get("metrics", {}) if isinstance(financial, dict) else {}

        lines = [
            "### Executive view",
            f"{company} ({ticker}) was analyzed using LangGraph across financial, news, sentiment, technical, RAG, risk, graph, and forecast nodes.",
            "",
            "### Financial picture",
            financial.get("summary", "No financial summary was produced.") if isinstance(financial, dict) else "No financial summary was produced.",
            f"- Revenue growth: {self._format_percent(metrics.get('revenue_growth_percent'))}",
            f"- Trailing PE: {self._format_number(metrics.get('trailing_pe'))}",
            f"- Market cap: {metrics.get('currency', 'USD')} {self._format_number(metrics.get('market_cap'))}",
            "",
            "### News and sentiment",
            f"- Sentiment: {sentiment.get('label', 'Neutral') if isinstance(sentiment, dict) else 'Neutral'}",
            f"- Headline count: {len(news.get('headlines', [])) if isinstance(news, dict) else 0}",
            "",
            "### Technical trend",
            f"- Trend: {technical.get('trend', 'N/A') if isinstance(technical, dict) else 'N/A'}",
            f"- Rationale: {technical.get('rationale', '') if isinstance(technical, dict) else ''}",
            "",
            "### Key risks",
        ]
        risks = risk.get("risks", []) if isinstance(risk, dict) else []
        lines.extend(f"- {item}" for item in risks[:5])
        lines.extend(
            [
                "",
                "### RAG findings",
                f"- Status: {rag.get('status', 'unknown') if isinstance(rag, dict) else 'unknown'}",
                f"- Answer: {rag.get('answer', 'No local documents were available.') if isinstance(rag, dict) else 'No local documents were available.'}",
                "",
                "### 30-day forecast",
                f"- Direction: {forecast.get('direction', 'N/A') if isinstance(forecast, dict) else 'N/A'}",
                f"- Expected return: {self._format_percent(forecast.get('expected_return_percent') if isinstance(forecast, dict) else None)}",
                f"- Confidence: {self._format_percent((forecast.get('confidence') * 100) if isinstance(forecast, dict) and isinstance(forecast.get('confidence'), int | float) else None)}",
                "",
                "### Bottom line",
                self._bottom_line(query, financial, sentiment, technical, forecast, rag),
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _headline_rationale(company: str, positive_hits: int, negative_hits: int, headlines: list[dict[str, object]]) -> str:
        if positive_hits > negative_hits:
            tone = "leans positive"
        elif negative_hits > positive_hits:
            tone = "leans negative"
        else:
            tone = "is balanced"
        sample = str(headlines[0].get("title", "the latest headlines")) if headlines else "recent headlines"
        return f"Headline tone for {company} {tone}; the sample headline was '{sample}'."

    @staticmethod
    def _extract_target_company(text: str, company: str) -> str:
        patterns = [
            r"(?:partner|supplier|customer|competitor|subsidiary|joint venture|distribution|collaboration|acquisition)\s+(?:with|of|for)?\s*([A-Z][A-Za-z0-9&\-. ]{2,})",
            r"([A-Z][A-Za-z0-9&\-. ]{2,})\s+(?:partner|supplier|customer|competitor|subsidiary|joint venture)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                candidate = match.group(1).strip(" ,.;:-")
                if candidate and candidate.lower() != company.lower():
                    return candidate
        return company

    @staticmethod
    def _format_percent(value: object) -> str:
        return f"{value:.2f}%" if isinstance(value, int | float) else "N/A"

    @staticmethod
    def _format_number(value: object) -> str:
        return f"{value:,.2f}" if isinstance(value, int | float) else "N/A"

    @staticmethod
    def _bottom_line(
        query: str,
        financial: dict[str, object],
        sentiment: dict[str, object],
        technical: dict[str, object],
        forecast: dict[str, object],
        rag: dict[str, object],
    ) -> str:
        company = financial.get("company_name", "the company") if isinstance(financial, dict) else "the company"
        sentiment_label = sentiment.get("label", "Neutral") if isinstance(sentiment, dict) else "Neutral"
        trend = technical.get("trend", "Neutral") if isinstance(technical, dict) else "Neutral"
        direction = forecast.get("direction", "Neutral") if isinstance(forecast, dict) else "Neutral"
        rag_status = rag.get("status", "empty") if isinstance(rag, dict) else "empty"
        return (
            f"For {company}, the combined LangGraph workflow indicates {sentiment_label.lower()} news tone, "
            f"{trend.lower()} technical momentum, and a {direction.lower()} short-term outlook. "
            f"Use the documented risks and {rag_status} RAG context alongside your own diligence before making a decision about {query}."
        )