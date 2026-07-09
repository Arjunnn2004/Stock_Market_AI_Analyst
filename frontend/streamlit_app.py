from __future__ import annotations

import os

import requests
import streamlit as st


def _get_api_base_url() -> str:
    try:
        secret_value = st.secrets.get("API_BASE_URL")
        if secret_value:
            return str(secret_value)
    except Exception:
        pass

    env_value = os.getenv("API_BASE_URL")
    if env_value:
        return env_value

    return "http://127.0.0.1:8000"


def _get_default_query() -> str:
    try:
        secret_value = st.secrets.get("DEFAULT_RESEARCH_QUERY")
        if secret_value:
            return str(secret_value)
    except Exception:
        pass

    return os.getenv("DEFAULT_RESEARCH_QUERY", "")


API_BASE_URL = _get_api_base_url()
API_URL = f"{API_BASE_URL.rstrip('/')}/multi-agent/analyze"


def format_large_number(value: object, currency: object = "USD") -> str:
    if not isinstance(value, int | float):
        return "N/A"
    prefix = str(currency or "USD")
    abs_value = abs(value)
    if abs_value >= 1_000_000_000_000:
        return f"{prefix} {value / 1_000_000_000_000:.2f}T"
    if abs_value >= 1_000_000_000:
        return f"{prefix} {value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"{prefix} {value / 1_000_000:.2f}M"
    return f"{prefix} {value:,.0f}"


def format_percent(value: object) -> str:
    return f"{value:.2f}%" if isinstance(value, int | float) else "N/A"


def format_ratio(value: object) -> str:
    return f"{value:.2f}" if isinstance(value, int | float) else "N/A"


st.set_page_config(page_title="Stock AI Analyst", page_icon=":chart_with_upwards_trend:", layout="wide")

st.title("Stock AI Analyst")

query = st.text_input("Research question", value=_get_default_query())
analyze = st.button("Analyze", type="primary")

if analyze and query.strip():
    with st.spinner("Running financial, news, sentiment, technical, RAG, risk, graph, and forecast agents..."):
        try:
            response = requests.get(API_URL, params={"query": query}, timeout=120)
        except requests.exceptions.RequestException as exc:
            st.error(
                "The backend API is not reachable. Start the FastAPI server first with `uvicorn backend.api.main:app --reload` or set API_BASE_URL to the correct host."
            )
            st.exception(exc)
            st.stop()

    if not response.ok:
        try:
            detail = response.json().get("detail", "Analysis failed.")
        except ValueError:
            detail = response.text or "Analysis failed."
        st.error(detail)
        st.stop()

    data = response.json()
    agents = data["agents"]
    financial = agents["financial"]
    metrics = financial["metrics"]

    st.subheader(f"{data['company_name']} ({data['ticker']})")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Revenue Growth", format_percent(metrics.get("revenue_growth_percent")))
    col2.metric("PE Ratio", format_ratio(metrics.get("trailing_pe")))
    col3.metric("Market Cap", format_large_number(metrics.get("market_cap"), metrics.get("currency")))
    col4.metric("Technical Trend", agents["technical"].get("trend", "N/A"))

    st.divider()
    st.subheader("Final Analyst")
    st.markdown(data["final_analysis"])

    tabs = st.tabs(["Financial", "News", "Sentiment", "Technical", "Risks", "RAG", "Graph", "Forecast"])

    with tabs[0]:
        st.write(financial["summary"])
        st.json(metrics)

    with tabs[1]:
        headlines = agents["news"].get("headlines", [])
        if headlines:
            for item in headlines:
                title = item.get("title", "Untitled")
                url = item.get("url")
                publisher = item.get("publisher") or "Unknown"
                if url:
                    st.markdown(f"- [{title}]({url})  \n  `{publisher}`")
                else:
                    st.markdown(f"- {title}  \n  `{publisher}`")
        else:
            st.info("No recent yfinance headlines returned.")
        st.json(agents["news"].get("sentiment", {}))

    with tabs[2]:
        sentiment = agents["sentiment"]
        st.metric("Sentiment", sentiment.get("label", "N/A"), sentiment.get("score", 0))
        st.json(sentiment)

    with tabs[3]:
        technical = agents["technical"]
        st.metric("Trend", technical.get("trend", "N/A"))
        st.json(technical.get("indicators", technical))

    with tabs[4]:
        for risk in agents["risk"].get("risks", []):
            st.markdown(f"- {risk}")

    with tabs[5]:
        rag = agents["rag"]
        st.write(rag.get("answer"))
        sources = rag.get("sources", [])
        if sources:
            for source in sources:
                st.caption(source.get("source"))
                st.write(source.get("text"))
        else:
            st.info("Drop .txt filings or transcripts into backend/data/filings to activate local RAG retrieval.")

    with tabs[6]:
        st.json(agents["graph"])

    with tabs[7]:
        st.json(agents["forecast"])
