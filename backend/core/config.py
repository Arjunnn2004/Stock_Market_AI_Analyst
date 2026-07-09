from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    analysis_model: str = ""
    news_api_key: str | None = None
    rag_documents_dir: str = ""
    max_headlines_considered: int = 8
    max_risk_items: int = 8
    max_relationship_sources: int = 5
    max_relationships: int = 5
    technical_rsi_period: int = 14
    technical_sma_short_window: int = 50
    technical_sma_long_window: int = 200
    technical_ema_window: int = 20
    technical_macd_fast_window: int = 12
    technical_macd_slow_window: int = 26
    technical_macd_signal_window: int = 9
    technical_bollinger_window: int = 20
    technical_bollinger_stddev: float = 2.0
    technical_rsi_overbought: float = 70.0
    technical_rsi_oversold: float = 30.0
    technical_history_period: str = "1y"
    technical_history_interval: str = "1d"
    forecast_history_period: str = "6mo"
    forecast_history_interval: str = "1d"
    forecast_min_history_points: int = 30
    forecast_return_5d_window: int = 5
    forecast_return_30d_window: int = 30
    forecast_short_ma_window: int = 20
    forecast_long_ma_window: int = 50
    forecast_recent_high_low_window: int = 30
    forecast_volatility_threshold: float = 0.35
    risk_trailing_pe_threshold: float = 35.0
    rag_chunk_size: int = 1200
    rag_ranked_chunks: int = 5
    rag_excerpt_chars: int = 280

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
