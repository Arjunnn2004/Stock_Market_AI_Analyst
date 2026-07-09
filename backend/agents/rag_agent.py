from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from backend.core.config import get_settings
class RagAgent:
    def __init__(self, documents_dir: str | None = None) -> None:
        settings = get_settings()
        resolved_dir = documents_dir or settings.rag_documents_dir
        self.documents_dir = Path(resolved_dir) if resolved_dir else None
        self.settings = settings

    def analyze(self, ticker: str, query: str) -> dict[str, Any]:
        documents = self._load_documents(ticker)
        if not documents:
            return {
                "agent": "rag",
                "status": "empty",
                "answer": "No local filings or transcripts indexed yet.",
                "sources": [],
                "risk_mentions": [],
            }

        chunks = self._chunk_documents(documents, self.settings.rag_chunk_size)
        query_terms = {term for term in re.findall(r"[A-Za-z0-9]+", query.lower()) if len(term) > 2}
        if not query_terms:
            query_terms = {term for term in re.findall(r"[A-Za-z0-9]+", ticker.lower()) if term}
        ranked = sorted(
            chunks,
            key=lambda chunk: sum(chunk["text"].lower().count(term) for term in query_terms),
            reverse=True,
        )[: self.settings.rag_ranked_chunks]

        risk_mentions = [
            chunk["text"][: self.settings.rag_excerpt_chars].strip()
            for chunk in ranked
        ][:3]

        return {
            "agent": "rag",
            "status": "ready",
            "answer": "Retrieved relevant local filing/transcript excerpts.",
            "sources": ranked,
            "risk_mentions": risk_mentions,
        }

    def _load_documents(self, ticker: str) -> list[dict[str, str]]:
        if not self.documents_dir or not self.documents_dir.exists():
            return []

        docs: list[dict[str, str]] = []
        patterns = [f"{ticker.upper()}*.txt", f"{ticker.lower()}*.txt", "*.txt"]
        seen: set[Path] = set()
        for pattern in patterns:
            for path in self.documents_dir.glob(pattern):
                if path in seen or not path.is_file():
                    continue
                seen.add(path)
                docs.append({"source": str(path), "text": path.read_text(encoding="utf-8", errors="ignore")})
        return docs

    def _chunk_documents(self, documents: list[dict[str, str]], size: int) -> list[dict[str, str]]:
        chunks: list[dict[str, str]] = []
        for doc in documents:
            text = " ".join(doc["text"].split())
            for start in range(0, len(text), size):
                chunk = text[start : start + size]
                if chunk:
                    chunks.append({"source": doc["source"], "text": chunk})
        return chunks

