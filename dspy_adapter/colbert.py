"""Client helpers for integrating the FastAPI ColBERT service with DSPy."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import requests

LOGGER = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Lightweight search result wrapper."""

    id: str
    text: str
    score: float


class ColBERTv2Client:
    """HTTP client for the FastAPI ColBERT service."""

    def __init__(self, url: str, timeout: float = 10.0, session: Optional[requests.Session] = None) -> None:
        self.base_url = url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def search(self, query: str, k: int) -> List[SearchResult]:
        response = self.session.post(
            f"{self.base_url}/search",
            json={"query": query, "k": k},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results", [])
        return [SearchResult(id=str(item["id"]), text=item["text"], score=float(item["score"])) for item in results]


def build_dspy_retriever(url: str, default_k: int = 10, timeout: float = 10.0):
    """Return a DSPy-compatible retrieval module bound to the service."""

    try:
        import dspy
    except ImportError as exc:  # pragma: no cover - guard for optional dependency
        raise RuntimeError("dspy must be installed to build the retrieval adapter") from exc

    retrieve_base = getattr(dspy, "Retrieve", None)
    retrieve_result_cls = getattr(dspy, "RetrieveResult", None)
    if retrieve_base is None or retrieve_result_cls is None:
        raise RuntimeError("The installed dspy package does not expose Retrieve/RetrieveResult")

    client = ColBERTv2Client(url=url, timeout=timeout)

    class _RemoteColBERT(retrieve_base):  # type: ignore[misc]
        def __init__(self):
            super().__init__(k=default_k)

        def forward(self, query: str, k: Optional[int] = None):  # type: ignore[override]
            top_k = k or self.k
            try:
                results = client.search(query=query, k=top_k)
            except requests.RequestException as exc:  # pragma: no cover - passthrough for runtime errors
                LOGGER.error("ColBERT remote search failed: %s", exc)
                raise

            return [
                retrieve_result_cls(
                    passage=item.text,
                    score=item.score,
                    metadata={"id": item.id},
                )
                for item in results
            ]

    return _RemoteColBERT()


__all__ = ["ColBERTv2Client", "SearchResult", "build_dspy_retriever"]
