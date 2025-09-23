"""FastAPI application exposing ColBERT semantic search."""

from __future__ import annotations

import logging
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .state import get_state


LOGGER = logging.getLogger("colbert_api")


class SearchHit(BaseModel):
    id: str
    text: str
    score: float


class SearchRequest(BaseModel):
    query: str = Field(..., description="Free-form natural language query.")
    k: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Number of results to return (defaults to COLBERT_SERVICE_DEFAULT_K).",
    )


class SearchResponse(BaseModel):
    results: List[SearchHit]


app = FastAPI(title="ColBERTv2 Wiki17 Search", version="0.1.0")


@app.on_event("startup")
async def startup_event() -> None:
    # Prime the service state so that the first request does not incur load time.
    LOGGER.info("Initialising ColBERT service state")
    try:
        get_state()
    except Exception:
        LOGGER.exception("Failed to initialise ColBERT service state during startup")
        raise


@app.get("/health")
async def healthcheck() -> dict:
    state = get_state()
    return {
        "status": "ok",
        "index_name": state.settings.index_name,
        "documents": len(state.documents),
    }


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    state = get_state()
    k = request.k or state.settings.default_k
    hits = state.search(request.query, k=k)
    return SearchResponse(results=hits)
