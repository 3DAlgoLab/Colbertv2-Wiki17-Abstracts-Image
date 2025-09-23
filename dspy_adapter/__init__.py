"""DSPy integration helpers for the ColBERT service."""

from .colbert import ColBERTv2Client, SearchResult, build_dspy_retriever

__all__ = ["ColBERTv2Client", "SearchResult", "build_dspy_retriever"]
