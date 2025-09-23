"""Lazy loading and shared state for the ColBERT FastAPI service."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from colbert import Searcher

from .config import Settings


LOGGER = logging.getLogger(__name__)


class DocumentStore:
    """In-memory view over the metadata file produced during download."""

    def __init__(self, metadata_path: Path):
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self._ids: List[str] = []
        self._texts: List[str] = []

        with metadata_path.open("r", encoding="utf-8") as stream:
            for idx, line in enumerate(stream):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {idx + 1} of {metadata_path}") from exc
                doc_id = record.get("id", idx)
                text = record.get("text")
                if text is None:
                    raise ValueError(f"Missing 'text' field on line {idx + 1} of {metadata_path}")
                self._ids.append(str(doc_id))
                self._texts.append(text)

    def __len__(self) -> int:
        return len(self._ids)

    def get(self, doc_idx: int) -> Tuple[str, str]:
        try:
            return self._ids[doc_idx], self._texts[doc_idx]
        except IndexError as exc:
            raise IndexError(f"Document index {doc_idx} out of bounds for collection of size {len(self._ids)}") from exc


class ServiceState:
    """Container for heavy objects that should be reused across requests."""

    def __init__(self, settings: Settings):
        self.settings = settings
        LOGGER.info("Loading ColBERT searcher from %s", settings.index_dir)
        if not settings.index_dir.exists():
            raise FileNotFoundError(f"ColBERT index directory not found: {settings.index_dir}")

        collection_path = settings.inferred_collection_path
        if not collection_path.exists():
            LOGGER.warning("Collection file missing at %s; ColBERT Searcher will rely on index metadata only.", collection_path)
            collection = None
        else:
            collection = str(collection_path)

        self.searcher = Searcher(index=str(settings.index_dir), checkpoint=settings.checkpoint, collection=collection)
        self.documents = DocumentStore(settings.metadata_path)

    def search(self, query: str, k: int) -> List[Dict[str, str]]:
        results = self.searcher.search(query, k=k)
        docids = results["docids"][0]
        scores = results["scores"][0]

        hits = []
        for raw_doc_id, score in zip(docids, scores):
            # Some ColBERT exports use 1-based docids.  Fallback to 0-based if needed.
            doc_idx = int(raw_doc_id)
            if doc_idx >= len(self.documents) and doc_idx > 0:
                doc_idx -= 1
            doc_id, text = self.documents.get(doc_idx)
            hits.append({"id": doc_id, "text": text, "score": float(score)})

        return hits


_state: ServiceState | None = None


def get_state() -> ServiceState:
    global _state
    if _state is None:
        settings = Settings()
        _state = ServiceState(settings)
    return _state
