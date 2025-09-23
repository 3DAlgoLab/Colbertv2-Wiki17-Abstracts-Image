"""Configuration handling for the ColBERT FastAPI service."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    index_root: Path = Field(default=Path("data/index"), env="COLBERT_SERVICE_INDEX_ROOT")
    index_name: str = Field(default="wiki17_abstracts", env="COLBERT_SERVICE_INDEX_NAME")
    checkpoint: str = Field(default="colbert-ir/colbertv2.0", env="COLBERT_SERVICE_CHECKPOINT")
    metadata_path: Path = Field(
        default=Path("data/raw/wiki17/documents.jsonl"), env="COLBERT_SERVICE_METADATA_PATH"
    )
    collection_path: Optional[Path] = Field(default=None, env="COLBERT_SERVICE_COLLECTION_PATH")
    default_k: int = Field(default=10, env="COLBERT_SERVICE_DEFAULT_K")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def index_dir(self) -> Path:
        return self.index_root / self.index_name

    @property
    def inferred_collection_path(self) -> Path:
        if self.collection_path is not None:
            return self.collection_path
        return self.metadata_path.parent / "collection.tsv"
