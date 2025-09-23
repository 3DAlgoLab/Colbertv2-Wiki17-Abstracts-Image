#!/usr/bin/env python3
"""Upload the built index artefacts to the Hugging Face Hub."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from huggingface_hub import HfApi


LOGGER = logging.getLogger("push_to_hf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id", help="Destination repository in the form namespace/name")
    parser.add_argument(
        "--repo-type",
        default="dataset",
        choices=["dataset", "model"],
        help="Type of Hugging Face repository to target.",
    )
    parser.add_argument(
        "--index-root",
        type=Path,
        default=Path("data/index"),
        help="Root directory containing ColBERT indices.",
    )
    parser.add_argument(
        "--index-name",
        default="wiki17_abstracts",
        help="Name of the index directory to upload (relative to index root).",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("data/raw/wiki17/documents.jsonl"),
        help="Metadata file to upload alongside the index.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload ColBERT index",
        help="Commit message to use for the upload.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    index_dir = args.index_root / args.index_name
    if not index_dir.exists():
        raise FileNotFoundError(f"Index directory not found: {index_dir}")
    if not args.metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {args.metadata_path}")

    LOGGER.info("Uploading %s to %s (repo_type=%s)", index_dir, args.repo_id, args.repo_type)
    api = HfApi()
    api.upload_folder(
        folder_path=str(index_dir),
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        path_in_repo=f"index/{args.index_name}",
        commit_message=args.commit_message,
        allow_patterns=["*.pt", "*.json", "*.info", "*.npz", "*.hdf5"],
    )
    api.upload_file(
        path_or_fileobj=str(args.metadata_path),
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        path_in_repo=f"metadata/{args.metadata_path.name}",
        commit_message=args.commit_message,
    )
    LOGGER.info("Upload complete")


if __name__ == "__main__":
    main()
