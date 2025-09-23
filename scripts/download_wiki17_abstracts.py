#!/usr/bin/env python3
"""Prepare the Wikipedia 2017 abstracts corpus for ColBERT indexing.

By default the script downloads the pre-packaged archive published by the Baleen
project (Stanford Future Data) and materialises two artifacts compatible with
ColBERT:

* ``documents.jsonl`` containing ``{"id", "text"}`` entries.
* ``collection.tsv`` mirroring the format expected by the ColBERT indexer.

The script can also fall back to Hugging Face Datasets if required.  Pass
``--source huggingface`` to enable that mode.
"""

from __future__ import annotations

import argparse
import json
import logging
import tarfile
from pathlib import Path
from typing import Iterator

import requests
from datasets import load_dataset


LOGGER = logging.getLogger("download_wiki17")
DEFAULT_ARCHIVE_URL = "https://downloads.cs.stanford.edu/nlp/data/colbert/baleen/wiki.abstracts.2017.tar.gz"
DEFAULT_DATASET = "colbert-ir/wikipedia-english-abstracts-2017"


def iter_rows_from_huggingface(dataset, id_column: str, text_column: str) -> Iterator[dict]:
    for idx, sample in enumerate(dataset):
        doc_id = sample.get(id_column, idx)
        text = sample.get(text_column)
        if text is None:
            raise KeyError(f"Missing text column '{text_column}' in row {idx}")
        yield {"id": doc_id, "text": text}


def download_archive(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".tmp")

    LOGGER.info("Downloading archive from %s", url)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as stream:
            for chunk in response.iter_content(chunk_size=1 << 20):  # 1 MiB
                if chunk:
                    stream.write(chunk)

    tmp_path.replace(dest)
    LOGGER.info("Saved archive to %s", dest)


def iter_rows_from_archive(archive_path: Path) -> Iterator[dict]:
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    LOGGER.info("Loading documents from archive %s", archive_path)
    with tarfile.open(archive_path, "r:gz") as tar:
        tsv_member = next((m for m in tar.getmembers() if m.name.endswith("collection.tsv")), None)
        if tsv_member is None:
            raise FileNotFoundError("collection.tsv not found inside archive")

        with tar.extractfile(tsv_member) as handle:
            if handle is None:
                raise RuntimeError("Failed to extract collection.tsv from archive")
            for idx, raw_line in enumerate(handle):
                line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
                if not line:
                    continue
                if "\t" in line:
                    doc_id, text = line.split("\t", 1)
                else:
                    doc_id, text = str(idx), line
                yield {"id": doc_id, "text": text}


def write_outputs(rows: Iterator[dict], jsonl_path: Path, text_path: Path) -> int:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with jsonl_path.open("w", encoding="utf-8") as jsonl_stream, text_path.open(
        "w", encoding="utf-8"
    ) as text_stream:
        for count, row in enumerate(rows, start=1):
            jsonl_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            text_stream.write(row["text"].replace("\n", " ").strip() + "\n")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=["archive", "huggingface"],
        default="archive",
        help="Where to pull the dataset from (Baleen archive or Hugging Face).",
    )
    parser.add_argument(
        "--archive-url",
        default=DEFAULT_ARCHIVE_URL,
        help="URL to the Baleen tar.gz archive.",
    )
    parser.add_argument(
        "--archive-path",
        type=Path,
        default=None,
        help="Optional local path for the downloaded archive (defaults to output_dir/<filename>).",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Hugging Face dataset repository to download (when source=huggingface).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to read (when source=huggingface).",
    )
    parser.add_argument(
        "--id-column",
        default="id",
        help="Column containing unique identifiers (Hugging Face mode).",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Column containing the document text (Hugging Face mode).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/wiki17"),
        help="Directory where outputs will be written.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    jsonl_path = args.output_dir / "documents.jsonl"
    text_path = args.output_dir / "collection.tsv"

    if args.source == "huggingface":
        LOGGER.info(
            "Loading dataset %s (split=%s) from Hugging Face",
            args.dataset,
            args.split,
        )
        dataset = load_dataset(args.dataset, split=args.split, streaming=False)
        rows = iter_rows_from_huggingface(dataset, args.id_column, args.text_column)
    else:
        archive_path = args.archive_path
        if archive_path is None:
            archive_path = args.output_dir / Path(args.archive_url).name
        archive_path.parent.mkdir(parents=True, exist_ok=True)

        if not archive_path.exists():
            download_archive(args.archive_url, archive_path)
        else:
            LOGGER.info("Archive already present at %s; skipping download", archive_path)

        rows = iter_rows_from_archive(archive_path)

    LOGGER.info("Writing outputs to %s", args.output_dir)
    count = write_outputs(rows, jsonl_path=jsonl_path, text_path=text_path)

    LOGGER.info("Preparation complete: %d documents", count)


if __name__ == "__main__":
    main()
