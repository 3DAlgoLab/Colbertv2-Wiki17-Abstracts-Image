#!/usr/bin/env python3
"""Create a ColBERT index for the Wikipedia 2017 abstracts corpus.

The script expects a ``collection.tsv`` file with one document per line and
leverages the official ColBERT indexer.  Run ``download_wiki17_abstracts.py``
first to materialise the corpus locally.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from colbert import Indexer
from colbert.infra import ColBERTConfig, Run, RunConfig


LOGGER = logging.getLogger("build_index")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection",
        type=Path,
        default=Path("data/raw/wiki17/collection.tsv"),
        help="Path to the collection file (one document per line).",
    )
    parser.add_argument(
        "--checkpoint",
        default="colbert-ir/colbertv2.0",
        help="Hugging Face checkpoint to use for indexing.",
    )
    parser.add_argument(
        "--index-root",
        type=Path,
        default=Path("data/index"),
        help="Directory where the ColBERT index will be stored.",
    )
    parser.add_argument(
        "--index-name",
        default="wiki17_abstracts",
        help="Name assigned to the ColBERT index.",
    )
    parser.add_argument(
        "--doc-maxlen",
        type=int,
        default=180,
        help="Document max length for ColBERT (number of tokens).",
    )
    parser.add_argument(
        "--nbits",
        type=int,
        default=2,
        help="Number of bits per dimension for residual compression (see ColBERT docs).",
    )
    parser.add_argument(
        "--nranks",
        type=int,
        default=1,
        help="Number of GPUs to distribute indexing across.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    LOGGER.info("Starting indexing: collection=%s, checkpoint=%s", args.collection, args.checkpoint)

    if not args.collection.exists():
        raise FileNotFoundError(f"Collection file not found: {args.collection}")

    config = ColBERTConfig(doc_maxlen=args.doc_maxlen, nbits=args.nbits)

    run_config = RunConfig(nranks=args.nranks, experiment=args.index_name, root=str(args.index_root))

    with Run().context(run_config):
        indexer = Indexer(checkpoint=args.checkpoint, config=config)
        indexer.index(
            name=args.index_name,
            collection=str(args.collection),
        )

    LOGGER.info("Index build complete: %s", args.index_root / args.index_name)


if __name__ == "__main__":
    main()
