#!/usr/bin/env python3
"""Fix the collection.tsv file format for ColBERT indexing.

ColBERT expects: id<TAB>text
Current format: text<TAB>title

This script converts the format and adds integer IDs.
"""

import sys
from pathlib import Path


def fix_collection_format(input_file: Path, output_file: Path) -> None:
    """Convert collection from text<TAB>title to id<TAB>text format."""
    with (
        open(input_file, "r", encoding="utf-8") as infile,
        open(output_file, "w", encoding="utf-8") as outfile,
    ):
        # Skip header line
        header = infile.readline()
        if not header.startswith("text"):
            print(f"Warning: Unexpected header: {header.strip()}")

        # Process each line and add ID
        doc_id = 0
        for line in infile:
            line = line.strip()
            if line:
                # Split by tab, take first column as text
                parts = line.split("\t")
                text = parts[0] if parts else line

                # Write in ColBERT format: id<TAB>text
                outfile.write(f"{doc_id}\t{text}\n")
                doc_id += 1

        print(f"Processed {doc_id} documents")


def main():
    input_file = Path("data/raw/wiki17/collection.tsv")
    output_file = Path("data/raw/wiki17/collection_fixed.tsv")

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    print(f"Converting {input_file} to {output_file}")
    fix_collection_format(input_file, output_file)
    print("Conversion complete!")


if __name__ == "__main__":
    main()
