# Build and package the ColBERTv2 Wiki17 search service.
FROM python:3.10-slim AS base

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies required by PyTorch / Faiss / tokenizers.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY pyproject.toml LICENSE ./
COPY app ./app
COPY dspy_adapter ./dspy_adapter
COPY scripts ./scripts

RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install 'faiss-cpu>=1.7.4' \
    && pip install .

# Optional build arguments to control dataset download and indexing.
ARG DATASET_NAME="colbert-ir/wikipedia-english-abstracts-2017"
ARG DEFAULT_K=10

# Materialise the corpus and build the ColBERT index under /workspace/data.
RUN python scripts/download_wiki17_abstracts.py --dataset "$DATASET_NAME" --output-dir data/raw/wiki17 \
    && python scripts/build_index.py --collection data/raw/wiki17/collection.tsv --index-root data/index --index-name wiki17_abstracts

ENV COLBERT_SERVICE_INDEX_ROOT=/workspace/data/index \
    COLBERT_SERVICE_INDEX_NAME=wiki17_abstracts \
    COLBERT_SERVICE_METADATA_PATH=/workspace/data/raw/wiki17/documents.jsonl \
    COLBERT_SERVICE_DEFAULT_K=$DEFAULT_K

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
