# ColBERTv2 Wiki17 Abstracts Service

This repository contains tooling to reproduce the ColBERTv2 retrieval service that ships
with DSPy.  It downloads the Wikipedia 2017 abstract corpus, builds a ColBERTv2 index,
serves it via FastAPI, and exposes a tiny DSPy adapter so existing code can switch to the
self-hosted endpoint with minimal changes.

## Features

- Scripts to download the Wikipedia 2017 abstracts dataset and export it in ColBERT format.
- Turn-key index building pipeline targeting the `colbert-ir/colbertv2.0` checkpoint.
- FastAPI service with a single `/search` endpoint returning `{id, text, score}` payloads.
- DSPy helper (`dspy_adapter.build_dspy_retriever`) for drop-in replacement of
  `dspy.ColBERTv2(url=...)`.
- Dockerfile that performs the full build (dataset download + indexing) and bakes the
  service into a deployable image.

## Prerequisites

- Python 3.10+
- PyTorch-compatible hardware (GPU optional; the provided Dockerfile defaults to CPU)
- Hugging Face account (optional) if you want to publish the resulting index/image

## Local Setup

Create a virtual environment and install the package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

Download the corpus and build the index:

```bash
python scripts/download_wiki17_abstracts.py --output-dir data/raw/wiki17
python scripts/build_index.py --collection data/raw/wiki17/collection.tsv --index-root data/index --index-name wiki17_abstracts
```

Launch the API (defaults to port 8000):

```bash
COLBERT_SERVICE_INDEX_ROOT=$(pwd)/data/index \
COLBERT_SERVICE_METADATA_PATH=$(pwd)/data/raw/wiki17/documents.jsonl \
uvicorn app:app --reload
```

Query the service:

```bash
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "barack obama", "k": 5}'
```

## DSPy Integration

```python
import dspy
from dspy_adapter import build_dspy_retriever

colbert_rm = build_dspy_retriever(url="http://localhost:8000", default_k=10)
dspy.settings.configure(rm=colbert_rm)
```

The adapter transparently calls the FastAPI service and returns `dspy.RetrieveResult`
instances, enabling existing DSPy pipelines to stay unchanged.

## Docker

Build the container (this step downloads the dataset and builds the index inside the
image; it can take 1–2 hours and ~30GB of disk):

```bash
docker build -t colbertv2-wiki17 .
```

Run the service:

```bash
docker run --rm -p 8000:8000 colbertv2-wiki17
```

### Customising the Build

- `--build-arg DATASET_NAME="namespace/dataset"` switches to a different Hugging Face corpus.
- `--build-arg DEFAULT_K=20` adjusts the server default for `k` when clients omit it.

To persist the index outside the container, you can mount a volume at `/workspace/data`.

## Publishing to Hugging Face

1. Create a private or public repository on the Hugging Face hub (dataset or space).
2. Use `huggingface_hub` with a write token to upload artifacts:

   ```bash
   pip install huggingface_hub
   python -m huggingface_hub login
   huggingface-cli upload <your-namespace>/<repo-name> data/index/wiki17_abstracts ./data/index/wiki17_abstracts
   ```

3. For Spaces, point the Space to this repository and set the runtime to Docker; the
   provided `Dockerfile` works out-of-the-box.  Store any required HF tokens in the Space
   secrets and set `DATASET_NAME` if you prefer a hosted dataset variant.

## Project Layout

```
app/                FastAPI application (entry point: app:app)
dspy_adapter/      DSPy client helper
scripts/           Dataset download and index build utilities
Dockerfile         Full pipeline container definition
pyproject.toml     Python packaging metadata
```

## Roadmap / Next Steps

- Add tests against the service layer with a small synthetic index.
- Automate publishes to the Hugging Face Hub via GitHub Actions.
- Parameterise index refresh cadence (cron-based rebuild).

## License

MIT License.  See [LICENSE](LICENSE).
