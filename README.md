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
- uv (recommended) for fast dependency management and package installation

## Local Setup

### Using uv (Recommended)

Set up the environment and install dependencies:

```bash
uv venv
source .venv/bin/activate
uv pip install --upgrade pip
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e .
```

### Traditional pip setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

Download the corpus (Baleen mirror) and build the index:

```bash
python scripts/download_wiki17_abstracts.py --source archive --output-dir data/raw/wiki17
python fix_collection_format.py  # Fix collection format for ColBERT compatibility
python scripts/build_index.py --collection data/raw/wiki17/collection_fixed.tsv --index-root data/index --index-name wiki17_abstracts
```

Or use the convenience script:

```bash
./build_index
```

Pass `--source huggingface` if you prefer to pull the corpus via Hugging Face Datasets instead of the Baleen archive.

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

- `--build-arg ARCHIVE_URL="https://example.com/wiki.abstracts.2017.tar.gz"` to point the build at an alternate mirror.
- `--build-arg DEFAULT_K=20` adjusts the server default for `k` when clients omit it.

To persist the index outside the container, you can mount a volume at `/workspace/data`.

## Publishing to Hugging Face

1. Mirror the Baleen archive once so future builds pull from the Hugging Face Hub instead of Stanford servers:

   ```bash
   pip install huggingface_hub
   python -m huggingface_hub login
   huggingface-cli upload <your-namespace>/<repo-name> data/raw/wiki17/wiki.abstracts.2017.tar.gz wiki.abstracts.2017.tar.gz --repo-type dataset
   ```

   Update `ARCHIVE_URL` (or set `COLBERT_ARCHIVE_URL` when running locally) to point at the new mirror, e.g. `https://huggingface.co/datasets/<your-namespace>/<repo-name>/resolve/main/wiki.abstracts.2017.tar.gz`.

2. Upload the ColBERT index and metadata with the provided helper:

   ```bash
   pip install .[deploy]
   python scripts/push_to_hf.py <your-namespace>/<repo-name> --repo-type dataset
   ```

3. For Spaces, point the Space to this repository and set the runtime to Docker; the
   provided `Dockerfile` works out-of-the-box.  Store the Hugging Face token and override
   `ARCHIVE_URL` as needed via Space secrets.

## Utilities

### Collection Format Fixer

The `fix_collection_format.py` utility converts the downloaded Wikipedia collection from the original `text<TAB>title` format to ColBERT's expected `id<TAB>text` format with sequential integer IDs. This fixes a common indexing error where ColBERT expects integer document IDs but finds string identifiers.

### Build Index Script

The `build_index` script provides a convenient one-command process that:
1. Cleans any existing index data
2. Runs the collection format fixer
3. Builds the ColBERT index with optimal settings

## Project Layout

```
app/                FastAPI application (entry point: app:app)
dspy_adapter/      DSPy client helper
scripts/           Dataset download and index build utilities
Dockerfile         Full pipeline container definition
pyproject.toml     Python packaging metadata
fix_collection_format.py  Utility to fix collection format for ColBERT
build_index       Convenience script for index building
```

## Recent Updates

- **Bug Fix**: Fixed collection format compatibility issue with ColBERT indexing
- **Utility Added**: Added `fix_collection_format.py` to handle format conversion
- **Build Script**: Added `build_index` convenience script for streamlined workflow
- **UV Support**: Added support for uv package manager for faster dependency management

## Roadmap / Next Steps

- Add tests against the service layer with a small synthetic index.
- Automate publishes to the Hugging Face Hub via GitHub Actions.
- Parameterise index refresh cadence (cron-based rebuild).

## License

MIT License.  See [LICENSE](LICENSE).
