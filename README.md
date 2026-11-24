# LLM Fine-Tuning & RAG-Based Technical Documentation Assistant

A document-grounded question-answering application for technical manuals, engineering references, and other domain-specific PDF documents.

> **Project provenance:** This repository is an adapted and extended implementation based on [`chulparyuvraj/rag-document-qa`](https://github.com/chulparyuvraj/rag-document-qa). The upstream MIT licence and copyright notice are retained in `LICENSE`. See [Acknowledgements](#acknowledgements).

## Current implementation

- PDF text extraction with PyMuPDF.
- Recursive text chunking with source and page metadata.
- Dense semantic retrieval using sentence-transformer embeddings and FAISS.
- Sparse keyword retrieval using BM25.
- Reciprocal Rank Fusion (RRF) to combine dense and sparse results.
- Context-grounded answer generation using a configurable Hugging Face causal language model.
- FastAPI endpoints for health checks, question answering, PDF upload, and document listing.
- A QLoRA fine-tuning script and dataset preparation utilities (training has to be run separately; no trained adapter is bundled).

## Architecture

```text
Technical PDF documents
          |
          v
  PyMuPDF extraction
          |
          v
  Recursive chunking + provenance metadata
          |
          +------------------------+
          |                        |
          v                        v
 Sentence embeddings             BM25 index
          |                        |
          v                        |
     FAISS index                  |
          |                        |
          +-----------+------------+
                      |
                      v
              Reciprocal Rank Fusion
                      |
                      v
                Top-k context
                      |
                      v
          Grounded prompt + Hugging Face LLM
                      |
                      v
                  FastAPI
```

## Repository layout

```text
src/
  ingestion/       PDF extraction and chunking
  retrieval/       Embeddings, FAISS, BM25, hybrid retrieval
  pipeline/        Prompt construction and RAG orchestration
  finetuning/      Dataset preparation and QLoRA training
  api/             FastAPI routes and Pydantic schemas
scripts/           Indexing and evaluation commands
tests/             Unit tests
notebooks/         Development and demonstration notebooks
deploy/            Container and Azure deployment examples
```

## Setup

Use Python 3.10 or 3.11 for the pinned dependency set. A CUDA-capable GPU is recommended for loading the default 7B model and for QLoRA training.

```bash
git clone <your-repository-url>
cd <your-repository-folder>
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set model configuration in `.env` as needed. Do not commit `.env`, access tokens, private documents, model weights, or generated indexes.

## Add documents and build the index

Place PDFs that you are permitted to use in `data/papers/`:

```bash
python scripts/ingest_docs.py --pdf_dir data/papers --index_path indexes/faiss
```

The ingestion command requires at least one PDF containing extractable text. The FAISS index is stored locally and is excluded from Git.

## Run the API

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation.

### Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Reports API and pipeline readiness |
| POST | `/query` | Answers a question using retrieved document context |
| POST | `/upload` | Validates, extracts, chunks, and indexes a PDF |
| GET | `/documents` | Lists indexed source documents |

Example query:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What does the selected manual say about the system architecture?","include_sources":true}'
```

## Fine-tuning status and workflow

The repository includes a QLoRA training implementation and dataset preparation code. It does **not** include a trained adapter or a verified training result. To run fine-tuning, first prepare a domain-specific question-answer dataset in the format expected by `src/finetuning/dataset_prep.py`, then follow the CLI arguments in `src/finetuning/train_qlora.py`.

Only describe fine-tuning as completed after a training run finishes successfully, the adapter is saved, and its use is verified. Keep generated weights out of Git unless there is a specific reason and sufficient storage.

## Evaluation

The evaluation script expects a labelled JSON test set. Run it only after building the index and preparing test examples with correct source filenames. Report measured results from your own run; this repository does not currently publish verified benchmark results.

```bash
python scripts/evaluate.py --test_file data/test_qa.json
```

## Configuration

See `.env.example` for the model, index, document-directory, and API settings. The upload limit defaults to 20 MB and can be changed with `MAX_UPLOAD_MB`. Set `CORS_ORIGINS` to a comma-separated list of trusted frontend origins before deployment.

## Security and deployment notes

- Use only documents you have permission to process.
- Keep secrets in environment variables; never commit `.env`.
- Restrict CORS origins for any non-local deployment.
- Put authentication, rate limiting, request logging controls, and storage retention policies in place before exposing the API publicly.
- FAISS persistence uses LangChain's local serialization. Load index files only from a trusted source.
- The included Azure deployment files are examples and do not mean this application is currently deployed.

## Acknowledgements

This project adapts code and design from [`chulparyuvraj/rag-document-qa`](https://github.com/chulparyuvraj/rag-document-qa), Copyright (c) 2026 Yuvraj Chulpar, under the MIT License. The original licence and copyright notice are preserved in `LICENSE`. Changes in this repository should be documented in its commit history.
