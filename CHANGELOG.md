# Changelog

## Adaptation work

This repository is an adaptation of `chulparyuvraj/rag-document-qa`. The upstream MIT licence and copyright notice are retained.

Changes made in this working copy:

- Corrected the malformed and incorrectly scoped RAG `query()` implementation.
- Added explicit RAG readiness checks and a reusable retriever rebuild path.
- Added incremental document ingestion that updates both FAISS and BM25 and persists the FAISS index.
- Corrected FastAPI health/readiness checks to use the pipeline's actual state.
- Hardened PDF upload handling with extension and PDF-signature checks, a configurable size limit, generated safe filenames, and ingestion error handling.
- Restricted default CORS origins to local development frontends; configurable through `CORS_ORIGINS`.
- Removed unsupported benchmark and fine-tuning performance claims from project documentation.
- Clarified that QLoRA training must be run and evaluated separately; no trained adapter is included.
- Added an explicit `langchain-text-splitters` dependency and modernized the test import for `Document`.

Validation performed in this working session:

- Python bytecode compilation passed for `src/`, `scripts/`, and `tests/`.
- Full pytest execution could not be completed because the current runtime does not have the project's LangChain dependencies installed (`ModuleNotFoundError: langchain`).

Further validation still required on a machine with the pinned dependencies, sample technical PDFs, and suitable model hardware: run unit tests, build a FAISS index, start the API, exercise all endpoints, and perform a real QLoRA training/evaluation run before making those claims in a resume.
