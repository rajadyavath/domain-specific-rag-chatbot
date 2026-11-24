"""
main.py — FastAPI REST API for the RAG Document QA System.

Endpoints:
  GET  /health          — liveness check
  POST /query           — ask a question over indexed papers
  POST /upload          — upload and index a new PDF
  GET  /documents       — list indexed documents

Run locally:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

Deploy on Azure App Service:
    See deploy/azure-deploy.sh
"""

import os
import time
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from dotenv import load_dotenv

from .schemas import QueryRequest, QueryResponse, SourceDocument, UploadResponse, HealthResponse
from ..pipeline.rag_chain import RAGPipeline

load_dotenv()

# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Document QA API",
    description="Domain-specific document question answering with hybrid retrieval and a language model",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",") if origin.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global Pipeline Instance ──────────────────────────────────────────────────
rag: RAGPipeline = None


@app.on_event("startup")
async def startup_event():
    """Initialize RAG pipeline on server start."""
    global rag
    logger.info("Initializing RAG pipeline...")
    rag = RAGPipeline(
        index_path=os.getenv("FAISS_INDEX_PATH", "indexes/faiss"),
        top_k=5,
    )
    pdf_dir = os.getenv("PDF_DIR", "data/papers")
    rag.build_index(pdf_dir=pdf_dir, save_index=True)
    logger.info("RAG pipeline ready. API is live.")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Liveness and readiness check."""
    ready = rag is not None and rag.is_ready
    return HealthResponse(
        status="healthy" if ready else "not_ready",
        index_loaded=ready,
        total_chunks=len(rag._chunks) if rag is not None else 0,
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Ask a question over the indexed technical documents.

    Returns the generated answer, optional source chunks, and measured latency.
    """
    if rag is None or not rag.is_ready:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    start = time.perf_counter()

    try:
        result = rag.query(request.question)
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail="Question answering failed. Check the server logs.")

    latency_ms = (time.perf_counter() - start) * 1000

    sources = []
    if request.include_sources:
        for doc in result.get("source_documents", []):
            sources.append(SourceDocument(
                content=doc.page_content[:500],   # truncate for response size
                source=doc.metadata.get("source", "unknown"),
                page=doc.metadata.get("page", 0),
                rrf_score=doc.metadata.get("rrf_score"),
            ))

    logger.info(f"Query answered in {latency_ms:.1f}ms")
    return QueryResponse(
        answer=result["result"],
        sources=sources,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
):
    """
    Upload a new research paper PDF and index it incrementally.
    The paper is available for querying immediately after indexing.
    """
    if rag is None or not rag.is_ready:
        raise HTTPException(status_code=503, detail="RAG pipeline is not initialized")
    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    max_upload_bytes = int(os.getenv("MAX_UPLOAD_MB", "20")) * 1024 * 1024
    content = await file.read(max_upload_bytes + 1)
    if len(content) > max_upload_bytes:
        raise HTTPException(status_code=413, detail="PDF exceeds the configured upload limit")
    if not content.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid PDF")

    # Use a generated filename to prevent path traversal and collisions.
    import uuid
    upload_dir = Path(os.getenv("PDF_DIR", "data/papers"))
    upload_dir.mkdir(parents=True, exist_ok=True)
    original_name = Path(file.filename).name.replace("\\", "_").replace("/", "_")
    safe_filename = f"{uuid.uuid4().hex}_{original_name}"
    save_path = upload_dir / safe_filename
    save_path.write_bytes(content)

    try:
        from ..ingestion.pdf_loader import PDFLoader
        from ..ingestion.chunker import ResearchPaperChunker

        pages = PDFLoader(source_dir=None).load_file(str(save_path))
        new_chunks = ResearchPaperChunker().chunk(pages)
        if not new_chunks:
            save_path.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail="No extractable text found in the PDF")
        indexed_count = rag.add_documents(new_chunks, save_index=True)
    except HTTPException:
        raise
    except Exception as exc:
        save_path.unlink(missing_ok=True)
        logger.exception("PDF ingestion failed")
        raise HTTPException(status_code=500, detail="PDF ingestion failed") from exc

    logger.info(f"Indexed {safe_filename}: {indexed_count} chunks")
    return UploadResponse(
        message="PDF uploaded and indexed successfully",
        filename=safe_filename,
        chunks_indexed=indexed_count,
    )


@app.get("/documents")
async def list_documents():
    """List all indexed research papers."""
    if rag is None or not rag._chunks:
        return {"documents": [], "total": 0}
    sources = sorted({c.metadata.get("source", "unknown") for c in rag._chunks})
    return {"documents": sources, "total": len(sources)}
