# src/rag/document_processor.py
"""Document processing utilities for SocrAItes.

PDF 텍스트 추출 및 청킹 파이프라인.
- PyMuPDF(fitz)가 설치되어 있으면 실제 PDF 파싱.
- 없을 경우 바이트 디코딩 플레이스홀더로 graceful fallback.
"""

from __future__ import annotations

import hashlib
import os
from typing import List, Dict

# PyMuPDF optional import
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

CHUNK_SIZE = 800    # characters per chunk
CHUNK_OVERLAP = 150 # overlap between chunks


# ---------------------------------------------------------------------------
# PDF Loader
# ---------------------------------------------------------------------------

def load_pdf(file_path: str) -> str:
    """PDF 파일에서 텍스트를 추출하여 반환합니다.

    PyMuPDF가 설치된 경우 실제 PDF 파싱을 수행하고,
    그렇지 않으면 바이너리 디코딩 방식으로 fallback합니다.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if PYMUPDF_AVAILABLE:
        return _load_pdf_pymupdf(file_path)
    else:
        return _load_pdf_fallback(file_path)


def _load_pdf_pymupdf(file_path: str) -> str:
    """PyMuPDF를 사용해 페이지별 텍스트를 추출합니다."""
    doc = fitz.open(file_path)
    pages_text = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            pages_text.append(f"[Page {page_num + 1}]\n{text}")
    doc.close()
    return "\n\n".join(pages_text)


def _load_pdf_fallback(file_path: str) -> str:
    """PyMuPDF 없이 바이너리 디코딩 방식으로 fallback."""
    with open(file_path, "rb") as f:
        raw = f.read()
    return raw.decode("utf-8", errors="ignore")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str) -> List[str]:
    """텍스트를 CHUNK_SIZE 크기의 겹치는 청크로 분할합니다."""
    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + CHUNK_SIZE, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ---------------------------------------------------------------------------
# Combined Processor
# ---------------------------------------------------------------------------

def process_pdf(file_path: str) -> List[Dict]:
    """PDF 파일을 로드하고 메타데이터가 포함된 청크 딕셔너리 리스트를 반환합니다.

    Returns:
        [{"text": str, "metadata": {"source": str, "chunk_index": int}}, ...]
    """
    filename = os.path.basename(file_path)
    raw_text = load_pdf(file_path)
    chunks = chunk_text(raw_text)

    return [
        {
            "text": chunk,
            "metadata": {
                "source": filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
            },
        }
        for i, chunk in enumerate(chunks)
    ]


def compute_file_hash(file_path: str) -> str:
    """파일의 SHA-256 해시를 반환합니다."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()
