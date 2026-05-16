# src/rag/document_processor.py
from __future__ import annotations

import hashlib
import os
from typing import List, Dict

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", " ", ""],
)


# ---------------------------------------------------------------------------
# PDF Loader
# ---------------------------------------------------------------------------

def load_pdf_pages(file_path: str) -> List[Dict]:
    """페이지별 텍스트와 페이지 번호를 반환합니다.

    Returns:
        [{"text": str, "page": int}, ...]
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if PYMUPDF_AVAILABLE:
        return _load_pages_pymupdf(file_path)
    else:
        return [{"text": _load_pdf_fallback(file_path), "page": 1}]


def _load_pages_pymupdf(file_path: str) -> List[Dict]:
    doc = fitz.open(file_path)
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            pages.append({"text": text, "page": page_num + 1})
    doc.close()
    return pages


def _load_pdf_fallback(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return f.read().decode("utf-8", errors="ignore")


# ---------------------------------------------------------------------------
# Combined Processor
# ---------------------------------------------------------------------------

def process_pdf(file_path: str) -> List[Dict]:
    """PDF를 페이지별로 로드하고 RecursiveCharacterTextSplitter로 청킹합니다.

    Returns:
        [{"text": str, "metadata": {"source": str, "page": int, "chunk_index": int}}, ...]
    """
    filename = os.path.basename(file_path)
    pages = load_pdf_pages(file_path)

    all_chunks = []
    chunk_index = 0
    for page_data in pages:
        page_chunks = _splitter.split_text(page_data["text"])
        for chunk in page_chunks:
            if chunk.strip():
                all_chunks.append({
                    "text": chunk,
                    "metadata": {
                        "source": filename,
                        "page": page_data["page"],
                        "chunk_index": chunk_index,
                    },
                })
                chunk_index += 1

    for chunk in all_chunks:
        chunk["metadata"]["total_chunks"] = chunk_index

    return all_chunks


def compute_file_hash(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()
