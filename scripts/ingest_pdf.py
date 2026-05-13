# scripts/ingest_pdf.py
# -*- coding: utf-8 -*-
"""강의 PDF를 ChromaDB에 인덱싱하는 스크립트.

Usage:
    # data/pdfs/ 폴더 내 모든 PDF 인덱싱 (기본)
    python scripts/ingest_pdf.py

    # 특정 파일만 인덱싱
    python scripts/ingest_pdf.py --file data/pdfs/Week5.pdf

    # 컬렉션 초기화 후 전체 재인덱싱
    python scripts/ingest_pdf.py --reset

    # 인덱싱 후 검색 테스트
    python scripts/ingest_pdf.py --query "CAP 정리란 무엇인가"
"""

from __future__ import annotations

import argparse
import io
import os
import sys

# 출력 인코딩 설정 (Windows 호환)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.rag.document_processor import process_pdf, compute_file_hash, PYMUPDF_AVAILABLE
from src.rag.vectorstore import add_documents, query, get_client, get_collection, PERSIST_DIR
from src.db.database import init_db, register_document, get_documents

# 기본 PDF 저장 폴더
DEFAULT_PDF_DIR = os.path.join(PROJECT_ROOT, "data", "pdfs")


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------

def _get_indexed_hashes() -> set[str]:
    """DB에 이미 등록된 파일 해시 집합을 반환합니다."""
    docs = get_documents()
    return {d["file_hash"] for d in docs if d.get("file_hash")}


def _ingest_single(file_path: str, indexed_hashes: set[str], force: bool = False) -> bool:
    """단일 PDF 파일을 인덱싱합니다. 중복이면 건너뜁니다.

    Returns:
        True if indexing was performed, False if skipped.
    """
    filename = os.path.basename(file_path)
    file_hash = compute_file_hash(file_path)

    if file_hash in indexed_hashes and not force:
        print(f"  [SKIP] {filename} — 이미 인덱싱된 파일 (hash: {file_hash[:8]}...)")
        return False

    print(f"  [PROC] {filename} ...")

    # 1. PDF 처리 (텍스트 추출 + 청킹)
    chunks = process_pdf(file_path)
    if not chunks:
        print(f"  [WARN] {filename} — 텍스트 없음, 건너뜁니다.")
        return False

    # 2. ChromaDB에 저장 (ID: {hash}_{chunk_index})
    texts = [c["text"] for c in chunks]
    metas = [c["metadata"] for c in chunks]
    ids = [f"{file_hash}_{c['metadata']['chunk_index']}" for c in chunks]

    added = add_documents(texts, metas, ids)
    print(f"  [OK]   {filename} — {len(chunks)} 청크 중 {added}개 신규 저장")

    # 3. SQLite DB에 메타데이터 등록
    register_document(
        filename=filename,
        num_chunks=len(chunks),
        display_name=os.path.splitext(filename)[0],
        file_hash=file_hash,
    )
    return True


# ---------------------------------------------------------------------------
# 메인 로직
# ---------------------------------------------------------------------------

def cmd_ingest(file_path: str | None, reset: bool):
    """PDF 인덱싱 실행."""
    # DB 초기화 보장
    init_db()

    if reset:
        client = get_client()
        try:
            client.delete_collection("socratic_docs")
            print("[RESET] 기존 컬렉션 삭제 완료.")
        except Exception:
            print("[RESET] 삭제할 컬렉션이 없습니다.")

    indexed_hashes = _get_indexed_hashes() if not reset else set()

    if file_path:
        # 단일 파일 인덱싱
        if not os.path.isfile(file_path):
            print(f"[ERROR] 파일을 찾을 수 없습니다: {file_path}")
            sys.exit(1)
        targets = [file_path]
    else:
        # 폴더 전체 스캔
        os.makedirs(DEFAULT_PDF_DIR, exist_ok=True)
        targets = [
            os.path.join(DEFAULT_PDF_DIR, f)
            for f in os.listdir(DEFAULT_PDF_DIR)
            if f.lower().endswith(".pdf")
        ]
        if not targets:
            print(f"\n[INFO] '{DEFAULT_PDF_DIR}' 폴더에 PDF 파일이 없습니다.")
            print("       강의 PDF를 해당 폴더에 복사한 후 다시 실행하세요.")
            return

    print(f"\n[INGEST] 총 {len(targets)}개 PDF 처리 시작...")
    print(f"         저장 경로: {os.path.abspath(PERSIST_DIR)}\n")

    success, skipped = 0, 0
    for fp in sorted(targets):
        result = _ingest_single(fp, indexed_hashes, force=reset)
        if result:
            success += 1
        else:
            skipped += 1

    # 최종 현황 출력
    col = get_collection()
    print(f"\n{'='*50}")
    print(f"[DONE] 신규 인덱싱: {success}개  /  건너뜀: {skipped}개")
    print(f"[STAT] ChromaDB 총 문서 수: {col.count()} 청크")
    print(f"{'='*50}\n")


def cmd_query(query_text: str):
    """검색 테스트."""
    print(f"\n[QUERY] '{query_text}' 검색 중...\n")
    results = query(query_text, k=3)
    if not results:
        print("  결과 없음 — 먼저 인덱싱을 실행하세요.")
        return
    for i, (doc, dist) in enumerate(results, 1):
        print(f"[{i}] 거리={dist:.4f}")
        print(f"     {doc[:200].replace(chr(10), ' ')}...")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SocrAItes PDF 인덱싱 도구")
    parser.add_argument("--file",  type=str, default=None,  help="인덱싱할 특정 PDF 파일 경로")
    parser.add_argument("--reset", action="store_true",      help="컬렉션 초기화 후 전체 재인덱싱")
    parser.add_argument("--query", type=str, default=None,  help="인덱싱 후 검색 테스트")
    args = parser.parse_args()

    cmd_ingest(file_path=args.file, reset=args.reset)

    if args.query:
        cmd_query(args.query)


if __name__ == "__main__":
    main()
