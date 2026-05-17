# RAG VectorDB 인덱싱 파이프라인 — 구현 완료 보고 (2026-05-10)

## 현황 요약

| 항목 | 이전 상태 | 현재 상태 |
|------|-----------|-----------|
| `chroma_db/` 디렉토리 | ❌ 없음 | ✅ 자동 생성됨 |
| PDF 인덱싱 스크립트 | ❌ 없음 | ✅ `scripts/ingest_pdf.py` 구현 |
| PDF 파서 | ⚠️ 플레이스홀더 | ✅ PyMuPDF 기반 실제 파서 |
| 중복 인덱싱 방지 | ❌ 없음 | ✅ SHA-256 해시 기반 |
| ChromaDB 저장 | ⚠️ ID 충돌 가능 | ✅ 파일해시+청크인덱스 ID 사용 |
| README 가이드 | ❌ 없음 | ✅ 인덱싱 단계 추가 |

---

## 변경된 파일 목록

1. **[NEW] `scripts/ingest_pdf.py`**
   - `data/pdfs/` 폴더 내 PDF를 자동 탐색하여 인덱싱
   - SHA-256 해시로 중복 파일 건너뜀
   - ChromaDB 및 SQLite `documents` 테이블에 동시 저장
   - `--file`, `--reset`, `--query` CLI 옵션 제공

2. **[MODIFY] `src/rag/document_processor.py`**
   - PyMuPDF(fitz) 기반 실제 PDF 텍스트 추출로 교체
   - PyMuPDF 미설치 시 바이너리 디코딩 방식으로 graceful fallback
   - `compute_file_hash()` 함수 추가

3. **[MODIFY] `src/rag/vectorstore.py`**
   - `add_documents()` 함수에 명시적 ID 지원 및 중복 필터링 로직 추가
   - 반환값으로 신규 저장된 문서 수(int) 반환

4. **[MODIFY] `requirements.txt`**
   - `PyMuPDF` 패키지 추가

5. **[MODIFY] `README.md`**
   - "시작하기" 가이드에 4단계 "강의 자료 인덱싱" 섹션 추가

---

## 인덱싱 실행 방법

```bash
# 1. PDF를 data/pdfs/ 폴더에 복사
# 2. 인덱싱 실행
python scripts/ingest_pdf.py

# 특정 파일만
python scripts/ingest_pdf.py --file data/pdfs/Week5.pdf

# 초기화 후 전체 재인덱싱
python scripts/ingest_pdf.py --reset

# 검색 테스트
python scripts/ingest_pdf.py --query "CAP 정리란 무엇인가"
```

## 다음 단계 (추천)
- 강의 PDF를 `data/pdfs/` 폴더에 복사하여 실제 인덱싱 테스트
- `PyMuPDF` 설치: `pip install PyMuPDF`
- OpenAI API Key 설정 시 더 높은 품질의 `text-embedding-3-small` 임베딩 사용 가능
