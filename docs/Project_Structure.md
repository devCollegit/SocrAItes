# Project Structure & Directory Overview

이 문서는 현재 SocrAItes 코드베이스의 디렉토리 구조와 역할을 설명합니다.

## 1. 최상위 구조

```text
SocrAItes/
├── src/
│   ├── agent/
│   ├── db/
│   ├── frontend/
│   ├── rag/
│   ├── tools/
│   └── api.py
├── docs/
│   ├── README.md
│   ├── System_Design.md
│   ├── Agent_Workflow.md
│   ├── Project_Structure.md
│   ├── Backend_Overview.md
│   ├── SRS.md
│   └── legacy/
├── scripts/
├── docker/
├── docker-compose.yml
└── requirements.txt
```

## 2. `src/` 상세 설명

### 2.1 `src/api.py`

FastAPI 엔트리포인트입니다.

1. `POST /chat`: LangGraph 실행 및 답변 반환
2. `POST /upload`: PDF 업로드/청킹/인덱싱
3. `GET /health`: 헬스체크

### 2.2 `src/agent/`

LangGraph 기반 에이전트 오케스트레이션 계층입니다.

1. `graph.py`: 노드 정의 및 그래프 라우팅
2. `state.py`: 공용 상태 스키마

### 2.3 `src/rag/`

RAG 파이프라인 구성 요소입니다.

1. `document_processor.py`: PDF 페이지 로드 + 청킹
2. `embeddings.py`: bge-m3 임베딩 생성
3. `vectorstore.py`: Elasticsearch 인덱싱/검색(BM25 + KNN + RRF)

### 2.4 `src/tools/`

학습 도구(Function Calling) 영역입니다.

1. `generate_quiz`
2. `schedule_review`
3. `save_weakness`
4. `escape_to_answer`

### 2.5 `src/db/`

SQLite 기반 보조 저장 계층입니다. 현재/향후 대화 기록 및 약점 데이터 저장에 사용됩니다.

### 2.6 `src/frontend/`

Vanilla HTML/CSS/JS 기반 웹 UI입니다.

## 3. `docs/` 운영 원칙

1. 최신 정본은 `docs/` 루트 문서로 관리
2. 과거 설계/실험 문서는 `docs/legacy/`로 이동
3. 새로운 LLM 협업 문서는 기존 정본 문서에 통합
