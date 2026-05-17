# Backend Overview Documentation

이 문서는 현재 SocrAItes 백엔드 구성 요소의 역할과 연결 방식을 설명합니다.

## 1. API Layer (`src/api.py`)

### 1.1 핵심 엔드포인트

1. `POST /chat`
	- 요청 메시지를 `AgentState`로 변환
	- `GRAPH.compile().invoke(state)` 실행
	- `answer`, `retrieved_docs`, `plan` 반환
2. `POST /upload`
	- PDF 파일 임시 저장
	- `process_pdf`로 텍스트 청킹
	- `add_documents`로 Elasticsearch 적재
3. `GET /health`
	- 서비스 상태 반환

### 1.2 정적 프론트엔드 서빙

`src/frontend/`를 `/frontend` 경로로 마운트하고, `/` 요청 시 `index.html`을 반환합니다.

## 2. Agent Layer (`src/agent/`)

### 2.1 `state.py`

LangGraph에서 공유되는 상태 스키마와 기본 상태를 제공합니다.

### 2.2 `graph.py`

현재 그래프 노드:

1. `coordinator`
2. `planner` 또는 `direct_response`
3. `retrieval`
4. `supervisor`
5. `evaluator`

노드별 LLM 요청/응답/결정은 `logs/agent_trace.log`에 저장합니다.

## 3. RAG Layer (`src/rag/`)

### 3.1 `document_processor.py`

1. PyMuPDF(`fitz`)로 PDF 페이지 텍스트 추출
2. `RecursiveCharacterTextSplitter`로 청킹
3. `source`, `page`, `chunk_index` 메타데이터 생성

### 3.2 `embeddings.py`

1. `BAAI/bge-m3` 로컬 모델 로드
2. 디바이스 자동 선택(`mps`, `cuda`, `cpu`)
3. 문서/질의 임베딩 벡터 생성

### 3.3 `vectorstore.py`

Elasticsearch 기반 인덱싱/검색을 담당합니다.

1. 인덱스: `socratic_docs`
2. 매핑: 한국어 analyzer + dense_vector(1024)
3. 검색: BM25 + Dense KNN + RRF 결합

## 4. Data Layer (`src/db/`)

SQLite 모듈은 대화 로그/약점 기록을 위한 보조 저장 계층입니다.
현재 기능 확장 시 tool 호출 결과와 리포팅 데이터를 저장하는 방향으로 사용합니다.

## 5. Tools Layer (`src/tools/`)

`learning_tools.py`는 function-calling 도구 스키마와 구현을 제공합니다.

1. `generate_quiz`
2. `schedule_review`
3. `save_weakness`
4. `escape_to_answer`

## 6. 실행 데이터 흐름

1. 사용자 질문 입력 (`/chat`)
2. coordinator/planner로 의도 분석 및 계획 생성
3. retrieval이 Elasticsearch에서 문서 검색
4. supervisor가 소크라테스식 응답 초안 생성
5. evaluator를 거쳐 최종 답변 반환

업로드 흐름은 별도로,

1. PDF 업로드 (`/upload`)
2. 텍스트 추출/청킹
3. 임베딩 생성
4. Elasticsearch 인덱싱

으로 동작합니다.
