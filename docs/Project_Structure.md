# Project Structure & Directory Overview

이 문서는 SocrAItes 프로젝트의 `src/` 디렉토리 하위 구조와 각 구성 요소의 역할을 상세히 설명합니다.

## 📁 디렉토리 구조 요약

```text
src/
├── agent/        # LangGraph 기반 에이전트 오케스트레이션 및 상태 관리
├── db/           # SQLite 기반 대화 이력 및 약점 데이터 저장소
├── frontend/     # 웹 인터페이스 (HTML/CSS/JS)
├── rag/          # RAG 파이프라인 (PDF 처리 및 벡터 스토어)
├── tools/        # 에이전트가 호출하는 기능 모듈 (Function Calling)
└── api.py        # FastAPI 서버 엔드포인트
```

---

## 🧩 상세 설명

### 1. `src/agent/` - Agent Orchestration
시스템의 두뇌 역할을 하며, 사용자의 질문에 따라 어떤 작업을 수행할지 결정하고 워크플로우를 관리합니다.
- **`graph.py`**: LangGraph를 사용하여 에이전트의 노드(Coordinator, Planner, SocraticAgent 등)와 엣지(흐름 제어)를 정의합니다. 복잡한 소크라테스식 대화 로직을 그래프 형태로 구현합니다.
- **`state.py`**: 에이전트 간에 공유되는 상태(`AgentState`) 객체를 정의합니다. 대화 이력, RAG 검색 결과, 현재 소크라테스식 질문 깊이(Depth), 사용자의 좌절도(Frustration level) 등이 포함됩니다.

### 2. `src/db/` - Database Layer
사용자의 학습 데이터를 영구적으로 저장하고 관리합니다.
- **`database.py`**: SQLite3를 사용하여 대화 메시지를 로그로 남기고, 에이전트가 진단한 사용자의 '약점(Weakness)' 정보를 저장 및 조회하는 기능을 제공합니다. 이 데이터는 나중에 주간 학습 리포트 생성에 활용됩니다.

### 3. `src/frontend/` - Frontend (Web UI)
사용자가 직접 상호작용하는 웹 기반 인터페이스입니다.
- **`index.html`**: 메인 레이아웃 및 챗봇 인터페이스 구조를 담당합니다.
- **`style.css`**: 현대적이고 프리미엄한 느낌의 디자인 시스템을 정의합니다. 글래스모피즘 효과와 부드러운 애니메이션이 포함되어 있습니다.
- **`app.js`**: 서버 API와의 통신을 관리합니다. 서버로부터의 스트리밍 응답 처리, PDF 파일 업로드 및 인덱싱 요청, UI 요소의 실시간 업데이트를 처리합니다.

### 4. `src/rag/` - RAG Pipeline
강의 자료(PDF)를 분석하여 답변의 근거가 되는 지식을 추출하고 관리합니다.
- **`document_processor.py`**: 업로드된 PDF 파일에서 텍스트를 추출하고(PyMuPDF 사용), 검색의 정확도를 높이기 위해 겹침(Overlap)이 있는 텍스트 청크로 쪼개는 역할을 합니다.
- **`vectorstore.py`**: ChromaDB와 연동하여 텍스트 청크를 고차원 벡터로 변환(Embedding)해 저장합니다. 사용자의 질문이 들어오면 벡터 유사도 검색을 통해 가장 관련성 높은 자료를 찾아냅니다.

### 5. `src/tools/` - Function Calling Tools
에이전트가 대화 도중 특정 작업이 필요할 때 직접 실행할 수 있는 도구들입니다.
- **`learning_tools.py`**: 다음과 같은 도구들의 스키마와 로직을 정의합니다.
  - **`generate_quiz`**: 학습 내용 기반 퀴즈 생성.
  - **`schedule_review`**: 복습 일정 등록.
  - **`save_weakness`**: 파악된 학습 취약점 기록.
  - **`escape_to_answer`**: 사용자가 매우 좌절했을 때 소크라테스식 대화를 중단하고 정답을 제공.

### 6. `src/api.py` - API Backend
프론트엔드와 백엔드 로직을 연결하는 관문입니다.
- FastAPI를 기반으로 구축되었으며, 실시간 스트리밍 대화 인터페이스(`POST /chat`), 파일 업로드 API(`POST /upload`), 그리고 정적 파일(Frontend) 서빙을 담당합니다.
