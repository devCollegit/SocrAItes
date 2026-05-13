# 🦉 SocrAItes: 소크라테스식 학습 코치 Agent

> "정답을 알려주지 않는 AI, 스스로 답을 찾게 하는 AI"

SocrAItes(소크라테스)는 단순한 정답 제공을 넘어, 소크라테스식 문답법을 통해 학습자 스스로 개념을 깨우치도록 돕는 RAG 기반 AI Agent입니다. 

![SocrAItes 실행 화면](docs/images/screenshot.png)

## 🚀 프로젝트 개요
- **목적:** 대학원생들이 강의 자료를 깊이 있게 이해하고 비판적 사고를 기를 수 있도록 돕는 메타인지 학습 도우미
- **주요 특징:** 
  - **정답 지연 & 반문:** 즉각적인 답 대신 반문, 예시 요구, 전제 검토를 통해 사고를 자극합니다.
  - **강의 자료 기반 (RAG):** 환각(Hallucination)을 최소화하고 교수님이 제공한 PDF 자료를 기반으로 답변합니다.
  - **적응형 깊이 조절 (Adaptive Socratic Depth):** 학습자의 좌절 신호를 감지하여 힌트를 제공하거나 난이도를 조절합니다.

## ✨ 주요 기능
- **강의자료 RAG:** PDF 업로드 및 지식 기반 질의응답
- **소크라테스식 대화 엔진:** 반문 및 사고 유도 대화
- **학습 도우미 (Function Calling):** 퀴즈 생성, 일정 등록, 약점 저장
- **약점 진단 & 리포트:** 대화 이력을 기반으로 한 주간 학습 리포트 제공

## 🛠️ 기술 스택
- **LLM:** GPT-4o-mini
- **RAG & Vector DB:** LangChain, OpenAI text-embedding-3-small, ChromaDB
- **Orchestration:** LangGraph (Agentic Workflow)
- **Backend:** FastAPI (Python)
- **Frontend:** Vanilla HTML/CSS/JS (Modern Aesthetic)
- **Database:** SQLite (History), ChromaDB (Vector)

## 📂 문서
상세 기획 및 설계 문서는 `docs/` 디렉토리를 참고하세요.
- [소프트웨어 요구사항 명세서 (SRS)](docs/SRS.md)
- [시스템 아키텍처 및 설계 문서](docs/System_Design.md)

## 🏃 시작하기 (Getting Started)

SocrAItes 서비스를 로컬 환경에서 실행하려면 아래 단계를 따르세요.

### 1. 환경 설정 (Setup)
Python 3.9 이상의 환경이 필요합니다. 가상환경을 생성하고 필요한 패키지를 설치합니다.

```bash
# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# 의존성 설치
pip install -r requirements.txt
pip install python-multipart pymupdf  # 추가 필수 라이브러리
```

### 2. 환경 변수 설정 (Configuration)
`.env.example` 파일을 복사하여 `.env` 파일을 생성하고, `OPENAI_API_KEY`를 입력합니다.

```bash
cp .env.example .env
```

### 3. 서비스 실행 (Run)
FastAPI 서버를 uvicorn으로 실행합니다. (권장)

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

서버가 구동되면 브라우저에서 **[http://localhost:8000](http://localhost:8000)**에 접속하여 서비스를 이용할 수 있습니다.

> **TIP: 임베딩 충돌 에러가 발생한다면?**  
> `Embedding function conflict` 에러가 발생할 경우 기존 DB 폴더를 삭제하고 다시 실행하세요.
> ```powershell
> # PowerShell 기준
> Remove-Item -Recurse -Force chroma_db
> ```

### 4. 강의 자료 등록 (PDF Upload)
웹 화면 왼쪽 하단의 **클립(첨부) 아이콘**을 클릭하여 PDF 파일을 직접 업로드할 수 있습니다. 업로드된 파일은 자동으로 인덱싱되어 대화 시 참조됩니다.

