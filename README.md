# 🦉 SocrAItes: 소크라테스식 학습 코치 Agent

> "정답을 알려주지 않는 AI, 스스로 답을 찾게 하는 학습 코치"

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
- **Embedding:** BAAI/bge-m3 (로컬, MPS/CUDA/CPU 자동 감지)
- **RAG & Vector DB:** LangChain, Elasticsearch (BM25 + Dense KNN + RRF 하이브리드 검색)
- **Orchestration:** LangGraph (Agentic Workflow)
- **Backend:** FastAPI (Python)
- **Frontend:** Vanilla HTML/CSS/JS (Modern Aesthetic)
- **Database:** SQLite (History), Elasticsearch (Vector)

## 📂 문서
상세 기획 및 설계 문서는 `docs/` 디렉토리를 참고하세요.

- [문서 허브 (읽기 순서/운영 규칙)](docs/README.md)
- [소프트웨어 요구사항 명세서 (SRS)](docs/SRS.md)
- [시스템 아키텍처 및 설계 문서](docs/System_Design.md)
- [프로젝트 디렉토리 구조 및 상세 설명](docs/Project_Structure.md)
- [에이전트 워크플로우 및 로직 상세](docs/Agent_Workflow.md)
- [백엔드 개요](docs/Backend_Overview.md)
- [레거시 문서 인덱스](docs/legacy/README.md)

## 🏃 시작하기 (Getting Started)

SocrAItes 서비스를 로컬 환경에서 실행하려면 아래 단계를 따르세요.

### 1. Elasticsearch 실행 (Docker)

#### Docker 설치 확인
[Docker Desktop](https://www.docker.com/products/docker-desktop/)이 설치되어 있어야 합니다.

```bash
docker --version  # Docker version 20.x 이상 필요
```

#### 컨테이너 실행

```bash
docker compose up -d
```

백그라운드에서 아래 두 컨테이너가 실행됩니다.

| 서비스 | 주소 | 용도 |
|---|---|---|
| Elasticsearch | http://localhost:9200 | 벡터 DB (BM25 + KNN) |
| Kibana | http://localhost:5601 | 인덱스/데이터 시각화 |

#### 실행 상태 확인

ES가 완전히 준비되는 데 약 30초 소요됩니다.

```bash
# 방법 1: 클러스터 상태 확인 (status가 green 또는 yellow면 정상)
curl http://localhost:9200/_cluster/health

# 방법 2: 컨테이너 상태 확인
docker compose ps
```

정상 응답 예시:
```json
{"cluster_name":"docker-cluster","status":"green", ...}
```

#### 트러블슈팅

```bash
# 로그 확인
docker compose logs elasticsearch

# 컨테이너 재시작
docker compose restart elasticsearch

# 포트 충돌 시 (9200 또는 5601이 이미 사용 중)
lsof -i :9200  # 해당 프로세스 확인 후 종료
```

### 2. 환경 설정 (Python)
Python 3.11 이상의 환경이 필요합니다.

```bash
# 가상환경 생성 및 활성화
python3.11 -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 3. 환경 변수 설정 (Configuration)
`.env.example` 파일을 복사하여 `.env` 파일을 생성하고 `OPENAI_API_KEY`를 입력합니다.

```bash
cp .env.example .env
```

### 4. 모델 다운로드 (최초 1회)

bge-m3 모델(약 570MB)을 미리 다운로드합니다. 한 번만 실행하면 이후엔 캐시에서 즉시 로드됩니다.

```bash
huggingface-cli download BAAI/bge-m3
```

### 5. 서비스 실행 (Run)

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

서버가 구동되면 브라우저에서 **[http://localhost:8000](http://localhost:8000)**에 접속합니다.

### 5. 강의 자료 등록 (PDF Upload)
웹 화면 왼쪽 하단의 **클립(첨부) 아이콘**을 클릭하여 PDF 파일을 업로드합니다. 업로드된 파일은 자동으로 인덱싱되어 대화 시 참조됩니다.

### Docker 종료

```bash
docker compose down        # 컨테이너만 종료 (데이터 유지)
docker compose down -v     # 컨테이너 + 데이터 전체 삭제
```

