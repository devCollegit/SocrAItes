# Software Requirements Specification (SRS) - SocrAItes

## 1. 서론 (Introduction)
### 1.1 목적
본 문서는 'SocrAItes(소크라테스식 학습 코치)' 시스템의 요구사항을 명확히 정의하여 프로젝트 개발의 기준을 삼기 위함이다.

### 1.2 시스템 개요
SocrAItes는 교수님이 제공한 강의 PDF를 기반으로, 학생에게 정답을 직접 주지 않고 소크라테스식 문답법으로 사고를 유도하는 RAG 기반 AI Agent이다.

## 2. 전반적 설명 (Overall Description)
### 2.1 사용자 클래스 및 특성
- **대학원생 (Primary User):** 강의 내용을 깊이 이해하고 싶으나, 질문할 곳이 마땅치 않은 학생. 즉각적인 답보다는 구조적 이해를 원함.

### 2.2 운영 환경
- Web Application (Streamlit 기반)
- PC 및 모바일 웹 브라우저 지원

## 3. 기능적 요구사항 (Functional Requirements)
### 3.1 [F1] 강의자료 RAG (Retrieval-Augmented Generation)
- **요구사항:** 사용자는 PDF 형태의 강의자료를 업로드할 수 있어야 한다.
- **세부기능:** PDF 텍스트 추출, 청킹(Chunking), 임베딩, Vector DB(ChromaDB) 저장 및 검색.

### 3.2 [F2] 소크라테스식 대화 엔진
- **요구사항:** 시스템은 사용자의 질문에 직접적인 정답을 제공하지 않고, 반문/예시 요구/전제 검토 형식으로 응답해야 한다.
- **세부기능:** CoT(Chain of Thought) 및 페르소나 프롬프트를 적용한 LLM 응답 생성.

### 3.3 [F3] Function Calling 기반 학습 도구
- **요구사항:** 대화 맥락에 따라 외부 도구를 호출하여 추가 기능을 제공한다.
- **세부기능:** 퀴즈 생성(`generate_quiz`), 학습 일정 등록(`schedule_review`), 약점 저장(`save_weakness`), 답변 모드 전환(`escape_to_answer`).

### 3.4 [F4] 약점 진단 & 리포트
- **요구사항:** 사용자의 대화 이력을 분석하여 이해가 부족한 개념을 식별하고 리포트를 제공한다.
- **세부기능:** 대화 로그 분석, 주간 학습 리포트 생성 기능.

### 3.5 [F5] Adaptive Socratic Depth (적응형 대화 깊이 조절)
- **요구사항:** 시스템은 사용자의 좌절 신호(Frustration signal)를 감지하고, 대화의 난이도를 동적으로 조절해야 한다.
- **세부기능:** 
  - 모드 선택: Light(1-2회 반문), Standard(3-4회), Deep(5회 이상)
  - 자동 전환: 사용자가 "모르겠어", "그냥 답 알려줘" 등의 입력 시 점진적 힌트(Scaffolding) 제공 또는 즉시 답변 모드로 전환.

## 4. 비기능적 요구사항 (Non-Functional Requirements)
### 4.1 성능 (Performance)
- 평균 응답 시간: 3초 미만
- p95 응답 시간: 5초 미만
- 동시 사용자 10명 처리 가능

### 4.2 품질 (Quality & Accuracy)
- RAG Faithfulness (정확성): 0.85 이상 (RAGAS 기준)
- 직답 회피율: 80% 이상
- 반문 비율: 60% 이상

### 4.3 확장성 및 유지보수 (Scalability & Maintainability)
- LangGraph를 활용한 Agent 계층 구조 분리 (Coordinator, Planner, Supervisor, Evaluator)
- 모듈화된 파이프라인 구성

## 5. MVP 범위 (Minimum Viable Product)
- [x] 1개 강의 PDF 업로드 및 RAG 질의응답
- [x] 소크라테스 페르소나 기반 5턴 이상 대화 유지
- [x] Socratic Depth 3단계 모드 + 좌절 신호 감지
- [x] Function Calling 4종 (퀴즈, 일정, 약점, 답변전환)
- [x] Streamlit Web UI
