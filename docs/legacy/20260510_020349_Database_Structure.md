# SocrAItes 데이터베이스 구조 및 용도 가이드

이 문서는 SocrAItes 프로젝트에서 사용하는 SQLite 데이터베이스(`data/socraites.db`)의 구조와 각 테이블의 용도를 설명합니다.

## 1. 개요
- **파일 경로:** `data/socraites.db`
- **유형:** SQLite 3
- **용도:** 사용자 세션, 대화 이력, 학습 약점, 복습 일정, 문서 메타데이터 및 분석 리포트를 영구적으로 저장하기 위한 관계형 데이터베이스입니다.

---

## 2. 테이블 상세 설명

### 2.1 sessions (대화 세션)
사용자와 AI 간의 개별 대화 세션을 관리합니다.
- `id`: 세션 고유 식별자 (UUID)
- `user_id`: 사용자 식별자
- `title`: 세션 제목 (예: "분산시스템론 Week5 학습")
- `socratic_mode`: 소크라테스식 문답 깊이 (`light`, `standard`, `deep`)
- `created_at`, `updated_at`: 생성 및 수정 시각

### 2.2 messages (대화 메시지)
각 세션 내에서 오가는 모든 메시지를 저장합니다.
- `id`: 메시지 고유 번호 (PK, Auto-increment)
- `session_id`: 소속 세션 ID (FK)
- `role`: 발화자 역할 (`user`, `assistant`, `system`, `tool`)
- `content`: 메시지 텍스트 내용
- `metadata`: 도구 호출 정보나 깊이 설정 등을 포함하는 JSON 데이터
- `created_at`: 메시지 생성 시각

### 2.3 weaknesses (학습 약점)
대화 중 AI가 감지한 사용자의 학습 취약 개념을 기록합니다.
- `id`: 약점 고유 번호
- `concept`: 취약한 개념 이름 (예: "Partition Tolerance")
- `details`: 구체적인 취약점 내용
- `severity`: 취약 정도 (1~5)
- `resolved`: 해결 여부 (0/1)
- `session_id`: 관련 세션 ID (FK)

### 2.4 schedules (복습 일정)
감지된 약점에 대한 복습 일정을 관리합니다 (망각 곡선 기반 학습 지원).
- `review_at`: 복습 권장 시각
- `description`: 복습 내용 설명
- `completed`: 완료 여부
- `weakness_id`: 관련 약점 ID (FK)

### 2.5 documents (문서 메타데이터)
시스템에 등록된 강의 자료(PDF)의 정보를 관리합니다. (실제 벡터 데이터는 ChromaDB에 저장됨)
- `filename`: 실제 파일명
- `display_name`: 화면에 표시될 이름
- `num_chunks`: 텍스트 분할(Chunking)된 개수
- `file_hash`: 파일 중복 체크를 위한 해시값

### 2.6 reports (진단 리포트)
대화 이력을 분석하여 생성된 주간/학습 진단 리포트를 저장합니다.
- `title`: 리포트 제목
- `body`: 마크다운 형식의 리포트 본문
- `user_id`: 대상 사용자 식별자

---

## 3. 데이터베이스 관리 스크립트
- `scripts/init_db.py`: 데이터베이스 테이블을 생성하거나 초기화합니다.
    - `--reset`: 기존 DB를 삭제하고 새로 생성
    - `--seed`: 테스트를 위한 샘플 데이터를 삽입
