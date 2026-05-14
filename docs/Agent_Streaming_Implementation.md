# 에이전트 진행 단계 실시간 표시 (Agent Progress Streaming)

이 문서는 SocrAItes 에이전트의 사고 과정을 실시간으로 사용자에게 시각화하는 기능의 설계 및 구현 내용을 기록합니다.

## 1. 목적
- 사용자가 질문을 던진 후 답변이 나오기까지의 대기 시간(Latency) 동안 에이전트가 어떤 작업을 수행 중인지 투명하게 보여줌으로써 사용자 경험(UX)을 개선합니다.
- LangGraph의 멀티 에이전트 워크플로우를 시각적으로 전달합니다.

## 2. 기술 설계

### 2.1 백엔드 (FastAPI + LangGraph)
- **엔드포인트**: `POST /chat`을 스트리밍 방식으로 전환하거나 `/chat`에서 응답 헤더에 따라 처리. (안정성을 위해 기존 `/chat` 로직을 스트리밍 친화적으로 변경)
- **스트리밍 방식**: `StreamingResponse`와 `ndjson` (Newline Delimited JSON) 형식을 사용합니다.
- **데이터 구조**:
  ```json
  {"type": "step", "node": "coordinator", "message": "질문 분석 중..."}
  {"type": "step", "node": "planner", "message": "학습 계획 수립 중..."}
  {"type": "final", "answer": "...", "plan": "...", "session_id": "..."}
  ```

### 2.2 프론트엔드 (Vanilla JS)
- **수신 방식**: `fetch` API의 `response.body.getReader()`를 사용하여 스트림을 한 줄씩 읽어 처리합니다.
- **UI 컴포넌트**: 
  - 답변 생성 전, 채팅창에 'Thinking Indicator'와 함께 현재 진행 중인 노드의 이름을 표시하는 Stepper UI를 노출합니다.
  - 각 노드가 완료될 때마다 체크 표시(✅)와 함께 다음 단계로 넘어가는 애니메이션을 추가합니다.

## 3. 에이전트 노드별 메시지 매핑
- **coordinator**: "질문 의도 분석 및 경로 결정 중"
- **planner**: "맞춤형 학습 계획 설계 중"
- **retrieval**: "강의 자료에서 관련 지식 검색 중"
- **supervisor**: "소크라테스식 답변 구성 중"
- **evaluator**: "답변 품질 검토 및 최종 조정 중"
- **direct_response**: "일상 대화 응답 준비 중"

## 4. 상세 구현 가이드
1. `src/api.py`: `GRAPH.stream()`을 사용하는 제너레이터 함수 구현 및 `StreamingResponse` 반환.
2. `src/frontend/app.js`: 스트림 파서 추가 및 UI 상태 관리 로직 업데이트.
3. `src/frontend/style.css`: `.thinking-process`, `.step-item` 관련 스타일 추가.
