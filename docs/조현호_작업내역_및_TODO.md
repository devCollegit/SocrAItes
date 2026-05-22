# 조현호 작업내역 및 TODO

> 최종 업데이트: 2026-05-22  
> 브랜치: `feature/learning-tools` → PR #dev 등록 완료

---

## ✅ 완료된 작업

### 1. 툴 개발 (LangChain Function Calling)

| 툴 이름 | 기능 | 파일 |
|--------|------|------|
| `generate_quiz` | 주제 기반 5문항 MCQ 생성 | `src/tools/learning_tools.py` |
| `schedule_review` | 학습 일정 등록 → SQLite `schedules` 테이블 저장 | `src/tools/learning_tools.py` |
| `save_weakness` | 약점 기록 → SQLite `weaknesses` 테이블 저장 (심각도 1–5) | `src/tools/learning_tools.py` |
| `escape_to_answer` | 답변 모드 전환 (소크라테스식 → 직접 답변) | `src/tools/learning_tools.py` |

**구현 포인트:**
- `LANGCHAIN_TOOLS`: `StructuredTool.from_function()`으로 LangChain 바인딩
- `TOOL_MAP`: 이름 → 함수 딕셔너리 (supervisor 노드에서 실행)
- LangChain ToolCall 포맷: `{"id": ..., "name": ..., "args": {...}}`  
  ※ OpenAI API raw 포맷(`function.name`)과 다름 — 혼동 주의

**연결 위치:**
- `src/agent/graph.py` → `supervisor` 노드에서 `llm.bind_tools(LANGCHAIN_TOOLS)` → `_extract_tool_calls()` → `_run_tool_calls()`
- `src/api.py` → SSE `final_result` 이벤트에 `tool_results` 포함

**테스트:** `scratch/comprehensive_tool_test.py` 4/4 통과 ✅

---

### 2. UI 버그 수정 (Korean 텍스트 렌더링)

**문제:** 퀴즈 응답에서 한국어 단어가 붙어서 출력되는 현상 (띄어쓰기 깨짐)

**수정 파일:** `src/frontend/style.css`

```css
/* 수정 전 */
word-break: break-word;

/* 수정 후 */
word-break: keep-all;        /* 한국어 단어 단위 줄바꿈 */
overflow-wrap: anywhere;
line-height: 1.65;
```

---

## ✅ 추가 완료 작업

### 3. UI 버그: 마지막 글자/토큰 누락 오류 수정

**파일:** `src/frontend/app.js`

**원인:** SSE 스트림 읽기 루프에서 `done=true`일 때 `buffer`에 남아있는 마지막 청크를 처리하지 않고 `break`

**수정 내용:**
- `processLine()` 헬퍼 함수로 라인 파싱 로직 분리
- `done=true` 시 `decoder.decode()`(flush 모드)로 남은 바이트 추출 후 buffer 처리

```js
if (done) {
    buffer += decoder.decode();   // TextDecoder flush
    buffer.split('\n').forEach(line => processLine(line.trim()));
    break;
}
```

**상태:** ✅ 완료

---

### 2. 퀴즈 개선 (선택사항 / 시간 여유 시)

현재 `generate_quiz`는 **미리 작성된 템플릿** 문제를 반환함.  
실제 유용성을 높이려면 LLM이 학습 내용 기반으로 동적 생성하도록 변경 필요.

**현재 구조:**
```python
# src/tools/learning_tools.py
QUIZ_TEMPLATES = [...]  # 고정 문제 리스트
def generate_quiz(topic: str) -> dict:
    return {"questions": QUIZ_TEMPLATES[:5]}
```

**개선 방향:**
- LLM에게 topic을 넘겨 직접 문제 생성 요청
- RAG로 관련 문서 검색 후 그 내용 기반 출제
- 중복 출제 방지: session-level 출제 기록 관리

**우선순위:** 낮음 (기능 자체는 동작 중, 고도화 단계에서 진행)

---

## 📌 팀원에게 설명할 내용 요약

### PR 내용 한 줄 요약
> "LangChain Function Calling을 이용해 퀴즈생성·학습일정·약점저장·답변모드 4개 툴을 구현하고, Supervisor 노드에 연결했습니다. 한국어 텍스트 렌더링 CSS 버그도 함께 수정했습니다."

### 팀원 리뷰 포인트
| 위치 | 확인 사항 |
|------|----------|
| `src/agent/graph.py` | `_extract_tool_calls` / `_run_tool_calls` LangChain 포맷 처리 |
| `src/tools/learning_tools.py` | 각 툴 함수 + TOOL_MAP + LANGCHAIN_TOOLS |
| `src/agent/state.py` | `tool_results` 필드 추가 |
| `src/api.py` | SSE `final_result` 이벤트에 `tool_results` 포함 |
| `src/frontend/style.css` | `word-break: keep-all` 적용 |

### 🔗 다른 팀원 작업과의 연결점
- **김우림 (히스토리)**: 히스토리 State가 추가되면 `generate_quiz`에서 이전 출제 기록 참조 가능
- **권지수 (Sub-agent)**: Sub-agent 구조 완성되면 툴 실행을 Sub-agent로 위임하는 구조로 전환 가능
- **신승윤 (세션/메모리)**: `schedule_review`, `save_weakness` 저장 데이터를 세션/유저별로 구분하려면 session_id 연동 필요

---

## 🗓 일정 참고
- **5월 25일 (일) 21:00**: 온라인 전체 회의
- **5월 31일 (일) 21:00**: 온라인 전체 회의 (잠정)
- **6월 9일**: 최종 제출

