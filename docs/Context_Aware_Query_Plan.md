# SocrAItes 문맥 인지형 쿼리(Context-Aware Query) 설계 계획서

이 문서는 SocrAItes 대화 에이전트에서 단건 쿼리가 아닌, 이전 대화 내용(Context)을 함께 고려하여 검색 및 에이전트 처리를 수행할 수 있도록 하는 **문맥 인지형 쿼리 재구성(Context-Aware Query Reformulation)** 도입 계획을 설명합니다.

> [!NOTE]
> 설계 초기에는 별도의 `query_contextualizer` 노드를 신설하려 하였으나, LLM 호출 횟수 감소 및 레이턴시 최적화를 위해 **코디네이터(`coordinator`) 노드에서 쿼리 재구성(Query Reformulation)과 의도 분류(Intent Classification)를 통합 처리**하도록 아키텍처를 개선하였습니다.

---

## 1. 도입 배경 및 필요성

기존에는 사용자가 보낸 **마지막 메시지(single query)**만을 바탕으로 질문 분류, 학습 계획 수립, 강의 자료 검색을 수행했습니다. 이로 인해 다음과 같은 한계가 발생했습니다:

1. **대명사 및 지시어 처리 불가:** 사용자가 "그거 더 자세히 설명해줘", "두 번째 조건은 왜 필요해?"와 같이 말할 때, "그거"나 "두 번째 조건"이 무엇인지 이전 대화 맥락을 알지 못해 오작동합니다.
2. **검색 품질 저하:** RAG(Retrieval-Augmented Generation) 시스템에서는 이전 대화의 핵심 학술 개념(예: "데드락", "CAP 정리")이 쿼리에 포함되어야만 벡터 스토어에서 정확한 강의 자료 청크를 찾아낼 수 있습니다.
3. **분류 및 계획 오류:** 문맥이 생략된 쿼리는 대화 코디네이터가 단순 일상 대화(`DIRECT`)로 오분류하거나 플래너가 정확한 학습 도메인을 지정하지 못하게 만듭니다.

---

## 2. 해결 방안: 통합 코디네이터 (`coordinator`) 구현

사용자의 마지막 메시지와 이전 대화 이력(History)을 결합하여, **독립적으로 검색 및 이해가 가능한 단일 검색용 쿼리(Standalone Query)**로 재구성함과 동시에 학습 경로(`PLAN`) 혹은 일상 경로(`DIRECT`) 분기를 함께 판단하는 통합 LLM 파이프라인을 구축합니다.

### 예시 흐름
* **이전 대화:**
  * **학생 (User):** "OS에서 데드락이 뭐야?"
  * **튜터 (Assistant):** "데드락은 프로세스들이 서로의 자원을 기다리며 멈춰있는 상태입니다..."
* **현재 입력:** "어떻게 해결해?"
* **재구성 결과 (JSON):**
  ```json
  {
    "contextualized_query": "OS 데드락 해결 방법",
    "routing_decision": "PLAN"
  }
  ```

---

## 3. 상세 설계 및 변경 사항

### 3.1 에이전트 상태 (`AgentState`) 변경
재구성된 쿼리를 보관할 `contextualized_query` 필드를 상태 정의에 추가합니다.

* **파일:** `src/agent/state.py`
```python
class AgentState(TypedDict, total=False):
    messages: List[Dict[str, Any]]
    socratic_depth: int
    frustration_level: int
    retrieved_docs: List[Any]
    next_step: str
    plan: str
    draft_answer: str
    evaluation: Dict[str, Any]
    session_id: str
    contextualized_query: str  # <- 문맥 재구성 쿼리 보관 필드
```

### 3.2 코디네이터 노드 (`coordinator`)의 역할 확장
에이전트 그래프의 진입점(Entry Point)인 `coordinator` 노드에서 쿼리 재구성과 라우팅 결정을 하나의 LLM 체인으로 통합 처리합니다.

* **파일:** `src/agent/core/coordinator.py`
* **역할:** 
  1. 이전 대화 내역(`messages`)과 마지막 입력을 파악하여 `contextualized_query`를 도출합니다.
  2. 질문의 성격에 따라 `PLAN` 또는 `DIRECT` 노드로 라우팅할 분기 기준(`next_step`)을 설정합니다.

#### 통합 프롬프트 디자인
```python
COMBINED_COORDINATOR_PROMPT = """You are the Coordinator and Query Reformulator for SocrAItes, a Socratic learning assistant.

Your task is to analyze the conversation history and the latest user message to do two things:
1. Reformulate the latest user message into a standalone, search-optimized query in Korean.
2. Classify whether the user's intent is study-related (academic concepts, lecture materials) or a casual interaction (greetings, gratitude, off-topic, simple navigation).

Instructions for Query Reformulation:
1. Identify the core academic concept or question the user is asking about.
2. Incorporate necessary context (concepts, terms, definitions) from the previous turns so the query is fully self-contained.
3. Keep the query concise, focused on key concepts, and ideal for retrieval from lecture materials (PDF).
4. If the latest message is a simple greeting, thank you, or casual chit-chat, return it exactly as-is.
5. If the latest message asks to continue or summarize "again" (e.g., "다시 요약해줘", "다음 내용 알려줘"), reformulate to search for the NEXT sequential topics rather than repeating what was already explained.

Instructions for Classification:
- Output 'PLAN' if the query is learning/study-related (lecture materials, academic concepts, quizzes).
- Output 'DIRECT' if the query is casual (greeting, thanks, off-topic).

Respond ONLY in JSON:
{{"contextualized_query": "<Korean query>", "routing_decision": "PLAN" | "DIRECT"}}

Conversation History:
{history_text}

Latest User Message:
{last_message}"""
```

#### 비용 및 속도 최적화 (Optimization)
사용자가 **첫 번째 메시지**를 보내는 경우(대화 이력이 없는 경우)에는 쿼리 재구성이 의미가 없으므로, LLM의 입력 및 연산 비용을 줄이기 위해 원본 입력을 그대로 `contextualized_query`로 사용하고 분기 과정만 진행할 수 있도록 설계합니다.

```python
if len(state["messages"]) <= 1:
    state["contextualized_query"] = state["messages"][-1]["content"] if state["messages"] else ""
```

---

## 4. 검증 및 테스트 계획

1. **단위 테스트:** `scripts/test_agent.py`를 활용하여 단건/기본 질문의 올바른 코디네이터 분기 검증.
2. **멀티턴 시나리오 테스트:** `scripts/test_agent_history.py` 스크립트를 활용하여, 대명사/지시어가 포함된 멀티턴 대화 이력에서 `contextualized_query`가 성공적으로 추출되는지 검사.
3. **통합 테스트:** Uvicorn API 서버를 구동하고 프론트엔드 채팅방에서 실시간 멀티턴 문맥 질의 시 Elasticsearch에 문맥화된 쿼리로 질의하여 올바른 강의 청크를 로드하는지 로그 확인.
