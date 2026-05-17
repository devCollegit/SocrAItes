# SocrAItes 문맥 인지형 쿼리(Context-Aware Query) 설계 계획서

이 문서는 SocrAItes 대화 에이전트에서 단건 쿼리가 아닌, 이전 대화 내용(Context)을 함께 고려하여 검색 및 에이전트 처리를 수행할 수 있도록 하는 **문맥 인지형 쿼리 재구성(Context-Aware Query Reformulation)** 도입 계획을 설명합니다.

---

## 1. 도입 배경 및 필요성

현재 SocrAItes는 사용자가 보낸 **최지막 메시지(single query)**만을 바탕으로 질문 분류(`coordinator`), 학습 계획 수립(`planner`), 강의 자료 검색(`retrieval_node`)을 수행합니다. 

이로 인해 다음과 같은 한계가 발생합니다.
1. **대명사 및 지시어 처리 불가:** 사용자가 "그거 더 자세히 설명해줘", "두 번째 조건은 왜 필요해?"와 같이 말할 때, "그거"나 "두 번째 조건"이 무엇인지 이전 대화 맥락을 알지 못하면 올바른 답변이나 검색을 수행할 수 없습니다.
2. **검색 품질 저하:** RAG(Retrieval-Augmented Generation) 시스템에서는 이전 대화의 핵심 학술 개념(예: "데드락", "CAP 정리")이 쿼리에 포함되어야만 벡터 스토어에서 정확한 강의 자료 청크를 찾아낼 수 있습니다.
3. **분류 및 계획 오류:** 문맥이 생략된 쿼리는 대화 코디네이터가 단순 일상 대화(`DIRECT`)로 오분류하거나 플래너가 정확한 학습 도메인을 지정하지 못하게 만듭니다.

---

## 2. 해결 방안: 문맥 인지형 쿼리 재구성 (Query Reformulation)

사용자의 마지막 메시지와 이전 대화 이력(History)을 결합하여, **독립적으로 검색 및 이해가 가능한 단일 검색용 쿼리(Standalone Query)**로 재구성하는 LLM 파이프라인을 추가합니다.

### 예시 흐름
* **이전 대화:**
  * **학생 (User):** "OS에서 데드락이 뭐야?"
  * **튜터 (Assistant):** "데드락은 프로세스들이 서로의 자원을 기다리며 멈춰있는 상태입니다..."
* **현재 입력:** "어떻게 해결해?"
* **재구성된 Standalone 쿼리:** `"OS 데드락 해결 방법"`

이렇게 재구성된 쿼리를 사용하여 문서 검색 및 학습 계획을 진행하되, 소크라테스 응답 생성기(`supervisor`)에는 **사용자의 원본 질문과 자연스러운 대화 이력**을 그대로 전달하여 자연스러운 대화 톤앤매너를 유지합니다.

---

## 3. 상세 설계 및 변경 사항

### 3.1 에이전트 상태 (`AgentState`) 변경
새롭게 재구성된 쿼리를 보관할 `contextualized_query` 필드를 상태 정의에 추가합니다.

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
    contextualized_query: str  # <- 신규 필드 추가
```

### 3.2 쿼리 재구성 노드 (`query_contextualizer`) 신설
대화의 시작점에 위치하여 입력 쿼리를 재구성하는 노드를 추가하고, 이를 에이전트 그래프의 새로운 진입점(Entry Point)으로 지정합니다.

* **파일:** `src/agent/graph.py`
```python
def query_contextualizer(state: AgentState) -> AgentState:
    """대화 이력과 마지막 메시지를 분석하여 독립된 검색용 쿼리를 생성하는 노드"""
    # ...
```

#### 프롬프트 디자인
```python
REWRITER_PROMPT = """You are an expert Query Reformulator for a Socratic learning assistant.
Your task is to analyze the conversation history and the latest user message, and reformulate it into a standalone, search-optimized query in Korean.

Instructions:
1. Identify the core academic concept or question the user is asking about.
2. Incorporate necessary context (concepts, terms, definitions) from the previous turns of the conversation so that the query is fully self-contained.
3. Keep the query concise, focused on key concepts, and ideal for retrieval from lecture materials (PDF).
4. If the latest message is a simple greeting, thank you, casual chit-chat, or does not require any context (it is already self-contained), return it exactly as-is.
5. Do NOT add any introductory text, explanations, or quotes. Output ONLY the reformulated query.

Conversation History:
{history_text}

Latest User Message:
{last_message}

Standalone Query (Korean):"""
```

#### 비용 및 속도 최적화 (Optimization)
사용자가 **첫 번째 메시지**를 보내는 경우(대화 이력이 없는 경우)에는 쿼리 재구성이 불필요하므로, LLM 호출을 생략하고 원본 입력을 그대로 `contextualized_query`로 사용하도록 최적화합니다.

```python
if len(state["messages"]) <= 1:
    state["contextualized_query"] = state["messages"][-1]["content"] if state["messages"] else ""
    return state
```

### 3.3 기존 노드들의 문맥 쿼리 적용
새로 생성된 `contextualized_query`를 기존 노드들의 입력으로 연결합니다.

1. **Coordinator:** `state["contextualized_query"]`를 기반으로 학습 쿼리 여부를 더 높은 정확도로 판별합니다.
2. **Planner:** 지시어가 배제된 완전한 쿼리를 통해 올바른 학습 계획을 수립합니다.
3. **Retrieval Node:** 문맥 정보가 포함된 쿼리를 사용하여 벡터 스토어에서 정확한 PDF 청크를 검색합니다.

---

## 4. 검증 및 테스트 계획

1. **단위 테스트:** `scripts/test_agent.py`를 활용하여 기본 단건 질문 작동 검증.
2. **멀티턴 시나리오 테스트:** 새롭게 `scripts/test_agent_history.py` 스크립트를 작성하여, 문맥이 필수적인 멀티턴 질문("OS 데드락이 뭐야?" -> "어떻게 해결해?" -> "그것의 예시를 들어줘")에 대해 쿼리가 성공적으로 재구성되고 관련 문서를 정확히 찾아오는지 검증.
3. **통합 테스트:** Uvicorn API 서버를 구동한 상태에서 프론트엔드 대화방을 이용해 실시간 멀티턴 문맥 질의가 매끄럽게 이루어지는지 모니터링.
