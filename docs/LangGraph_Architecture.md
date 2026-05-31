# SocrAItes LangGraph Architecture

이 문서는 `src/agent/graph.py` 및 `src/agent/state.py` 코드를 바탕으로 현재 구현된 LangGraph의 노드 구성, 라우팅 흐름, 그리고 각 노드의 상태(Input/Output)를 시각화한 문서입니다.

## 1. 노드 라우팅 및 제어 흐름 (Control Flow)

다음은 전체 에이전트의 실행 경로를 보여주는 플로우차트입니다. `router` 노드에서 사용자의 입력 및 상태를 분석하여 알맞은 경로로 분기하며, 각 단계를 거쳐 최종적으로 응답을 생성(`END`)합니다.

```mermaid
flowchart TD
    START(["&nbsp; START &nbsp;"])
    
    router["<span style='font-size:16px'><b>Router</b></span><br><span style='font-size:11px'>쿼리 재작성 및 라우팅</span><hr><b>In:</b> messages, user_profile<br><b>Out:</b> route, rewritten_query,<br>active_agents, subtask"]
    
    responder["<span style='font-size:16px'><b>Responder</b></span><br><span style='font-size:11px'>캐주얼 / 잡담 응답</span><hr><b>In:</b> messages<br><b>Out:</b> response"]
    
    retrieval["<span style='font-size:16px'><b>Retrieval Agent</b></span><br><span style='font-size:11px'>벡터 스토어 검색</span><hr><b>In:</b> rewritten_query, messages<br><b>Out:</b> retrieved_docs, selected_docs"]
    
    socratic["<span style='font-size:16px'><b>Socratic Agent</b></span><br><span style='font-size:11px'>소크라테스식 응답 생성</span><hr><b>In:</b> retrieved_docs, user_profile,<br>frustration_level, socratic_depth<br><b>Out:</b> tutor_response, pending_quiz"]
    
    tool["<span style='font-size:16px'><b>Tool Agent</b></span><br><span style='font-size:11px'>학습 도구 호출</span><hr><b>In:</b> messages, subtask, active_agents<br><b>Out:</b> tool_result, tool_results"]
    
    composer["<span style='font-size:16px'><b>Composer</b></span><br><span style='font-size:11px'>서브에이전트 결과 조립</span><hr><b>In:</b> tutor_response, tool_result,<br>messages, pending_quiz<br><b>Out:</b> response"]
    
    reviewer["<span style='font-size:16px'><b>Reviewer</b></span><br><span style='font-size:11px'>품질 검사 및 평가</span><hr><b>In:</b> response, messages,<br>user_profile, retry_count<br><b>Out:</b> evaluation, retry_count"]
    
    END(["&nbsp; END &nbsp;"])

    %% 진입점
    START --> router

    %% Router 조건 분기
    router -->|"chat"| responder
    router -->|"tools"| tool
    router -->|"escape + pending_quiz"| composer
    router -->|"learn / escape"| retrieval

    %% Learn 파이프라인
    retrieval --> socratic
    socratic --> tool
    tool --> composer

    %% Chat 종료
    responder --> END

    %% Composer 분기
    composer -->|"검수 필요"| reviewer
    composer -->|"검수 불필요"| END

    %% Reviewer 피드백 루프
    reviewer -->|"실패 & retry < MAX"| socratic
    reviewer -->|"통과 / 재시도 초과"| END

    %% 스타일
    classDef startEnd fill:#2c3e50,stroke:#2c3e50,color:#fff,font-weight:bold;
    classDef routerNode fill:#e74c3c,stroke:#c0392b,color:#fff,stroke-width:2px;
    classDef agentNode fill:#2980b9,stroke:#1a5276,color:#fff,stroke-width:2px;
    classDef outputNode fill:#27ae60,stroke:#1e8449,color:#fff,stroke-width:2px;
    classDef chatNode fill:#8e44ad,stroke:#6c3483,color:#fff,stroke-width:2px;

    class START,END startEnd;
    class router routerNode;
    class retrieval,socratic,tool agentNode;
    class composer,reviewer outputNode;
    class responder chatNode;

    linkStyle 0 stroke:#2c3e50,stroke-width:2px
    linkStyle 1 stroke:#8e44ad,stroke-width:2px
    linkStyle 2 stroke:#e67e22,stroke-width:2px
    linkStyle 3 stroke:#e67e22,stroke-width:2px
    linkStyle 4 stroke:#2980b9,stroke-width:2px
    linkStyle 5 stroke:#2980b9,stroke-width:2px
    linkStyle 6 stroke:#2980b9,stroke-width:2px
    linkStyle 7 stroke:#2980b9,stroke-width:2px
    linkStyle 8 stroke:#8e44ad,stroke-width:2px
    linkStyle 9 stroke:#27ae60,stroke-width:2px
    linkStyle 10 stroke:#27ae60,stroke-width:2px
    linkStyle 11 stroke:#e74c3c,stroke-width:2px,stroke-dasharray:5
    linkStyle 12 stroke:#27ae60,stroke-width:2px
```

---

## 2. 노드별 상태 접근 및 데이터 플로우 (Data Flow)

각 노드는 공유되는 `AgentState` 객체를 읽고(Input) 업데이트(Output) 하는 방식으로 데이터를 교환합니다.

```mermaid
classDiagram
    class AgentState {
        +List messages
        +String session_id
        +Dict user_profile
        +String route
        +String rewritten_query
        +List active_agents
        +String subtask
        +Int socratic_depth
        +Int frustration_level
        +Int retry_count
        +List retrieved_docs
        +String tutor_response
        +Dict tool_result
        +String response
        +Dict evaluation
        +List tool_results
        +List pending_quiz
        +Bool force_explain
        +List selected_docs
    }

    class router {
        <<Node>>
    }
    class responder {
        <<Node>>
    }
    class retrieval_agent {
        <<Node>>
    }
    class socratic_agent {
        <<Node>>
    }
    class tool_agent {
        <<Node>>
    }
    class composer {
        <<Node>>
    }
    class reviewer {
        <<Node>>
    }

    %% 연관관계 (노드가 상태를 어떻게 갱신하는지 표시)
    AgentState <.. router : Updates (route, subtask...)
    AgentState <.. responder : Updates (response)
    AgentState <.. retrieval_agent : Updates (retrieved_docs)
    AgentState <.. socratic_agent : Updates (tutor_response)
    AgentState <.. tool_agent : Updates (tool_result)
    AgentState <.. composer : Updates (response)
    AgentState <.. reviewer : Updates (evaluation)
```

### 상세 상태 매핑표

| 노드 이름 | 주요 Input (읽기) | 주요 Output (쓰기) | 설명 |
|---|---|---|---|
| **`router`** | `messages`, `user_profile` | `route`, `rewritten_query`, `active_agents`, `subtask` | 대화 맥락을 파악하고 최적의 경로와 에이전트들을 결정합니다. |
| **`responder`** | `messages` | `response` | "chat" 경로 시 일상적인 대화나 인사말을 즉시 생성합니다. |
| **`retrieval_agent`** | `rewritten_query`, `messages` | `retrieved_docs`, `selected_docs` | RAG를 위해 재작성된 쿼리로 벡터 스토어를 검색하여 문서를 가져옵니다. |
| **`socratic_agent`** | `messages`, `retrieved_docs`, `user_profile`, `frustration_level`, `socratic_depth`, `pending_quiz`, `retry_count` | `tutor_response`, `pending_quiz` | 소크라테스식 문답을 구성하거나 퀴즈를 출제합니다. |
| **`tool_agent`** | `messages`, `subtask`, `active_agents` | `tool_result`, `tool_results` | 필요한 학습 도구를 실행하고 결과를 반환합니다. |
| **`composer`** | `tutor_response`, `tool_result`, `messages`, `pending_quiz` | `response` | 하위 에이전트들의 실행 결과(`tutor_response`, `tool_result` 등)를 하나의 텍스트 응답으로 조립합니다. |
| **`reviewer`** | `response`, `messages`, `user_profile`, `retry_count` | `evaluation`, `retry_count` | `response`의 품질(정답 누설 여부 등)을 검증하고, 실패 시 `retry_count`를 증가시킵니다. |
