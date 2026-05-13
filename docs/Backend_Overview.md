# Backend Overview Documentation

This document explains the purpose and design of the core backend components we have created for **SocrAItes**.  Each section references the actual source files in the repository so you can quickly locate the implementation.

---

## 1. Agent Layer (`src/agent/`)

| File | Role | Key Concepts |
|------|------|--------------|
| `state.py` | Defines the **state schema** (`AgentState`) that flows through the LangGraph workflow. It holds the conversation history, current Socratic depth, frustration level, and retrieved document chunks. | TypedDict ensures static typing while remaining JSON‑serializable. Provides `DEFAULT_STATE` for easy initialization. |
| `graph.py` | Constructs the **LangGraph StateGraph** that orchestrates the agent pipeline: **Coordinator → Planner → RetrievalAgent → SocraticAgent → DiagnosisAgent → Supervisor → Evaluator**. | Node functions are currently stubs that forward the state; they will later be wired to actual LLM calls, RAG queries, and analysis logic. The graph uses conditional edges to model the flow and finishes at the `evaluator` node. |

**Purpose**: The agent layer is the brain of SocrAItes. It manages the turn‑by‑turn execution, decides which sub‑agents to run, and combines their outputs into a draft response.  By separating concerns into distinct nodes, we get clear debugging boundaries and can replace each stub with production logic incrementally.

---

## 2. Database Layer (`src/db/`)

| File | Role | Key Functions |
|------|------|---------------|
| `database.py` | Provides a **lightweight SQLite** store for persisting conversation logs and weakness records. | `init_db()`, `log_message()`, `get_recent_messages()`, `save_weakness()`, `get_all_weaknesses()` |

**Purpose**: Persistent storage is required for two reasons:
1. **Conversation history** – needed by the Diagnosis Agent to analyse weakness patterns and generate weekly reports.
2. **Weakness tracking** – each time the `save_weakness` tool is called we store a record that can later be visualised or used for spaced‑repetition scheduling.

The module is deliberately dependency‑free (uses only the built‑in `sqlite3`) to keep setup simple for the MVP.

---

## 3. RAG (Retrieval‑Augmented Generation) Layer (`src/rag/`)

| File | Role | Highlights |
|------|------|------------|
| `document_processor.py` | Minimal **PDF loader & chunker**.  In the MVP it reads a PDF as UTF‑8 text and splits it into overlapping character chunks (`CHUNK_SIZE = 1000`, `CHUNK_OVERLAP = 200`). | Functions: `load_pdf`, `chunk_text`, `process_pdf`. Replace with a proper PDF parsing library (e.g., PyMuPDF) for production. |
| `vectorstore.py` | Wrapper around **ChromaDB** for embedding storage and similarity search. | Functions: `get_client`, `get_collection`, `add_documents`, `query`. Uses OpenAI `text-embedding-3-small` via `chromadb.utils.embedding_functions`. |

**Purpose**: The RAG pipeline turns uploaded lecture PDFs into vector embeddings that can be retrieved during a conversation.  `document_processor` prepares raw text chunks, `vectorstore` indexes them and provides a `query` API for the Retrieval Agent.

---

## 4. Function‑Calling Tools (`src/tools/`)

| File | Role | Exposed Tools |
|------|------|---------------|
| `learning_tools.py` | Defines **Pydantic schemas** and **stub implementations** for the four function‑calling utilities required by the SocrAItes workflow. | `generate_quiz`, `schedule_review`, `save_weakness`, `escape_to_answer` (exposed via `TOOL_MAP`). |

Each tool currently logs the request and returns a deterministic placeholder response.  When the full system is ready, these stubs will be replaced with real logic:
- **Quiz generation** – LLM‑driven multiple‑choice questions.
- **Review scheduling** – write to a calendar or DB.
- **Weakness persistence** – call `src.db.database.save_weakness`.
- **Escape to answer** – bypass Socratic flow for direct answers.

**Purpose**: Function calling enables the agent to trigger external actions (quiz creation, scheduling, logging weaknesses) without hard‑coding the behavior inside the LLM prompts.  The clear schema makes it easy for LangChain/LangGraph to validate and serialize calls.

---

## 5. How the Pieces Fit Together

1. **User uploads a PDF** → `document_processor.process_pdf` creates text chunks.
2. Chunks are stored in **ChromaDB** via `vectorstore.add_documents`.
3. During a chat turn, the **Coordinator** receives the user query and passes the state to the **Planner**.
4. The Planner triggers the **Retrieval Agent**, which queries ChromaDB (`vectorstore.query`) and populates `state["retrieved_docs"]`.
5. The **Socratic Agent** (future implementation) uses the retrieved docs and the current `socratic_depth` to craft a Socratic response.
6. The **Diagnosis Agent** examines `state["messages"]` to detect frustration; if needed it calls one of the **Tools** (e.g., `escape_to_answer`).
7. The **Supervisor** aggregates everything into a draft answer, which the **Evaluator** validates before sending back to the user.
8. All messages and any identified weaknesses are persisted via the **DB layer**.

---

## 6. Next Steps
- Choose the LLM provider (OpenAI vs Anthropic) and set the appropriate API keys in `.env`.
- Replace placeholder logic in the agent nodes with real calls to the LLM and RAG utilities.
- Upgrade `document_processor` to use a proper PDF parser and improve chunking (semantic splitting, metadata extraction).
- Connect the `learning_tools` stubs to real implementations (quiz generation, calendar integration, etc.).

---

*Document location*: `docs/Backend_Overview.md`
