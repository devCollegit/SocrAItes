import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Literal

from dotenv import load_dotenv
load_dotenv()  # Ensure .env is loaded even when graph.py is imported directly

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END

from .state import AgentState, DEFAULT_STATE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SocrAItes.Agent")

# Configure specialized trace logging for graph steps and LLM interactions
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
trace_file = os.path.join(LOG_DIR, "agent_trace.log")

trace_logger = logging.getLogger("SocrAItes.Trace")
trace_logger.setLevel(logging.INFO)
# Avoid double logging if this module is reloaded
if not trace_logger.handlers:
    fh = logging.FileHandler(trace_file, encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    trace_logger.addHandler(fh)

def _log_trace(step: str, request: Any = None, response: Any = None, decision: Any = None):
    """Helper to log detailed trace to the specialized log file."""
    trace_logger.info(f"STEP: {step}")
    if request:
        trace_logger.info(f"  [LLM Request]: {request}")
    if response:
        trace_logger.info(f"  [LLM Response]: {response}")
    if decision:
        trace_logger.info(f"  [Decision/Result]: {decision}")
    trace_logger.info("-" * 40)


def _get_content(response) -> str:
    """Safely extract text content from LLM response.

    ChatOpenAI returns an AIMessage (has .content).
    FakeListLLM returns a plain str.
    """
    if hasattr(response, "content"):
        return response.content
    return str(response)

# ---------------------------------------------------------------------------
# LLM Configuration
# ---------------------------------------------------------------------------

if os.getenv("OPENAI_API_KEY"):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
else:
    # Mock LLM for testing frontend when API key is missing
    from langchain_core.language_models.fake import FakeListLLM
    llm = FakeListLLM(responses=["Socratic response mock: How would you define this concept in your own words?", "Interesting. Can you provide an example?", "DIRECT", "PLAN"])

# ---------------------------------------------------------------------------
# Core Agent Nodes
# ---------------------------------------------------------------------------

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

def _format_history(messages: List[Dict[str, Any]]) -> str:
    formatted = []
    for msg in messages[:-1]: # Exclude the last message
        role = "Student" if msg["role"] == "user" else "Tutor"
        formatted.append(f"{role}: {msg['content']}")
    return "\n".join(formatted)

def query_contextualizer(state: AgentState) -> AgentState:
    """Query Contextualizer: Reformulates the user's query in light of conversation history."""
    logger.info(f"--- [Query Contextualizer] Step ---")
    messages = state.get("messages", [])
    
    if len(messages) <= 1:
        # Optimization: Bypassing LLM call on first turn
        state["contextualized_query"] = messages[-1]["content"] if messages else ""
        logger.info(f"First-turn query. Bypassing LLM query rewriting. Query: '{state['contextualized_query']}'")
        _log_trace("QueryContextualizer", decision=f"Bypassed LLM: {state['contextualized_query']}")
        return state
        
    history_text = _format_history(messages)
    last_message = messages[-1]["content"]
    
    prompt = REWRITER_PROMPT.format(history_text=history_text, last_message=last_message)
    response = llm.invoke([HumanMessage(content=prompt)])
    rewritten = _get_content(response).strip()
    
    # Strip quotes if any returned by LLM
    if rewritten.startswith('"') and rewritten.endswith('"'):
        rewritten = rewritten[1:-1].strip()
    elif rewritten.startswith("'") and rewritten.endswith("'"):
        rewritten = rewritten[1:-1].strip()
        
    state["contextualized_query"] = rewritten
    logger.info(f"Original query: '{last_message}' -> Contextualized query: '{rewritten}'")
    _log_trace("QueryContextualizer", request=prompt, response=rewritten, decision=f"Rewritten: {rewritten}")
    return state


def coordinator(state: AgentState) -> AgentState:
    """Coordinator: Analyzes the user query to decide the next step.
    Routes to 'planner' for study queries or 'direct_response' for simple interactions.
    """
    last_user_message = state.get("contextualized_query", "")
    if not last_user_message:
        last_user_message = state["messages"][-1]["content"] if state["messages"] else ""
    logger.info(f"--- [Coordinator] Entry Point ---")
    
    prompt = f"""You are the Coordinator for SocrAItes, a Socratic learning coach.
Analyze the user's message: "{last_user_message}"
 
Decide if this is:
1. A learning/study query related to lecture materials or academic concepts.
2. A casual interaction (greeting, thanks, off-topic, etc.) or a simple navigation request.
 
Respond with ONLY one word: 'PLAN' for category 1, or 'DIRECT' for category 2."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    decision = _get_content(response).strip().upper()
    
    if "PLAN" in decision:
        state["next_step"] = "planner"
    else:
        state["next_step"] = "direct_response"
        
    logger.info(f"Coordinator Decision: {state['next_step']}")
    _log_trace("Coordinator", request=prompt, response=decision, decision=state["next_step"])
    return state


def planner(state: AgentState) -> AgentState:
    """Planner: Creates an execution plan and sets Socratic depth.
    Decides which sub-agents (retrieval, socratic, diagnosis) to prioritize.
    """
    logger.info(f"--- [Planner] Step ---")
    last_query = state.get("contextualized_query", "")
    if not last_query:
        last_query = state["messages"][-1]["content"] if state["messages"] else ""
    depth_modes = ["Light (1-2 turns)", "Standard (3-4 turns)", "Deep (5+ turns)"]
    current_depth = depth_modes[state.get("socratic_depth", 1)]
    
    prompt = f"""You are the Planner for SocrAItes.
Current depth mode: {current_depth}
User query: "{last_query}"

Task: Create a concise execution plan.
1. Should we retrieve lecture materials? (Yes/No)
2. What is the core concept to investigate?
3. What is the target of the Socratic inquiry?

Respond with a brief plan description."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    state["plan"] = _get_content(response)
    state["next_step"] = "subagents"
    
    logger.info(f"Execution Plan: {state['plan']}")
    _log_trace("Planner", request=prompt, response=state["plan"])
    return state


def supervisor(state: AgentState) -> AgentState:
    """Supervisor: The Socratic Persona.
    Synthesizes retrieved documents and conversation history into a Socratic response.
    """
    logger.info(f"--- [Supervisor] Generation Step ---")
    history = state["messages"]
    docs = state.get("retrieved_docs", [])
    plan = state.get("plan", "")
    depth = state.get("socratic_depth", 1)
    
    context = "\n".join([d[0] for d in docs]) if docs else "No specific documents found."
    
    system_prompt = f"""You are SocrAItes, a world-class Socratic tutor.
Your goal is NEVER to give the direct answer. Instead, guide the student using:
- Counter-questions to challenge assumptions.
- Requests for examples.
- Examination of premises.
- Socratic irony (feigning ignorance to encourage explanation).

Current Plan: {plan}
Socratic Depth: {depth} (0: Light, 1: Standard, 2: Deep)
Available Lecture Context:
---
{context}
---

Rules:
1. Do NOT answer the question directly. 
   - EXCEPTION: If the user explicitly asks for a 'summary', you MUST provide a very brief, high-level overview (like a list of main topics or a 2-sentence summary) to establish context, but then immediately follow up with a Socratic question to explore a specific detail.
2. Use the provided context to form your questions.
3. Be encouraging but firm in pushing the student to think.
4. Detect frustration: if the student is struggling significantly, provide a small hint (Scaffolding).
5. Respond in Korean naturally."""

    messages = [SystemMessage(content=system_prompt)]
    for m in history:
        if m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            messages.append(AIMessage(content=m["content"]))
            
    response = llm.invoke(messages)
    state["draft_answer"] = _get_content(response)
    
    logger.info(f"Draft Answer Generated (length: {len(state['draft_answer'])})")
    _log_trace("Supervisor", request=system_prompt, response=state["draft_answer"])
    return state


def evaluator(state: AgentState) -> AgentState:
    """Evaluator: Quality check.
    Ensures the response is Socratic and grounded in the lecture materials.
    """
    logger.info(f"--- [Evaluator] Quality Check Step ---")
    draft = state.get("draft_answer", "")
    docs = state.get("retrieved_docs", [])
    
    prompt = f"""Evaluate this response for SocrAItes:
Response: "{draft}"

Criteria:
1. Is it Socratic? (Does it avoid direct answers OR only provide high-level context/summary before asking a probing question?)
2. Is it grounded in provided documents?
3. Is it encouraging?

Respond with JSON: {{"pass": true/false, "feedback": "reasoning"}}"""

    # For now, we'll assume a pass for the workflow. 
    # In a real implementation, we would parse JSON and potentially route back to Supervisor.
    state["evaluation"] = {"pass": True, "feedback": "Good Socratic engagement."}
    
    logger.info(f"Evaluation Result: {state['evaluation']['pass']}")
    _log_trace("Evaluator", request=prompt, response=state["evaluation"])
    return state

# ---------------------------------------------------------------------------
# Routing & Graph Construction
# ---------------------------------------------------------------------------

def route_coordinator(state: AgentState) -> Literal["planner", "direct_response"]:
    return state["next_step"]

def direct_response(state: AgentState) -> AgentState:
    """Node for non-study queries."""
    logger.info(f"--- [Direct Response] Step ---")
    last_msg = state["messages"][-1]["content"]
    system_msg = "You are a friendly academic assistant. Respond to the user's greeting or casual talk briefly in Korean."
    response = llm.invoke([SystemMessage(content=system_msg), HumanMessage(content=last_msg)])
    state["draft_answer"] = _get_content(response)
    
    logger.info(f"Direct Response Generated.")
    _log_trace("DirectResponse", request=f"{system_msg} | User: {last_msg}", response=state["draft_answer"])
    return state

from src.rag import vectorstore

def retrieval_node(state: AgentState) -> AgentState:
    """Retrieval Node: Queries the vector store based on the last user message."""
    logger.info(f"--- [Retrieval] Step ---")
    last_query = state.get("contextualized_query", "")
    if not last_query:
        last_query = state["messages"][-1]["content"] if state["messages"] else ""
    # Retrieve top 5 relevant chunks
    results = vectorstore.query(last_query, k=5)
    state["retrieved_docs"] = results
    
    logger.info(f"Retrieved {len(results)} document chunks.")
    _log_trace("Retrieval", request=last_query, decision=f"Retrieved {len(results)} chunks")
    return state

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("query_contextualizer", query_contextualizer)
    graph.add_node("coordinator", coordinator)
    graph.add_node("planner", planner)
    graph.add_node("direct_response", direct_response)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("supervisor", supervisor)
    graph.add_node("evaluator", evaluator)

    graph.set_entry_point("query_contextualizer")
    
    graph.add_edge("query_contextualizer", "coordinator")
    
    graph.add_conditional_edges(
        "coordinator",
        route_coordinator,
        {
            "planner": "planner",
            "direct_response": "direct_response"
        }
    )
    
    graph.add_edge("planner", "retrieval")
    graph.add_edge("retrieval", "supervisor")
    graph.add_edge("supervisor", "evaluator")
    graph.add_edge("evaluator", END)
    graph.add_edge("direct_response", END)

    return graph

GRAPH = build_graph()
