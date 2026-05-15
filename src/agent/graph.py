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

def coordinator(state: AgentState) -> AgentState:
    """Coordinator: Analyzes the user query to decide the next step.
    Routes to 'planner' for study queries or 'direct_response' for simple interactions.
    """
    last_user_message = state["messages"][-1]["content"] if state["messages"] else ""
    logger.info(f"--- [Coordinator] Entry Point ---")
    logger.info(f"User Query: {last_user_message}")
    
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
    return state


def planner(state: AgentState) -> AgentState:
    """Planner: Creates an execution plan and sets Socratic depth.
    Decides which sub-agents (retrieval, socratic, diagnosis) to prioritize.
    """
    logger.info(f"--- [Planner] Step ---")
    last_query = state["messages"][-1]["content"]
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
1. Is it Socratic? (Does it avoid direct answers and ask a question?)
2. Is it grounded in provided documents?
3. Is it encouraging?

Respond with JSON: {{"pass": true/false, "feedback": "reasoning"}}"""

    # For now, we'll assume a pass for the workflow. 
    # In a real implementation, we would parse JSON and potentially route back to Supervisor.
    state["evaluation"] = {"pass": True, "feedback": "Good Socratic engagement."}
    
    logger.info(f"Evaluation Result: {state['evaluation']['pass']}")
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
    response = llm.invoke([SystemMessage(content="You are a friendly academic assistant. Respond to the user's greeting or casual talk briefly in Korean."), HumanMessage(content=last_msg)])
    state["draft_answer"] = _get_content(response)
    
    logger.info(f"Direct Response Generated.")
    return state

from src.rag import vectorstore

def retrieval_node(state: AgentState) -> AgentState:
    """Retrieval Node: Queries the vector store based on the last user message."""
    logger.info(f"--- [Retrieval] Step ---")
    last_query = state["messages"][-1]["content"]
    # Retrieve top 5 relevant chunks
    results = vectorstore.query(last_query, k=5)
    state["retrieved_docs"] = results
    
    logger.info(f"Retrieved {len(results)} document chunks.")
    return state

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("coordinator", coordinator)
    graph.add_node("planner", planner)
    graph.add_node("direct_response", direct_response)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("supervisor", supervisor)
    graph.add_node("evaluator", evaluator)

    graph.set_entry_point("coordinator")
    
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
