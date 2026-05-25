"""LangGraph wiring: assembles all nodes and defines routing logic."""
import logging
from typing import Literal

from langgraph.graph import StateGraph, END

from src.agent.state import AgentState
from src.agent.core import coordinator, planner, supervisor, evaluator, direct_response
from src.agent.sub_agents import retrieval_agent, socratic_dialogue_agent, diagnosis_agent

logger = logging.getLogger("SocrAItes.Agent")

MAX_EVAL_RETRIES = 2


def route_coordinator(state: AgentState) -> Literal["planner", "direct_response"]:
    return state["next_step"]


def route_evaluator(state: AgentState) -> Literal["socratic_dialogue_agent", "end"]:
    evaluation = state.get("evaluation", {})
    retry_count = state.get("retry_count", 0)
    if not evaluation.get("pass", True) and retry_count < MAX_EVAL_RETRIES:
        logger.info(f"Evaluator retry #{retry_count} — re-running Socratic Dialogue Agent.")
        return "socratic_dialogue_agent"
    return "end"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Core Agent
    graph.add_node("coordinator", coordinator)
    graph.add_node("planner", planner)
    graph.add_node("supervisor", supervisor)
    graph.add_node("evaluator", evaluator)
    graph.add_node("direct_response", direct_response)

    # Sub Agents
    graph.add_node("retrieval_agent", retrieval_agent)
    graph.add_node("socratic_dialogue_agent", socratic_dialogue_agent)
    graph.add_node("diagnosis_agent", diagnosis_agent)

    graph.set_entry_point("coordinator")

    graph.add_conditional_edges(
        "coordinator",
        route_coordinator,
        {"planner": "planner", "direct_response": "direct_response"},
    )

    # Planner → Sub Agents → Supervisor → Evaluator
    graph.add_edge("planner", "retrieval_agent")
    graph.add_edge("retrieval_agent", "socratic_dialogue_agent")
    graph.add_edge("socratic_dialogue_agent", "diagnosis_agent")
    graph.add_edge("diagnosis_agent", "supervisor")
    graph.add_edge("supervisor", "evaluator")

    graph.add_conditional_edges(
        "evaluator",
        route_evaluator,
        {"socratic_dialogue_agent": "socratic_dialogue_agent", "end": END},
    )

    graph.add_edge("direct_response", END)

    return graph


GRAPH = build_graph()
