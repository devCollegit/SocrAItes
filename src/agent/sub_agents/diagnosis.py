"""Sub Agent — Diagnosis: analyzes conversation and calls learning tools."""
import logging
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.agent.state import AgentState
from src.agent.llm import llm, _get_content, _extract_tool_calls, _run_tool_calls, LANGCHAIN_TOOLS
from src.agent.helpers import _log_trace

logger = logging.getLogger("SocrAItes.Agent")

DIAGNOSIS_PROMPT = """You are the Diagnosis Agent for SocrAItes.

Analyze the conversation and call the appropriate learning tool if clearly needed:
- generate_quiz: student explicitly asked for a quiz or practice problems. Infer the topic from recent conversation.
- save_weakness: a clear knowledge gap was identified in the conversation. Infer concept and details from conversation.
- schedule_review: student wants to schedule a review session. Call immediately — datetime is optional (defaults to 7 days from now). Infer description from recent topic.
- escape_to_answer: student explicitly asked for the direct answer (e.g., "그냥 답 알려줘")

IMPORTANT: Call tools proactively. Missing optional parameters (like datetime) are fine — use defaults. Do NOT ask clarifying questions; just call the tool with what you know.
If no tool is needed, respond with "No tools needed."

Current subtask: {subtask}
Frustration level: {frustration_level}"""


def diagnosis_agent(state: AgentState) -> AgentState:
    """Calls learning tools when 'diagnosis' is in sub_agents."""
    logger.info("--- [Diagnosis Agent] Step ---")
    if "diagnosis" not in state.get("sub_agents", []):
        state["diagnosis_result"] = {}
        _log_trace(step="DiagnosisAgent", purpose="Skipped.", decision="Not in sub_agents.")
        return state

    subtask = state.get("plan", "")
    frustration_level = state.get("frustration_level", 0)
    history = state.get("messages", [])

    system_content = DIAGNOSIS_PROMPT.format(subtask=subtask, frustration_level=frustration_level)
    messages_for_llm = [SystemMessage(content=system_content)]
    for m in history:
        if m["role"] == "user":
            messages_for_llm.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            messages_for_llm.append(AIMessage(content=m["content"]))

    response = llm.bind_tools(LANGCHAIN_TOOLS).invoke(messages_for_llm)
    analysis_text = _get_content(response)
    tool_calls = _extract_tool_calls(response)
    tool_results = _run_tool_calls(tool_calls) if tool_calls else []

    state["diagnosis_result"] = {
        "analysis": analysis_text,
        "tool_results": tool_results,
        "tool_calls_made": len(tool_calls),
    }
    state["tool_results"] = tool_results

    logger.info(f"Diagnosis done. Tools called: {len(tool_calls)}.")
    _log_trace(
        step="DiagnosisAgent",
        purpose="Analyze conversation and call appropriate learning tools.",
        inputs={"Subtask": subtask, "Frustration": frustration_level},
        prompt_details=system_content,
        response=analysis_text,
        decision={"tool_calls": len(tool_calls), "results": tool_results},
    )
    return state
