"""Sub Agent — Socratic Dialogue: generates Socratic questions and hints."""
import logging
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.state import AgentState
from src.agent.llm import llm, _get_content
from src.agent.helpers import _log_trace, _is_summary_request

logger = logging.getLogger("SocrAItes.Agent")

SOCRATIC_DIALOGUE_PROMPT = """You are SocrAItes, a world-class Socratic tutor.

Your task: Generate a Socratic response that guides the student to think critically.
- Do NOT give direct answers. Provide a brief hint or high-level context (1-2 sentences), then ask a concrete, thought-provoking Socratic question.
- Exception: If the student requested a summary, provide a comprehensive structured summary followed by a Socratic question.
- If frustration_level >= 2, offer more scaffolding (a more detailed hint) before asking the question.
- Ask about specific sub-components, mechanisms, or key distinctions — never vague opinion questions.
- Respond naturally in Korean.

Socratic depth: {depth} (0=Light 1-2 turns, 1=Standard 3-4 turns, 2=Deep 5+ turns)
Frustration level: {frustration_level}
Evaluator feedback (improve if non-empty): "{eval_feedback}"

Lecture context:
---
{context}
---

Conversation history:
{history}"""


def socratic_dialogue_agent(state: AgentState) -> AgentState:
    """Generates Socratic questions/hints when 'dialogue' is in sub_agents."""
    logger.info("--- [Socratic Dialogue Agent] Step ---")
    if "dialogue" not in state.get("sub_agents", ["dialogue"]):
        state["socratic_response"] = ""
        _log_trace(step="SocraticDialogueAgent", purpose="Skipped.", decision="Not in sub_agents.")
        return state

    docs = state.get("retrieved_docs", [])
    history = state.get("messages", [])
    depth = state.get("socratic_depth", 1)
    frustration_level = state.get("frustration_level", 0)
    eval_feedback = state.get("evaluation", {}).get("feedback", "")

    context = "\n".join(d["text"] for d in docs) if docs else "No lecture materials found."
    last_msg = history[-1]["content"] if history else ""

    is_summary = _is_summary_request(last_msg) or _is_summary_request(state.get("contextualized_query", ""))
    summary_override = (
        "IMPORTANT: The student requested a summary. "
        "Provide a comprehensive, structured, detailed summary of the lecture context first, "
        "then close with one concrete Socratic question.\n"
        if is_summary else ""
    )

    history_text = "\n".join(
        f"{'Student' if m['role'] == 'user' else 'Tutor'}: {m['content']}"
        for m in history
    )

    system_content = summary_override + SOCRATIC_DIALOGUE_PROMPT.format(
        depth=depth,
        frustration_level=frustration_level,
        eval_feedback=eval_feedback,
        context=context,
        history=history_text,
    )

    response = llm.invoke([SystemMessage(content=system_content), HumanMessage(content=last_msg)])
    socratic_response = _get_content(response)
    state["socratic_response"] = socratic_response

    logger.info(f"Socratic response generated (len={len(socratic_response)}).")
    _log_trace(
        step="SocraticDialogueAgent",
        purpose="Generate Socratic questions/hints from retrieved lecture content.",
        inputs={"Depth": depth, "Frustration": frustration_level, "Docs": len(docs)},
        prompt_details=system_content,
        response=socratic_response,
        decision="socratic_response saved.",
    )
    return state
