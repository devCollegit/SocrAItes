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
    session_id = state.get("session_id")
    user_profile = state.get("user_profile", {})
    user_id = user_profile.get("user_id", "default")
    if session_id and tool_calls:
        for tc in tool_calls:
            args = tc.get("args", {})
            if isinstance(args, dict):
                args["session_id"] = session_id
                if tc.get("name") in ["save_weakness", "save_strength"]:
                    args["user_id"] = user_id
                tc["args"] = args
    
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


BACKGROUND_DIAGNOSIS_PROMPT = """You are the Background Diagnosis Agent for SocrAItes, a Socratic learning assistant.

Your task is to analyze the recent conversation history to implicitly diagnose and update the student's learning profile by calling `update_user_profile`.

Specifically, you MUST manage and update these two summaries inside `update_user_profile`:
- strengths_summary: A 1-3 sentence paragraph summarizing active conceptual masteries.
- weaknesses_summary: A 1-3 sentence paragraph summarizing active conceptual struggles or misconceptions.

Rules for updating summaries (CRITICAL):
1. If the student displays ANY conceptual error, confusion, or incorrect understanding in the conversation, you MUST add/update that concept in the weaknesses_summary.
2. If the student explains a concept correctly, resolves a previous misconception, or demonstrates mastery, you MUST add it to the strengths_summary and REMOVE/FADE OUT that concept from the weaknesses_summary.
3. Fade-out: Maintain a concise summary (max 3 sentences). Drop old or resolved topics.
4. If there is ANY update to the strengths/weaknesses or learning details, you MUST call `update_user_profile`. Do NOT output text explanations. Only call the tool.

Current User Profile State:
- Academic Background: {academic_background}
- Learning Style: {learning_style}
- Preferred Tone: {preferred_tone}
- AI Notes: {profile_notes}
- Current Strengths Summary: "{current_strengths_summary}"
- Current Weaknesses Summary: "{current_weaknesses_summary}"

Session ID: {session_id}
User ID: {user_id}"""


def run_background_diagnosis(session_id: str, user_id: str, messages: List[Dict[str, Any]]) -> None:
    """Runs asynchronously in the background after the API response is sent.
    
    Uses LLM with function calling to implicitly update user profile, strengths, and weaknesses in SQLite.
    """
    logger.info(f"--- [Background Diagnosis Task] Started for session={session_id}, user={user_id} ---")
    
    # Load current user profile to feed to the background diagnosis prompt
    from src.db.database import get_user_profile
    profile = get_user_profile(user_id)
    
    system_content = BACKGROUND_DIAGNOSIS_PROMPT.format(
        session_id=session_id,
        user_id=user_id,
        academic_background=profile.get("academic_background", "대학원생"),
        learning_style=profile.get("learning_style", "conceptual"),
        preferred_tone=profile.get("preferred_tone", "encouraging"),
        profile_notes=profile.get("notes", ""),
        current_strengths_summary=profile.get("strengths_summary", ""),
        current_weaknesses_summary=profile.get("weaknesses_summary", ""),
    )
    messages_for_llm = [SystemMessage(content=system_content)]
    for m in messages:
        if m["role"] == "user":
            messages_for_llm.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            messages_for_llm.append(AIMessage(content=m["content"]))
            
    try:
        # Bind ONLY update_user_profile tool to force updating the summaries and details
        from src.tools.learning_tools import LANGCHAIN_TOOLS
        profiling_tools = [t for t in LANGCHAIN_TOOLS if t.name == "update_user_profile"]
        
        response = llm.bind_tools(profiling_tools).invoke(messages_for_llm)
        tool_calls = _extract_tool_calls(response)
        
        if tool_calls:
            # Inject user_id where appropriate
            for tc in tool_calls:
                args = tc.get("args", {})
                if isinstance(args, dict):
                    if tc.get("name") == "update_user_profile":
                        args["user_id"] = user_id
                    tc["args"] = args
                    
            logger.info(f"[Background Diagnosis] Executing {len(tool_calls)} profiling tools...")
            results = _run_tool_calls(tool_calls)
            logger.info(f"[Background Diagnosis] Results: {results}")
        else:
            logger.info("[Background Diagnosis] No profile summary updates needed.")
            
    except Exception as e:
        logger.error(f"[Background Diagnosis] Failed: {e}")

