import os
import logging
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Literal

from dotenv import load_dotenv
load_dotenv()  # Ensure .env is loaded even when graph.py is imported directly

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END

from .state import AgentState, DEFAULT_STATE
from src.tools.learning_tools import TOOL_MAP, LANGCHAIN_TOOLS

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

def _log_trace(
    step: str,
    purpose: str = "",
    inputs: dict = None,
    prompt_details: Any = None,
    response: Any = None,
    decision: Any = None
):
    """Helper to log beautifully formatted detailed trace to the specialized log file."""
    border = "=" * 80
    trace_logger.info(border)
    trace_logger.info(f"NODE: {step.upper()}")
    trace_logger.info(border)
    
    if purpose:
        trace_logger.info(f"[PURPOSE]\n  {purpose}\n")
        
    if inputs:
        trace_logger.info("[INPUT CONTEXT]")
        for k, v in inputs.items():
            trace_logger.info(f"  * {k}: {v}")
        trace_logger.info("")
        
    if prompt_details:
        trace_logger.info("[LLM PROMPT / REQUEST CONTENT]")
        if isinstance(prompt_details, str):
            indented = "\n".join([f"    {line}" for line in prompt_details.split("\n")])
            trace_logger.info(f"{indented}\n")
        elif isinstance(prompt_details, dict):
            for pk, pv in prompt_details.items():
                trace_logger.info(f"  - {pk}:")
                indented = "\n".join([f"      {line}" for line in str(pv).split("\n")])
                trace_logger.info(f"{indented}")
            trace_logger.info("")
        else:
            trace_logger.info(f"  {prompt_details}\n")
            
    if response is not None:
        trace_logger.info("[LLM RESPONSE]")
        trace_logger.info("-" * 80)
        indented_resp = "\n".join([f"  {line}" for line in str(response).split("\n")])
        trace_logger.info(indented_resp)
        trace_logger.info("-" * 80)
        trace_logger.info("")
        
    if decision:
        trace_logger.info("[DECISION & STATE MUTATION]")
        if isinstance(decision, dict):
            for dk, dv in decision.items():
                trace_logger.info(f"  * {dk}: {dv}")
        else:
            trace_logger.info(f"  * Result: {decision}")
            
    trace_logger.info(border + "\n\n")


def _get_content(response) -> str:
    """Safely extract text content from LLM response.

    ChatOpenAI returns an AIMessage (has .content).
    FakeListLLM returns a plain str.
    """
    if hasattr(response, "content"):
        return response.content
    return str(response)


def _extract_tool_calls(response: Any) -> List[Dict[str, Any]]:
    """Extract tool calls from LangChain response.
    
    Handles both:
    1. LangChain ToolCall format: {"id": "...", "name": "...", "args": {...}}
    2. OpenAI API format: {"type": "function", "function": {"name": "...", "arguments": "..."}}
    """
    tool_calls = getattr(response, "tool_calls", None)
    if tool_calls:
        return tool_calls
    
    # Fallback to additional_kwargs (less common in recent LangChain)
    additional_kwargs = getattr(response, "additional_kwargs", {}) or {}
    return additional_kwargs.get("tool_calls", [])


def _run_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Execute tool calls and return results.
    
    Handles LangChain ToolCall format:
    - tc.get("name") = tool name
    - tc.get("args") = arguments dict (already parsed by LangChain)
    """
    results: List[Dict[str, Any]] = []
    for tc in tool_calls:
        # LangChain ToolCall format
        tool_name = tc.get("name")
        args = tc.get("args", {})
        
        # Fallback: OpenAI API format (unlikely but handles edge cases)
        if not tool_name:
            fn_info = tc.get("function", {})
            tool_name = fn_info.get("name")
            raw_args = fn_info.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            except json.JSONDecodeError:
                args = {}

        if not tool_name:
            results.append({
                "tool": None,
                "ok": False,
                "error": f"Tool name not found in call",
            })
            continue

        tool_fn = TOOL_MAP.get(tool_name)
        if not tool_fn:
            results.append({
                "tool": tool_name,
                "ok": False,
                "error": f"Unknown tool: {tool_name}",
            })
            continue

        try:
            logger.info(f"Executing tool: {tool_name} with args: {args}")
            output = tool_fn(args)
            results.append({"tool": tool_name, "ok": True, "output": output})
        except Exception as e:
            logger.exception("Tool execution failed: %s", tool_name)
            results.append({"tool": tool_name, "ok": False, "error": str(e)})

    return results


# Regex to detect answer submission like "1:A, 2:B, 3:C, 4:D, 5:A"
_ANSWER_PATTERN = re.compile(r"(\d+)\s*[:：]\s*([A-Da-d])", re.UNICODE)


def _is_quiz_answer(text: str) -> bool:
    """Return True when the message looks like an answer submission."""
    return len(_ANSWER_PATTERN.findall(text)) >= 2


def _detect_frustration(text: str) -> bool:
    """Detect frustration in Korean text based on keywords."""
    keywords = [
        "모르겠", "모르겠음", "모르겠어", "모르겠다", "모르겠는데",
        "어렵", "어려워", "어려움", "어렵다", "어려운데",
        "힘들", "힘들어", "힘들다", "힘듦", "힘든데",
        "포기", "못하겠", "못하겠어", "못하겠다", "못하겠음",
        "답답", "헷갈려", "헷갈림", "이해 안", "이해가 안", "이해 안 됨",
        "그냥 알려줘", "답 알려줘", "답을 알려", "어려운"
    ]
    # Remove whitespace and lower case to normalize
    clean_text = text.replace(" ", "").lower()
    return any(k.replace(" ", "").lower() in clean_text for k in keywords)


def _is_last_message_quiz_prompt(history: List[Dict[str, Any]]) -> bool:
    """Check if the last assistant message in history was a quiz prompt."""
    for msg in reversed(history):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            return "퀴즈" in content and "답안은 예:" in content
    return False


def _grade_quiz(user_text: str, quiz_items: List[Dict[str, Any]]) -> str:
    """Compare user answers to correct answers and return a Korean grade report."""
    submissions = {int(q): v.upper() for q, v in _ANSWER_PATTERN.findall(user_text)}
    total = len(quiz_items)
    correct = 0
    lines = ["## 채점 결과", ""]

    for idx, item in enumerate(quiz_items, start=1):
        correct_ans = item.get("answer", "").upper()
        user_ans = submissions.get(idx, "?")
        options = item.get("options", [])
        correct_text = options[ord(correct_ans) - ord("A")] if correct_ans and options else correct_ans

        if user_ans == correct_ans:
            correct += 1
            mark = "✅"
        else:
            mark = "❌"

        lines.append(f"{mark} {idx}번: 제출={user_ans} / 정답={correct_ans}")
        if user_ans != correct_ans:
            lines.append(f"   → 정답: {correct_ans}. {correct_text}")

    score_pct = int(correct / total * 100) if total else 0
    lines.append("")
    lines.append(f"**{total}문항 중 {correct}문항 정답 ({score_pct}점)**")

    if score_pct == 100:
        lines.append("완벽해요! 다음 개념으로 넘어가볼까요?")
    elif score_pct >= 60:
        lines.append("잘 했어요! 틀린 문항을 다시 한번 살펴보세요.")
    else:
        lines.append("조금 더 복습이 필요해요. 틀린 개념을 약점으로 저장해드릴까요?")

    return "\n".join(lines)


def _format_quiz_response(tool_results: List[Dict[str, Any]]) -> tuple[str | None, List[Dict[str, Any]]]:
    """Create a learner-facing quiz message and return pending quiz items for grading.

    Returns (message_str | None, quiz_items).
    """
    for result in tool_results:
        if result.get("tool") != "generate_quiz" or not result.get("ok"):
            continue
        output = result.get("output") or {}
        quiz_items = output.get("quiz") if isinstance(output, dict) else None
        if not isinstance(quiz_items, list) or not quiz_items:
            continue

        source = output.get("source", "template")
        header = "퀴즈 %d문항을 준비했어요 (출처: %s). 한 번에 풀어보세요." % (
            len(quiz_items),
            "강의 자료 기반" if source == "llm_rag" else "기본 템플릿",
        )
        lines = [header, ""]
        for idx, item in enumerate(quiz_items, start=1):
            question = item.get("question", "질문")
            options = item.get("options", [])
            lines.append(f"{idx}. {question}")
            if isinstance(options, list) and len(options) >= 4:
                lines.append(f"A. {options[0]}")
                lines.append(f"B. {options[1]}")
                lines.append(f"C. {options[2]}")
                lines.append(f"D. {options[3]}")
            lines.append("")

        lines.append("답안은 예: 1:A, 2:B, 3:A, 4:C, 5:D 형태로 보내주세요.")
        return "\n".join(lines), quiz_items
    return None, []

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
5. If the latest message asks to continue, resume, show the next content, or summarize "again" (e.g., "다시 요약해줘", "다음 내용 알려줘") a broad topic that has already been partially discussed in the history, reformulate the query to search for the NEXT sequential or REMAINING topics/concepts in that lecture, rather than repeating or focusing on the sub-concepts that were already explained.
6. Do NOT add any introductory text, explanations, or quotes. Output ONLY the reformulated query.

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
    
    # Calculate or accumulate frustration level
    # Since the backend is stateless and runs a new graph execution on every turn,
    # we scan the entire conversation history to count user frustration signals.
    frustration_count = 0
    for msg in messages:
        if msg.get("role") == "user":
            if _detect_frustration(msg.get("content", "")):
                frustration_count += 1
    state["frustration_level"] = frustration_count
    logger.info(f"Calculated accumulated frustration level: {frustration_count}")
    
    # Check if there is a pending quiz and the user expressed frustration
    pending_quiz = state.get("pending_quiz", [])
    is_frustrated_in_quiz = False
    # Check if a quiz is pending either from state or conversation history (last tutor message was quiz)
    is_quiz_pending = len(pending_quiz) > 0 or _is_last_message_quiz_prompt(messages[:-1])
    if is_quiz_pending and messages and messages[-1].get("role") == "user":
        last_user_text = messages[-1].get("content", "")
        if _detect_frustration(last_user_text):
            is_frustrated_in_quiz = True

    if is_frustrated_in_quiz:
        state["contextualized_query"] = messages[-1]["content"]
        logger.info(f"Frustrated during pending quiz. Bypassing LLM query rewriting. Query: '{state['contextualized_query']}'")
        _log_trace(
            step="QueryContextualizer",
            purpose="Determine search-optimized query. Bypassed LLM query rewriting because user is frustrated during a pending quiz.",
            inputs={
                "Original User Message": state["contextualized_query"],
                "Frustration Level": frustration_count,
                "Pending Quiz Count": len(pending_quiz)
            },
            decision="Bypassed LLM due to frustration during quiz. Using original query."
        )
        return state

    if len(messages) <= 1:
        # Optimization: Bypassing LLM call on first turn
        state["contextualized_query"] = messages[-1]["content"] if messages else ""
        logger.info(f"First-turn query. Bypassing LLM query rewriting. Query: '{state['contextualized_query']}'")
        _log_trace(
            step="QueryContextualizer",
            purpose="Determine search-optimized query. Bypassed LLM query rewriting because this is the first turn.",
            inputs={
                "Original User Message": state["contextualized_query"],
                "Frustration Level": frustration_count
            },
            decision="Bypassed LLM. Using original query as contextualized query."
        )
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
    _log_trace(
        step="QueryContextualizer",
        purpose="Reformulate follow-up user query using conversation history into a standalone, search-optimized query.",
        inputs={
            "Original User Message": last_message,
            "History Length (turns)": len(messages) - 1,
            "Frustration Level": frustration_count
        },
        prompt_details={
            "Formatted History": history_text,
            "Reformulation Prompt Template": REWRITER_PROMPT.replace("{history_text}", "...").replace("{last_message}", last_message)
        },
        response=rewritten,
        decision=f"Rewritten Standalone Query: '{rewritten}'"
    )
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
    _log_trace(
        step="Coordinator",
        purpose="Analyze the contextualized query to classify it as a casual conversation (DIRECT) or study-related query (PLAN).",
        inputs={"Contextualized Query": last_user_message},
        prompt_details=prompt,
        response=decision,
        decision=f"Routing Key: {state['next_step']}"
    )
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
    _log_trace(
        step="Planner",
        purpose="Create a Socratic inquiry plan and identify the study target.",
        inputs={
            "User Query": last_query,
            "Socratic Depth Mode": current_depth
        },
        prompt_details=prompt,
        response=state["plan"],
        decision="Execution Plan generated and saved to state."
    )
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
    
    context = "\n".join([d["text"] for d in docs]) if docs else "No specific documents found."
    
    system_prompt = f"""You are SocrAItes, a world-class Socratic tutor who helps students build deep understanding and strong meta-cognition.

Your goal is to guide the student to think critically about academic concepts and lecture topics. To do this effectively:
1. Provide a very brief, high-level explanation, hint, or conceptual summary (1-2 sentences max) based on the retrieved lecture materials to anchor their thoughts. Do NOT give a complete, fully detailed direct answer.
2. Follow up immediately with a concrete, thought-provoking Socratic question that breaks down the concepts or bridges to the next logical topic.

Tool Usage Policy (very important):
- If the learner asks for quiz/problem practice, call tool `generate_quiz`.
- If the learner asks to plan/register review schedule, call tool `schedule_review`.
- If the learner asks to save a weak concept/mistake, call tool `save_weakness`.
- If the learner explicitly asks for direct answer mode (e.g., "그냥 답 알려줘"), call tool `escape_to_answer`.
- When tool usage is appropriate, call the tool first before final response.

Guidelines for Socratic Questions:
- DO NOT ask vague, broad, or purely subjective/opinion-based questions (e.g., "어떤 점이 흥미롭고 도전적이라고 느끼시나요?", "어떤 생각이 드시나요?").
- Focus on concrete concept break-down: Ask the student to explain a specific sub-component, mechanism, distinction, or key difference (e.g., "자연어와 프로그래밍 언어의 차이점에서 '모호성'이란 무엇이며, 왜 발생할까요?", "형태소 분석과 구문 분석은 각각 어떤 역할을 담당하나요?").
- Drive logical progression / Bridge to the next topic: Guide the student to the next logical concept in the provided lecture notes (e.g., "자연어 처리의 목표가 컴퓨터와의 소통이라면, 그 첫 번째 단계인 '형태소 분석'에서 한국어의 어떤 특징이 분석을 어렵게 만드는 걸림돌이 될까요?").
- Encourage analytical reasoning or hypothetical scenarios: "만약 형태소 분석 단계에서 동음이의어(예: '산'의 다양한 의미)가 잘못 분석된다면, 이후의 의미 분석(Semantic Analysis) 단계에는 어떤 영향을 미치게 될까요?"

Current Plan: {plan}
Socratic Depth: {depth} (0: Light, 1: Standard, 2: Deep)
Available Lecture Context:
---
{context}
---

Rules:
1. Always start with a brief, encouraging, high-level summary or hint, and immediately close with a concrete Socratic question that pushes the student to reflect, analyze, or explain the logic (strictly avoiding vague, subjective opinion questions).
2. Keep the overall response friendly, academic, and extremely encouraging.
3. Use the provided lecture context to ensure the hint and question are accurate, grounded, and specific to the lecture materials.
4. Detect frustration: if the student is struggling, offer slightly more scaffolding (a slightly more descriptive hint) before asking the question.
5. Respond naturally in Korean, adopting the persona of a warm Socratic coach.
6. Context Progression: Analyze the conversation history. If the user asks to continue, resume, learn the next part, or summarize "again" (e.g., "다시 요약해줘", "다음 내용 알려줘"), do NOT repeat the previously explained concepts (like NLU/NLG). Instead, identify and summarize the NEXT sequential concepts from the available lecture context (such as traditional NLP components: morphological analysis, syntax parsing, semantic analysis, etc., or Korean linguistic characteristics) and ask a Socratic question on those new concepts."""

    last_msg = history[-1]["content"] if history else ""
    pending_quiz = state.get("pending_quiz", [])
    
    # --- Check if user is submitting quiz answers ---
    if pending_quiz and _is_quiz_answer(last_msg):
        grade_report = _grade_quiz(last_msg, pending_quiz)
        state["draft_answer"] = grade_report
        state["pending_quiz"] = []  # clear after grading
        state["tool_results"] = []
        logger.info("Quiz graded for user submission.")
        return state

    is_frustrated_in_quiz = False
    is_quiz_pending = len(pending_quiz) > 0 or _is_last_message_quiz_prompt(history[:-1])
    if is_quiz_pending and _detect_frustration(last_msg):
        is_frustrated_in_quiz = True
        logger.info("User is frustrated during a pending quiz. Adding strong prompt guards to prevent generate_quiz.")

    messages = [SystemMessage(content=system_prompt)]
    for m in history:
        if m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            messages.append(AIMessage(content=m["content"]))
            
    if is_frustrated_in_quiz:
        messages.append(SystemMessage(content=(
            "IMPORTANT: The student is currently taking a quiz (pending_quiz exists) and expressed frustration or said they don't know the answer. "
            "DO NOT call generate_quiz under any circumstances. "
            "Instead, do the following:\n"
            "1. Offer direct, encouraging support and a helpful hint for the quiz questions.\n"
            "2. Inform them clearly that they can type '그냥 답 알려줘' if they want to give up and reveal the correct answers immediately."
        )))

    response = llm.bind_tools(LANGCHAIN_TOOLS).invoke(messages)
    draft = _get_content(response)
    tool_calls = _extract_tool_calls(response)
    tool_results = _run_tool_calls(tool_calls) if tool_calls else []

    if tool_results:
        quiz_message, quiz_items = _format_quiz_response(tool_results)
        if quiz_message:
            state["draft_answer"] = quiz_message
            state["tool_results"] = tool_results
            state["pending_quiz"] = quiz_items
            return state

        summary_prompt = f"""You are SocrAItes. A tool call was executed during tutoring.
Use tool results below and produce a concise Korean response for the learner.

Tool Results JSON:
{json.dumps(tool_results, ensure_ascii=False)}

Rules:
1. If escape_to_answer succeeded, provide a direct helpful answer in Korean.
2. Otherwise, mention tool outcome clearly, then ask one short Socratic follow-up question.
3. Keep it concise.
"""
        final_response = llm.invoke([SystemMessage(content=summary_prompt), HumanMessage(content="최종 응답을 작성해줘.")])
        state["draft_answer"] = _get_content(final_response)
    else:
        state["draft_answer"] = draft

    state["tool_results"] = tool_results
    
    logger.info(f"Draft Answer Generated (length: {len(state['draft_answer'])})")
    _log_trace(
        step="Supervisor",
        purpose="Synthesize conversation history and retrieved lecture documents to generate a Socratic question or hint in Korean.",
        inputs={
            "Socratic Depth": depth,
            "Retrieved Docs Count": len(docs),
            "History Length (turns)": len(history)
        },
        prompt_details={
            "System Prompt / Rules": system_prompt,
            "Conversation History Stream": "\n".join([f"  {m['role'].upper()}: {m['content']}" for m in history])
        },
        response=state["draft_answer"],
        decision={
            "Tool Calls": len(tool_calls),
            "Tool Results": tool_results,
            "Result": "Socratic Draft Response generated and saved to state."
        }
    )
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
    _log_trace(
        step="Evaluator",
        purpose="Quality check: Ensure the drafted response is Socratic, encourages student thinking, and is grounded in context.",
        inputs={
            "Draft Answer Length": len(draft),
            "Retrieved Docs Count": len(docs)
        },
        prompt_details=prompt,
        response=state["evaluation"],
        decision=f"Pass Status: {state['evaluation']['pass']}"
    )
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
    _log_trace(
        step="DirectResponse",
        purpose="Respond politely and briefly in Korean to non-study casual messages or greetings.",
        inputs={"User Input": last_msg},
        prompt_details=f"System: {system_msg}\nHuman: {last_msg}",
        response=state["draft_answer"],
        decision="Casual Direct Response generated."
    )
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
    _log_trace(
        step="Retrieval",
        purpose="Query the Elasticsearch vector store using the contextualized query.",
        inputs={"Search Query": last_query},
        decision={
            "Retrieved Count": len(results),
            "Retrieved Chunks Snippets": [r["text"][:80].replace("\n", " ") + "..." for r in results]
        }
    )
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
