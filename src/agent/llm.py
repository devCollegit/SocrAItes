"""LLM instance and tool-call utilities shared across all agent nodes."""
import os
import json
import logging
from typing import Any, List, Dict

from langchain_openai import ChatOpenAI
from src.tools.learning_tools import TOOL_MAP, LANGCHAIN_TOOLS

logger = logging.getLogger("SocrAItes.Agent")

# ---------------------------------------------------------------------------
# LLM instance
# ---------------------------------------------------------------------------

if os.getenv("OPENAI_API_KEY"):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # 성능 핵심 노드(socratic_agent) 전용 상위 모델
    llm_strong = ChatOpenAI(model="gpt-4o", temperature=0)
else:
    # Mock LLM for testing frontend when API key is missing
    from langchain_core.language_models.fake import FakeListLLM
    llm = FakeListLLM(responses=[
        '{"rewritten_query": "안녕하세요!", "route": "chat", "active_agents": [], "subtask": "인사 응답"}',
        "안녕하세요! 무엇을 공부하고 싶으신가요?",
        '{"rewritten_query": "CAP 정리에 대해 알고 싶어요.", "route": "learn", "active_agents": ["retrieval", "socratic"], "subtask": "CAP 정리 탐구"}',
        "How would you explain the CAP theorem in your own words?",
        '{"scores": {"socratic": 4, "grounding": 3, "encouragement": 4, "clarity": 4}, "pass": true, "feedback": ""}',
    ])
    llm_strong = llm  # API 키 없으면 동일 mock 사용


# ---------------------------------------------------------------------------
# Response utilities
# ---------------------------------------------------------------------------

def _get_content(response: Any) -> str:
    if hasattr(response, "content"):
        return response.content
    return str(response)


def _extract_tool_calls(response: Any) -> List[Dict[str, Any]]:
    tool_calls = getattr(response, "tool_calls", None)
    if tool_calls:
        return tool_calls
    # Fallback to additional_kwargs (older LangChain versions)
    additional_kwargs = getattr(response, "additional_kwargs", {}) or {}
    return additional_kwargs.get("tool_calls", [])


def _run_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for tc in tool_calls:
        tool_name = tc.get("name")
        args = tc.get("args", {})

        # Fallback: OpenAI API format
        if not tool_name:
            fn_info = tc.get("function", {})
            tool_name = fn_info.get("name")
            raw_args = fn_info.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            except json.JSONDecodeError:
                args = {}

        if not tool_name:
            results.append({"tool": None, "ok": False, "error": "Tool name not found"})
            continue

        tool_fn = TOOL_MAP.get(tool_name)
        if not tool_fn:
            results.append({"tool": tool_name, "ok": False, "error": f"Unknown tool: {tool_name}"})
            continue

        try:
            logger.info(f"Executing tool: {tool_name} with args: {args}")
            output = tool_fn(args)
            results.append({"tool": tool_name, "ok": True, "output": output})
        except Exception as e:
            logger.exception("Tool execution failed: %s", tool_name)
            results.append({"tool": tool_name, "ok": False, "error": str(e)})

    return results
