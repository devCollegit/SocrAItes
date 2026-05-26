"""Core Agent — Responder: 캐주얼(비학습) 메시지에 대한 간단한 응답 생성.

기존 evaluator.py 내 direct_response 함수를 독립 모듈로 분리.
필드명: draft_answer → response
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.state import AgentState
from src.agent.llm import llm, _get_content
from src.agent.helpers import _log_trace

logger = logging.getLogger("SocrAItes.Agent")

# 캐주얼 응답용 시스템 메시지: 친근하고 간결한 한국어 응답을 유도
_RESPONDER_SYSTEM = (
    "You are a friendly academic assistant. "
    "Respond to the user's greeting or casual talk briefly in Korean."
)


def responder(state: AgentState) -> AgentState:
    """chat route에서 호출. 인사/잡담에 짧고 친근한 한국어 응답을 생성한다.

    학습 내용을 다루지 않으므로 retrieval, socratic, reviewer 노드를 거치지 않는다.
    응답은 state["response"]에 저장되고 그래프는 END로 종료된다.
    """
    logger.info("--- [Responder] Step ---")
    last_msg = state["messages"][-1]["content"]

    # LLM에 시스템 + 사용자 메시지를 직접 전달해 간결한 응답을 얻는다
    state["response"] = _get_content(
        llm.invoke([SystemMessage(content=_RESPONDER_SYSTEM), HumanMessage(content=last_msg)])
    )

    _log_trace(
        step="Responder",
        purpose="인사/잡담에 대한 간단한 한국어 응답 생성.",
        inputs={"Input": last_msg},
        response=state["response"],
    )
    return state
