import logging
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.agent.llm import llm, _extract_tool_calls, _run_tool_calls

logger = logging.getLogger("SocrAItes.Agent")

# 백그라운드 프로필 업데이트용 확장된 프롬프트
BACKGROUND_PROMPT = """You are the Background Profile Updater for SocrAItes.

최근 대화를 분석해 `update_user_profile` 도구를 호출해 학생 프로필을 업데이트하세요.

반드시 아래 요소들을 관리하세요:
1. strengths_summary: 현재 개념 숙달 목록 요약 (1-3문장)
2. weaknesses_summary: 현재 개념 혼동/오류 요약 (1-3문장)
3. notes: 학습 관련 외에 발견된 개인 신상(이름, 나이, 직업 등), 현재 상황(시험 기간, 기분 등), 학습 목표를 자연스럽게 기록 및 누적.

업데이트 규칙 (CRITICAL):
1. 개념 오류/혼동/잘못된 이해 → weaknesses_summary에 추가/업데이트
2. 개념 정확 설명/오개념 자가 수정/숙달 → strengths_summary에 추가, weaknesses_summary에서 해당 항목 제거
3. 단순 인사나 잡담에서 개인 정보나 현재 기분, 상태가 나오면 notes 필드에 기존 내용을 유지하며 추가. 정보가 없다면 기존 값을 유지하세요.
4. 업데이트할 내용이 전혀 없으면 빈 문자열을 반환하고 도구를 호출하지 마세요. 업데이트할 내용이 있으면 반드시 `update_user_profile` 호출. 텍스트 설명 불필요.

현재 프로필:
- Academic Background: {academic_background}
- Learning Style: {learning_style}
- Preferred Tone: {preferred_tone}
- AI Notes: {profile_notes}
- Current Strengths Summary: "{current_strengths_summary}"
- Current Weaknesses Summary: "{current_weaknesses_summary}"

Session ID: {session_id}
User ID: {user_id}"""


def update_profile_in_background(
    session_id: str,
    user_id: str,
    messages: List[Dict[str, Any]],
) -> None:
    """API 응답 전송 후 daemon thread에서 사용자 프로필을 묵시적으로 업데이트한다.

    LLM + update_user_profile 도구를 사용해 강점/약점/신상정보를 자동으로 갱신한다.
    daemon=True로 실행하므로 메인 프로세스 종료 시 함께 종료된다.
    """
    logger.info(
        f"--- [Background Profile Update] session={session_id}, user={user_id} ---"
    )

    # 현재 프로필 로드 (백그라운드이므로 실패해도 무시)
    try:
        from src.db.database import get_user_profile
        profile = get_user_profile(user_id)
    except Exception as e:
        logger.error(f"[BackgroundUpdate] 프로필 로드 실패: {e}")
        return

    system_content = BACKGROUND_PROMPT.format(
        session_id=session_id,
        user_id=user_id,
        academic_background=profile.get("academic_background", "대학원생"),
        learning_style=profile.get("learning_style", "conceptual"),
        preferred_tone=profile.get("preferred_tone", "encouraging"),
        profile_notes=profile.get("notes", ""),
        current_strengths_summary=profile.get("strengths_summary", ""),
        current_weaknesses_summary=profile.get("weaknesses_summary", ""),
    )

    # 대화 히스토리를 LangChain 메시지 형식으로 변환
    messages_for_llm = [SystemMessage(content=system_content)]
    for m in messages:
        if m["role"] == "user":
            messages_for_llm.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            messages_for_llm.append(AIMessage(content=m["content"]))

    try:
        # update_user_profile 도구만 바인딩해서 프로필 업데이트에 집중
        from src.tools.learning_tools import LANGCHAIN_TOOLS as ALL_TOOLS
        profiling_tools = [t for t in ALL_TOOLS if t.name == "update_user_profile"]

        response = llm.bind_tools(profiling_tools).invoke(messages_for_llm)
        tool_calls = _extract_tool_calls(response)

        if tool_calls:
            # user_id를 모든 tool call에 주입
            for tc in tool_calls:
                args = tc.get("args", {})
                if isinstance(args, dict):
                    if tc.get("name") == "update_user_profile":
                        args["user_id"] = user_id
                    tc["args"] = args
            logger.info(f"[BackgroundUpdate] {len(tool_calls)}개 프로필 도구 실행 중...")
            results = _run_tool_calls(tool_calls)
            logger.info(f"[BackgroundUpdate] 완료: {results}")
        else:
            logger.info("[BackgroundUpdate] 업데이트할 프로필 변경 없음.")
    except Exception as e:
        logger.error(f"[BackgroundUpdate] 실패: {e}")
