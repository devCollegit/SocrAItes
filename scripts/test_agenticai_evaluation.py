#!/usr/bin/env python3
"""SocrAItes Agentic AI 평가 스크립트.

이 스크립트는 Agentic AI 평가 기준을 기준으로 현재 프로젝트를 점검한다.
- 추론 과정: 소크라테스식 질문, 근거성, 자기 수정 여부
- 행동 및 도구 사용: tools 라우팅, 도구 호출 결과
- 최종 출력: 직답 회피, 자연스러운 한국어, 안전성 징후
- 운영 및 검증: /report-data, /generate-report 응답 여부

실행 전제:
- 로컬 FastAPI 서버가 http://localhost:8000 에서 실행 중이어야 한다.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import requests


BASE_URL = "http://localhost:8000"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CRITERIA_FILE = PROJECT_ROOT / "docs" / "AgenticAI_Evaluation_Criteria.md"


@dataclass
class ScoreItem:
    name: str
    score: float
    max_score: float = 5.0
    note: str = ""

    @property
    def passed(self) -> bool:
        return self.score >= 3.0


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _call_chat(query: str, depth: int = 1) -> Dict[str, Any]:
    resp = requests.post(
        f"{BASE_URL}/chat",
        json={"messages": [{"role": "user", "content": query}], "socratic_depth": depth},
        stream=True,
        timeout=60,
    )

    final: Dict[str, Any] = {}
    events: List[Dict[str, Any]] = []
    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line or not raw_line.startswith("data: "):
            continue
        payload = raw_line[len("data: ") :]
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        events.append(event)
        if event.get("type") == "final_result":
            final = event

    return {
        "status_code": resp.status_code,
        "final": final,
        "events": events,
        "answer": final.get("answer", ""),
        "route": final.get("route", "unknown"),
        "retrieved_docs": len(final.get("retrieved_docs", [])),
        "evaluation": final.get("evaluation", {}),
        "tool_results": final.get("tool_results", []),
        "error": None if resp.status_code == 200 else f"HTTP {resp.status_code}",
    }


def _call_json(method: str, path: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    url = f"{BASE_URL}{path}"
    if method == "GET":
        resp = requests.get(url, params=payload, timeout=30)
    else:
        resp = requests.post(url, json=payload or {}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def evaluate_reasoning() -> ScoreItem:
    cases = [
        "Attention이 뭐야?",
        "InstructGPT가 왜 중요한지 설명해줘",
        "그냥 답 알려줘",
    ]

    direct_answers = 0
    question_like = 0
    grounded = 0
    for query in cases:
        result = _call_chat(query, depth=1)
        answer = result["answer"]
        if any(token in answer for token in ["정답은", "답은", "맞습니다", "맞아요"]):
            direct_answers += 1
        if "?" in answer or any(token in answer for token in ["왜", "어떻게", "생각해"]):
            question_like += 1
        if result["retrieved_docs"] > 0:
            grounded += 1

    score = 5.0
    score -= direct_answers * 1.0
    score -= max(0, 1 - question_like / len(cases)) * 1.5
    score -= max(0, 1 - grounded / len(cases)) * 1.0
    score = max(1.0, min(5.0, score))
    return ScoreItem(
        name="추론 과정",
        score=round(score, 1),
        note=f"direct_answers={direct_answers}, question_like={question_like}, grounded={grounded}/{len(cases)}",
    )


def evaluate_tool_use() -> ScoreItem:
    cases = {
        "퀴즈 내줘": "tools",
        "약점으로 저장해줘": "tools",
        "복습 일정 잡아줘": "tools",
    }

    hits = 0
    results: List[str] = []
    for query, expected_route in cases.items():
        result = _call_chat(query, depth=1)
        route = result["route"]
        if route == expected_route:
            hits += 1
        results.append(f"{query}:{route}")

    score = 2.0 + hits * 1.0
    score = max(1.0, min(5.0, score))
    return ScoreItem(name="행동 및 도구 사용", score=round(score, 1), note=f"routes={', '.join(results)}")


def evaluate_output_quality() -> ScoreItem:
    result = _call_chat("GPT 시리즈 모델의 발전 흐름을 설명해줘", depth=1)
    answer = result["answer"]

    score = 5.0
    if len(answer) < 40:
        score -= 1.0
    if any(token in answer for token in ["정답은", "답은"]):
        score -= 0.5
    if not any(token in answer for token in ["왜", "어떻게", "생각해"]):
        score -= 0.5
    if any(token in answer.lower() for token in ["\n\n\n", "<answer>", "<feedback>"]):
        score -= 0.5

    score = max(1.0, min(5.0, score))
    return ScoreItem(name="최종 생성 결과", score=round(score, 1), note=f"len={len(answer)}, route={result['route']}")


def evaluate_operations() -> ScoreItem:
    checks: List[str] = []
    ok = 0

    try:
        report_data = _call_json("GET", "/report-data", {"user_id": "default"})
        checks.append(f"report-data:{bool(report_data.get('stats'))}")
        ok += 1 if report_data.get("stats") else 0
    except Exception as exc:
        checks.append(f"report-data:error:{exc}")

    try:
        report = _call_json("POST", "/generate-report", {"user_id": "default"})
        checks.append(f"generate-report:{bool(report.get('body'))}")
        ok += 1 if report.get("body") else 0
    except Exception as exc:
        checks.append(f"generate-report:error:{exc}")

    score = 2.0 + ok * 1.5
    score = max(1.0, min(5.0, score))
    return ScoreItem(name="검증 환경 및 운영", score=round(score, 1), note="; ".join(checks))


def load_criteria() -> str:
    try:
        return CRITERIA_FILE.read_text(encoding="utf-8")
    except Exception:
        return ""


def main() -> int:
    if not CRITERIA_FILE.exists():
        print(f"[ERROR] 기준 파일을 찾을 수 없습니다: {CRITERIA_FILE}")
        return 1

    print("=" * 80)
    print("SocrAItes Agentic AI 평가")
    print(f"실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"기준 파일: {CRITERIA_FILE}")
    print("=" * 80)

    scores = [
        evaluate_reasoning(),
        evaluate_tool_use(),
        evaluate_output_quality(),
        evaluate_operations(),
    ]

    total = sum(item.score for item in scores)
    passed = sum(1 for item in scores if item.passed)

    print("\n[평가 결과]")
    for item in scores:
        status = "PASS" if item.passed else "FAIL"
        print(f"- {item.name}: {item.score:.1f}/5.0 [{status}] | {item.note}")

    print("\n[종합]")
    print(f"- 평균 점수: {total / len(scores):.2f}/5.0")
    print(f"- 통과 항목 수: {passed}/{len(scores)}")

    report_lines = [
        "# SocrAItes Agentic AI 평가 리포트",
        f"- 실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 기준 파일: {CRITERIA_FILE.name}",
        "",
    ]
    for item in scores:
        report_lines.append(f"- {item.name}: {item.score:.1f}/5.0 | {item.note}")
    report_lines.extend([
        "",
        f"- 평균 점수: {total / len(scores):.2f}/5.0",
        f"- 통과 항목 수: {passed}/{len(scores)}",
    ])

    out_path = PROJECT_ROOT / "docs" / "AgenticAI_Runtime_Evaluation_Report.md"
    out_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n[저장] {out_path}")

    return 0 if passed >= 3 else 2


if __name__ == "__main__":
    sys.exit(main())
