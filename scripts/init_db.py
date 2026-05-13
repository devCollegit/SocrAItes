# scripts/init_db.py
# -*- coding: utf-8 -*-
"""Initialise (or reset) the SocrAItes SQLite database.

Usage:
    python scripts/init_db.py          # create tables (idempotent)
    python scripts/init_db.py --seed   # create tables + insert sample data
    python scripts/init_db.py --reset  # drop everything and recreate

The DB file is stored at  data/socraites.db  by default.
Override with the SOCRAITES_DB_PATH environment variable.
"""

from __future__ import annotations

import argparse
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Ensure the project root is on sys.path so we can import src.*
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.db.database import (
    DB_PATH,
    get_connection,
    init_db,
    create_session,
    log_message,
    save_weakness,
    add_schedule,
    register_document,
    save_report,
    get_messages,
    get_weaknesses,
    get_pending_schedules,
    get_documents,
    get_reports,
    list_sessions,
)


def _reset_db() -> None:
    """Drop the database file entirely and reinitialise."""
    abs_path = os.path.abspath(DB_PATH)
    if os.path.exists(abs_path):
        os.remove(abs_path)
        print(f"[RESET] Deleted existing database: {abs_path}")
    # Also remove WAL / SHM sidecar files if present.
    for suffix in ("-wal", "-shm"):
        sidecar = abs_path + suffix
        if os.path.exists(sidecar):
            os.remove(sidecar)


def _seed_db() -> None:
    """Insert sample data for quick testing / demo."""
    print("[SEED] Inserting sample data …")

    # 1. Session
    sid = create_session(
        user_id="student_kim",
        title="분산시스템론 Week5 학습",
        socratic_mode="standard",
    )
    print(f"  [OK] Session created: {sid}")

    # 2. Messages
    turns = [
        ("user",      "CAP 정리가 뭔지 잘 모르겠어요"),
        ("assistant", "좋은 질문이에요. 먼저, 분산 시스템에서 일관성과 가용성이 동시에 필요한 상황을 하나 떠올려볼래요?"),
        ("user",      "은행 거래?"),
        ("assistant", "은행 거래에서 만약 네트워크가 끊긴다면, 두 지점이 잔액 정보를 다르게 가지고 있어도 거래를 허용해야 할까요?"),
        ("user",      "음... 잘 모르겠어요"),
        ("assistant", "괜찮아요, 힌트를 드릴게요. CAP의 P는 Partition Tolerance인데, 네트워크 분할 상황에서 시스템은 일관성(C)과 가용성(A) 중 하나를 선택해야 합니다."),
    ]
    for role, content in turns:
        log_message(sid, role, content)
    print(f"  [OK] {len(turns)} messages logged")

    # 3. Weakness
    wid = save_weakness(
        concept="Partition Tolerance",
        details="CAP 정리에서 P의 의미와 실제 적용 사례를 혼동함",
        severity=3,
        session_id=sid,
    )
    print(f"  [OK] Weakness saved (id={wid})")

    # 4. Schedule
    schedule_id = add_schedule(
        review_at="2026-05-16T10:00:00+09:00",
        description="CAP 정리 복습 – Partition Tolerance 중심",
        weakness_id=wid,
    )
    print(f"  [OK] Schedule created (id={schedule_id})")

    # 5. Document
    doc_id = register_document(
        filename="Week5_분산시스템론.pdf",
        num_chunks=42,
        display_name="분산시스템론 Week5",
        file_hash="e3b0c44298fc1c149afbf4c8996fb924",
    )
    print(f"  [OK] Document registered (id={doc_id})")

    # 6. Report
    report_id = save_report(
        title="주간 학습 진단 리포트 (Week 1)",
        body=(
            "## 요약\n"
            "- 학습 세션: 1회\n"
            "- 주요 약점: Partition Tolerance 개념 이해 부족\n"
            "- 권장 복습: CAP 정리 전반, 실제 분산 시스템 사례 학습\n"
        ),
        user_id="student_kim",
    )
    print(f"  [OK] Report saved (id={report_id})")


def _verify_db() -> None:
    """Print a summary of all table contents to verify the DB."""
    print("\n" + "=" * 60)
    print("  DATABASE VERIFICATION")
    print("=" * 60)

    sessions = list_sessions("student_kim")
    print(f"\n[Sessions] {len(sessions)} row(s)")
    for s in sessions:
        print(f"  - {s['id'][:8]}... | mode={s['socratic_mode']} | {s['title']}")

    if sessions:
        sid = sessions[0]["id"]
        msgs = get_messages(sid)
        print(f"\n[Messages] {len(msgs)} row(s) in session {sid[:8]}...")
        for m in msgs:
            preview = m["content"][:50].replace("\n", " ")
            print(f"  {m['role']:>10}: {preview}...")

    weaks = get_weaknesses()
    print(f"\n[Weaknesses] {len(weaks)} row(s)")
    for w in weaks:
        print(f"  - [{w['severity']}] {w['concept']} (resolved={w['resolved']})")

    scheds = get_pending_schedules()
    print(f"\n[Schedules] {len(scheds)} pending")
    for s in scheds:
        print(f"  - {s['review_at']} - {s['description']}")

    docs = get_documents()
    print(f"\n[Documents] {len(docs)} row(s)")
    for d in docs:
        print(f"  - {d['display_name']} ({d['num_chunks']} chunks)")

    reports = get_reports("student_kim")
    print(f"\n[Reports] {len(reports)} row(s)")
    for r in reports:
        print(f"  - {r['title']}")

    print("\n" + "=" * 60)
    print(f"  DB location: {os.path.abspath(DB_PATH)}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SocrAItes DB initialiser")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate DB")
    parser.add_argument("--seed", action="store_true", help="Insert sample data after init")
    args = parser.parse_args()

    if args.reset:
        _reset_db()

    print(f"[INIT] Initialising database at {os.path.abspath(DB_PATH)} ...")
    init_db()
    print("[INIT] [OK] All tables created successfully.")

    if args.seed:
        _seed_db()

    _verify_db()


if __name__ == "__main__":
    main()
