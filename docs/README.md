# SocrAItes Docs Guide

이 문서는 SocrAItes 문서의 정본(Single Source of Truth) 가이드입니다.
여러 개발자/LLM이 동시에 작업할 때 문서 불일치를 줄이기 위해, 최신 기준 문서와 레거시 문서를 분리합니다.

## 1) 먼저 읽을 문서 (Current)

1. `System_Design.md`
   - 시스템 아키텍처, 데이터 흐름, 런타임 구성
2. `Agent_Workflow.md`
   - LangGraph 노드 흐름 및 응답 생성 로직
3. `Project_Structure.md`
   - 현재 코드베이스 디렉토리 구조
4. `Backend_Overview.md`
   - 모듈 단위 구현 개요
5. `SRS.md`
   - 요구사항 및 비기능 목표

## 2) 레거시 문서

과거 설계/실험 문서는 `legacy/` 하위에 보관합니다.
현재 개발 판단의 기준으로 사용하지 말고, 히스토리 참고 용도로만 사용하세요.

## 2-1) 보조 문서

1. `Agent_Streaming_Implementation.md`
   - 스트리밍 UI/응답 설계 메모
   - 현재 정본 아키텍처 문서는 아니며, 기능 확장 참고용

## 3) 문서 운영 규칙 (LLM 협업 권장)

1. 아키텍처 변경 시 `System_Design.md`를 가장 먼저 갱신합니다.
2. 워크플로우 변경 시 `Agent_Workflow.md`를 함께 갱신합니다.
3. 파일/모듈 구조 변경 시 `Project_Structure.md`를 갱신합니다.
4. 과거 문서를 덮어쓰지 말고 `legacy/`로 이동합니다.
5. 동일 주제의 문서를 중복 생성하지 않습니다. 기존 정본 문서에 통합합니다.
