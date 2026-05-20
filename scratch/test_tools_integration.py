#!/usr/bin/env python3
"""
Integration test for learning tools (quiz, schedule, weakness, direct answer mode)
Tests tool calling via /chat endpoint with different prompts
"""
import requests
import json
import time
from typing import List, Dict, Any

BASE_URL = "http://localhost:8000"
ENDPOINT = f"{BASE_URL}/chat"

# Test cases: (description, user_message, expected_tool)
TEST_CASES = [
    (
        "Test 1: Generate Quiz",
        "데드락 현상에 대해 퀴즈를 만들어줄래?",
        "generate_quiz"
    ),
    (
        "Test 2: Schedule Review",
        "내일 오전 10시에 상호배제 개념 복습을 예약해줘",
        "schedule_review"
    ),
    (
        "Test 3: Save Weakness",
        "뮤텍스와 세마포어의 차이를 아직 잘 이해 못해. 약점으로 저장해줄래?",
        "save_weakness"
    ),
    (
        "Test 4: Direct Answer Mode",
        "그냥 세마포어가 뭔지 설명 좀 해줄래? 답변 모드로",
        "escape_to_answer"
    ),
    (
        "Test 5: Socratic (no tool)",
        "프로세스와 스레드의 차이가 뭐야?",
        None  # Expected to use regular response without tool
    ),
]

def test_tool_call(test_case: tuple, session_id: str = None):
    """Test a single tool call"""
    description, message, expected_tool = test_case
    
    if session_id is None:
        session_id = f"test_{int(time.time() * 1000)}"
    
    payload = {
        "messages": [
            {"role": "user", "content": message}
        ],
        "socratic_depth": 1,
        "session_id": session_id
    }
    
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'-'*70}")
    print(f"User: {message}")
    print(f"Session: {session_id}")
    
    try:
        # SSE streaming request
        response = requests.post(ENDPOINT, json=payload, stream=True, timeout=60)
        response.raise_for_status()
        
        final_result = None
        node_outputs = []
        
        # Parse SSE stream
        for line in response.iter_lines():
            if line and line.startswith(b"data: "):
                try:
                    data = json.loads(line[6:])
                    
                    if data.get("type") == "final_result":
                        final_result = data
                    elif data.get("type") == "node_end":
                        node_name = data.get("node")
                        output = data.get("output", {})
                        if output:
                            node_outputs.append((node_name, output))
                            
                except json.JSONDecodeError:
                    pass
        
        # Display results
        if final_result:
            print(f"\n✅ Response received:")
            print(f"  Answer: {final_result.get('answer', 'N/A')[:200]}...")
            
            tool_results = final_result.get("tool_results", [])
            if tool_results:
                print(f"\n🔧 Tool Results:")
                for tool_result in tool_results:
                    print(f"  Tool: {tool_result.get('tool')}")
                    print(f"  Status: {'OK' if tool_result.get('ok') else 'ERROR'}")
                    output = tool_result.get('output') or tool_result.get('error')
                    print(f"  Output: {str(output)[:150]}")
                    
                    # Verify expected tool was called
                    if expected_tool and tool_result.get('tool') == expected_tool:
                        print(f"  ✅ Expected tool '{expected_tool}' was called!")
            else:
                print(f"\n❌ No tool was triggered (expected: {expected_tool})")
        else:
            print(f"❌ No response received")
            
    except requests.exceptions.Timeout:
        print(f"⏱️  Timeout (60s) - likely embedding/inference delay")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("LEARNING TOOLS INTEGRATION TEST")
    print("="*70)
    print(f"Testing: {len(TEST_CASES)} scenarios")
    print(f"Endpoint: {ENDPOINT}")
    print("="*70)
    
    # Health check
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5)
        health.raise_for_status()
        print(f"✅ Backend is healthy")
    except Exception as e:
        print(f"❌ Backend is not accessible: {e}")
        return
    
    # Run tests
    for i, test_case in enumerate(TEST_CASES, 1):
        test_tool_call(test_case)
        
        # Small delay between tests
        if i < len(TEST_CASES):
            print(f"\n⏳ Waiting 2s before next test...")
            time.sleep(2)
    
    print(f"\n{'='*70}")
    print("TEST SUITE COMPLETED")
    print("="*70)

if __name__ == "__main__":
    main()
