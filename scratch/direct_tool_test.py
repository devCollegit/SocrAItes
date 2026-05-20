#!/usr/bin/env python3
"""Quick direct test for tool calling"""
import requests
import json

def test_tool(message):
    data = {
        'messages': [{'role': 'user', 'content': message}],
        'socratic_depth': 1
    }
    
    print(f"Testing: {message}")
    try:
        r = requests.post('http://localhost:8000/chat', json=data, stream=True, timeout=90)
        print(f"Status: {r.status_code}")
        
        for line in r.iter_lines():
            if line and b'final_result' in line:
                final = json.loads(line[6:])
                tools = final.get('tool_results', [])
                if tools:
                    for t in tools:
                        print(f"  Tool: {t.get('tool')}, OK: {t.get('ok')}")
                        if not t.get('ok'):
                            print(f"  Error: {t.get('error')}")
                else:
                    print("  No tool triggered")
                break
    except Exception as e:
        print(f"Error: {e}")

# Test cases
tests = [
    "퀴즈 만들어줄래?",
    "내일 10시에 복습",
    "약점으로 저장",
]

for msg in tests:
    test_tool(msg)
    print()
