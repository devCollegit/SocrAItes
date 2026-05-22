#!/usr/bin/env python3
"""
Optimized integration test for learning tools
Tests tool calling with proper SSE parsing and longer timeout
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"
ENDPOINT = f"{BASE_URL}/chat"

# Simpler test cases
TESTS = [
    ("Test 1: Quiz", "퀴즈 만들어줄래?"),
    ("Test 2: Schedule", "내일 10시에 복습 등록해줘"),
    ("Test 3: Weakness", "약점으로 저장해줄래"),
    ("Test 4: Direct Answer", "그냥 답해줄래"),
]

def parse_sse_response(response):
    """Parse SSE streaming response"""
    final_result = None
    node_count = 0
    
    for line in response.iter_lines():
        if line and line.startswith(b"data: "):
            try:
                data = json.loads(line[6:].decode('utf-8'))
                if data.get("type") == "final_result":
                    final_result = data
                elif data.get("type") == "node_end":
                    node_count += 1
            except:
                pass
    
    return final_result, node_count

def run_test(description, message):
    """Run a single test"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"Message: {message}")
    print(f"{'-'*60}")
    
    payload = {
        "messages": [{"role": "user", "content": message}],
        "socratic_depth": 1
    }
    
    try:
        print("Sending request...")
        response = requests.post(ENDPOINT, json=payload, stream=True, timeout=120)
        response.raise_for_status()
        
        print("Parsing streaming response...")
        final_result, node_count = parse_sse_response(response)
        
        if final_result:
            print(f"✅ Response received ({node_count} nodes)")
            print(f"   Answer: {final_result.get('answer', 'N/A')[:100]}...")
            
            tools = final_result.get("tool_results", [])
            if tools:
                print(f"   🔧 Tools called: {len(tools)}")
                for t in tools:
                    print(f"      - {t.get('tool')}: {'OK' if t.get('ok') else 'ERROR'}")
            else:
                print(f"   ℹ️  No tools triggered")
        else:
            print(f"❌ No final result received")
            
    except requests.exceptions.Timeout:
        print(f"⏱️  Timeout (120s)")
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n" + "="*60)
print("LEARNING TOOLS TEST")
print("="*60)

# Health check
try:
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    print(f"✅ Backend healthy")
except:
    print(f"❌ Backend not accessible")
    exit(1)

# Run tests
for desc, msg in TESTS:
    run_test(desc, msg)
    time.sleep(1)

print(f"\n{'='*60}")
print("TEST COMPLETED")
print("="*60)
