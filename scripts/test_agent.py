# scripts/test_agent.py
"""Test script for the SocrAItes Core Agent.
Simulates a user query and runs the LangGraph workflow.
"""

import os
import sys
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

load_dotenv()

from src.agent.graph import GRAPH
from src.agent.state import DEFAULT_STATE

def test_query(query: str):
    print(f"\n[USER]: {query}")
    print("-" * 50)
    
    # Initialize state
    state = DEFAULT_STATE.copy()
    state["messages"] = [{"role": "user", "content": query}]
    
    # Run the graph
    # graph.invoke returns the final state
    run_graph = GRAPH.compile()
    final_state = run_graph.invoke(state)
    
    print(f"[COORDINATOR ROUTE]: {final_state.get('next_step')}")
    if final_state.get("plan"):
        print(f"[PLAN]: {final_state.get('plan')}")
    
    if final_state.get("retrieved_docs"):
        print(f"[RETRIEVED DOCS]: Found {len(final_state['retrieved_docs'])} chunks")
        
    print(f"[SOCRAITES]: {final_state.get('draft_answer')}")
    print("-" * 50)
    if final_state.get("evaluation"):
        print(f"[EVALUATION]: {final_state['evaluation']}")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment.")
        sys.exit(1)
        
    # Test 1: Casual greeting
    test_query("안녕하세요!")
    
    # Test 2: Study query (assuming DB is seeded or it will just say no docs found)
    test_query("CAP 정리에 대해 알고 싶어요.")
