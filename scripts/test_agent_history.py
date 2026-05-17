# scripts/test_agent_history.py
"""Test script for SocrAItes Core Agent in a multi-turn conversation.
Simulates a conversation history and runs the LangGraph workflow to verify context-aware query reformulation.
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

def safe_print(text: str):
    """Safely print text to terminal, replacing unencodable characters on Windows."""
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or 'utf-8'
        print(text.encode(encoding, errors='replace').decode(encoding))

def test_multiturn_conversation(turns: list):
    safe_print("\n" + "=" * 60)
    safe_print("STARTING MULTI-TURN CONVERSATION TEST")
    safe_print("=" * 60)
    
    # Initialize rolling state
    state = DEFAULT_STATE.copy()
    state["messages"] = []
    
    run_graph = GRAPH.compile()
    
    for i, user_query in enumerate(turns, start=1):
        safe_print(f"\n--- Turn {i} ---")
        safe_print(f"[USER]: {user_query}")
        
        # Append the new user message to the rolling history
        state["messages"].append({"role": "user", "content": user_query})
        
        # Run the graph
        final_state = run_graph.invoke(state)
        
        # Display results
        safe_print(f"[CONTEXTUALIZED QUERY]: '{final_state.get('contextualized_query')}'")
        safe_print(f"[COORDINATOR ROUTE]: {final_state.get('next_step')}")
        if final_state.get("plan"):
            safe_print(f"[PLAN]: {final_state.get('plan')}")
        
        if final_state.get("retrieved_docs"):
            safe_print(f"[RETRIEVED DOCS]: Found {len(final_state['retrieved_docs'])} chunks")
            if len(final_state['retrieved_docs']) > 0:
                snippet = final_state['retrieved_docs'][0][0][:100].replace('\n', ' ')
                safe_print(f"  -> Top Snippet: {snippet}...")
            
        safe_print(f"[SOCRAITES]: {final_state.get('draft_answer')}")
        safe_print("-" * 50)
        
        # Prepare state for the next turn by updating messages with SocrAItes' response
        state = final_state.copy()
        state["messages"].append({"role": "assistant", "content": final_state.get("draft_answer", "")})

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment.")
        sys.exit(1)
        
    # Simulated conversation turns:
    # 1. Greeting (first turn - should bypass rewriter or evaluate as greeting)
    # 2. Concept introduction (study topic)
    # 3. Follow-up with pronouns/context (needs query contextualization!)
    # 4. Another follow-up (needs query contextualization!)
    conversation = [
        "안녕하세요!",
        "OS에서 데드락에 대해 알고 싶어요.",
        "그거 발생 조건이 뭐야?",
        "예방(prevention)은 어떻게 해?"
    ]
    
    test_multiturn_conversation(conversation)
