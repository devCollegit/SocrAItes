# scratch/test_filter.py
import io
import sys
import os

# Set output encoding to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.rag.vectorstore import query, SIMILARITY_THRESHOLD

def main():
    print(f"Current Similarity Threshold: {SIMILARITY_THRESHOLD}")
    
    # Test 1: Completely unrelated query (should be filtered out and return empty list)
    print("\n--- Test 1: Searching for '데드락 예방 방법' (Unrelated to index contents) ---")
    results_unrelated = query("데드락 예방 방법", k=3)
    print(f"Retrieved Chunks Count: {len(results_unrelated)}")
    for i, (text, score) in enumerate(results_unrelated, 1):
        print(f"[{i}] score={score:.4f} | {text[:150]}...")
        
    # Test 2: Highly related query (should NOT be filtered and return chunks)
    print("\n--- Test 2: Searching for '자연어처리 정의 및 NLU' (Highly related to index contents) ---")
    results_related = query("자연어처리 정의 및 NLU", k=3)
    print(f"Retrieved Chunks Count: {len(results_related)}")
    for i, (text, score) in enumerate(results_related, 1):
        print(f"[{i}] score={score:.4f} | {text[:150]}...")

if __name__ == "__main__":
    main()
