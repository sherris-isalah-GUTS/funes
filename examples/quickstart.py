# -*- coding: utf-8 -*-
"""
Funes Quickstart
================
Basic usage of Funes: Ingest conversational turns and perform
ultra-fast, zero-embedding biomimetic retrieval.
"""

from funes import FunesEngine

def main():
    # 1. Initialize Funes in-memory (or pass db_path="funes_memory.db" for persistent storage)
    engine = FunesEngine()

    # 2. Ingest conversational history
    print("Ingesting conversation turns...")
    engine.add_turn(
        session_id="session_001",
        date_str="2026-05-01",
        role="user",
        content="I bought a new 16-inch MacBook Pro in Seattle yesterday for $2,499."
    )
    engine.add_turn(
        session_id="session_001",
        date_str="2026-05-01",
        role="assistant",
        content="That is a powerful workstation for systems programming and architecture!"
    )
    engine.add_turn(
        session_id="session_002",
        date_str="2026-05-12",
        role="user",
        content="Planning a camping trip to Mount Rainier next weekend. Need to bring warm gear."
    )

    # 3. Perform dual-stream biomimetic retrieval (0 GPU, 0 Tokens, ~6ms)
    query = "What laptop did I purchase in Seattle and how much did it cost?"
    print(f"\nQuerying: '{query}'")
    
    result = engine.retrieve(query, reference_date="2026-05-15")

    # 4. Inspect retrieved evidence
    print("\n--- [Exact Facts Stream] ---")
    for turn, score, probe in result.exact_facts:
        print(f"[{turn.date_str}] (Score: {score:.2f}) {turn.role}: {turn.content}")

    # 5. Build high-SNR prompt for downstream LLM
    print("\n--- [Formatted LLM Context Prompt] ---")
    prompt = engine.build_prompt(query, result)
    print(prompt)

if __name__ == "__main__":
    main()
