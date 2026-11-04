# -*- coding: utf-8 -*-
"""
Funes Edge Search Engine Demo
=============================
Demonstrates how Funes serves as a zero-embedding, zero-VRAM
instant local document search engine.
"""

from funes import FunesEngine
import datetime

def main():
    print("Initializing Funes Edge Search Engine...")
    engine = FunesEngine()

    # Ingest local articles or notes as chronological documents
    docs = [
        ("doc_1", "2026-01-10", "Quantum Computing Breakthrough: IBM announces new 1,000-qubit Condor chip with error mitigation."),
        ("doc_2", "2026-02-14", "Personal Diary: Upgraded server cluster with 128GB ECC RAM. Investigated high memory latency in B-tree lookups."),
        ("doc_3", "2026-03-22", "Clinical Research Note: Diclofenac sodium suppositories administered for acute fever relief in pediatric settings."),
        ("doc_4", "2026-04-05", "Travel Log: Flight to Zurich booked for conference on biomimetic algorithms and computational neuroscience."),
        ("doc_5", "2026-05-18", "System Architecture Note: Replaced continuous vector database with discrete chromatin phase separation, dropping RAM to 18MB.")
    ]

    for doc_id, date_str, text in docs:
        engine.add_turn(
            session_id=doc_id,
            date_str=date_str,
            role="document",
            content=text
        )
    print(f"Successfully indexed {len(docs)} documents natively on CPU (0 tokens, 0 embeddings).\n")

    # Perform instantaneous discrete search
    queries = [
        "What medication was used for acute fever relief?",
        "When was the Zurich conference on biomimetic algorithms?",
        "How did we drop RAM to 18MB?"
    ]

    for q in queries:
        print(f"Search Query: \"{q}\"")
        res = engine.retrieve(q, reference_date="2026-06-01")
        
        matches = [t for t, sc, pr in res.exact_facts] + [t for t, sc, pr in res.fuzzy_contexts]
        seen = set()
        for t in matches:
            if t.turn_id not in seen:
                seen.add(t.turn_id)
                print(f"  -> [{t.date_str}] {t.content}")
        print()

if __name__ == "__main__":
    main()
