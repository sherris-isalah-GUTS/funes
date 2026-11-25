# -*- coding: utf-8 -*-
"""
Funes Benchmark Suite: LongMemEval-S Evaluation
==============================================
Reproduces the empirical retrieval benchmark on the ICLR 2025 LongMemEval-S suite.

Usage:
    python benchmark.py --limit 20       # Fast smoke test across 20 questions
    python benchmark.py                  # Full 500-question benchmark evaluation
    python benchmark.py --data path.json # Custom dataset path
"""

import argparse
import collections
import json
import os
import re
import sys
import time

from funes import FunesEngine, STOP_WORDS

def normalize_text(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return " ".join(t.split())

def evaluate(data_file: str, limit: int = None):
    if not os.path.exists(data_file):
        print(f"Error: Dataset file not found at: {data_file}")
        print("Please download longmemeval_s.json and place it in the data/ directory,")
        print("or pass the path via: python benchmark.py --data <path_to_json>")
        return

    print(f"Loading LongMemEval-S dataset from: {data_file}")
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_q = len(data)
    eval_subset = data[:limit] if limit else data
    eval_count = len(eval_subset)

    print(f"\n{'='*75}")
    print(f"  Funes 1.1.0 Empirical Evaluation on LongMemEval-S")
    print(f"  Total Questions: {eval_count} / {total_q}")
    print(f"{'='*75}\n")

    ingest_latencies = []
    retrieval_latencies = []
    by_type = collections.defaultdict(lambda: {
        "total": 0, "full_sess": 0, "any_sess": 0, "gold_norm": 0, "latencies": []
    })

    t_all_start = time.time()

    for idx, q in enumerate(eval_subset):
        qid = q["question_id"]
        q_type = q.get("question_type", "unknown")
        question = q["question"]
        gold_ans = q.get("answer", "")
        answer_sids = set(q.get("answer_session_ids", []))
        ref_date = q.get("question_date", "")

        # 1. Ingest question haystack
        t0_ingest = time.perf_counter()
        engine = FunesEngine()
        for sid, d_str, turns in zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"]):
            for t in turns:
                engine.add_turn(sid, d_str, t["role"], t["content"])
        t_ingest_ms = (time.perf_counter() - t0_ingest) * 1000.0
        ingest_latencies.append(t_ingest_ms)

        # 2. Retrieve
        t0_ret = time.perf_counter()
        res = engine.retrieve(question, reference_date=ref_date)
        t_ret_ms = (time.perf_counter() - t0_ret) * 1000.0
        retrieval_latencies.append(t_ret_ms)

        # 3. Assess recall metrics
        retrieved_items = [t for t, sc, pr in res.exact_facts] + [t for t, sc, pr in res.fuzzy_contexts]
        retrieved_sids = set(t.session_id for t in retrieved_items)

        full_sess_hit = 1 if (answer_sids and answer_sids.issubset(retrieved_sids)) else 0
        any_sess_hit = 1 if (answer_sids and len(answer_sids.intersection(retrieved_sids)) > 0) else 0

        norm_gold = normalize_text(str(gold_ans))
        gold_match = 0
        if norm_gold and len(norm_gold) >= 2:
            combined_text = normalize_text(" ".join(t.content for t in retrieved_items))
            if norm_gold in combined_text:
                gold_match = 1

        # Track stats
        st = by_type[q_type]
        st["total"] += 1
        st["full_sess"] += full_sess_hit
        st["any_sess"] += any_sess_hit
        st["gold_norm"] += gold_match
        st["latencies"].append(t_ret_ms)

        # Progress indicator
        if (idx + 1) % 10 == 0 or (idx + 1) == eval_count:
            curr_elapsed = time.time() - t_all_start
            q_per_sec = (idx + 1) / curr_elapsed if curr_elapsed > 0 else 0
            print(f"[{idx+1:3d}/{eval_count:3d}] "
                  f"Any-Sess: {sum(x['any_sess'] for x in by_type.values())/(idx+1)*100:5.1f}% | "
                  f"Full-Sess: {sum(x['full_sess'] for x in by_type.values())/(idx+1)*100:5.1f}% | "
                  f"Gold-Match: {sum(x['gold_norm'] for x in by_type.values())/(idx+1)*100:5.1f}% | "
                  f"Throughput: {q_per_sec:4.1f} q/s | "
                  f"Elapsed: {curr_elapsed:5.1f}s")

    t_total_time = time.time() - t_all_start
    avg_ret_lat = sum(retrieval_latencies) / len(retrieval_latencies) if retrieval_latencies else 0.0
    avg_ing_lat = sum(ingest_latencies) / len(ingest_latencies) if ingest_latencies else 0.0

    tot_full = sum(x["full_sess"] for x in by_type.values())
    tot_any = sum(x["any_sess"] for x in by_type.values())
    tot_gold = sum(x["gold_norm"] for x in by_type.values())

    print(f"\n{'='*75}")
    print(f"  FINAL EMPIRICAL RESULTS SUMMARY")
    print(f"{'='*75}")
    print(f"  • Total Evaluated Questions:         {eval_count}")
    print(f"  • Total Suite Execution Time:        {t_total_time:.2f} s ({t_total_time/eval_count:.3f} s/query, {eval_count/t_total_time:.1f} q/s)")
    print(f"  • Haystack Ingestion Latency:        {avg_ing_lat:6.2f} ms/question")
    print(f"  • Retrieval Query Latency:           {avg_ret_lat:6.2f} ms/query")
    print(f"  • Any-Session Hit Rate:              {tot_any / eval_count * 100:6.2f}% ({tot_any}/{eval_count})")
    print(f"  • Full-Session Recall:               {tot_full / eval_count * 100:6.2f}% ({tot_full}/{eval_count})")
    print(f"  • Direct Gold Substring Match:       {tot_gold / eval_count * 100:6.2f}% ({tot_gold}/{eval_count})")
    print(f"{'='*75}\n")

    print("Breakdown by Question Type:")
    for q_type, st in sorted(by_type.items()):
        cnt = st["total"]
        f_rate = st["full_sess"] / cnt * 100.0 if cnt else 0
        a_rate = st["any_sess"] / cnt * 100.0 if cnt else 0
        g_rate = st["gold_norm"] / cnt * 100.0 if cnt else 0
        lat = sum(st["latencies"]) / cnt if cnt else 0
        print(f"  - {q_type:<26} | Count: {cnt:3d} | Any: {a_rate:5.1f}% | Full: {f_rate:5.1f}% | Gold: {g_rate:5.1f}% | Lat: {lat:5.1f}ms")

def main():
    parser = argparse.ArgumentParser(description="Funes LongMemEval-S Evaluation Runner")
    parser.add_argument("--data", type=str, default=None, help="Path to longmemeval_s.json")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions to evaluate")
    args = parser.parse_args()

    data_path = args.data or os.path.join(os.path.dirname(__file__), "data", "longmemeval_s.json")

    evaluate(data_path, limit=args.limit)

if __name__ == "__main__":
    main()
