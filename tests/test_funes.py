# -*- coding: utf-8 -*-
"""Unit tests for the pure-stdlib Funes dual-stream memory engine."""

import os
import tempfile
import unittest

from funes import FunesEngine, UniversalShingleTokenizer


class TestFunesEngine(unittest.TestCase):
    """Core regression and smoke tests for FunesEngine."""

    def test_exact_factual_retrieval_and_prompt(self):
        engine = FunesEngine()
        engine.add_turn("s1", "2026-05-01", "user", "I bought a 16-inch MacBook Pro in Seattle for $2,499.")
        engine.add_turn("s1", "2026-05-01", "assistant", "Great workstation choice.")
        engine.add_turn("s2", "2026-05-10", "user", "I also love pouring single-origin Ethiopian coffee.")

        res = engine.retrieve("What laptop did I buy in Seattle?", reference_date="2026-05-15")
        self.assertGreater(len(res.exact_facts), 0)
        self.assertTrue(any("MacBook Pro" in t.content for t, _, _ in res.exact_facts))

        prompt = engine.build_prompt("What laptop did I buy in Seattle?", result=res, reference_date="2026-05-15")
        self.assertIn("STREAM 1: Exact Factual Anchors", prompt)
        self.assertIn("STREAM 2: Associative Context & Preferences", prompt)
        self.assertIn("2026-05-15", prompt)
        engine.close()

    def test_bilingual_chinese_retrieval(self):
        engine = FunesEngine()
        engine.batch_add_turns([
            {"session_id": "cn1", "date_str": "2026-06-01", "role": "user", "content": "周末我打算去杭州西湖骑行，顺便喝杯龙井茶。"},
            {"session_id": "cn2", "date_str": "2026-06-05", "role": "user", "content": "周三晚上我要去北京出差参加技术峰会。"},
        ])

        tokens = UniversalShingleTokenizer.extract_keywords("杭州西湖")
        self.assertIn("杭州", tokens)
        self.assertIn("西湖", tokens)

        res = engine.retrieve("我周末要去哪里骑行喝龙井茶？", reference_date="2026-06-06")
        combined = [t.content for t, _, _ in (res.exact_facts + res.fuzzy_contexts)]
        self.assertTrue(any("杭州西湖" in c for c in combined))
        engine.close()

    def test_relative_temporal_window(self):
        engine = FunesEngine()
        engine.add_turn("t1", "2026-05-01", "user", "Visited the museum exhibit.")
        engine.add_turn("t2", "2026-05-12", "user", "Went hiking in the mountains.")

        res = engine.retrieve("What did I do 3 days ago?", reference_date="2026-05-15")
        self.assertIn("2026-05-12", res.opened_blocks)
        engine.close()

    def test_sqlite_persistence_and_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "funes_test.db")
            eng1 = FunesEngine(db_path=db_path)
            eng1.add_turn("p1", "2026-03-22", "user", "Diclofenac sodium was prescribed for fever relief.")
            eng1.add_turn("p2", "2026-04-05", "user", "Booked flight to Zurich for the neuroscience workshop.")
            eng1.close()

            eng2 = FunesEngine(db_path=db_path)
            res = eng2.retrieve("When is the Zurich flight?", reference_date="2026-04-10")
            combined = [t.content for t, _, _ in (res.exact_facts + res.fuzzy_contexts)]
            self.assertTrue(any("Zurich" in c for c in combined))
            eng2.close()


if __name__ == "__main__":
    unittest.main()
