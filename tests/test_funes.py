# -*- coding: utf-8 -*-
"""Unit tests for the pure-stdlib Funes dual-stream memory engine."""

import os
import sqlite3
import tempfile
import time
import unittest

from funes import FunesEngine, Transducer, UniversalShingleTokenizer


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

    def test_month_matching_without_substring_false_positives(self):
        engine = FunesEngine()
        engine.add_turn("m1", "2026-05-10", "user", "Playing chess at home.")
        engine.add_turn("m2", "2026-01-15", "user", "January snowstorm covered the driveway.")
        engine.add_turn("m3", "2026-11-20", "user", "November thanksgiving dinner was delicious.")

        # Query with 'maybe' should NOT match 'may' (-05-)
        res_maybe = engine.retrieve("Maybe I should learn piano?")
        self.assertNotIn("2026-05-10", res_maybe.opened_blocks)

        # Explicit 'May' should match
        res_may = engine.retrieve("What did I do in May?")
        self.assertIn("2026-05-10", res_may.opened_blocks)

        # Query with '十一月' should match 11 and NOT falsely match 01 (一月)
        res_nov = engine.retrieve("十一月有什么晚餐安排？")
        self.assertIn("2026-11-20", res_nov.opened_blocks)
        self.assertNotIn("2026-01-15", res_nov.opened_blocks)
        engine.close()

    def test_dehydrated_block_append_no_collision(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "funes_dehydrate_test.db")
            engine = FunesEngine(db_path=db_path)
            d = "2026-05-01"
            for i in range(10):
                engine.add_turn(f"sess_{i}", d, "user", f"Turn number {i} on the first of May.")

            # Force dehydration of 2026-05-01
            engine.store.dehydrate_inactive_blocks(keep_date_strs=set())
            self.assertFalse(engine.store.blocks[d].is_open)

            # Append two new turns to the same dehydrated date
            t11 = engine.add_turn("sess_10", d, "user", "Turn number 10 appended after dehydration.")
            t12 = engine.add_turn("sess_11", d, "user", "Turn number 11 appended after dehydration.")

            # Indices should be strictly monotonic without collision
            self.assertEqual(t11.turn_id + 1, t12.turn_id)
            self.assertEqual(engine.store.blocks[d].turn_count, 12)

            # Query should successfully retrieve turns
            res = engine.retrieve("What was appended after dehydration?")
            self.assertTrue(any("Turn number 10" in t.content for t, _, _ in (res.exact_facts + res.fuzzy_contexts)))
            engine.close()

    def test_thousands_comma_number_matching(self):
        engine = FunesEngine()
        engine.add_turn("c1", "2026-05-01", "user", "I bought a custom workstation for $2,499 in cash.")
        res = engine.retrieve("What did I buy for $2,499?", reference_date="2026-05-15")
        self.assertGreater(len(res.exact_facts), 0)
        self.assertTrue(any("$2,499" in t.content for t, _, _ in res.exact_facts))
        engine.close()

    def test_long_turn_not_truncated_beyond_16_keywords(self):
        engine = FunesEngine()
        long_prefix = " ".join(f"fillerword{i}" for i in range(25))
        engine.add_turn("s1", "2026-05-01", "assistant", f"{long_prefix} secret_isotope_xenon133 confirmed.")
        res = engine.retrieve("What happened with secret_isotope_xenon133?", reference_date="2026-05-05")
        self.assertTrue(any("xenon133" in t.content for t, _, _ in res.exact_facts))
        engine.close()

    def test_pure_temporal_query_returns_turns(self):
        engine = FunesEngine()
        engine.add_turn("t1", "2026-05-12", "user", "Went hiking in the mountains.")
        engine.add_turn("t2", "2026-05-01", "user", "Visited the museum exhibit.")

        res_ago = engine.retrieve("What did I do 3 days ago?", reference_date="2026-05-15")
        combined_ago = [t.content for t, _, _ in (res_ago.exact_facts + res_ago.fuzzy_contexts)]
        self.assertTrue(any("mountains" in c for c in combined_ago))

        res_may = engine.retrieve("What did I do in May?", reference_date="2026-06-01")
        combined_may = [t.content for t, _, _ in (res_may.exact_facts + res_may.fuzzy_contexts)]
        self.assertTrue(any("museum" in c for c in combined_may))
        engine.close()

    def test_da_qian_tian_and_cross_year_month_range(self):
        sig_dqt = Transducer.transduce("我大前天买了什么？", reference_date="2026-05-15")["remo_signal"]
        self.assertEqual(sig_dqt["target_ago_days"], 3)

        sig_range = Transducer.transduce("What happened from November to February?")["remo_signal"]
        self.assertEqual(sig_range["target_months"], ["01", "02", "11", "12"])

    def test_double_s_stemming_and_mi_word_boundaries(self):
        engine = FunesEngine()
        engine.add_turn("s1", "2026-05-01", "user", "I visited the company apartment in Japan.")
        engine.add_turn("s2", "2026-05-02", "user", "The boarding pass was ready.")

        res = engine.retrieve("Did I get the pass?", reference_date="2026-05-05")
        exact_texts = [t.content for t, _, _ in res.exact_facts]
        fuzzy_texts = [t.content for t, _, _ in res.fuzzy_contexts]
        self.assertTrue(any("boarding pass" in c for c in exact_texts))
        self.assertFalse(any("company apartment" in c for c in exact_texts + fuzzy_texts))
        engine.close()

    def test_sqlite_hebbian_persists_without_close(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "funes_crash.db")
            eng1 = FunesEngine(db_path=db_path)
            eng1.add_turn("s1", "2026-05-01", "user", "My favorite espresso uses ethiopian yirgacheffe beans.")
            eng1.store._conn.close()
            eng1.store._conn = None

            eng2 = FunesEngine(db_path=db_path)
            self.assertEqual(eng2.store.hebbian_net.total_turns, 1)
            self.assertIn("yirgacheffe", eng2.store.hebbian_net.adj.get("espresso", {}))
            eng2.close()

    def test_chinese_seeded_association_and_role_tags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "funes_seed.db")
            eng = FunesEngine(db_path=db_path)
            eng.add_turn("s1", "2026-05-01", "system", "下周二需要前往美国领事馆面试。")
            eng.close()

            conn = sqlite3.connect(db_path)
            conn.execute(
                "INSERT OR REPLACE INTO associations (a_token, b_token, weight, last_used) VALUES (?, ?, ?, ?)",
                ("面签", "领事馆面试", 3.0, time.time())
            )
            conn.commit()
            conn.close()

            eng2 = FunesEngine(db_path=db_path)
            res = eng2.retrieve("面签需要准备什么？", reference_date="2026-05-02")
            combined = [t.content for t, _, _ in (res.exact_facts + res.fuzzy_contexts)]
            self.assertTrue(any("领事馆面试" in c for c in combined))
            prompt = eng2.build_prompt("面签需要准备什么？", result=res, reference_date="2026-05-02")
            self.assertIn("[2026-05-01 | System]", prompt)
            eng2.close()

    def test_causal_bubble_neighbor_context_retained(self):
        engine = FunesEngine()
        engine.add_turn("s1", "2026-05-01", "user", "I decided on the MacBook Pro today.")
        engine.add_turn("s1", "2026-05-01", "assistant", "Nice! It cost $2,499 at the downtown store.")
        res = engine.retrieve("What did I decide on about the MacBook Pro?", reference_date="2026-05-05")
        combined = [t.content for t, _, _ in (res.exact_facts + res.fuzzy_contexts)]
        self.assertTrue(any("$2,499" in c for c in combined))
        engine.close()


if __name__ == "__main__":
    unittest.main()
