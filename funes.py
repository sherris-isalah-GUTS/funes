# -*- coding: utf-8 -*-
"""
Funes: Ultra-Lightweight Biomimetic Dual-Stream Memory Engine
=============================================================
A zero-embedding, CPU-native long-term memory and edge search engine
implemented purely in the Python standard library.

Author: Sherris Isalah (艾浩喆) <sherjia192@gmail.com>
Preprint DOI: 10.5281/zenodo.23921389
License: GNU AGPLv3 (AGPL-3.0); commercial closed-source license available via email.
"""

from collections import defaultdict
from dataclasses import dataclass, field
import datetime
import json
import math
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional, Set, Tuple

__version__ = "1.1.0"

STOP_WORDS: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "to", "in", "on", "of", "and", "or",
    "it", "i", "you", "my", "your", "for", "with", "at", "by", "this", "that", "do",
    "did", "have", "has", "had", "be", "been", "me", "we", "us", "they", "them",
    "what", "when", "where", "how", "who", "which", "will", "would", "can", "could",
    "so", "but", "about", "there", "their", "then", "some", "any", "no", "not",
    "just", "like", "get", "got", "know", "think", "tell", "say", "said", "from",
    "as", "up", "out", "if", "all", "also", "into", "than", "very"
}


def _stem_plural(word: str) -> str:
    """Strip a single trailing plural 's' without mangling words ending in ss/us/is."""
    if len(word) >= 4 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _is_cjk(text: str) -> bool:
    """Return True if text contains any CJK Unified Ideograph."""
    for ch in text:
        if "\u4e00" <= ch <= "\u9fa5":
            return True
    return False


def normalize_date_key(raw_date: Any) -> str:
    """Normalize heterogeneous date strings to YYYY-MM-DD."""
    if not raw_date:
        return "1970-01-01"
    s = str(raw_date).strip().replace("/", "-")
    m = re.search(r"(\d{4})[^\d](\d{1,2})[^\d](\d{1,2})", s)
    if m:
        y, month, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{y:04d}-{month:02d}-{d:02d}"
    return s[:10]


def parse_date_dt(raw_date: Any) -> datetime.datetime:
    """Parse a date string into a datetime object."""
    norm = normalize_date_key(raw_date)
    try:
        return datetime.datetime.strptime(norm, "%Y-%m-%d")
    except Exception:
        return datetime.datetime(1970, 1, 1)


@dataclass
class Turn:
    """Single conversational turn or document unit."""
    turn_id: int
    session_id: str
    date_str: str
    role: str
    content: str
    timestamp: float = 0.0


@dataclass
class DateBlock:
    """Calendar-partitioned Topological Associating Domain (TAD) block."""
    date_str: str
    turns: List[Turn] = field(default_factory=list)
    turn_count: int = 0
    exposed_vocab: Dict[str, List[int]] = field(default_factory=lambda: defaultdict(list))
    prefix4: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    is_open: bool = False


@dataclass
class TranscriptionBubble:
    """Causal window slice unwound around a hit anchor."""
    date_str: str
    session_id: str
    anchor_turn_idx: int
    turns: List[Turn]
    anchor_words: List[str]


@dataclass
class DualStreamResult:
    """Structured dual-stream retrieval output."""
    exact_facts: List[Tuple[Turn, float, str]]
    fuzzy_contexts: List[Tuple[Turn, float, str]]
    opened_blocks: List[str]
    active_bubbles: int
    latency_ms: float = 0.0
    reference_date: Optional[str] = None


class UniversalShingleTokenizer:
    """Zero-dependency bilingual (EN/ZH) lexical tokenizer and shingle generator."""

    ZH_STOPS = set("的了我你他在是有这那个就都要会什么哪去与及从和对把让被于以使得可以帮请问怎么谁何")

    @classmethod
    def extract_keywords(cls, text: str, max_kws: Optional[int] = 16) -> List[str]:
        """Extract normalized English tokens and Chinese 2-gram shingles."""
        if not text:
            return []
        text_clean = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
        has_zh = bool(re.search(r"[\u4e00-\u9fa5]", text_clean))
        kws: List[str] = []
        seen: Set[str] = set()

        def _add_kw(tok: str):
            if tok not in seen:
                seen.add(tok)
                kws.append(tok)

        for w in re.findall(r"(?<![A-Za-z0-9])(?:\$?\d+(?:\.\d+)+|[a-zA-Z0-9\-\'\$]{2,})(?![A-Za-z0-9])", text_clean):
            wl = re.sub(r"^[\-\']+|[\-\'\$]+$", "", w.lower())
            if wl.startswith("$") and (len(wl) < 2 or not wl[1].isdigit()):
                wl = wl.lstrip("$")
            if len(wl) < 2 or not any(ch.isalnum() for ch in wl):
                continue
            if wl not in STOP_WORDS:
                _add_kw(wl)
                if "-" in wl:
                    for sub in wl.split("-"):
                        sub_clean = sub.strip("'$")
                        if len(sub_clean) >= 2 and sub_clean not in STOP_WORDS:
                            _add_kw(sub_clean)

        if has_zh:
            for run in re.findall(r"[\u4e00-\u9fa5]+", text_clean):
                if len(run) == 1:
                    continue
                elif len(run) == 2:
                    if run[0] not in cls.ZH_STOPS and run[1] not in cls.ZH_STOPS:
                        _add_kw(run)
                elif len(run) == 3:
                    if sum(ch not in cls.ZH_STOPS for ch in run) >= 2:
                        _add_kw(run)
                    for i in range(2):
                        bg = run[i:i + 2]
                        if bg[0] not in cls.ZH_STOPS and bg[1] not in cls.ZH_STOPS:
                            _add_kw(bg)
                else:
                    for i in range(len(run) - 1):
                        bg = run[i:i + 2]
                        if bg[0] not in cls.ZH_STOPS and bg[1] not in cls.ZH_STOPS:
                            _add_kw(bg)

        return kws if max_kws is None else kws[:max_kws]

    @classmethod
    def get_shingles(cls, word: str) -> List[str]:
        """Generate morphological stem and prefix shingles for fuzzy alignment."""
        if _is_cjk(word):
            return [word]
        shingles = {word}
        if word.endswith("'s"):
            clean = word[:-2]
        elif word.endswith("'"):
            clean = word[:-1]
        else:
            clean = word
        if clean != word and len(clean) >= 2:
            shingles.add(clean)
        if len(clean) >= 4:
            shingles.add(clean[:4])
        stemmed = _stem_plural(clean)
        if stemmed != clean:
            shingles.add(stemmed)
        return list(shingles)


extract_keywords = UniversalShingleTokenizer.extract_keywords
get_shingles = UniversalShingleTokenizer.get_shingles


class HebbianAssociationNetwork:
    """Synaptic co-occurrence graph with epigenetic high-frequency silencing."""

    def __init__(self, step: float = 0.1, max_weight: float = 5.0):
        self.step = step
        self.max_weight = max_weight
        self.adj: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.doc_freq: Dict[str, int] = defaultdict(int)
        self.total_turns: int = 0

    def _reinforce_turn_delta(
        self,
        tokens: List[str],
        track_edges: bool = False
    ) -> Tuple[List[Tuple[str, str, float]], List[str]]:
        """Strengthen co-occurrence edges for a turn and return updated edges and unique tokens."""
        unique_tokens = list(dict.fromkeys(tokens))
        updated_edges: List[Tuple[str, str, float]] = []
        self.total_turns += 1
        doc_freq = self.doc_freq
        for t in unique_tokens:
            doc_freq[t] += 1

        # Cap Hebbian edge wiring on very long turns so ingestion stays fast while exposed_vocab keeps all tokens.
        hebbian_tokens = unique_tokens if len(unique_tokens) <= 48 else unique_tokens[:48]
        n = len(hebbian_tokens)
        if n < 2:
            return updated_edges, unique_tokens

        adj = self.adj
        step = self.step
        max_weight = self.max_weight
        max_span = n if n <= 20 else 8
        for i in range(n):
            ti = hebbian_tokens[i]
            adj_i = adj[ti]
            j_end = min(n, i + max_span)
            for j in range(i + 1, j_end):
                tj = hebbian_tokens[j]
                w_ij = adj_i[tj] + step
                if w_ij > max_weight:
                    w_ij = max_weight
                adj_i[tj] = w_ij
                adj[tj][ti] = w_ij
                if track_edges:
                    updated_edges.append((ti, tj, w_ij))
        return updated_edges, unique_tokens

    def reinforce_turn(self, tokens: List[str]):
        """Strengthen pairwise edge weights for co-occurring tokens in a turn."""
        self._reinforce_turn_delta(tokens)

    def get_associated_probes(self, seed_keywords: List[str], top_k: int = 12) -> List[str]:
        """Traverse 1-hop synaptic neighbors with 8% DF cutoff and inverse sqrt decay."""
        if not self.adj or not seed_keywords:
            return []

        scores: Dict[str, float] = defaultdict(float)
        seeds_set = set(k.lower() for k in seed_keywords)
        max_df = max(12, int(self.total_turns * 0.08))

        for seed in seeds_set:
            neighbors = self.adj.get(seed, {})
            if not neighbors and len(seed) >= 4:
                seed_prefix = seed[:4]
                for known_tok, n_dict in self.adj.items():
                    if len(known_tok) >= 4 and known_tok[:4] == seed_prefix:
                        for nbr, w in n_dict.items():
                            if nbr not in seeds_set:
                                df = self.doc_freq.get(nbr, 1)
                                if df > max_df:
                                    continue
                                scores[nbr] += (w * 0.7) / math.sqrt(df + 1.0)
            else:
                for nbr, w in neighbors.items():
                    if nbr not in seeds_set:
                        df = self.doc_freq.get(nbr, 1)
                        if df > max_df:
                            continue
                        scores[nbr] += w / math.sqrt(df + 1.0)

        sorted_probes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [tok for tok, _ in sorted_probes[:top_k]]


class Transducer:
    """Query signal transducer synthesizing rigid, flexible, and temporal probes."""

    extract_keywords = staticmethod(UniversalShingleTokenizer.extract_keywords)

    @classmethod
    def transduce(
        cls,
        query: str,
        hebbian_net: Optional[HebbianAssociationNetwork] = None,
        reference_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Convert natural language query into deterministic dual-track probes."""
        q_low = query.lower()
        kws = UniversalShingleTokenizer.extract_keywords(query)

        is_single_sess = any(phrase in q_low for phrase in [
            "previous conversation", "our chat", "previous chat", "remind me",
            "you recommended", "you mentioned", "you gave", "you suggest", "our discussion",
            "in our conversation", "what happened",
            "之前聊过", "上次聊", "上次说", "记得", "推荐过", "提过", "你说过", "我们的对话", "发生了什么"
        ])

        is_multi_sess = any(phrase in q_low for phrase in [
            "how many", "total number", "across all", "how many times", "list all",
            "how many days", "total hours", "how much time",
            "一共", "总共", "统计", "几次", "多少次", "多少天", "多少个", "所有", "清单", "列出", "花了多少"
        ])

        remo_signal: Dict[str, Any] = {
            "query_keywords": kws,
            "temporal_hints": [],
            "target_months": [],
            "relative_scope_days": None,
            "target_ago_days": None,
            "expand_adjacent_days": False,
            "reference_date": reference_date,
            "is_single_sess": is_single_sess,
            "is_multi_sess": is_multi_sess,
            "target_exact_dates": []
        }

        num_map = {
            "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
            "twelve": 12, "couple": 2, "few": 3
        }

        def parse_num_token(token: str) -> int:
            return int(token) if token.isdigit() else num_map.get(token.lower(), 1)

        m_ago = re.search(
            r"\b(?:about\s+)?(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|couple|few)\s+(?:of\s+)?(day|week|month)s?\s+ago\b",
            q_low
        )
        cn_num = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
        m_cn_ago = re.search(r"(\d+|一|二|两|三|四|五|六|七|八|九|十)\s*(天|周|个?星期|个?月|年)前", q_low)

        if m_ago:
            n = parse_num_token(m_ago.group(1))
            unit = m_ago.group(2)
            remo_signal["target_ago_days"] = n if unit == "day" else (n * 7 if unit == "week" else n * 30)
        elif m_cn_ago:
            val_str, unit = m_cn_ago.group(1), m_cn_ago.group(2)
            n = int(val_str) if val_str.isdigit() else cn_num.get(val_str, 1)
            remo_signal["target_ago_days"] = (
                n if "天" in unit else (n * 7 if ("周" in unit or "星期" in unit) else (n * 30 if "月" in unit else n * 365))
            )
        elif "大前天" in q_low:
            remo_signal["target_ago_days"] = 3
        elif "day before yesterday" in q_low or "前天" in q_low:
            remo_signal["target_ago_days"] = 2
        elif "yesterday" in q_low or "昨天" in q_low:
            remo_signal["target_ago_days"] = 1

        remo_signal["expand_adjacent_days"] = bool(
            re.search(r"\b(?:the\s+)?day\s+(?:before|after)\b", q_low) or
            "the next day" in q_low or "the previous day" in q_low or
            "第二天" in q_low or "前一天" in q_low or "隔天" in q_low
        )

        m_last = re.search(
            r"\b(?:last|past)\s+(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|couple|few)\s+(?:of\s+)?(day|week|month)s?\b",
            q_low
        )
        if m_last:
            n = parse_num_token(m_last.group(1))
            unit = m_last.group(2)
            days = n if unit == "day" else (n * 7 if unit == "week" else n * 30)
            remo_signal["relative_scope_days"] = days + (3 if unit == "week" else 5)
        elif re.search(r"\b(?:last|past)\s+week\b", q_low) or "上周" in q_low or "上个星期" in q_low:
            remo_signal["relative_scope_days"] = 8
        elif re.search(r"\b(?:last|past)\s+month\b", q_low) or "上个月" in q_low:
            remo_signal["relative_scope_days"] = 35
        elif "last year" in q_low or "去年" in q_low:
            remo_signal["relative_scope_days"] = 730
        elif "this year" in q_low or "今年" in q_low:
            remo_signal["relative_scope_days"] = 365

        months_map = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12
        }
        zh_months_map = {
            "一月": 1, "二月": 2, "三月": 3, "四月": 4,
            "五月": 5, "六月": 6, "七月": 7, "八月": 8,
            "九月": 9, "十月": 10, "十一月": 11, "十二月": 12
        }

        m_range = re.search(r"from\s+([a-z]+)\s+to\s+([a-z]+)", q_low)
        if m_range and m_range.group(1) in months_map and m_range.group(2) in months_map:
            m1, m2 = months_map[m_range.group(1)], months_map[m_range.group(2)]
            month_indices = range(m1, m2 + 1) if m1 <= m2 else (list(range(m1, 13)) + list(range(1, m2 + 1)))
            for m_idx in month_indices:
                remo_signal["target_months"].append(f"{m_idx:02d}")
        else:
            for m_name, m_num in months_map.items():
                if re.search(r"\b" + m_name + r"\b", q_low):
                    remo_signal["target_months"].append(f"{m_num:02d}")

        for m_match in re.findall(r"(?:十一|十二|[一二三四五六七八九十])月", query):
            if m_match in zh_months_map:
                remo_signal["target_months"].append(f"{zh_months_map[m_match]:02d}")
        for m_num_str in re.findall(r"(?<![\d])(1[0-2]|[1-9])月", query):
            remo_signal["target_months"].append(f"{int(m_num_str):02d}")

        remo_signal["target_months"] = sorted(list(set(remo_signal["target_months"])))

        gu_probes = set()
        q_clean = re.sub(r"(?<=\d),(?=\d{3}\b)", "", query)
        for w in re.findall(r"(?<![A-Za-z0-9])[A-Z][a-z0-9\-\'\$]{2,}(?![A-Za-z0-9])", q_clean):
            wl = re.sub(r"^[\-\']+|[\-\'\$]+$", "", w.lower())
            if wl.startswith("$") and (len(wl) < 2 or not wl[1].isdigit()):
                wl = wl.lstrip("$")
            if len(wl) >= 2 and any(ch.isalnum() for ch in wl) and wl not in STOP_WORDS:
                gu_probes.add(wl)

        for w in re.findall(r"(?<![A-Za-z0-9])(?:\$\d+(?:\.\d+)*|\d+(?:\.\d+)?(?:[a-zA-Z%]+|\$)?)(?![A-Za-z0-9])", q_clean):
            wl = w.lower().strip()
            if len(wl) >= 2 and wl not in STOP_WORDS and not wl.isdigit():
                gu_probes.add(wl)

        for kw in kws:
            gu_probes.add(kw)

        mi_probes = set()
        if hebbian_net is not None:
            seed_tokens = list(kws)
            # Also probe multi-char CJK substrings present directly in the synaptic table (e.g. Strategy 2).
            for run in re.findall(r"[\u4e00-\u9fa5]{2,}", query):
                max_sub = min(len(run), 8)
                for span in range(2, max_sub + 1):
                    for i in range(len(run) - span + 1):
                        sub = run[i:i + span]
                        if sub in hebbian_net.adj and sub not in seed_tokens:
                            seed_tokens.append(sub)
            for sp in hebbian_net.get_associated_probes(seed_tokens, top_k=10):
                mi_probes.add(sp)

        for kw in kws:
            for shingle in UniversalShingleTokenizer.get_shingles(kw):
                mi_probes.add(shingle)

        gu_list = sorted(list(gu_probes))
        mi_list = sorted(list(mi_probes))
        remo_signal["gu_probes"] = gu_list
        remo_signal["mi_probes"] = mi_list

        return {
            "remo_signal": remo_signal,
            "gu_probes": gu_list,
            "mi_probes": mi_list,
            "is_single_sess": is_single_sess,
            "is_multi_sess": is_multi_sess
        }


class DNAMemoryStore:
    """Chromatin storage kernel with date-partitioned A/B compartment dehydration."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
        self.blocks: Dict[str, DateBlock] = {}
        self._next_turn_id: int = 0
        self._conn: Optional[sqlite3.Connection] = None
        self.hebbian_net = HebbianAssociationNetwork()

        if self.db_path is not None:
            self._init_sqlite()
            self._load_meta_from_db()

    def _init_sqlite(self):
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.cursor()

        cur.execute("PRAGMA journal_mode = WAL;")
        cur.execute("PRAGMA synchronous = NORMAL;")
        cur.execute("PRAGMA cache_size = -4096;")
        cur.execute("PRAGMA mmap_size = 0;")
        cur.execute("PRAGMA temp_store = MEMORY;")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS dna_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                date_str TEXT,
                local_idx INTEGER,
                role TEXT,
                content TEXT,
                timestamp REAL
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_dna_turns_date ON dna_turns(date_str)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_dna_turns_session ON dna_turns(session_id)")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS dna_blocks (
                date_str TEXT PRIMARY KEY,
                turn_count INTEGER,
                exposed_vocab TEXT,
                updated_at REAL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS associations (
                a_token TEXT,
                b_token TEXT,
                weight REAL,
                last_used REAL,
                PRIMARY KEY (a_token, b_token)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS hebbian_meta (
                key TEXT PRIMARY KEY,
                val TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS token_df (
                token TEXT PRIMARY KEY,
                df INTEGER
            )
        """)
        self._conn.commit()

    def _load_meta_from_db(self):
        if not self._conn:
            return
        cur = self._conn.cursor()

        cur.execute("SELECT date_str, turn_count, exposed_vocab FROM dna_blocks")
        for row in cur.fetchall():
            d_str = row["date_str"]
            v_dict = json.loads(row["exposed_vocab"])
            block = DateBlock(date_str=d_str, turn_count=row["turn_count"], is_open=False)
            block.exposed_vocab = defaultdict(list, v_dict)
            for tok in block.exposed_vocab.keys():
                if len(tok) >= 4 and not _is_cjk(tok):
                    block.prefix4[tok[:4]].add(tok)
            self.blocks[d_str] = block

        cur.execute("SELECT a_token, b_token, weight FROM associations")
        for a_row in cur.fetchall():
            self.hebbian_net.adj[a_row["a_token"]][a_row["b_token"]] = float(a_row["weight"])

        cur.execute("SELECT val FROM hebbian_meta WHERE key = 'total_turns'")
        meta_row = cur.fetchone()
        if meta_row:
            try:
                self.hebbian_net.total_turns = int(meta_row["val"])
            except Exception:
                pass

        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='token_df'")
        if cur.fetchone():
            for df_row in cur.execute("SELECT token, df FROM token_df").fetchall():
                self.hebbian_net.doc_freq[df_row["token"]] = int(df_row["df"])

        cur.execute("SELECT COALESCE(MAX(id), 0) AS max_id FROM dna_turns")
        max_row = cur.fetchone()
        self._next_turn_id = int(max_row["max_id"]) if max_row else 0

        if not self.blocks:
            cur.execute("SELECT id, session_id, date_str, local_idx, role, content, timestamp FROM dna_turns ORDER BY id ASC")
            turn_rows = cur.fetchall()
            if turn_rows:
                rebuild_hebbian = (self.hebbian_net.total_turns == 0)
                for r in turn_rows:
                    norm_date = r["date_str"]
                    if norm_date not in self.blocks:
                        self.blocks[norm_date] = DateBlock(date_str=norm_date, is_open=False)
                    block = self.blocks[norm_date]
                    local_idx = r["local_idx"]
                    tokens = UniversalShingleTokenizer.extract_keywords(r["content"], max_kws=None)
                    if rebuild_hebbian:
                        self.hebbian_net.reinforce_turn(tokens)
                    for tok in tokens:
                        for s in UniversalShingleTokenizer.get_shingles(tok):
                            lst = block.exposed_vocab[s]
                            if not lst or lst[-1] != local_idx:
                                lst.append(local_idx)
                            if len(s) >= 4 and not _is_cjk(s):
                                block.prefix4[s[:4]].add(s)
                    block.turn_count = max(block.turn_count, local_idx + 1)
                self.flush_meta()

    def append_turn(self, session_id: str, date_str: str, role: str, content: str, timestamp: float = 0.0, commit: bool = True) -> Turn:
        """Append a single dialogue turn and update TAD + Hebbian indices."""
        norm_date = normalize_date_key(date_str)
        if norm_date not in self.blocks:
            self.blocks[norm_date] = DateBlock(date_str=norm_date, is_open=True)

        block = self.blocks[norm_date]
        if self._conn and not block.is_open and block.turn_count > 0 and not block.turns:
            self.load_active_turns(block)

        local_idx = len(block.turns) if block.is_open else block.turn_count
        turn_id = self._next_turn_id

        if self._conn:
            cur = self._conn.cursor()
            cur.execute("""
                INSERT INTO dna_turns (session_id, date_str, local_idx, role, content, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (session_id, norm_date, local_idx, role, content, timestamp))
            turn_id = cur.lastrowid or turn_id
        self._next_turn_id = max(self._next_turn_id + 1, turn_id + 1)

        t = Turn(
            turn_id=turn_id,
            session_id=session_id,
            date_str=norm_date,
            role=role,
            content=content,
            timestamp=timestamp
        )

        tokens = UniversalShingleTokenizer.extract_keywords(content, max_kws=None)
        updated_edges, unique_tokens = self.hebbian_net._reinforce_turn_delta(
            tokens, track_edges=bool(self._conn and commit)
        )

        block.turns.append(t)
        block.turn_count = max(block.turn_count, local_idx + 1)

        for tok in tokens:
            for s in UniversalShingleTokenizer.get_shingles(tok):
                lst = block.exposed_vocab[s]
                if not lst or lst[-1] != local_idx:
                    lst.append(local_idx)
                if len(s) >= 4 and not _is_cjk(s):
                    block.prefix4[s[:4]].add(s)

        if self._conn and commit:
            now_ts = time.time()
            cur = self._conn.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO dna_blocks (date_str, turn_count, exposed_vocab, updated_at)
                VALUES (?, ?, ?, ?)
            """, (norm_date, block.turn_count, json.dumps(dict(block.exposed_vocab), ensure_ascii=False), now_ts))
            if updated_edges:
                edge_rows = []
                for a_tok, b_tok, weight in updated_edges:
                    edge_rows.append((a_tok, b_tok, weight, now_ts))
                    edge_rows.append((b_tok, a_tok, weight, now_ts))
                cur.executemany(
                    "INSERT OR REPLACE INTO associations (a_token, b_token, weight, last_used) VALUES (?, ?, ?, ?)",
                    edge_rows
                )
            if unique_tokens:
                cur.executemany(
                    "INSERT OR REPLACE INTO token_df (token, df) VALUES (?, ?)",
                    [(tok, self.hebbian_net.doc_freq[tok]) for tok in unique_tokens]
                )
            cur.execute(
                "INSERT OR REPLACE INTO hebbian_meta (key, val) VALUES ('total_turns', ?)",
                (str(self.hebbian_net.total_turns),)
            )
            self._conn.commit()

        return t

    def batch_append_turns(self, turns_data: List[Tuple[str, str, str, str, float]]):
        """Bulk append turns and commit index metadata in a single transaction."""
        for sid, d_str, r, c, ts in turns_data:
            self.append_turn(sid, d_str, r, c, ts, commit=False)
        self.flush_meta()

    def flush_meta(self):
        """Persist TAD vocabulary and Hebbian adjacency metadata to SQLite."""
        if not self._conn:
            return
        cur = self._conn.cursor()
        now_ts = time.time()
        for d_str, block in self.blocks.items():
            cur.execute("""
                INSERT OR REPLACE INTO dna_blocks (date_str, turn_count, exposed_vocab, updated_at)
                VALUES (?, ?, ?, ?)
            """, (d_str, block.turn_count, json.dumps(dict(block.exposed_vocab), ensure_ascii=False), now_ts))

        for a_tok, neighbors in self.hebbian_net.adj.items():
            for b_tok, weight in neighbors.items():
                cur.execute("""
                    INSERT OR REPLACE INTO associations (a_token, b_token, weight, last_used)
                    VALUES (?, ?, ?, ?)
                """, (a_tok, b_tok, weight, now_ts))

        cur.execute("CREATE TABLE IF NOT EXISTS token_df (token TEXT PRIMARY KEY, df INTEGER)")
        for tok, df_val in self.hebbian_net.doc_freq.items():
            cur.execute("""
                INSERT OR REPLACE INTO token_df (token, df)
                VALUES (?, ?)
            """, (tok, df_val))

        cur.execute("INSERT OR REPLACE INTO hebbian_meta (key, val) VALUES ('total_turns', ?)", (str(self.hebbian_net.total_turns),))
        self._conn.commit()

    def load_active_turns(self, block: DateBlock):
        """Hydrate target block turns from SQLite into active A-compartment memory."""
        if block.turns and len(block.turns) == block.turn_count:
            block.is_open = True
            return

        if not self._conn:
            block.is_open = True
            return

        cur = self._conn.cursor()
        cur.execute("""
            SELECT id, session_id, date_str, local_idx, role, content, timestamp
            FROM dna_turns WHERE date_str = ? ORDER BY local_idx ASC
        """, (block.date_str,))
        rows = cur.fetchall()
        block.turns = [
            Turn(
                turn_id=r["id"],
                session_id=r["session_id"],
                date_str=r["date_str"],
                role=r["role"],
                content=r["content"],
                timestamp=r["timestamp"]
            )
            for r in rows
        ]
        block.turn_count = len(block.turns)
        block.is_open = True

    def dehydrate_inactive_blocks(self, keep_date_strs: Set[str]):
        """Dehydrate inactive B-compartment blocks to maintain low RAM footprint."""
        for d_str, block in self.blocks.items():
            if d_str not in keep_date_strs and block.is_open:
                if self._conn:
                    block.turns.clear()
                block.is_open = False

    def close(self):
        """Flush pending metadata and close the database handle."""
        if self._conn:
            try:
                self.flush_meta()
            except Exception:
                pass
            self._conn.close()
            self._conn = None


def _cjk_bigram_intersection(term: str, exposed_vocab: Dict[str, List[int]]) -> Set[int]:
    """Match 3+ char CJK phrases against 2-gram exposed_vocab entries by turn-index intersection."""
    valid_bgs: List[str] = []
    for run in re.findall(r"[\u4e00-\u9fa5]{2,}", term):
        for i in range(len(run) - 1):
            bg = run[i:i + 2]
            if bg[0] not in UniversalShingleTokenizer.ZH_STOPS and bg[1] not in UniversalShingleTokenizer.ZH_STOPS:
                valid_bgs.append(bg)
    if not valid_bgs:
        return set()
    sets = []
    for bg in valid_bgs:
        if bg not in exposed_vocab:
            return set()
        sets.append(set(exposed_vocab[bg]))
    result = sets[0]
    for s in sets[1:]:
        result = result.intersection(s)
        if not result:
            return set()
    return result


def _ensure_prefix4(block: DateBlock):
    """Lazily build prefix4 buckets if exposed_vocab was populated externally."""
    if not block.prefix4 and block.exposed_vocab:
        for tok in block.exposed_vocab.keys():
            if len(tok) >= 4 and not _is_cjk(tok):
                block.prefix4[tok[:4]].add(tok)


def _vocab_matching_turns(term: str, block: DateBlock) -> Set[int]:
    """Return all local turn indices in a block matching term (deduplicated via O(1) buckets)."""
    exposed_vocab = block.exposed_vocab
    hits: Set[int] = set()
    if term in exposed_vocab:
        hits.update(exposed_vocab[term])
    if _is_cjk(term):
        if not hits and len(term) >= 3:
            hits.update(_cjk_bigram_intersection(term, exposed_vocab))
        return hits

    term_stem = _stem_plural(term)
    if term_stem != term and term_stem in exposed_vocab:
        hits.update(exposed_vocab[term_stem])
    if len(term) >= 4:
        _ensure_prefix4(block)
        for v_word in block.prefix4.get(term[:4], ()):
            v_stem = _stem_plural(v_word)
            if term_stem == v_stem or v_word.startswith(term) or term.startswith(v_word):
                hits.update(exposed_vocab[v_word])
    return hits


def _compute_temporal_targets(all_dates: List[str], signal: Dict[str, Any]) -> Tuple[Set[str], Set[str]]:
    """Compute (all_temporally_activated_dates, narrow_temporal_target_dates)."""
    activated: Set[str] = set()
    narrow_targets: Set[str] = set()
    if not all_dates:
        return activated, narrow_targets

    ref_d = signal.get("reference_date")
    ref_dt = parse_date_dt(ref_d or all_dates[-1])

    target_ago = signal.get("target_ago_days")
    if target_ago is not None:
        target_dt = ref_dt - datetime.timedelta(days=target_ago)
        closest_date = min(all_dates, key=lambda d: abs((parse_date_dt(d) - target_dt).total_seconds()))
        activated.add(closest_date)
        narrow_targets.add(closest_date)
        closest_dt = parse_date_dt(closest_date)
        for d in all_dates:
            if abs((parse_date_dt(d) - closest_dt).days) <= 1:
                activated.add(d)

    rel_scope = signal.get("relative_scope_days")
    if rel_scope is not None:
        for d in all_dates:
            diff = (ref_dt - parse_date_dt(d)).days
            if -1 <= diff <= rel_scope:
                activated.add(d)
                if rel_scope <= 35:
                    narrow_targets.add(d)

    target_months = signal.get("target_months", [])
    if target_months:
        for d in all_dates:
            if any(f"-{m}-" in d for m in target_months):
                activated.add(d)
                narrow_targets.add(d)

    return activated, narrow_targets


class Remo:
    """Stage 2: Chromatin remodeler executing TAD phase separation and scoring."""

    @classmethod
    def remodel(cls, store: DNAMemoryStore, signal: Dict[str, Any]) -> List[DateBlock]:
        """Select and hydrate top-scoring DateBlocks using IDF and turn-level synergy."""
        all_dates = sorted(list(store.blocks.keys()))
        if not all_dates:
            return []

        activated_date_set, _ = _compute_temporal_targets(all_dates, signal)
        kws = signal.get("query_keywords", [])
        gu_probes = signal.get("gu_probes", kws)
        mi_probes = signal.get("mi_probes", [])
        gu_set = set(gu_probes)
        mi_only = [p for p in mi_probes if p not in gu_set]

        total_turns = max(1, store.hebbian_net.total_turns)
        doc_freq = store.hebbian_net.doc_freq

        gu_weights = [
            (term, 2.0 * math.log(1.0 + total_turns / (doc_freq.get(term, 0) + 1.0)))
            for term in gu_probes
        ]
        mi_weights = [
            (term, 1.0 * math.log(1.0 + total_turns / (doc_freq.get(term, 0) + 1.0)))
            for term in mi_only
        ]

        date_scores: Dict[str, float] = defaultdict(float)

        for d_str, block in store.blocks.items():
            turn_cohit: Dict[int, float] = defaultdict(float)
            block_idf_sum = 0.0
            matched_gu_prefixes: Set[str] = set()

            for term, w in gu_weights:
                t_hits = _vocab_matching_turns(term, block)
                if t_hits:
                    block_idf_sum += w
                    if len(term) >= 4 and not _is_cjk(term):
                        matched_gu_prefixes.add(term[:4])
                    for t_idx in t_hits:
                        turn_cohit[t_idx] += w

            for term, w in mi_weights:
                if len(term) == 4 and term in matched_gu_prefixes:
                    continue
                t_hits = _vocab_matching_turns(term, block)
                if t_hits:
                    block_idf_sum += w
                    for t_idx in t_hits:
                        turn_cohit[t_idx] += w

            if block_idf_sum > 0:
                max_turn_score = max(turn_cohit.values()) if turn_cohit else 0.0
                date_scores[d_str] = block_idf_sum + 1.5 * max_turn_score

        sorted_date_by_score = sorted(date_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_blocks = 8 if signal.get("is_multi_sess") else 5
        for d_str, _ in sorted_date_by_score[:top_k_blocks]:
            activated_date_set.add(d_str)

        if signal.get("expand_adjacent_days") and activated_date_set:
            adjacent_adds = set()
            for d_str in activated_date_set:
                curr_dt = parse_date_dt(d_str)
                for other_d in all_dates:
                    if abs((parse_date_dt(other_d) - curr_dt).days) == 1:
                        adjacent_adds.add(other_d)
            activated_date_set.update(adjacent_adds)

        if not activated_date_set:
            activated_date_set.add(all_dates[-1])

        activated_blocks = []
        for d_str in sorted(list(activated_date_set)):
            blk = store.blocks[d_str]
            store.load_active_turns(blk)
            activated_blocks.append(blk)

        store.dehydrate_inactive_blocks(activated_date_set)
        return activated_blocks


class Heli:
    """Stage 3: Helicase unwinding session-insulated asymmetric [-2, +3] causal bubbles."""

    @classmethod
    def unwind(
        cls,
        active_blocks: List[DateBlock],
        candidate_terms: Set[str],
        is_single_sess: bool = False,
        temporal_target_dates: Optional[Set[str]] = None
    ) -> List[TranscriptionBubble]:
        """Extract non-overlapping causal context slices constrained by session_id."""
        bubbles = []

        for block in active_blocks:
            if not block.is_open:
                continue

            hit_turn_sets: Dict[int, Set[str]] = defaultdict(set)
            for term in candidate_terms:
                for turn_idx in _vocab_matching_turns(term, block):
                    hit_turn_sets[turn_idx].add(term)

            if not hit_turn_sets:
                # Fallback for pure-temporal queries when an explicitly targeted DateBlock has no lexical hits.
                if temporal_target_dates and block.date_str in temporal_target_dates and block.turns:
                    sess_buckets: Dict[str, List[Turn]] = defaultdict(list)
                    for t in block.turns:
                        sess_buckets[t.session_id].append(t)
                    for sid, s_turns in sess_buckets.items():
                        bubbles.append(TranscriptionBubble(
                            date_str=block.date_str,
                            session_id=sid,
                            anchor_turn_idx=0,
                            turns=s_turns,
                            anchor_words=["__temporal__"]
                        ))
                continue

            unwound_turn_sets = set()
            sorted_anchors = sorted(hit_turn_sets.keys(), key=lambda idx: len(hit_turn_sets[idx]), reverse=True)

            for anchor_idx in sorted_anchors:
                if anchor_idx in unwound_turn_sets or anchor_idx >= len(block.turns):
                    continue

                anchor_turn = block.turns[anchor_idx]
                target_sid = anchor_turn.session_id
                same_sess_indices = [i for i, t in enumerate(block.turns) if t.session_id == target_sid]

                if is_single_sess or len(hit_turn_sets[anchor_idx]) >= 4:
                    bubble_indices = same_sess_indices
                else:
                    pos_in_sess = same_sess_indices.index(anchor_idx) if anchor_idx in same_sess_indices else 0
                    p_start = max(0, pos_in_sess - 2)
                    p_end = min(len(same_sess_indices), pos_in_sess + 4)
                    bubble_indices = same_sess_indices[p_start:p_end]

                bubble_turns = [block.turns[p] for p in bubble_indices]
                for p in bubble_indices:
                    unwound_turn_sets.add(p)

                bubbles.append(TranscriptionBubble(
                    date_str=block.date_str,
                    session_id=target_sid,
                    anchor_turn_idx=anchor_idx,
                    turns=bubble_turns,
                    anchor_words=sorted(hit_turn_sets[anchor_idx])
                ))

        return bubbles


class MolecularSoup:
    """Stage 4: Dual-stream thermodynamic hybridization with BM25 length normalization."""

    @staticmethod
    def _compile_gu_patterns(gu_probes: List[str]) -> List[Tuple[str, str, bool, Optional[re.Pattern]]]:
        compiled = []
        for probe in gu_probes:
            if _is_cjk(probe):
                compiled.append((probe, probe, True, None))
            else:
                probe_stem = _stem_plural(probe)
                if len(probe) >= 4:
                    pat = re.compile(
                        r"(?<![A-Za-z0-9])(?:" + re.escape(probe) + r"|" + re.escape(probe_stem) + r"s?)(?![A-Za-z0-9])"
                    )
                else:
                    pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(probe) + r"(?![A-Za-z0-9])")
                compiled.append((probe, probe_stem, False, pat))
        return compiled

    @staticmethod
    def _compile_mi_patterns(mi_probes: List[str]) -> List[Tuple[str, str, bool, Optional[re.Pattern]]]:
        compiled = []
        for probe in mi_probes:
            if _is_cjk(probe):
                compiled.append((probe, probe, True, None))
            else:
                probe_stem = _stem_plural(probe)
                if len(probe) >= 4:
                    pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(probe_stem))
                else:
                    pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(probe) + r"s?(?![A-Za-z0-9])")
                compiled.append((probe, probe_stem, False, pat))
        return compiled

    @classmethod
    def hybridize(
        cls,
        bubbles: List[TranscriptionBubble],
        gu_probes: List[str],
        mi_probes: List[str],
        target_dt: Optional[datetime.datetime] = None
    ) -> Tuple[List[Tuple[Turn, float, str]], List[Tuple[Turn, float, str]]]:
        """Score candidate turns across rigid exact and flexible association streams."""
        exact_facts_dict: Dict[int, Tuple[Turn, float, str]] = {}
        fuzzy_contexts_dict: Dict[int, Tuple[Turn, float, str]] = {}

        gu_set = set(gu_probes)
        gu_patterns = cls._compile_gu_patterns(gu_probes)
        mi_patterns = cls._compile_mi_patterns([p for p in mi_probes if p not in gu_set])

        seen_turns: Dict[int, Turn] = {}
        for b in bubbles:
            for t in b.turns:
                seen_turns[t.turn_id] = t

        causal_candidates: Dict[int, Tuple[int, List[str]]] = {}
        hit_turn_ids_in_bubbles: Set[int] = set()

        for turn_id, turn in seen_turns.items():
            content_low = turn.content.lower()
            content_norm = re.sub(r"(?<=\d),(?=\d{3}\b)", "", content_low)

            gu_hit_score = 0.0
            gu_hit_probes: List[str] = []
            for probe, probe_stem, is_cn, pat in gu_patterns:
                if is_cn:
                    hit = probe in content_low or probe in content_norm
                else:
                    hit = (probe_stem in content_norm) and bool(pat.search(content_norm))
                if hit:
                    gu_hit_score += 5.0
                    gu_hit_probes.append(probe)

            if gu_hit_score > 0:
                hit_turn_ids_in_bubbles.add(turn_id)
                text_len = max(1, len(content_low))
                b_penalty = 0.55 + 0.45 * (text_len / 35.0)
                exact_facts_dict[turn_id] = (turn, gu_hit_score / b_penalty, ", ".join(gu_hit_probes))

            matched_gu_prefixes = {p[:4] for p in gu_hit_probes if not _is_cjk(p) and len(p) >= 4}
            mi_hit_score = 0.0
            mi_hit_probes: List[str] = []
            for probe, probe_stem, is_cn, pat in mi_patterns:
                if probe in matched_gu_prefixes:
                    continue
                if is_cn:
                    hit = probe in content_low or probe in content_norm
                else:
                    hit = (probe_stem in content_norm) and bool(pat.search(content_norm))
                if hit:
                    mi_hit_score += 2.0
                    mi_hit_probes.append(probe)

            if mi_hit_score > 0:
                hit_turn_ids_in_bubbles.add(turn_id)
                text_len = max(1, len(content_low))
                b_penalty = 0.55 + 0.45 * (text_len / 35.0)
                fuzzy_contexts_dict[turn_id] = (turn, mi_hit_score / b_penalty, ", ".join(mi_hit_probes))

        # Preserve [-2, +3] causal neighbors around matched turns and pure-temporal fallback turns.
        for b in bubbles:
            if b.anchor_words == ["__temporal__"]:
                for t in b.turns:
                    if t.turn_id not in exact_facts_dict and t.turn_id not in fuzzy_contexts_dict:
                        b_penalty = 0.55 + 0.45 * (max(1, len(t.content)) / 35.0)
                        fuzzy_contexts_dict[t.turn_id] = (t, 1.0 / b_penalty, "temporal_window")
                continue

            anchor_indices = [i for i, t in enumerate(b.turns) if t.turn_id in hit_turn_ids_in_bubbles]
            if not anchor_indices:
                continue
            for i, t in enumerate(b.turns):
                if t.turn_id in exact_facts_dict or t.turn_id in fuzzy_contexts_dict:
                    continue
                # Respect asymmetric [-2, +3] window around any matched anchor inside the bubble.
                for a_i in anchor_indices:
                    offset = i - a_i
                    if -2 <= offset <= 3:
                        dist = abs(offset)
                        prev = causal_candidates.get(t.turn_id)
                        if prev is None or dist < prev[0]:
                            causal_candidates[t.turn_id] = (dist, b.anchor_words)
                        break

        for turn_id, (dist, anchor_words) in causal_candidates.items():
            if turn_id not in exact_facts_dict and turn_id not in fuzzy_contexts_dict:
                turn = seen_turns[turn_id]
                b_penalty = 0.55 + 0.45 * (max(1, len(turn.content)) / 35.0)
                score = 0.35 / (b_penalty * (1.0 + 0.25 * dist))
                motifs = ", ".join(anchor_words[:3])
                fuzzy_contexts_dict[turn_id] = (turn, score, f"causal_context({motifs})")

        exact_facts = list(exact_facts_dict.values())
        fuzzy_contexts = list(fuzzy_contexts_dict.values())

        if target_dt is not None:
            def _apply_decay(items: List[Tuple[Turn, float, str]]) -> List[Tuple[Turn, float, str]]:
                decayed = []
                for turn, sc, pr in items:
                    day_diff = abs((parse_date_dt(turn.date_str) - target_dt).days)
                    penalty = max(0.2, 1.0 - (day_diff / 60.0) * 0.8)
                    decayed.append((turn, sc * penalty, pr))
                return decayed

            exact_facts = _apply_decay(exact_facts)
            fuzzy_contexts = _apply_decay(fuzzy_contexts)

        return exact_facts, fuzzy_contexts


class DualStreamPresenter:
    """Formats Top-K scored turns chronologically into an LLM context prompt."""

    _ROLE_MAP = {
        "user": "User",
        "assistant": "Assistant",
        "system": "System",
        "document": "Document",
    }

    @classmethod
    def _format_role(cls, role: str) -> str:
        if not role:
            return "Assistant"
        return cls._ROLE_MAP.get(role.lower(), role.capitalize())

    @staticmethod
    def format_presentation(
        query: str,
        exact_facts: List[Tuple[Turn, float, str]],
        fuzzy_contexts: List[Tuple[Turn, float, str]],
        max_facts: int = 20,
        max_contexts: int = 20,
        reference_date: Optional[str] = None
    ) -> str:
        """Assemble isolated exact-fact and associative-context streams."""
        prompt_parts = [
            "You are answering the user's question based on retrieved dual-stream memory records.",
            f"Current Reference Date: {reference_date or 'Unknown'}",
            "Respect the distinct roles of both memory streams:\n",
            "=" * 30 + " [STREAM 1: Exact Factual Anchors (Rigid Probes)] " + "=" * 30,
            "[Role: High-precision factual records, numbers, entities, and verifiable events.]\n"
        ]

        if exact_facts:
            top_facts = sorted(exact_facts, key=lambda x: (x[1], x[0].turn_id), reverse=True)[:max_facts]
            top_facts.sort(key=lambda x: (x[0].date_str, x[0].timestamp, x[0].turn_id))
            for turn, _, probes in top_facts:
                role_tag = DualStreamPresenter._format_role(turn.role)
                prompt_parts.append(f"- [{turn.date_str} | {role_tag}]: \"{turn.content}\"  (Probes: {probes})")
        else:
            prompt_parts.append("(No exact factual records retrieved)")

        prompt_parts.extend([
            "\n" + "=" * 30 + " [STREAM 2: Associative Context & Preferences (Flexible Probes)] " + "=" * 30,
            "[Role: Synaptic context, background details, habits, and implicit user preferences.]\n"
        ])

        if fuzzy_contexts:
            top_contexts = sorted(fuzzy_contexts, key=lambda x: (x[1], x[0].turn_id), reverse=True)[:max_contexts]
            top_contexts.sort(key=lambda x: (x[0].date_str, x[0].timestamp, x[0].turn_id))
            for turn, _, probes in top_contexts:
                role_tag = DualStreamPresenter._format_role(turn.role)
                prompt_parts.append(f"- [{turn.date_str} | {role_tag}]: \"{turn.content}\"  (Motifs: {probes})")
        else:
            prompt_parts.append("(No associative context records retrieved)")

        prompt_parts.extend([
            "\n" + "=" * 80,
            "[Reasoning & Answering Instructions]",
            "1. FACTUAL & COUNTING: Rely strictly on verbatim timestamps and entities. State the direct answer in the first sentence.",
            "2. KNOWLEDGE UPDATES: Always prioritize the record with the MOST RECENT timestamp.",
            "3. DATE & TIME CALCULATIONS: Perform step-by-step calendar arithmetic using [YYYY-MM-DD] and the Current Reference Date.",
            "4. PREFERENCES: Synthesize clues from Stream 2 to give tailored recommendations; do not abstain if clues exist.",
            "-" * 80,
            f"Question: {query}",
            "Answer:"
        ])

        return "\n".join(prompt_parts)


class FunesEngine:
    """High-level interface for the Funes biomimetic dual-stream memory engine."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize engine in-memory (db_path=None) or persistent SQLite WAL mode."""
        self.store = DNAMemoryStore(db_path=db_path)

    def add_turn(self, session_id: str, date_str: str, role: str, content: str, timestamp: float = 0.0) -> Turn:
        """Ingest a single conversational turn."""
        return self.store.append_turn(session_id, date_str, role, content, timestamp)

    def add_conversation_turn(self, session_id: str, date_str: str, role: str, content: str, timestamp: float = 0.0) -> Turn:
        """Alias for add_turn."""
        return self.add_turn(session_id, date_str, role, content, timestamp)

    def batch_add_turns(self, turns_data: List[Dict[str, Any]]):
        """Bulk ingest a list of turn dictionaries."""
        tuples_data = [
            (item["session_id"], item["date_str"], item["role"], item["content"], item.get("timestamp", 0.0))
            for item in turns_data
        ]
        self.store.batch_append_turns(tuples_data)

    def batch_add_conversation_turns(self, turns_data: List[Dict[str, Any]]):
        """Alias for batch_add_turns."""
        self.batch_add_turns(turns_data)

    def retrieve(self, query: str, reference_date: Optional[str] = None) -> DualStreamResult:
        """Execute 4-stage dual-stream retrieval for a given query."""
        t0 = time.perf_counter()

        parsed = Transducer.transduce(query, hebbian_net=self.store.hebbian_net, reference_date=reference_date)
        signal = parsed["remo_signal"]
        gu_probes = parsed["gu_probes"]
        mi_probes = parsed["mi_probes"]

        all_dates = sorted(list(self.store.blocks.keys()))
        _, narrow_temporal_targets = _compute_temporal_targets(all_dates, signal)

        active_blocks = Remo.remodel(self.store, signal)
        candidate_terms = set(gu_probes + mi_probes)
        bubbles = Heli.unwind(
            active_blocks,
            candidate_terms,
            is_single_sess=parsed["is_single_sess"],
            temporal_target_dates=narrow_temporal_targets
        )

        target_dt = None
        if signal.get("target_ago_days") is not None:
            effective_ref = signal.get("reference_date") or (all_dates[-1] if all_dates else "1970-01-01")
            ref_dt = parse_date_dt(effective_ref)
            target_dt = ref_dt - datetime.timedelta(days=signal["target_ago_days"])

        exact_facts, fuzzy_contexts = MolecularSoup.hybridize(bubbles, gu_probes, mi_probes, target_dt=target_dt)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return DualStreamResult(
            exact_facts=exact_facts,
            fuzzy_contexts=fuzzy_contexts,
            opened_blocks=[b.date_str for b in active_blocks],
            active_bubbles=len(bubbles),
            latency_ms=latency_ms,
            reference_date=reference_date
        )

    def build_prompt(self, query: str, result: Optional[DualStreamResult] = None, reference_date: Optional[str] = None) -> str:
        """Format retrieved memory streams into a structured LLM prompt."""
        if result is None:
            result = self.retrieve(query, reference_date=reference_date)
        ref_d = reference_date or result.reference_date
        return DualStreamPresenter.format_presentation(query, result.exact_facts, result.fuzzy_contexts, reference_date=ref_d)

    def build_llm_prompt(self, query: str, result: DualStreamResult, reference_date: Optional[str] = None) -> str:
        """Alias for build_prompt."""
        return self.build_prompt(query, result=result, reference_date=reference_date)

    def close(self):
        """Release storage resources cleanly."""
        self.store.close()
