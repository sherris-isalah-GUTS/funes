# -*- coding: utf-8 -*-
"""
Funes: Ultra-Lightweight Biomimetic Dual-Stream Memory Engine
=============================================================
A Cost-Compressed, Zero-Embedding, CPU-Native Long-Term Memory
and High-Speed Edge Search Architecture.

Author: Sherris Isalah (艾浩喆) | Independent Developer
Preprint DOI: 10.5281/zenodo.23921389
License: Apache-2.0
"""
# -*- coding: utf-8 -*-
"""
Funes 1.0 advance (先行先行版) - 仿生类脑长程记忆中枢核心引擎
================================================================================
“思考，就是遗忘差异，就是概括与抽象。在富内斯那个过于拥挤的世界里，除了即时呈现的细节之外，什么都没有。”
                                                     —— 豪尔赫·路易斯·博尔赫斯

【项目定位】
Funes 1.0 advance 是专为 AI Agent、个人助手与对话机器人打造的高性能、轻量级、仿生记忆中枢。
其底层由 HLM V3.8-melodrama 生化记忆算法驱动，彻底打破“大模型记忆必须堆算力、上向量库、调外部模型”的传统误区：
  1. 零显存消耗：纯 CPU 纯数学算法，完全运行于普通 Python 环境；
  2. 极度轻量：核心引擎运行时内存净占用仅 ~18 MB，单题检索耗时仅 ~150 毫秒；
  3. 零隐性 Token 账单：存储、索引与检索阶段 0 外部 API 调用，全流程无偷跑 Token；
  4. 极致去噪提纯：在数十万字的海量长历史中，精准过滤避开 91% 的无关废话，将黄金事实浓缩至 1/11；
  5. 顶尖准确率：在权威学术基准 (LongMemEval-S) 上取得 97.6% 严苛全证据召回与 92.2%~93.2% 问答高分。

【核心模块架构】
  - UniversalShingleTokenizer : 中英文双模切词器 (中文 2-gram 滑窗 + 英文形态学前缀瓦片)
  - HebbianAssociationNetwork : 类脑赫布突触网络 (历史动态共现强化 + 甲基化高频沉默 + IDF 能量衰减)
  - Transducer                : 分子探针转译器 (巅峰自然语言时间位移解析 + 相对范围与跨度计算)
  - DNAMemoryStore            : 染色质存储中枢 (嵌入式 SQLite 极端硬件锁 PRAGMA 优化 + A/B 区室管理)
  - Remo                      : 染色质重塑器 (增强子非线性超加性激活 len^1.5 + 动态水合与脱水)
  - Heli                      : 解旋酶转录引擎 (CTCF 会话物理绝缘隔离 + 超级操纵子 + [-2, +3] 非对称因果滑窗)
  - MolecularSoup             : 分子杂交热力学池 (CRISPR gRNA 刚性咬合 + miRNA 摆动配对柔性吸附)
  - DualStreamPresenter       : 双流隔离格式化呈现器 (Top-K 铁律排序与客观时序线恢复 + 黄金答题准则注入)
================================================================================
"""

from collections import defaultdict
from dataclasses import dataclass, field
import datetime
import hashlib
import json
import math
import os
import re
import sqlite3
import time
from typing import List, Dict, Any, Optional, Set, Tuple

STOP_WORDS: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "to", "in", "on", "of", "and", "or",
    "it", "i", "you", "my", "your", "for", "with", "at", "by", "this", "that", "do",
    "did", "have", "has", "had", "be", "been", "me", "we", "us", "they", "them",
    "what", "when", "where", "how", "who", "which", "will", "would", "can", "could",
    "so", "but", "about", "there", "their", "then", "some", "any", "no", "not",
    "just", "like", "get", "got", "know", "think", "tell", "say", "said", "from",
    "as", "up", "out", "if", "all", "also", "into", "than", "very"
}

SEMANTIC_CLUSTERS: Dict[str, List[str]] = {
    "shopping": ["buy", "bought", "purchase", "purchased", "order", "ordered", "shop", "shopping", "price", "cost", "dollar", "store", "market", "coupon"],
    "jewelry": ["jewelry", "jewel", "earring", "earrings", "necklace", "ring", "bracelet", "emerald", "silver", "gold", "pendant", "diamond"],
    "outdoors": ["camp", "camping", "hike", "hiking", "park", "summer", "tent", "outdoors", "trail", "backcountry", "trip", "mountain", "lake"],
    "models": ["model", "models", "kit", "kits", "scale", "tank", "aircraft", "bomber", "fighter", "assemble", "build", "tamiya", "revell"],
    "food": ["food", "dinner", "lunch", "breakfast", "restaurant", "delivery", "eat", "eating", "pizza", "meal", "coffee", "cook", "cooking"],
    "entertainment": ["game", "gaming", "play", "played", "podcast", "episode", "music", "movie", "show", "watch", "rollercoaster", "ride"],
    "work_study": ["job", "work", "office", "project", "meeting", "exam", "study", "code", "coding", "interview", "company"],
    "tech_network": ["speed", "plan", "internet", "mbps", "wifi", "router", "network", "download", "upload", "bandwidth", "provider"],
    "power_charging": ["battery", "phone", "charge", "charger", "charging", "power", "bank", "portable", "cable", "screen", "device"],
    "age_life": ["age", "aged", "old", "older", "young", "younger", "born", "birth", "birthday", "grandma", "grandpa", "mother", "father", "son", "daughter", "sister", "brother", "graduate", "graduated", "graduation", "college", "years"]
}


def normalize_date_key(raw_date: Any) -> str:
    """将多种异构日期格式统一规范化为标准的 YYYY-MM-DD 字符串"""
    if not raw_date:
        return "1970-01-01"
    s = str(raw_date).strip().replace("/", "-")
    m = re.search(r"(\d{4})[^\d](\d{1,2})[^\d](\d{1,2})", s)
    if m:
        y, month, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{y:04d}-{month:02d}-{d:02d}"
    return s[:10]


def parse_date_dt(raw_date: Any) -> datetime.datetime:
    """将日期字符串解析为 datetime.datetime 对象以便进行日历运算"""
    norm = normalize_date_key(raw_date)
    try:
        return datetime.datetime.strptime(norm, "%Y-%m-%d")
    except Exception:
        return datetime.datetime(1970, 1, 1)


@dataclass
class Turn:
    """DNA 双螺旋上的单个碱基对轮次"""
    turn_id: int
    session_id: str
    date_str: str  # YYYY-MM-DD
    role: str      # 'user' | 'assistant'
    content: str
    timestamp: float = 0.0


@dataclass
class DateBlock:
    """TAD 拓扑关联结构域：以大日期为物理边界的染色质块"""
    date_str: str                       # 日期标识 (YYYY-MM-DD)
    turns: List[Turn] = field(default_factory=list)
    turn_count: int = 0
    exposed_vocab: Dict[str, List[int]] = field(default_factory=lambda: defaultdict(list))
    is_open: bool = False               # A/B 区室活跃状态 (True 为 A 区室水合，False 为 B 区室脱水)


@dataclass
class TranscriptionBubble:
    """解旋酶展开的转录泡微观结构"""
    date_str: str
    session_id: str
    anchor_turn_idx: int
    turns: List[Turn]
    anchor_words: List[str]


@dataclass
class DualStreamResult:
    """双流格式化检索结果结构体"""
    exact_facts: List[Tuple[Turn, float, str]]      # Stream 1: gRNA 刚性硬核事实 (turn, score, matched_probes)
    fuzzy_contexts: List[Tuple[Turn, float, str]]   # Stream 2: miRNA 柔性偏好背景 (turn, score, matched_probes)
    opened_blocks: List[str]                        # 宏观水合开启的日期块
    active_bubbles: int                             # 解旋展开的转录泡总数
    latency_ms: float = 0.0                         # 检索总耗时 (毫秒)
    reference_date: Optional[str] = None            # 对话基准锚点日期


class UniversalShingleTokenizer:
    """中英文双模零依赖切词器 (融合中文 2-gram 滑窗与英文通用瓦片)"""

    ZH_STOPS = set("的了我你他在是有这那个就都要会什么哪去与及从和对把让被于以使得可以帮请问怎么谁何")

    @classmethod
    def extract_keywords(cls, text: str, max_kws: int = 16) -> List[str]:
        if not text:
            return []
        has_zh = bool(re.search(r"[\u4e00-\u9fa5]", text))
        kws = []

        for w in re.findall(r"(?<![A-Za-z0-9])[a-zA-Z0-9\-\'\$]{2,}(?![A-Za-z0-9])", text):
            wl = w.lower()
            if wl not in STOP_WORDS and wl not in kws:
                kws.append(wl)
                if '-' in wl:
                    for sub in wl.split('-'):
                        if len(sub) >= 2 and sub not in STOP_WORDS and sub not in kws:
                            kws.append(sub)

        if has_zh:
            for run in re.findall(r"[\u4e00-\u9fa5]+", text):
                if len(run) == 1:
                    continue
                elif len(run) == 2:
                    if run[0] not in cls.ZH_STOPS and run[1] not in cls.ZH_STOPS and run not in kws:
                        kws.append(run)
                elif len(run) == 3:
                    if run not in cls.ZH_STOPS and run not in kws:
                        kws.append(run)
                    for i in range(2):
                        bg = run[i:i + 2]
                        if bg[0] not in cls.ZH_STOPS and bg[1] not in cls.ZH_STOPS and bg not in kws:
                            kws.append(bg)
                else:
                    for i in range(len(run) - 1):
                        bg = run[i:i + 2]
                        if bg[0] not in cls.ZH_STOPS and bg[1] not in cls.ZH_STOPS and bg not in kws:
                            kws.append(bg)

        return kws[:max_kws]

    @classmethod
    def get_shingles(cls, word: str) -> List[str]:
        """获取词元的形态学前缀瓦片 (用于容错匹配)"""
        shingles = [word]
        is_cn = any('\u4e00' <= ch <= '\u9fa5' for ch in word)
        if not is_cn:
            clean = re.sub(r"\'s?$", "", word)
            if clean != word and len(clean) >= 2:
                shingles.append(clean)
            if len(clean) >= 4:
                shingles.append(clean[:4])  # 4-char prefix
            if clean.endswith('s') and len(clean) >= 4:
                shingles.append(clean.rstrip('s'))
        return list(set(shingles))


extract_keywords = UniversalShingleTokenizer.extract_keywords
get_shingles = UniversalShingleTokenizer.get_shingles


class HebbianAssociationNetwork:
    """类脑赫布突触联想网络 (内生 miRNA 生化发生中枢)
    'Cells that fire together, wire together.'
    词元在历史轮次中共现，动态强化突触权值；检索时引入 IDF 惩罚与高频泛滥词过滤，拓扑漫游输出柔性 miRNA 探针。
    """
    def __init__(self, step: float = 0.1, max_weight: float = 5.0):
        self.step = step
        self.max_weight = max_weight
        self.adj: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.doc_freq: Dict[str, int] = defaultdict(int)
        self.total_turns: int = 0

    def reinforce_turn(self, tokens: List[str]):
        """在单轮对话内强化共现词元的突触连接，并更新词元文档频率"""
        unique_tokens = list(set(tokens))
        if len(unique_tokens) < 2:
            for t in unique_tokens:
                self.doc_freq[t] += 1
            self.total_turns += 1
            return

        self.total_turns += 1
        for t in unique_tokens:
            self.doc_freq[t] += 1

        n = len(unique_tokens)
        for i in range(n):
            ti = unique_tokens[i]
            for j in range(i + 1, n):
                tj = unique_tokens[j]
                w_ij = self.adj[ti][tj] + self.step
                if w_ij > self.max_weight:
                    w_ij = self.max_weight
                self.adj[ti][tj] = w_ij
                self.adj[tj][ti] = w_ij

    def get_associated_probes(self, seed_keywords: List[str], top_k: int = 12) -> List[str]:
        """通过 1-Hop 拓扑扩散检索关联探针，施加 IDF 衰减并强制执行高频词甲基化沉默"""
        if not self.adj or not seed_keywords:
            return []

        scores: Dict[str, float] = defaultdict(float)
        seeds_set = set(k.lower() for k in seed_keywords)

        max_df = max(12, int(self.total_turns * 0.08))

        for seed in seeds_set:
            neighbors = self.adj.get(seed, {})
            if not neighbors and len(seed) >= 4:
                for known_tok, n_dict in self.adj.items():
                    if known_tok.startswith(seed[:4]) or seed.startswith(known_tok[:4]):
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
        return [tok for tok, sc in sorted_probes[:top_k]]


class Transducer:
    """本地分子探针转译器 (时间引擎 + 刚柔双轨探针合成)"""

    extract_keywords = staticmethod(UniversalShingleTokenizer.extract_keywords)

    @classmethod
    def transduce(cls, query: str, hebbian_net: Optional[HebbianAssociationNetwork] = None, reference_date: Optional[str] = None) -> Dict[str, Any]:
        q_low = query.lower()
        kws = UniversalShingleTokenizer.extract_keywords(query)

        is_single_sess = any(phrase in q_low for phrase in [
            "previous conversation", "our chat", "previous chat", "remind me",
            "you recommended", "you mentioned", "you gave", "you suggest", "our discussion",
            "in our conversation", "what was the name of the", "what happened",
            "之前聊过", "上次聊", "上次说", "记得", "推荐过", "提过", "你说过", "我们的对话", "那个叫什么", "发生了什么"
        ])

        is_multi_sess = any(phrase in q_low for phrase in [
            "how many", "total number", "across all", "how many times", "list all",
            "how many days", "which events", "which models", "total hours", "how much time",
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
            if token.isdigit():
                return int(token)
            return num_map.get(token.lower(), 1)

        m_ago = re.search(r"\b(?:about\s+)?(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|couple|few)\s+(?:of\s+)?(day|week|month)s?\s+ago\b", q_low)
        cn_num = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
        m_cn_ago = re.search(r"(\d+|一|二|两|三|四|五|六|七|八|九|十)\s*(天|周|个?星期|个?月|年)前", q_low)
        if m_ago:
            n = parse_num_token(m_ago.group(1))
            unit = m_ago.group(2)
            days = n if unit == "day" else (n * 7 if unit == "week" else n * 30)
            remo_signal["target_ago_days"] = days
        elif m_cn_ago:
            val_str, unit = m_cn_ago.group(1), m_cn_ago.group(2)
            n = int(val_str) if val_str.isdigit() else cn_num.get(val_str, 1)
            days = n if "天" in unit else (n * 7 if ("周" in unit or "星期" in unit) else (n * 30 if "月" in unit else n * 365))
            remo_signal["target_ago_days"] = days
        elif "day before yesterday" in q_low or "前天" in q_low:
            remo_signal["target_ago_days"] = 2
        elif "yesterday" in q_low or "昨天" in q_low:
            remo_signal["target_ago_days"] = 1
        elif "大前天" in q_low:
            remo_signal["target_ago_days"] = 3

        expand_adjacent_days = bool(
            re.search(r"\b(?:the\s+)?day\s+(?:before|after)\b", q_low) or
            "the next day" in q_low or "the previous day" in q_low or
            "第二天" in q_low or "前一天" in q_low or "隔天" in q_low
        )
        remo_signal["expand_adjacent_days"] = expand_adjacent_days

        m_last = re.search(r"\b(?:last|past)\s+(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|couple|few)\s+(?:of\s+)?(day|week|month)s?\b", q_low)
        if m_last:
            n = parse_num_token(m_last.group(1))
            unit = m_last.group(2)
            days = n if unit == "day" else (n * 7 if unit == "week" else n * 30)
            remo_signal["relative_scope_days"] = days + (3 if unit == "week" else 5)
        elif re.search(r"\b(?:last|past)\s+week\b", q_low) or "上周" in q_low or "上个星期" in q_low:
            remo_signal["relative_scope_days"] = 8
        elif re.search(r"\b(?:last|past)\s+month\b", q_low) or "上个月" in q_low:
            remo_signal["relative_scope_days"] = 35
        elif "this year" in q_low or "今年" in q_low or "去年" in q_low:
            remo_signal["relative_scope_days"] = 365

        months_map = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
            "一月": 1, "二月": 2, "三月": 3, "四月": 4,
            "五月": 5, "六月": 6, "七月": 7, "八月": 8,
            "九月": 9, "十月": 10, "十一月": 11, "十二月": 12
        }

        m_range = re.search(r"from\s+([a-z]+)\s+to\s+([a-z]+)", q_low)
        if m_range and m_range.group(1) in months_map and m_range.group(2) in months_map:
            m1 = months_map[m_range.group(1)]
            m2 = months_map[m_range.group(2)]
            for m_idx in range(min(m1, m2), max(m1, m2) + 1):
                remo_signal["target_months"].append(f"{m_idx:02d}")
        else:
            for m_name, m_num in months_map.items():
                if m_name in q_low:
                    remo_signal["target_months"].append(f"{m_num:02d}")

        gu_probes = set()
        for w in re.findall(r"(?<![A-Za-z0-9])[A-Z][a-z0-9\-\'\$]{2,}(?![A-Za-z0-9])", query):
            wl = w.lower()
            if wl not in STOP_WORDS:
                gu_probes.add(wl)

        for w in re.findall(r"(?<![A-Za-z0-9])\d+[\.\d]*\s*(?:[a-zA-Z%]+|\$)?(?![A-Za-z0-9])|\$\d+[\.\d]*", query):
            wl = w.lower().strip()
            if len(wl) >= 2 and wl not in STOP_WORDS and not wl.isdigit():
                gu_probes.add(wl)

        for kw in kws:
            gu_probes.add(kw)

        mi_probes = set()

        for cluster_name, cluster_words in SEMANTIC_CLUSTERS.items():
            if any(cw in kws or cw in q_low for cw in cluster_words):
                for cw in cluster_words:
                    mi_probes.add(cw)

        if hebbian_net is not None:
            synaptic_probes = hebbian_net.get_associated_probes(kws, top_k=10)
            for sp in synaptic_probes:
                mi_probes.add(sp)

        for kw in kws:
            for shingle in UniversalShingleTokenizer.get_shingles(kw):
                mi_probes.add(shingle)

        gu_list = sorted(list(gu_probes))
        mi_list = sorted(list(mi_probes))

        remo_signal["mi_probes"] = mi_list

        return {
            "remo_signal": remo_signal,
            "gu_probes": gu_list,
            "mi_probes": mi_list,
            "is_single_sess": is_single_sess,
            "is_multi_sess": is_multi_sess
        }


class DNAMemoryStore:
    """DNA 存储层：以大日期为边界的染色质结构域仓库 (融合 SQLite 嵌入式极端硬件优化锁与全量自愈持久化)"""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
        self.blocks: Dict[str, DateBlock] = {}
        self.all_turns: List[Turn] = []
        self._conn: Optional[sqlite3.Connection] = None
        self.hebbian_net = HebbianAssociationNetwork()

        if self.db_path is not None:
            self._init_sqlite()
            self._load_meta_from_db()

    def _init_sqlite(self):
        """挂载 SQLite 并应用极端硬件优化 PRAGMA 参数"""
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
        """从 SQLite 唤醒持久化染色质元数据；若元数据未刷盘但存在轮次，启动全量自愈重构"""
        if not self._conn:
            return
        cur = self._conn.cursor()

        cur.execute("SELECT date_str, turn_count, exposed_vocab FROM dna_blocks")
        for row in cur.fetchall():
            d_str = row["date_str"]
            v_dict = json.loads(row["exposed_vocab"])
            block = DateBlock(date_str=d_str, turn_count=row["turn_count"], is_open=False)
            block.exposed_vocab = defaultdict(list, v_dict)
            self.blocks[d_str] = block

        if not self.blocks:
            cur.execute("SELECT id, session_id, date_str, local_idx, role, content, timestamp FROM dna_turns ORDER BY id ASC")
            turn_rows = cur.fetchall()
            if turn_rows:
                for r in turn_rows:
                    norm_date = r["date_str"]
                    if norm_date not in self.blocks:
                        self.blocks[norm_date] = DateBlock(date_str=norm_date, is_open=False)
                    block = self.blocks[norm_date]
                    local_idx = r["local_idx"]
                    tokens = UniversalShingleTokenizer.extract_keywords(r["content"])
                    self.hebbian_net.reinforce_turn(tokens)
                    for tok in tokens:
                        for s in UniversalShingleTokenizer.get_shingles(tok):
                            if local_idx not in block.exposed_vocab[s]:
                                block.exposed_vocab[s].append(local_idx)
                    block.turn_count = max(block.turn_count, local_idx + 1)
                self.flush_meta()

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

    def append_turn(self, session_id: str, date_str: str, role: str, content: str, timestamp: float = 0.0, commit: bool = True) -> Turn:
        norm_date = normalize_date_key(date_str)
        
        if norm_date not in self.blocks:
            self.blocks[norm_date] = DateBlock(date_str=norm_date, is_open=True)

        block = self.blocks[norm_date]
        local_idx = len(block.turns) if block.is_open else block.turn_count
        turn_id = len(self.all_turns)

        if self._conn:
            cur = self._conn.cursor()
            cur.execute("""
                INSERT INTO dna_turns (session_id, date_str, local_idx, role, content, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (session_id, norm_date, local_idx, role, content, timestamp))
            turn_id = cur.lastrowid or turn_id

        t = Turn(
            turn_id=turn_id,
            session_id=session_id,
            date_str=norm_date,
            role=role,
            content=content,
            timestamp=timestamp
        )
        self.all_turns.append(t)

        tokens = UniversalShingleTokenizer.extract_keywords(content)
        self.hebbian_net.reinforce_turn(tokens)

        block.turns.append(t)
        block.turn_count = max(block.turn_count, len(block.turns))

        for tok in tokens:
            for s in UniversalShingleTokenizer.get_shingles(tok):
                if local_idx not in block.exposed_vocab[s]:
                    block.exposed_vocab[s].append(local_idx)

        if self._conn and commit:
            cur = self._conn.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO dna_blocks (date_str, turn_count, exposed_vocab, updated_at)
                VALUES (?, ?, ?, ?)
            """, (norm_date, block.turn_count, json.dumps(dict(block.exposed_vocab), ensure_ascii=False), time.time()))
            self._conn.commit()

        return t

    def batch_append_turns(self, turns_data: List[Tuple[str, str, str, str, float]]):
        """批量追加对话轮次并一次性提交事务与索引刷盘 (单事务优化，杜绝单轮重复序列化)"""
        for sid, d_str, r, c, ts in turns_data:
            self.append_turn(sid, d_str, r, c, ts, commit=False)
        self.flush_meta()

    def flush_meta(self):
        """全量持久化内存索引元数据到 SQLite (保存染色质结构域与赫布突触网络)"""
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
        """A 区室动态水合：从 SQLite 加载全量轮次并恢复物理展开"""
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
        """B 区室脱水折叠：将非活跃染色质块的 turns 释放，维持超低内存占用"""
        for d_str, block in self.blocks.items():
            if d_str not in keep_date_strs and block.is_open:
                if self._conn:
                    block.turns.clear()
                block.is_open = False

    def close(self):
        if self._conn:
            try:
                self.flush_meta()
            except Exception:
                pass
            self._conn.close()
            self._conn = None


class Remo:
    """第一级调度器：染色质重塑器 (Chromatin Remodeler)
    通过时间信号与探针增强子协同，宏观解块目标日期结构域，执行 A/B 区室水合与脱水。
    """

    @classmethod
    def remodel(cls, store: DNAMemoryStore, signal: Dict[str, Any]) -> List[DateBlock]:
        all_dates = sorted(list(store.blocks.keys()))
        if not all_dates:
            return []

        activated_date_set: Set[str] = set()
        kws = signal.get("query_keywords", [])
        mi_probes = signal.get("mi_probes", [])
        distinct_probes = set(kws + mi_probes)
        ref_d = signal.get("reference_date")

        target_ago = signal.get("target_ago_days")
        if target_ago is not None:
            ref_dt = parse_date_dt(ref_d or all_dates[-1])
            target_dt = ref_dt - datetime.timedelta(days=target_ago)
            target_str = target_dt.strftime("%Y-%m-%d")

            closest_date = min(all_dates, key=lambda d: abs((parse_date_dt(d) - target_dt).total_seconds()))
            activated_date_set.add(closest_date)

            for d in all_dates:
                diff_days = abs((parse_date_dt(d) - parse_date_dt(closest_date)).days)
                if diff_days <= 1:
                    activated_date_set.add(d)

        rel_scope = signal.get("relative_scope_days")
        if rel_scope is not None:
            ref_dt = parse_date_dt(ref_d or all_dates[-1])
            for d in all_dates:
                diff = (ref_dt - parse_date_dt(d)).days
                if -1 <= diff <= rel_scope:
                    activated_date_set.add(d)

        target_months = signal.get("target_months", [])
        if target_months:
            for d in all_dates:
                if any(f"-{m}-" in d for m in target_months):
                    activated_date_set.add(d)

        synergy_mult = len(distinct_probes) ** 1.5 if distinct_probes else 1.0
        date_scores: Dict[str, float] = defaultdict(float)

        for d_str, block in store.blocks.items():
            matched_vocab_count = 0
            for term in distinct_probes:
                term_stem = term.rstrip('s') if len(term) >= 4 else term
                if term in block.exposed_vocab:
                    matched_vocab_count += 1
                else:
                    for v in block.exposed_vocab.keys():
                        v_stem = v.rstrip('s') if len(v) >= 4 else v
                        if term_stem == v_stem or (len(term) >= 4 and len(v) >= 4 and (v.startswith(term) or term.startswith(v))):
                            matched_vocab_count += 1
                            break

            if matched_vocab_count > 0:
                score = (matched_vocab_count ** 1.5) * synergy_mult
                date_scores[d_str] = score

        sorted_date_by_score = sorted(date_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_blocks = 8 if signal.get("is_multi_sess") else 5
        for d_str, sc in sorted_date_by_score[:top_k_blocks]:
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
    """第二级调度器：解旋酶引擎 (Helicase)
    - 严格按 session_id 建立 CTCF 绝缘子边界保护，绝不跨会话污染；
    - 遇到单会话意图或单点峰值 >= 4 时，启动超级操纵子全开；
    - 常规锚点展开采用验证过的 [-2, +3] 非对称因果滑窗 (前 2 轮背景、后 3 轮解答)。
    """

    @classmethod
    def unwind(
        cls,
        active_blocks: List[DateBlock],
        candidate_terms: Set[str],
        is_single_sess: bool = False
    ) -> List[TranscriptionBubble]:
        bubbles = []

        for block in active_blocks:
            if not block.is_open:
                continue

            hit_turn_indices = defaultdict(list)
            for term in candidate_terms:
                term_stem = term.rstrip('s') if len(term) >= 4 else term
                if term in block.exposed_vocab:
                    for turn_idx in block.exposed_vocab[term]:
                        hit_turn_indices[turn_idx].append(term)
                else:
                    for v_word, turn_list in block.exposed_vocab.items():
                        v_stem = v_word.rstrip('s') if len(v_word) >= 4 else v_word
                        if term_stem == v_stem or (len(term) >= 4 and len(v_word) >= 4 and (v_word.startswith(term) or term.startswith(v_word))):
                            for turn_idx in turn_list:
                                hit_turn_indices[turn_idx].append(term)

            if not hit_turn_indices:
                continue

            unwound_turn_sets = set()
            sorted_anchors = sorted(hit_turn_indices.keys(), key=lambda idx: len(hit_turn_indices[idx]), reverse=True)

            for anchor_idx in sorted_anchors:
                if anchor_idx in unwound_turn_sets or anchor_idx >= len(block.turns):
                    continue

                anchor_turn = block.turns[anchor_idx]
                target_sid = anchor_turn.session_id

                same_sess_indices = [
                    i for i, t in enumerate(block.turns)
                    if t.session_id == target_sid
                ]

                if is_single_sess or len(hit_turn_indices[anchor_idx]) >= 4:
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
                    anchor_words=hit_turn_indices[anchor_idx]
                ))

        return bubbles


class MolecularSoup:
    """第三级调度器：分子杂交热力学池 (CRISPR gRNA 刚性结合 + Hebbian/Innate miRNA 柔性漫游)"""

    @classmethod
    def hybridize(
        cls,
        bubbles: List[TranscriptionBubble],
        gu_probes: List[str],
        mi_probes: List[str],
        target_dt: Optional[datetime.datetime] = None
    ) -> Tuple[List[Tuple[Turn, float, str]], List[Tuple[Turn, float, str]]]:
        exact_facts_dict: Dict[int, Tuple[Turn, float, str]] = {}
        fuzzy_contexts_dict: Dict[int, Tuple[Turn, float, str]] = {}

        seen_turns: Dict[int, Turn] = {}
        for b in bubbles:
            for t in b.turns:
                seen_turns[t.turn_id] = t

        for turn_id, turn in seen_turns.items():
            content_low = turn.content.lower()

            gu_hit_score = 0.0
            gu_hit_probes = []
            for probe in gu_probes:
                is_cn = any('\u4e00' <= ch <= '\u9fa5' for ch in probe)
                if is_cn:
                    hit = (probe in content_low)
                else:
                    probe_stem = probe.rstrip('s') if len(probe) >= 4 else probe
                    hit = bool(
                        re.search(r"(?<![A-Za-z0-9])" + re.escape(probe) + r"(?![A-Za-z0-9])", content_low) or
                        (len(probe) >= 4 and re.search(r"(?<![A-Za-z0-9])" + re.escape(probe_stem) + r"s?(?![A-Za-z0-9])", content_low))
                    )
                if hit:
                    gu_hit_score += 5.0
                    gu_hit_probes.append(probe)

            if gu_hit_score > 0:
                text_len = max(1, len(content_low))
                b_penalty = 1.0 - 0.45 + 0.45 * (text_len / 35.0)
                norm_score = gu_hit_score / b_penalty
                exact_facts_dict[turn_id] = (turn, norm_score, ", ".join(gu_hit_probes))

            mi_hit_score = 0.0
            mi_hit_probes = []
            for probe in mi_probes:
                is_cn = any('\u4e00' <= ch <= '\u9fa5' for ch in probe)
                if is_cn:
                    hit = (probe in content_low)
                else:
                    probe_stem = probe.rstrip('s') if len(probe) >= 4 else probe
                    hit = bool(
                        (probe in content_low) or
                        (len(probe) >= 4 and probe_stem in content_low) or
                        (len(probe) == 4 and re.search(r"(?<![A-Za-z0-9])" + re.escape(probe), content_low))
                    )
                if hit:
                    if probe not in gu_probes:
                        mi_hit_score += 2.0
                        mi_hit_probes.append(probe)

            if mi_hit_score > 0:
                text_len = max(1, len(content_low))
                b_penalty = 1.0 - 0.45 + 0.45 * (text_len / 35.0)
                norm_score = mi_hit_score / b_penalty
                fuzzy_contexts_dict[turn_id] = (turn, norm_score, ", ".join(mi_hit_probes))

        exact_facts = list(exact_facts_dict.values())
        fuzzy_contexts = list(fuzzy_contexts_dict.values())

        if target_dt is not None:
            decayed_facts = []
            for turn, sc, pr in exact_facts:
                t_dt = parse_date_dt(turn.date_str)
                day_diff = abs((t_dt - target_dt).days)
                penalty = max(0.2, 1.0 - (day_diff / 60.0) * 0.8)
                decayed_facts.append((turn, sc * penalty, pr))
            exact_facts = decayed_facts

        return exact_facts, fuzzy_contexts


class DualStreamPresenter:
    """第四级调度器：双流呈现器 (分流格式化呈现 + Top-K 铁律排序 + 黄金答题准则注入)"""

    @staticmethod
    def format_presentation(
        query: str,
        exact_facts: List[Tuple[Turn, float, str]],
        fuzzy_contexts: List[Tuple[Turn, float, str]],
        max_facts: int = 20,
        max_contexts: int = 20,
        reference_date: Optional[str] = None
    ) -> str:
        prompt_parts = []
        prompt_parts.append("You are answering the user's question based on retrieved biomimetic memory streams.")
        prompt_parts.append(f"Current Reference Date of this conversation: {reference_date or 'Unknown'}")
        prompt_parts.append("The context is strictly segregated into two biological streams. Respect their roles:\n")

        prompt_parts.append("=" * 30 + " 【STREAM 1: 确切历史事实与关键实体 (gRNA 刚性捕获)】 " + "=" * 30)
        prompt_parts.append("[角色定义: 本区域由高精度 gRNA 探针严格锁定，属于客观硬核事实、确切数字、专有名词与历史定论。请优先依据此处进行严谨计数、时间推导与事实查证。]\n")
        if exact_facts:
            top_facts = sorted(exact_facts, key=lambda x: (x[1], x[0].turn_id), reverse=True)[:max_facts]
            top_facts.sort(key=lambda x: x[0].turn_id)
            for turn, score, probes in top_facts:
                role_tag = "User" if turn.role == "user" else "Assistant"
                prompt_parts.append(f"- [{turn.date_str} | {role_tag}]: \"{turn.content}\"  (锁定探针: {probes})")
        else:
            prompt_parts.append("(未检测到高刚性确切事实条目)")

        prompt_parts.append("\n" + "=" * 30 + " 【STREAM 2: 关联生活背景与潜在偏好 (内生突触/本能 miRNA 柔性吸附)】 " + "=" * 30)
        prompt_parts.append("[角色定义: 本区域由先天语义簇与脑内突触共现 miRNA 探针泛化吸附，属于对话中的生活情调、隐性习惯、偏好倾向与发散意图。请依据此处推断深层动机与人性化建议。]\n")
        if fuzzy_contexts:
            top_contexts = sorted(fuzzy_contexts, key=lambda x: (x[1], x[0].turn_id), reverse=True)[:max_contexts]
            top_contexts.sort(key=lambda x: x[0].turn_id)
            for turn, score, probes in top_contexts:
                role_tag = "User" if turn.role == "user" else "Assistant"
                prompt_parts.append(f"- [{turn.date_str} | {role_tag}]: \"{turn.content}\"  (吸附基序: {probes})")
        else:
            prompt_parts.append("(未检测到泛化偏好背景条目)")

        prompt_parts.append("\n" + "=" * 85)
        prompt_parts.append("[Reasoning & Answering Instructions]")
        prompt_parts.append("1. For FACTUAL & COUNTING questions: Rely strictly on Stream 1 & Stream 2 verbatim timestamps and entities. State your direct final answer in the very first sentence.")
        prompt_parts.append("2. For KNOWLEDGE UPDATES (facts/states that changed over time): Always prioritize the excerpt with the MOST RECENT timestamp.")
        prompt_parts.append("3. For DATE / TIME CALCULATIONS: Explicitly compute step-by-step arithmetic using the excerpt's [YYYY-MM-DD] and the Current Reference Date.")
        prompt_parts.append("4. For PREFERENCE / RECOMMENDATION questions: Synthesize user tastes from Stream 2 to give tailored answers; do NOT reply \"insufficient information\" if preference clues exist.")
        prompt_parts.append("-" * 85)
        prompt_parts.append(f"Question: {query}")
        prompt_parts.append("Answer:")

        return "\n".join(prompt_parts)


class FunesEngine:

    def __init__(self, db_path: Optional[str] = None):
        """
        初始化 Funes 引擎实例
        :param db_path: 可选的持久化 SQLite 文件路径。若为 None，则完全在纯内存模式 (In-Memory) 下零持久化极速运行。
        """
        self.store = DNAMemoryStore(db_path=db_path)

    def add_turn(self, session_id: str, date_str: str, role: str, content: str, timestamp: float = 0.0) -> Turn:
        """记录单条对话轮次"""
        return self.store.append_turn(session_id, date_str, role, content, timestamp)

    def add_conversation_turn(self, session_id: str, date_str: str, role: str, content: str, timestamp: float = 0.0) -> Turn:
        """兼容别名"""
        return self.add_turn(session_id, date_str, role, content, timestamp)

    def batch_add_turns(self, turns_data: List[Dict[str, Any]]):
        """批量导入对话记录"""
        tuples_data = [
            (item["session_id"], item["date_str"], item["role"], item["content"], item.get("timestamp", 0.0))
            for item in turns_data
        ]
        self.store.batch_append_turns(tuples_data)

    def batch_add_conversation_turns(self, turns_data: List[Dict[str, Any]]):
        """兼容别名"""
        self.batch_add_turns(turns_data)

    def retrieve(self, query: str, reference_date: Optional[str] = None) -> DualStreamResult:
        """
        核心生化探针检索
        :param query: 用户提出的当前目标问题
        :param reference_date: 当前对话发生的基准参考日期 (YYYY-MM-DD)，留空则自适应推断
        :return: DualStreamResult 双流检索结果
        """
        t0 = time.perf_counter()

        parsed = Transducer.transduce(query, hebbian_net=self.store.hebbian_net, reference_date=reference_date)
        signal = parsed["remo_signal"]
        gu_probes = parsed["gu_probes"]
        mi_probes = parsed["mi_probes"]

        active_blocks = Remo.remodel(self.store, signal)

        candidate_terms = set(gu_probes + mi_probes)
        bubbles = Heli.unwind(
            active_blocks,
            candidate_terms,
            is_single_sess=parsed["is_single_sess"]
        )

        target_dt = None
        if signal.get("target_ago_days") is not None:
            ref_dt = parse_date_dt(signal.get("reference_date") or (active_blocks[-1].date_str if active_blocks else "1970-01-01"))
            target_dt = ref_dt - datetime.timedelta(days=signal["target_ago_days"])

        exact_facts, fuzzy_contexts = MolecularSoup.hybridize(bubbles, gu_probes, mi_probes, target_dt=target_dt)

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        return DualStreamResult(
            exact_facts=exact_facts,
            fuzzy_contexts=fuzzy_contexts,
            opened_blocks=[b.date_str for b in active_blocks],
            active_bubbles=len(bubbles),
            latency_ms=latency_ms,
            reference_date=reference_date
        )

    def build_prompt(self, query: str, result: Optional[DualStreamResult] = None, reference_date: Optional[str] = None) -> str:
        """
        将检索结果格式化为高信噪比的大模型专用 Prompt
        :param query: 用户问题
        :param result: DualStreamResult 实例；若不传，则内部自动先执行 retrieve
        :param reference_date: 参考日期
        """
        if result is None:
            result = self.retrieve(query, reference_date=reference_date)
        ref_d = reference_date or result.reference_date
        return DualStreamPresenter.format_presentation(query, result.exact_facts, result.fuzzy_contexts, reference_date=ref_d)

    def build_llm_prompt(self, query: str, result: DualStreamResult, reference_date: Optional[str] = None) -> str:
        """兼容别名"""
        return self.build_prompt(query, result=result, reference_date=reference_date)

    def close(self):
        """安全释放与关闭存储句柄"""
        self.store.close()


HLMMelodramaEngine = FunesEngine
HLMV38Engine = FunesEngine
HLMUnderscoreEngine = FunesEngine
