<div align="center">

# 🧠 FUNES (HLM)
### An Ultra-Lightweight, Zero-Embedding Biomimetic Memory Retrieval Engine & Fast Edge Search Architecture
### 一种超轻量、零向量双流仿生记忆检索与端侧高速搜索架构

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23921389.svg)](https://doi.org/10.5281/zenodo.23921389)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)]()
[![Zero Dependencies](https://img.shields.io/badge/dependencies-0%20(Pure%20Python)-brightgreen.svg)]()
[![Hardware](https://img.shields.io/badge/GPU%20VRAM-0%20MB-success.svg)]()
[![Memory](https://img.shields.io/badge/RAM-~18.6%20MB-purple.svg)]()

**Author / 作者**: Sherris Isalah (艾浩喆) · `sherjia192@gmail.com`  
**Preprint Paper / 预印本论文**: [Zenodo Record #23921389](https://zenodo.org/records/23921389) · [DOI: 10.5281/zenodo.23921389](https://doi.org/10.5281/zenodo.23921389)

[ **English** ](#-english) | [ **中文说明** ](#-中文说明) | [ **📌 Design & Benchmark Notes** ](#-usage-principles--benchmark-disclaimer)

</div>

---

<a name="usage-principles--benchmark-disclaimer"></a>
## 📌 Usage Principles & Benchmark Disclaimer / 核心使用原则与评测说明

> [!IMPORTANT]
> **1. Query with Unmodified Natural Language / 直接使用原生自然语言提问**  
> Pass user questions **as-is** in natural language (`engine.retrieve(query)` / `engine.build_prompt(query)`). Do **not** rewrite queries into keyword tricks or hardcode domain trigger words.
>
> **2. Associative Memory Architecture / 联想检索架构定位**  
> Funes is fundamentally an **associative memory retrieval architecture**. Its role is to rapidly recall relevant historical context streams for an LLM to reference and reason over—rather than acting as a standalone answer oracle without downstream model reasoning.
>
> **3. Independent Benchmark Reference (Not Official Leaderboard) / 非官方指定模型跑分声明**  
> End-to-end QA benchmarks reported below were evaluated using **third-party / open-weight models** (e.g., `Qwen/Qwen3.5-4B` with `glm-5.3-flash` judge, or `DeepSeek-V4.1-Flash` in the preprint) rather than LongMemEval's officially designated leaderboard model/judge (GPT-4o). Because they do not use the official leaderboard model setup, these numbers are provided as **empirical reference points only** and have **not** been submitted to the official LongMemEval leaderboard.

---

<a name="english"></a>
## 🌐 English

### ⚡ Highlights & Engineering Advantages

Funes (internal codename **HLM / Human-Like Memory**) provides a CPU-native alternative to heavy vector databases and LLM-driven memory loops. Built with **100% pure Python standard library (zero external dependencies)**, Funes models long-term memory through calendar-partitioned chromatin blocks, Hebbian co-occurrence associations, and square-root document-frequency silencing.

- 🚀 **~18.6 MB Resident RAM & Ultra-Fast Ingestion**: Compact in-memory or SQLite WAL footprint (`add_turn` / `batch_add_turns` ingest via lightweight lexical/Hebbian updates with zero neural inference delay).
- ❄️ **0 MB GPU VRAM & 0 Retrieval Tokens**: No embedding model weights, no LLM extraction/reflection loops during ingestion or search (`$0.00` retrieval cost).
- 🎯 **Pure Natural-Language Associative Retrieval**: Feed unmodified user queries directly—no query rewriting or keyword engineering needed (`~98%` Any-Session Hit Rate & `~92%` Full-Session Recall on LongMemEval-S 500Q).
- 📜 **Lossless Verbatim Storage & Verbatim Recall**: Stores and returns exact original turns with timestamps and roles—eliminating hallucination or detail loss caused by lossy LLM summarization.
- 🔍 **Fully Auditable & Transparently Tunable**: Deterministic scoring across explicit probe hits, opened `DateBlock`s, Hebbian steps, and causal window/decay constants—every retrieved turn is traceable with zero black-box embeddings, and parameters are clean to tune.
- 🗂️ **Structured Isolation (`session_id` / `date_str`)**: Indexed turn-level SQLite rows (`dna_turns`) avoid irreversible merged summaries or opaque vector graphs, making per-session/document-level isolation and cleanup straightforward to manage or extend.
- 🔒 **100% Local & Privacy-First**: Runs strictly offline with single-file SQLite WAL persistence—no data ever leaves your machine.
- 🌐 **Native EN/ZH Bilingual & Easy Script Extension**: `UniversalShingleTokenizer` natively handles English stems/shingles, Chinese 2-gram shingles, and mixed EN/ZH text, with simple regex hooks to extend to additional languages.
- 🛠️ **Tiny Footprint, High Maintainability & C++-Ready**: The entire core engine lives in a single ~993-line [`funes.py`](funes.py) file; because it relies solely on string scanning, hash maps, and scalar scoring, porting it to **C++ or Rust** is straightforward and expected to slash retrieval latency even further.

---

### 📊 Benchmark Overview (ICLR 2025 LongMemEval-S, 500 Questions)

#### 1. Retrieval Layer (CPU-Only, Clean Zero-Keyword Public Release)

Evaluated across all 500 questions of LongMemEval-S with engine state reset between questions:

| Architecture | Paradigm | Any-Session Hit Rate | Full-Session Recall | Total 500-Q Runtime | Resource Profile |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Vanilla BM25** | Lexical Inverted Index | ~73.4% | ~52.8% | ~48 s | ~16 MB RAM / 0 VRAM |
| **Dense Vector RAG (ChromaDB)** | `text-embedding-3-small` | ~86.4% | ~68.2% | ~340 s | ~1.45 GB RAM / ~4 GB VRAM |
| **Mem0 (Hybrid Graph+Vector)** | Dense + Triplet Graph | ~89.6% | ~71.0% | ~1,820 s | ~3.2 GB RAM / ~8 GB VRAM |
| **Funes 1.0 (Public Release)** | **Biomimetic Dual-Stream** | **~98.0% – 98.2%** | **~91.8% – 92.0%** | **~71 – 120 s** | **~18.6 MB RAM / 0 VRAM ($0.00)** |

*Note: Direct normalized substring overlap on raw retrieved turns via `benchmark.py` is roughly `~43%` in the clean keyword-free release (as many LongMemEval answers require synthesis, preference alignment, or temporal arithmetic by the downstream LLM rather than verbatim substring matching).*

#### 2. Downstream End-to-End QA Reference (Edge 4B & Cloud Models)

| Memory Engine + Answering Model | Judge Model | Overall Accuracy (500 Q) | Avg Prompt Tokens / Q | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Funes (Public Release) + `Qwen/Qwen3.5-4B`** | `glm-5.3-flash` | **70.40%** (352/500) | **~4,815** | Pure zero-keyword release, `enable_thinking=False` |
| HLM V1.6.0 + `Qwen/Qwen3.5-4B` (V2 Prompt) | `DeepSeek-V4.1-Flash` | 64.80% (324/500) | ~8,743 | Legacy architecture reference |
| HLM V1.6.0 + `Qwen/Qwen3.5-4B` (V1 Prompt) | `DeepSeek-V4.1-Flash` | 49.40% (247/500) | ~8,675 | Legacy architecture baseline |
| Funes + `DeepSeek-V4.1-Flash` (Preprint) | `glm-5.3-flash` | ~93.20% | — | Preprint cloud-reader reference |

> *Disclaimer: None of the end-to-end runs above used LongMemEval's official GPT-4o evaluation setup; figures are shared for transparent reference only and are not official leaderboard submissions.*

---

### 💡 What Can You Build With Funes? (Use Cases)

Because Funes is a **single-file, zero-dependency Python kernel** with SQLite WAL persistence and `0 MB` GPU VRAM overhead, it drops cleanly into existing LLM workflows:

- **🧩 Edge & Local LLM Long-Term Memory**: Pair directly with local models deployed via Ollama, `llama.cpp`, vLLM, or LM Studio (e.g., 4B–8B edge models). Retrieval uses `0 MB` of GPU VRAM—leaving all memory for model weights—while compressing long histories into concise prompts (`~4,815` avg tokens in our `Qwen/Qwen3.5-4B` reference run).
- **🤖 Drop-In AI Agent Tool / MCP Memory Backend**: Expose `FunesEngine` (`add_turn`, `retrieve`, `build_prompt`) as a `save_memory` / `recall_memory` tool for agent frameworks or MCP servers—just `from funes import FunesEngine` with a single local `.db` file, no Docker or external vector-DB stack required.
- **💸 Prompt Token Saver for Cloud APIs**: Place Funes on CPU in front of cloud APIs (DeepSeek, GPT, Claude, etc.) for assistants, roleplay, or customer-support bots. Instead of stuffing huge chat histories into every API call, retrieve only relevant causal conversation snippets at `$0.00` retrieval cost.
- **🔒 Offline & Privacy-First Knowledge Archives**: Run completely air-gapped on laptops, mini-PCs, or edge devices—ideal for private journals, local personal assistants, or searching across massive long-form drafts, worldbuilding notes, and character timelines.

---

### 🧬 Biomimetic Architecture

```mermaid
flowchart TD
    Q["Raw Natural Language Query (q)"] --> T["1. Transducer (Signal Transduction)"]

    subgraph S1 ["Multi-Track Probe Synthesis"]
        T --> T1["Track A: Rigid Probes (gRNA Tokens)"]
        T --> T2["Track B: Hebbian Synaptic Network (miRNA Probes)"]
        T --> T3["Track C: Relative Temporal Offset Vector"]
    end

    T1 & T2 & T3 --> R["2. Remo (Chromatin Remodeling Engine)"]

    subgraph S2 ["Topological Domain Activation"]
        R --> R1["TAD Calendar DateBlocks (A/B Compartment Hydration)"]
        R --> R2["Super-Enhancer Synergy: len^1.5"]
        R --> R3["Inactive Block Dehydration"]
    end

    R2 --> H["3. Heli (Transcription Helicase)"]

    subgraph S3 ["Causal Transcription Bubble"]
        H --> H1["Session-Boundary Isolation"]
        H --> H2["Asymmetric Causal Window [-2, +3]"]
    end

    H2 --> M["4. MolecularSoup (Dual-Stream Hybridization)"]

    subgraph S4 ["Dual-Stream Scoring"]
        M --> M1["Stream 1: Exact Probe Boundary Match + BM25 Norm"]
        M --> M2["Stream 2: Associative Resonance + sqrt(df+1) Silencing"]
    end

    M1 & M2 --> P["5. DualStreamPresenter (Chronological Context Assembly)"]
    P --> Out["Dual-Stream Reference Context for LLM"]
```

---

### 🚀 Quickstart & Examples

Zero installation dependencies required (Python 3.8+):

```bash
# Run quickstart demo
python examples/quickstart.py

# Run edge local-search demo
python examples/local_search.py

# Run unit tests
python -m unittest discover -s tests -v
```

Minimal embedded usage ([examples/quickstart.py](examples/quickstart.py)):

```python
from funes import FunesEngine

# Use db_path=None for pure RAM, or pass a path for persistent SQLite WAL mode
engine = FunesEngine(db_path="./agent_memory.db")

engine.add_turn(
    session_id="sess_01",
    date_str="2026-05-01",
    role="user",
    content="I bought a 16-inch MacBook Pro in Seattle for $2,499.",
)

# Query directly with raw natural language
prompt = engine.build_prompt(
    "What laptop did I buy in Seattle?",
    reference_date="2026-05-15",
)
# Pass `prompt` to any LLM as memory context
```

You can also inspect [examples/local_search.py](examples/local_search.py), [tests/test_funes.py](tests/test_funes.py), and [benchmark.py](benchmark.py).

---

### 🎯 Deep Domain Adaptation & Specialized Terminology Tuning Guide

The public release of Funes is intentionally configured with clean, unbiased standard-library rules to ensure zero dataset cheating for general academic benchmarks. When deploying Funes in specialized, high-barrier vertical domains (such as **visas/immigration, medicine, law, finance, or industrial engineering**), queries often feature **long multi-word technical compounds** (e.g., *"certificate of non-criminal record"* / *"无犯罪记录证明"*), **special-symbol codes** (e.g., `B1/B2`, `DS-160`, `214(b)`), and **industry slang/synonyms** (e.g., *"consular interview"* ↔ *"visa interview"*).

You can easily adapt Funes to deep vertical domains **without modifying a single line of internal engine code** using the following zero-overhead strategies:

#### 1. Cold-Start Synaptic Seeding (Recommended — Pure Conversation Turns)
Funes models human-like Hebbian associative memory: concepts that co-occur in the same conversational turn automatically form strong synaptic links. You can inject domain knowledge and synonyms simply by ingesting 1–2 background glossary turns during system setup:

```python
# Ingest domain terminology mappings as background seed turns
engine.add_turn(
    session_id="system_knowledge_seed",
    date_str="2026-01-01",
    role="system",
    content=(
        "DS-160 is the online nonimmigrant visa application form. "
        "B1/B2 visa consular interview is commonly referred to as visa interview or in-person interview. "
        "Administrative processing 221(g) is known as check. "
        "Police clearance certificate is also called certificate of non-criminal record."
    )
)
```
*How it works*: Funes automatically wires strong associative synaptic edges (`adj[token_a][token_b]`) between these terms. When a user asks *"Has the DS-160 been submitted before the visa interview?"*, the associative stream instantly fires and recalls historical records that used *"consular interview"* without needing an LLM loop.

#### 2. Direct Pre-population of the SQLite `associations` Synapse Table
For enterprise systems with pre-existing domain dictionaries or synonym graphs, you can batch-insert domain associations directly into Funes's native SQLite table (`associations`):

```python
import sqlite3, time

# Connect to the persistent Funes SQLite database
conn = sqlite3.connect("./agent_memory.db")
cur = conn.cursor()

# Insert domain synonym pairs with prior associative weight (e.g., 2.5 - 3.5)
domain_pairs = [
    ("面签", "领事馆面试", 3.0),
    ("check", "行政审查", 3.0),
    ("资金证明", "银行流水", 2.5),
    ("i20", "录取证明", 2.5)
]

now_ts = time.time()
for a, b, weight in domain_pairs:
    cur.execute("INSERT OR REPLACE INTO associations (a_token, b_token, weight, last_used) VALUES (?, ?, ?, ?)", (a, b, weight, now_ts))
    cur.execute("INSERT OR REPLACE INTO associations (a_token, b_token, weight, last_used) VALUES (?, ?, ?, ?)", (b, a, weight, now_ts))

conn.commit()
conn.close()
```

#### 3. Surface-Level Symbol Normalization
Standard lexical tokenizers split on punctuation marks like `/` and `()`. For domain acronyms featuring symbols (`B1/B2` → `B1B2` or `B1_B2`, `214(b)` → `214b`), applying a simple 2-line regex replacement or dictionary mapping before calling `engine.add_turn` and `engine.retrieve` keeps them intact as atomic, high-priority rigid probes.

---

### 🧪 Reproducing the Retrieval Benchmark

1. Place `longmemeval_s.json` under `data/longmemeval_s.json` (see [data/README.md](data/README.md)).
2. Run a quick 10-question smoke check or the full 500-question evaluation:
```bash
python benchmark.py --limit 10
python benchmark.py --data data/longmemeval_s.json
```
3. Typical clean-release output across 500 questions:
   - `Any-Session Hit Rate`: `~98%` (`~490–491 / 500`)
   - `Full-Session Recall`: `~92%` (`~459–460 / 500`)
   - `Total Runtime`: `~70–120 s` (depending on CPU)

---

<a name="中文说明"></a>
## 🇨🇳 中文说明

### 🌟 核心特性与工程优势

Funes（内部代号 **HLM / Human-Like Memory**）是一套 **100% 纯 Python 标准库实现、零外部依赖** 的轻量化长期记忆检索与端侧搜索架构。它不依赖任何向量数据库或嵌入模型，而是利用“日历拓扑分区（DateBlock）＋ 赫布共现联想网络 ＋ 表观遗传高频词衰减 ＋ 非对称因果滑窗”，在普通 CPU 上毫秒级捞出记忆线索，交给大模型做参考。

- 🚀 **极低内存与极速写入**：常驻内存约 `~18.6 MB`；写入阶段（`add_turn` / `batch_add_turns`）仅做轻量分词、共现统计与 SQLite WAL 插入，不跑任何向量推理或大模型提取，吞吐极快。
- ❄️ **零显存 / 零检索 Token 费用**：建库与检索阶段均不调用大模型或向量模型（检索算力成本 `$0.00`）。
- 🎯 **纯自然语言联想检索**：设计为**联想式记忆中枢**——无需改写问题或拼接检索词，直接传入用户原生自然语言，即可结合多探针与赫布共现网络召回相关线索。
- 📜 **原文写入、原文召回（零有损压缩）**：不做容易丢失细节或产生幻觉的“摘要改写”，SQLite 原文落盘、检索直接返回原始对话记录与精确日期角色信息。
- 🔍 **极强可审计性与参数可调**：全链路确定性打分（可清晰回溯命中的探针词、激活的 `DateBlock` 日期块与权重衰减），告别向量空间黑箱；赫布步长、窗口边界、衰减系数等核心参数全部明文透明，极易按业务微调。
- 🗂️ **按会话/文档结构化隔离**：底层 `dna_turns` 按 `session_id` 和 `date_str` 建索引存储（区别于合并摘要黑盒），天然支持按文档/会话精准定位、隔离与清理扩展。
- 🔒 **纯本地存储与检索，隐私性极强**：纯内存或单文件 SQLite WAL 本地运行，数据不出本地机器，零外部网络请求。
- 🌐 **原生支持中英双语且易扩多语言**：内置 `UniversalShingleTokenizer` 原生支持英文词干/shingle 与中文 2-gram 切分（及中英混排），只需补充字符集规则即可平滑扩展更多语种。
- 🛠️ **体量极小、易维护、极适于转译 C++/Rust**：整套核心引擎仅一个约 **993 行**的 [`funes.py`](funes.py) 单文件；且算法本质只涉及字符串扫描、哈希表与标量运算，几乎零迁移阻碍地转译为 **C++ / Rust** 后，可预期检索延迟还会进一步大幅下降。
- 🧪 **零作弊干净代码**：公开发布版 [`funes.py`](funes.py) 不含任何面向评测集特调的关键词词表。

---

### 💡 可以用来做什么？（典型应用场景）

由于 [`funes.py`](funes.py) 本质上是一个 **纯标准库单文件引擎 + SQLite WAL 单文件存储**，不占显卡显存也不需要额外拉起向量数据库服务，非常适合直接当作底层组件嵌入各类系统：

- **📱 端侧 / 本地部署大模型的长期记忆方案**：直接挂在 Ollama、`llama.cpp`、vLLM、LM Studio 部署的本地小模型（如 4B–8B）前面。检索阶段占用 `0 MB` 显存，不和推理抢 GPU 资源，又能把漫长历史对话浓缩成高信噪比上下文（例如我们实测接 `Qwen/Qwen3.5-4B` 时单题平均仅约 `4,815` tokens 即可跑出 `70.40%` 准确率）。
- **🤖 外挂为 AI Agent 工具 / MCP 记忆后端**：可以把 `FunesEngine` 的 `add_turn`、`retrieve`、`build_prompt` 直接封装成 Agent 的 `save_memory` / `recall_memory` 函数调用工具，或作为 MCP Server 的本地记忆库——一行 `from funes import FunesEngine` 就能跑，不用维护 Docker 和向量库集群。
- **💰 云端大模型对话省 Token 降本**：在调用商业 API（DeepSeek、GPT、Claude 等）的个人助理、陪伴或客服系统前充当零成本 CPU 预筛选器，避免每次对话都把几十万字全量历史硬塞进窗口导致账单爆炸。
- **🔒 纯离线、重隐私的本地资料与设定库检索**：全程不联网、不调外部 Embedding 接口、数据全留在本地 `.db` 文件中——适合个人私密笔记/日记助手、低配边缘设备，以及长篇创作中的人物设定、伏笔与时间线秒级召回。

---

### 📊 实测参考数据（LongMemEval-S 全量 500 题）

#### 1. 纯检索层表现（当前干净公开版实测）
- **Any-Session Hit Rate（任意目标会话命中率）**：**约 `98%`**（`~490–491 / 500`）
- **Full-Session Recall（完整目标会话全召回率）**：**约 `92%`**（`~459–460 / 500`）
- **500 题全量运行耗时**：**约 `70 – 120 秒`**（视本地 CPU 性能而定）
- **硬件与开销**：`~18.6 MB` 内存 / `0 MB` GPU 显存 / `0` 检索 Token

#### 2. 接下游大模型端到端问答参考（非官方榜单跑分）

| 记忆底座 + 答题模型 | 裁判模型 | 500 题准确率 | 单题平均输入 Token | 说明 |
| :--- | :---: | :---: | :---: | :--- |
| **Funes 公开版 + `Qwen/Qwen3.5-4B`** | `glm-5.3-flash` | **70.40%** (352/500) | **~4,815** | 4B 端侧小模型（关闭思考），零关键词调优 |
| HLM V1.6.0 + `Qwen/Qwen3.5-4B` (V2) | `DeepSeek-V4.1-Flash` | 64.80% (324/500) | ~8,743 | 历史版本对照 |
| HLM V1.6.0 + `Qwen/Qwen3.5-4B` (V1) | `DeepSeek-V4.1-Flash` | 49.40% (247/500) | ~8,675 | 历史初始基线 |
| Funes + `DeepSeek-V4.1-Flash`（论文） | `glm-5.3-flash` | 约 93.20% | — | 预印本云端大模型参考结果 |

> **声明**：以上端到端测试均使用第三方/开源模型进行独立对照实验，**未使用 LongMemEval 官方指定的标准评测模型（GPT-4o）**，因此不能直接等同于官方榜单排名，亦未上传至官方跑分榜，仅作工程与学术研究参考。

---

### 🛠️ 快速上手与复测

无需 `pip install` 任何第三方依赖（需 Python 3.8+）：

```bash
# 运行快速示例与本地检索示例
python examples/quickstart.py
python examples/local_search.py

# 运行单元测试
python -m unittest discover -s tests -v

# 复现 LongMemEval-S 基准测试（需按 data/README.md 放入 longmemeval_s.json）
python benchmark.py --data data/longmemeval_s.json
```

---

### 🎯 深度垂直领域专业词汇特调指导 (Domain Adaptation Guide)

Funes 公开版本为了严格保证学术基准评测（如 LongMemEval-S 500 题）的公信力，杜绝任何针对特定评测集的人工调优与作弊嫌疑，采用了通用、无偏的纯标准库规则。

当在**签证移民、法律合同、临床医疗、金融审计或工业工程**等高壁垒垂直领域落地时，用户输入往往包含大量**长复合专业词**（如“无犯罪记录证明”）、**带特殊符号的代码**（如 `B1/B2`、`DS-160`、`214(b)`）以及**行业行话/同义词**（如“面签” ↔ “领馆面试”、“check” ↔ “行政审查”）。

得益于 Funes 100% 透明的白盒架构，企业与开发者**无需修改底层任何代码**，即可通过以下三种策略完成即插即用的领域特调：

#### 策略 1：冷启动知识注入（推荐 —— 纯对话轮次即可完成）
Funes 具备人类记忆般的赫布共现特性（在同一对话轮次中共同出现的词汇，会自动拉起强突触关联）。系统部署或冷启动时，只需在后台作为背景轮次写入 1~2 条行业术语对照说明：

```python
from funes import FunesEngine

engine = FunesEngine(db_path="./agent_memory.db")

# 写入行业背景知识或同义词映射
engine.add_turn(
    session_id="system_domain_seed",
    date_str="2026-01-01",
    role="system",
    content=(
        "DS-160是非移民签证在线申请表；"
        "B1/B2签证领事馆面试通常被称为面签或面谈；"
        "行政审查221(g)在行业内被称为check或被check；"
        "无犯罪记录证明也称为未受刑事处分公证书。"
    )
)
```
*原理解析*：Funes 会立刻在这些词汇之间建立起强突触边（Hebbian Synapses）。之后即使客户提问说“面签需要带什么”，底层第二记忆流也会自然联想并共振召回包含“领事馆面试”的历史记录，完全不需要下游大模型额外做查询改写！

#### 策略 2：直接向 SQLite `associations` 突触表批量导入行业词库
对于已有专业术语表或同义词图谱的企业，可以直接利用 Funes 原生 SQLite 中的 `associations (a_token, b_token, weight, last_used)` 表进行批量注入：

```python
import sqlite3, time

conn = sqlite3.connect("./agent_memory.db")
cur = conn.cursor()

# 领域同义词/关联对及先验突触权重（建议 2.5 ~ 3.5）
domain_pairs = [
    ("面签", "领事馆面试", 3.0),
    ("check", "行政审查", 3.0),
    ("资金证明", "银行流水", 2.5),
    ("i20", "录取证明", 2.5)
]

now_ts = time.time()
for a, b, weight in domain_pairs:
    cur.execute("INSERT OR REPLACE INTO associations (a_token, b_token, weight, last_used) VALUES (?, ?, ?, ?)", (a, b, weight, now_ts))
    cur.execute("INSERT OR REPLACE INTO associations (a_token, b_token, weight, last_used) VALUES (?, ?, ?, ?)", (b, a, weight, now_ts))

conn.commit()
conn.close()
```

#### 策略 3：浅层符号归一化（保留特殊代码完整性）
通用分词器在遇到 `/` 或 `()` 等符号时会天然切断。若领域内存在大量形如 `B1/B2`、`214(b)` 的代码，建议在写入与检索前加入一行极简的正则归一化（如 `re.sub(r'(\w+)/(\w+)', r'\1\2', text)`，将 `B1/B2` 映射为 `B1B2`），即可确保作为高优先级硬探针一击即中。

---

## 📜 Preprint Paper & Citation / 预印本论文与引用

- **Zenodo Open Access**: [https://zenodo.org/records/23921389](https://zenodo.org/records/23921389)
- **DOI**: [https://doi.org/10.5281/zenodo.23921389](https://doi.org/10.5281/zenodo.23921389)

```bibtex
@article{isalah2026funes,
  title     = {When LLMs Fall into Forgetting, Where Do the Memories Go? A Cost-Compressed, Ultra-Lightweight Biomimetic Memory Retrieval Engine and Fast Edge Search Architecture},
  author    = {Isalah, Sherris (艾浩喆)},
  journal   = {Zenodo Preprint},
  year      = {2026},
  month     = {November},
  doi       = {10.5281/zenodo.23921389},
  url       = {https://zenodo.org/records/23921389}
}
```

---

## ⚖️ License & Commercial Dual-Licensing / 开源协议与商业双授权

- **Open-Source License ([GNU AGPLv3 / AGPL-3.0](LICENSE))**: Free for personal, academic, research, and open-source use under the terms of the GNU Affero General Public License v3.0. In accordance with AGPL-3.0, any modified version or derivative work—including **network/SaaS deployments**—must disclose its full source code under AGPL-3.0 and retain original attribution.
- **Commercial / Closed-Source Licensing（商业闭源双授权）**: If you wish to embed Funes into a proprietary closed-source product, commercial OEM firmware, or hosted commercial API without AGPL-3.0 source-disclosure obligations, please contact **`sherjia192@gmail.com`** for a commercial license.
- **Defensive Prior Art**: Core biomimetic four-stage pipeline disclosed openly as prior art alongside the Zenodo preprint (`10.5281/zenodo.23921389`).
- **Author & Contact**: Sherris Isalah (艾浩喆) · `sherjia192@gmail.com`

If you find Funes helpful for your agent or edge-AI project, feel free to leave a **Star**, open an **Issue / Pull Request**, or cite the Zenodo preprint!
