"""agent-memory-demo · 一个带遗忘、强化与冲突消解的 Agent 长期记忆层。

模块：
  embed         零依赖向量化（词 TF-IDF + 字 n-gram 双通道）
  store         SQLite 记忆存储（强度/访问次数/最后访问时间/状态/记忆关系）
  decay         遗忘衰减 + 使用强化（记忆的生命周期）
  retrieval     混合检索（相关性 × 强度，只召回 active 记忆）
  conflict      冲突消解（检测 / 判向 / 消解，自动 + 待裁决两档）
  ingest        把 md 知识库抽成记忆单元
  eval          记忆生命周期评测（Recall@k / 权重扫描 / 新旧竞争）
  conflict_eval 冲突消解评测（污染消除 / 误伤率）
"""
from .conflict import ConflictParams, resolve
from .decay import DecayParams, decayed_strength, reinforce
from .embed import HybridEmbedder
from .retrieval import HybridRetriever, Scored
from .semantic import SemanticEmbedder
from .store import Memory, MemoryStore

__all__ = [
    "Memory",
    "MemoryStore",
    "DecayParams",
    "decayed_strength",
    "reinforce",
    "HybridEmbedder",
    "SemanticEmbedder",
    "HybridRetriever",
    "Scored",
    "ConflictParams",
    "resolve",
]
