"""混合检索 —— 把「文本相关性」和「记忆强度」合成最终排序。

排序分数 = 相关度权重 × 混合相关性 + 强度权重 × 记忆强度

为什么要混强度：纯相关度检索会让「三个月前的过时结论」和「昨天刚校准的结论」
打平甚至超过。加上强度项后，越常用、越新的记忆天然占优，过时记忆自动沉底。
这正是在真实 Agent 里「召回该记住的、淡出该忘的」的工程实现。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .decay import DecayParams, compute_all_strengths
from .embed import HybridEmbedder
from .semantic import SemanticEmbedder
from .store import Memory, MemoryStore


@dataclass
class Scored:
    memory: Memory
    score: float          # 最终融合分
    relevance: float      # 纯文本相关度
    strength: float       # 记忆强度


def _minmax(x: np.ndarray) -> np.ndarray:
    """把一维数组 min-max 归一到 [0,1]；全相等时返回全 0（该项不产生区分度）。"""
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


class HybridRetriever:
    def __init__(
        self,
        store: MemoryStore,
        word_weight: float = 0.6,
        char_weight: float = 0.4,
        strength_weight: float = 0.35,
        params: Optional[DecayParams] = None,
        only_active: bool = True,
        backend: str = "lexical",
        semantic_weight: float = 0.0,
    ):
        """backend:
          - "lexical"（默认）：词 TF-IDF + 字 n-gram 双通道，零重依赖
          - "semantic"：ONNX 语义嵌入；**加载失败自动降级 lexical**，绝不中断
          - "hybrid"：词法 + 语义融合（semantic_weight 控制语义占比）
        """
        assert backend in ("lexical", "semantic", "hybrid")
        self.store = store
        self.word_weight = word_weight
        self.char_weight = char_weight
        self.strength_weight = strength_weight
        self.params = params or DecayParams()
        # 默认只检索 active 记忆：被取代（superseded）的记忆不再进入结果，
        # 这是冲突消解真正生效的地方——检测只是找到矛盾，不召回才是解决矛盾。
        self.only_active = only_active
        self.backend = backend
        self.semantic_weight = semantic_weight
        self.active_backend = "lexical"  # 实际生效的后端（可能因降级而变）
        self.degrade_reason = ""
        self.embedder: Optional[HybridEmbedder] = None
        self.semantic: Optional[SemanticEmbedder] = None
        self._mem_ids: List[int] = []
        self._mem_by_id: dict[int, Memory] = {}
        self._word_mat: Optional[np.ndarray] = None
        self._char_mat: Optional[np.ndarray] = None
        self._sem_mat: Optional[np.ndarray] = None

    # ---- 建索引 ------------------------------------------------------
    def build(self):
        mems = self.store.actives() if self.only_active else self.store.all()
        self._mem_ids = [m.id for m in mems]
        self._mem_by_id = {m.id: m for m in mems}
        corpus = [m.text for m in mems]

        # 词法通道总是建（它也是语义失败时的兜底）
        self.embedder = HybridEmbedder().fit(corpus) if corpus else HybridEmbedder()
        if corpus:
            self._word_mat, self._char_mat = self.embedder.transform(corpus)
        else:
            self._word_mat = np.zeros((0, 0))
            self._char_mat = np.zeros((0, 0))

        # 语义通道按需建；失败则记录原因并降级
        self.active_backend = "lexical"
        if self.backend in ("semantic", "hybrid") and corpus:
            self.semantic = SemanticEmbedder()
            if self.semantic._ensure_loaded():
                try:
                    self._sem_mat = self.semantic.transform(corpus)
                    self.active_backend = self.backend
                except Exception as e:  # noqa: BLE001 - 兜底，绝不中断
                    self.degrade_reason = f"语义编码失败，降级词法：{e}"
                    self._sem_mat = None
            else:
                self.degrade_reason = self.semantic.reason
        return self

    # ---- 相关度计算 --------------------------------------------------
    def _relevance(self, query: str) -> np.ndarray:
        """按当前生效后端算 query 对每条记忆的相关度。"""
        qw, qc = self.embedder.transform_one(query)
        rel_w = (
            self._word_mat @ qw
            if self._word_mat is not None and self._word_mat.shape[1] == qw.shape[0]
            else np.zeros(len(self._mem_ids))
        )
        rel_c = (
            self._char_mat @ qc
            if self._char_mat is not None and self._char_mat.shape[1] == qc.shape[0]
            else np.zeros(len(self._mem_ids))
        )
        lexical = self.word_weight * rel_w + self.char_weight * rel_c

        if self.active_backend in ("semantic", "hybrid") and self._sem_mat is not None:
            # 查询侧用 transform_query：BGE 系列要求 query 加指令前缀、文档不加，
            # 两侧编码方式本就不同，不能图省事用同一个函数。
            qs = self.semantic.transform_query(query)
            sem = self._sem_mat @ qs
            if self.active_backend == "semantic":
                return sem
            return (1 - self.semantic_weight) * lexical + self.semantic_weight * sem
        return lexical

    # ---- 检索 --------------------------------------------------------
    def search(
        self,
        query: str,
        k: int = 5,
        now: Optional[float] = None,
        min_strength: float = 0.0,
    ) -> List[Scored]:
        if not self._mem_ids or self.embedder is None:
            return []
        relevance = self._relevance(query)

        strengths = compute_all_strengths(list(self._mem_by_id.values()), now, self.params)
        s_vec = np.array([strengths.get(i, self.params.floor) for i in self._mem_ids])

        # 关键：两个信号量纲不同。TF-IDF 相关度在短文本里通常落在 [0, 0.3]，
        # 而强度在 [0, 1]。直接线性混合会让强度一项压倒相关度（权重还没调就失衡）。
        # 先把两者各自 min-max 归一化到 [0,1]，strength_weight 才是一个真正的旋钮。
        rel_n = _minmax(relevance)
        str_n = _minmax(s_vec)

        scores = (1 - self.strength_weight) * rel_n + self.strength_weight * str_n

        order = np.argsort(-scores)
        out: List[Scored] = []
        for idx in order:
            mid = self._mem_ids[idx]
            s = float(strengths.get(mid, self.params.floor))
            if s < min_strength:
                continue
            if float(relevance[idx]) <= 0 and float(scores[idx]) <= 0:
                continue
            out.append(
                Scored(
                    memory=self._mem_by_id[mid],
                    score=float(scores[idx]),
                    relevance=float(relevance[idx]),
                    strength=float(s),
                )
            )
            if len(out) >= k:
                break
        return out

    def recall(
        self,
        query: str,
        k: int = 5,
        now: Optional[float] = None,
        reinforce: bool = True,
    ) -> List[Scored]:
        """检索 + 强化：这一步让记忆「用进废退」真正闭环。

        被召回的每条记忆都会 touch() 一次（记访问日志）并抬升强度。
        reinforce=False 可用于纯评测（避免评测本身改变记忆状态）。
        """
        results = self.search(query, k=k, now=now)
        if reinforce:
            for r in results:
                self.store.touch(r.memory.id, query=query, score=r.score, now=now)
        return results
