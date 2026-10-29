"""记忆冲突消解 —— 本项目技术深度最高的一块。

为什么需要它
------------
只做「写入 + 检索 + 遗忘」，库里的记忆会互相矛盾而系统毫无察觉：

  · 你三个月前记下「用像素法判断水印」，昨天校准出「像素法是错的，要用服务端参数」；
  · 两条都会被召回，模型同时拿到两个矛盾的结论，只能瞎猜；
  · 更糟的是旧的那条可能因为被引用得多，强度还更高，排在前面。

这就是「记忆中毒」：错误或过时的记忆一旦进入库，会持续污染后续所有推理。
遗忘衰减能缓解（旧记忆会淡出），但它是**时间驱动**的，对付不了「新记忆刚写入、
旧记忆还很新鲜」这种即时冲突。

冲突消解做三件事
----------------
  1. **检测**：判断两条记忆是不是在说同一件事（同主题 + 高相似度）。
  2. **判向**：谁取代谁？—— **晚写入的优先**（新知识覆盖旧知识），
     但同批次、无法区分先后时，用「来源可信度 + 具体程度」做仲裁。
  3. **消解**：把旧的标记为 superseded 并链接到新的，检索默认不再返回旧的。

判向的边界（诚实说清）
--------------------
「新的覆盖旧的」不是永远对的——旧记忆可能才是对的（比如新记忆是模型幻觉写进去的）。
所以这里只做**结构化的候选提示 + 保守自动消解**：
  · 相似度极高 + 明确的时间先后 → 自动 supersede；
  · 相似度高但证据不足 → 只记 relation=contradicts，交给上游（人/LLM）裁决，不擅自删。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .embed import char_ngrams, tokenize
from .store import Memory, MemoryStore


@dataclass
class ConflictParams:
    """冲突判定阈值。

    重要教训（踩过的坑）：TF-IDF 余弦相似度是**语料相关**的——同一个真冲突对，
    在 3 条的小库里算出来 0.26，在 413 条的库里能到 0.7。所以绝不能硬编一个
    绝对阈值。这里改用**语料无关的复合信号 + 按分布自校准的阈值**：

      composite = 0.5·词重叠系数 + 0.3·字重叠系数 + 0.2·标题命中

    重叠系数(|A∩B|/min(|A|,|B|))不依赖 IDF，对长度差异更宽容，小语料也稳。
    阈值则取「观测分布的百分位」与「经验下限」的较大者——有数据时自适应，
    没数据时退回保守下限。
    """

    similarity_threshold: float = 0.40   # 经验下限（实测真冲突 ≥0.43，无关 ≈0）
    auto_resolve_threshold: float = 0.62  # 自动取代的较高门槛
    min_time_gap_days: float = 1.0        # 认定「新覆盖旧」所需的最小时间差（天）
    adaptive: bool = True                 # 是否用分布百分位抬高阈值
    adaptive_percentile: float = 99.0     # 自适应阈值取这个百分位


@dataclass
class ConflictPair:
    old: Memory
    new: Memory
    similarity: float
    action: str            # "supersede" | "flag"（只标记待裁决）
    reason: str


@dataclass
class ResolutionReport:
    n_scanned: int = 0
    n_pairs: int = 0
    superseded: List[ConflictPair] = field(default_factory=list)
    flagged: List[ConflictPair] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"扫描 {self.n_scanned} 条 / 发现 {self.n_pairs} 对疑似冲突 → "
            f"自动取代 {len(self.superseded)} 对，待裁决 {len(self.flagged)} 对"
        )


def _title(text: str) -> str:
    """取记忆的标题段（本项目约定用「标题｜正文」）。"""
    return text.split("｜", 1)[0].strip()


def _overlap(a: set, b: set) -> float:
    """重叠系数 = |A∩B| / min(|A|,|B|)。比 Jaccard 宽容，且不依赖 IDF。"""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _composite_similarity(
    sets_a: Tuple[set, set, str], sets_b: Tuple[set, set, str]
) -> float:
    """语料无关的复合相似度，落在 [0,1]。入参是**预计算好的** (词集, 字集, 标题)。

    三个互补信号加权：
      · 词重叠系数（0.5）：抓核心术语是否一致
      · 字 2-gram 重叠系数（0.3）：兜分词错误、生造词
      · 标题命中（0.2）：「标题｜正文」约定下，标题相同是「同主题」的强证据

    注意：分词是 O(文本长度) 的贵操作，必须**在配对循环外预计算一次**。
    早期版本在双重循环里对每个对重新分词，413 条库要跑 17 万次 jieba，直接卡死。
    """
    wa, wb = sets_a[0], sets_b[0]
    ca, cb = sets_a[1], sets_b[1]
    title_hit = 1.0 if sets_a[2] and sets_a[2] == sets_b[2] else 0.0
    return 0.5 * _overlap(wa, wb) + 0.3 * _overlap(ca, cb) + 0.2 * title_hit


def _prepare(text: str) -> Tuple[set, set, str]:
    """把一条记忆预处理成可复用的比较单元（词集 / 字集 / 标题）。"""
    return set(tokenize(text)), set(char_ngrams(text, 2)), _title(text)


def _pairwise_similarity(texts: Sequence[str]) -> np.ndarray:
    n = len(texts)
    sim = np.zeros((n, n), dtype=np.float64)
    prepared = [_prepare(t) for t in texts]  # 只分词一次
    for i in range(n):
        for j in range(i + 1, n):
            s = _composite_similarity(prepared[i], prepared[j])
            sim[i, j] = sim[j, i] = s
    return sim


def _effective_threshold(pairs: np.ndarray, base: float, params: ConflictParams) -> float:
    """自适应阈值：取「经验下限」与「观测分布百分位」的较大者。

    有数据时（库里记忆够多），分布百分位能把阈值抬到真正该报警的位置，避免
    小语料/大批量时把无关记忆误判为冲突；数据太少时退回保守下限。
    """
    if not params.adaptive or pairs.size < 50:
        return base
    pct = float(np.percentile(pairs, params.adaptive_percentile))
    return max(base, pct)


def detect_conflicts(
    memories: Sequence[Memory],
    params: ConflictParams = ConflictParams(),
) -> List[Tuple[Memory, Memory, float]]:
    """两两比对，找出相似度超阈值的记忆对（potential conflict）。

    复杂度 O(n²)。n 很大时应先按 kind/标签分桶或上向量索引做近邻召回，
    这里为保持零依赖、逻辑透明，直接全比——评测库几百条量级够用。
    """
    if len(memories) < 2:
        return []
    texts = [m.text for m in memories]
    sim = _pairwise_similarity(texts)
    n = len(memories)
    iu = np.triu_indices(n, 1)
    all_pairs = sim[iu]
    thr = _effective_threshold(all_pairs, params.similarity_threshold, params)

    out: List[Tuple[Memory, Memory, float]] = []
    for i, j in zip(*iu):
        s = float(sim[i, j])
        if s >= thr:
            a, b = memories[i], memories[j]
            old, new = (a, b) if a.created_at <= b.created_at else (b, a)
            out.append((old, new, s))
    return out


def _decide(old: Memory, new: Memory, sim: float, params: ConflictParams) -> Tuple[str, str]:
    """判定这对冲突怎么处理。返回 (action, reason)。"""
    dt_days = (new.created_at - old.created_at) / 86400.0
    same_source = bool(old.source) and old.source == new.source

    if sim >= params.auto_resolve_threshold and dt_days >= params.min_time_gap_days:
        return "supersede", (
            f"相似度 {sim:.2f} 极高且新记忆晚 {dt_days:.1f} 天写入 → 新知识覆盖旧知识"
        )
    if same_source and dt_days >= params.min_time_gap_days:
        return "supersede", (
            f"同源文档内的更新（晚 {dt_days:.1f} 天）→ 新版本覆盖旧版本"
        )
    # 其余情况证据不足，只标记，不擅自判决
    return "flag", (
        f"相似度 {sim:.2f} 达到冲突线但时间/来源证据不足 → 仅标记待裁决，不自动删"
    )


def resolve(
    store: MemoryStore,
    params: ConflictParams = ConflictParams(),
    auto_apply: bool = True,
    only_active: bool = True,
) -> ResolutionReport:
    """扫描全库、检测冲突、按策略消解。

    auto_apply=True 时，「supersede」判决会真正落地：旧记忆 state→superseded，
    并写一条 memory_links 记录。False 时只报告不改库（用于先看效果）。
    """
    mems = store.actives() if only_active else store.all()
    report = ResolutionReport(n_scanned=len(mems))
    pairs = detect_conflicts(mems, params)
    report.n_pairs = len(pairs)

    for old, new, sim in pairs:
        action, reason = _decide(old, new, sim, params)
        pair = ConflictPair(old=old, new=new, similarity=sim, action=action, reason=reason)
        if action == "supersede":
            report.superseded.append(pair)
            if auto_apply:
                store.set_state(old.id, "superseded")
                store.link(new.id, old.id, "supersedes", score=sim, reason=reason)
        else:
            report.flagged.append(pair)
            if auto_apply:
                store.link(new.id, old.id, "contradicts", score=sim, reason=reason)
    return report


def apply_flag_decision(store: MemoryStore, keep_id: int, drop_id: int, reason: str = ""):
    """对一条待裁决（flagged）的冲突做人工/上游裁决：保留 keep，取代 drop。"""
    store.set_state(drop_id, "superseded")
    store.link(keep_id, drop_id, "supersedes", score=1.0, reason=reason or "人工裁决保留")
