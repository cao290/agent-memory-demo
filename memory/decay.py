"""遗忘衰减 + 使用强化 —— 本项目最核心、也最区别于普通 RAG 的部分。

普通 RAG 的所有文档块权重相等，永远躺在库里等着被检索，于是：
  - 库里越堆越多，检索噪声越来越大；
  - 三个月前的一条过时结论和昨天刚校准的一条，被一视同仁地召回；
  - 冲突信息同时召回，模型对着两条矛盾的记忆瞎猜。

记忆系统的解法是给每条记忆一个会随时间变化的 **强度 (strength)**：

    strength(t) = 衰减(时间) × 强化(使用)

  · 衰减：艾宾浩斯式遗忘曲线，strength = exp(-Δt / τ)，久不用就自然淡出。
  · 强化：被召回一次就「复习」一次，抬高记忆的稳定度；
          强化遵循间距效应——刚记住马上又用，增益小；隔一阵再用，增益大。

最终强度同时影响 **检索排序**（强度低的排在后面）和 **淘汰**（低于阈值进入待遗忘区）。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List

from .store import Memory


@dataclass
class DecayParams:
    """遗忘曲线参数。默认值来自「一天粒度、两周半衰」的直觉设定，可调。"""

    half_life_days: float = 14.0      # 记忆强度衰减到一半所需天数（未强化状态下）
    reinforcement_gain: float = 0.35  # 每次被召回抬升稳定度的基准幅度
    spacing_bonus: float = 0.25       # 间距效应：间隔越久再复习，增益越大
    floor: float = 0.02               # 强度下限，避免完全归零（归零就无法分辨谁更该被淘汰）
    ceiling: float = 1.0              # 强度上限

    @property
    def decay_lambda(self) -> float:
        # exp(-Δt/τ) 的半衰期换算：τ = half_life / ln2
        return math.log(2) / max(self.half_life_days, 1e-6)


def days_between(a: float, b: float) -> float:
    return max(0.0, (a - b) / 86400.0)


def decayed_strength(
    mem: Memory,
    now: float,
    params: DecayParams = DecayParams(),
) -> float:
    """计算一条记忆在 `now` 时刻的强度。

    - 从未被访问过的记忆，从创建时间算起衰减；
    - 被访问过的记忆，因为每次访问都抬高了「记忆稳定度」，衰减更慢
      （用 effective_tau 体现：访问越多，遗忘越慢，这是「长时记忆」的直觉）。
    """
    last = mem.last_access if mem.access_count > 0 else mem.created_at
    dt = days_between(now, last)
    # 稳定度随访问次数增长（1 + gain*count），使常被用的记忆衰减更慢
    stability = 1.0 + params.reinforcement_gain * mem.access_count
    tau_eff = params.half_life_days * stability
    lam = math.log(2) / max(tau_eff, 1e-6)
    base = mem.strength if mem.access_count == 0 else 1.0
    val = base * math.exp(-lam * dt)
    return max(params.floor, min(params.ceiling, val))


def reinforce(
    mem: Memory,
    now: float,
    hits: int = 1,
    params: DecayParams = DecayParams(),
) -> float:
    """一次召回后，返回该记忆更新后的强度。

    强化幅度 = 基础增益 × 间距系数：
      - 距上次访问越久（但还没忘干净），这次复习的增益越大（间距效应）；
      - 反之刚访问完又访问，增益很小（冗余复习）。
    之后仍立即叠加一点衰减，保证「同一天反复召回」不会无限刷分。
    """
    if mem.access_count == 0:
        gap_days = days_between(now, mem.created_at)
    else:
        gap_days = days_between(now, mem.last_access)
    # 间距系数：gap 越大越接近 1 + spacing_bonus，越小越接近 1
    spacing = 1.0 + params.spacing_bonus * (1 - math.exp(-gap_days))
    cur = decayed_strength(mem, now, params)
    boosted = cur + params.reinforcement_gain * hits * spacing
    return max(params.floor, min(params.ceiling, boosted))


def compute_all_strengths(
    memories: List[Memory],
    now: float | None = None,
    params: DecayParams = DecayParams(),
) -> dict[int, float]:
    """给全库记忆批量算当前强度，返回 {id: strength}。"""
    now = now if now is not None else time.time()
    return {m.id: decayed_strength(m, now, params) for m in memories}


def consolidation_candidates(
    memories: List[Memory],
    now: float,
    params: DecayParams = DecayParams(),
    weak_threshold: float = 0.15,
) -> tuple[List[Memory], List[Memory]]:
    """把记忆分成两类，供「压缩」和「遗忘」两条策略使用：

    - weak:    强度 < 阈值 —— 候选遗忘，或压缩成一句话后归档；
    - strong:  强度 >= 阈值 —— 保留，参与正常检索。
    """
    weak, strong = [], []
    for m in memories:
        s = decayed_strength(m, now, params)
        (weak if s < weak_threshold else strong).append(m)
    return weak, strong
