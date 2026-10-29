"""评测：把「记忆层到底强在哪」量化出来，而不是自说自话。

关键设计：所有实验都跑在**老化过的**记忆库上。
刚写入的库里每条强度都是 1.0，强度项自然不起作用——这正是纯 RAG 的适用场景，
也是为什么很多 demo 看不出记忆层的价值。真实 Agent 跑几周后来问，才是记忆层的战场。

三段实验，对应面试时会被追问的三个问题：

  实验一 · 召回率   : 三种通道 Recall@k 谁高？（混合检索 > 纯词法；记忆层 > 无强度）
  实验二 · 新旧竞争  : 同主题有「旧版」和「新版」两条记忆时，谁能把新版排前？
                     —— 记忆层独有，纯 RAG 会把两条一视同仁甚至排反。
  实验三 · 上下文成本: 召回同样 k 条，谁的上下文更短？

评测集局限（诚实标注）：gold 由关键词自动生成，属弱标注；真实评测需人工标注金标准。
目标可复现、看相对差异，不宣布绝对分数。
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path
from typing import Dict, List, Set

from .decay import DecayParams
from .retrieval import HybridRetriever
from .store import MemoryStore

DAY = 86400.0
SEED = 42

QUERIES = {
    "流式下载卡住了怎么排查": ["看门狗", "停滞"],
    "水印怎么验证有没有去掉": ["水印"],
    "接口限流了怎么办": ["限流", "429", "403"],
    "公众号起号阅读量上不去": ["公众号"],
    "小说分镜怎么写": ["分镜"],
    "下载大文件速度慢": ["分段", "Range", "并行"],
}


def recall_at_k(retrieved: List[int], gold: Set[int]) -> float:
    return len(set(retrieved) & gold) / len(gold) if gold else 0.0


def build_gold(store: MemoryStore) -> Dict[str, Set[int]]:
    mems = store.all()
    gold = {}
    for q, keys in QUERIES.items():
        ids = {m.id for m in mems if any(k in m.text for k in keys)}
        if ids:
            gold[q] = ids
    return gold


def age_store(store: MemoryStore, now: float, stale_ratio: float = 0.6):
    """把一部分记忆「做旧」：拨回很久没用过。

    没有这一步，强度项是死代码——所有记忆都新鲜，记忆层和纯 RAG 表现完全一样。
    真实系统里「库里有大量陈旧记忆」才是常态，所以评测必须模拟它。
    """
    mems = store.all()
    rng = random.Random(SEED)
    idx = list(range(len(mems)))
    rng.shuffle(idx)
    n_stale = int(len(mems) * stale_ratio)
    stale_ids = set()
    for i in idx[:n_stale]:
        m = mems[i]
        # 陈旧记忆：250 天前最后访问、访问次数低
        back = now - rng.uniform(150, 300) * DAY
        store.conn.execute(
            "UPDATE memories SET last_access=?, access_count=?, created_at=? WHERE id=?",
            (back, rng.randint(0, 1), back, m.id),
        )
        stale_ids.add(m.id)
    # 新鲜记忆：最近刚用过几次
    for i in idx[n_stale:]:
        m = mems[i]
        store.conn.execute(
            "UPDATE memories SET last_access=?, access_count=?, created_at=? WHERE id=?",
            (now, rng.randint(2, 6), now - rng.uniform(1, 10) * DAY, m.id),
        )
    store.conn.commit()
    return stale_ids


def _make_retriever(store, mode: str, params: DecayParams) -> HybridRetriever:
    """三个通道 = 三种配置，其余代码完全一样，保证对照干净。"""
    if mode == "plain":     # 朴素 RAG：只按词相关度，无强度
        cfg = dict(word_weight=1.0, char_weight=0.0, strength_weight=0.0)
    elif mode == "hybrid":  # 词+字混合，仍无强度
        cfg = dict(word_weight=0.6, char_weight=0.4, strength_weight=0.0)
    else:                   # 完整记忆层：混合相关度 × 强度
        cfg = dict(word_weight=0.6, char_weight=0.4, strength_weight=0.35)
    return HybridRetriever(store, params=params, **cfg).build()


# ── 实验一：召回率（老化库上）+ 强度权重扫描 ────────────────────────
def exp_recall(store, gold, k: int, params: DecayParams, now: float) -> dict:
    out = {}
    for mode in ("plain", "hybrid", "memory"):
        r = _make_retriever(store, mode, params)
        scores, lens = [], []
        for q, gids in gold.items():
            hits = r.search(q, k=k, now=now)
            scores.append(recall_at_k([h.memory.id for h in hits], gids))
            lens.append(sum(len(h.memory.text) for h in hits))
        out[mode] = {
            "recall@k": round(sum(scores) / len(scores), 4),
            "avg_context_chars": round(sum(lens) / len(lens), 1),
        }
    return out


def exp_weight_sweep(store, gold, k: int, params: DecayParams, now: float) -> dict:
    """扫描强度权重，把「召回率 vs 新鲜度」的取舍曲线画出来。

    这是本项目最诚实、也最有价值的一张图：
      - 强度权重 = 0   等价于纯 RAG，raw recall 最高（因为 gold 是「所有含关键词的记忆」）
      - 强度权重 ↑     raw recall 下降（陈旧记忆被压下去），但「优先召回当前有效信息」上升
    没有免费的午餐：强度权重是一个**按任务调的旋钮**，不是越大越好。
    """
    out = {}
    for w in (0.0, 0.1, 0.2, 0.35, 0.5, 0.7):
        r = HybridRetriever(store, word_weight=0.6, char_weight=0.4,
                            strength_weight=w, params=params).build()
        scores = []
        for q, gids in gold.items():
            hits = r.search(q, k=k, now=now)
            scores.append(recall_at_k([h.memory.id for h in hits], gids))
        out[f"w={w}"] = round(sum(scores) / len(scores), 4)
    return out


# ── 实验二：新旧竞争 ────────────────────────────────────────────────
def exp_staleness(store: MemoryStore, params: DecayParams, now: float) -> dict:
    """给库里成对插入「同主题、同关键词」的旧版/新版记忆，看谁排前。

    旧版：250 天前写入、从没用过 → 强度接近 0
    新版：刚刚写入、用过几次     → 强度接近 1
    纯 RAG 只看文本相关度，两条分数几乎一样，谁排前面看运气；
    记忆层会把新版顶上去。用 top-1 命中新版的比例衡量。
    """
    topics = [
        ("下载加速", "旧的下载加速方案：单线程直连，经常超时。"),
        ("下载加速", "最新的下载加速方案：分段并行 + Range 切片 + 段内续传。"),
        ("发布流程", "旧的发布流程：手动复制粘贴，易漏。"),
        ("发布流程", "最新的发布流程：脚本化一键发布，带校验重试。"),
        ("限速处理", "旧的限速处理：撞到 429 就死等 60 秒。"),
        ("限速处理", "最新的限速处理：令牌桶预限速 + 指数退避 + 换源。"),
    ]
    topic_ids: Dict[str, dict] = {}
    for i in range(0, len(topics), 2):
        old_text = topics[i][1]
        new_text = topics[i + 1][1]
        topic = topics[i][0]
        old_id = store.add(old_text, kind="经验", source="staleness_probe")
        new_id = store.add(new_text, kind="经验", source="staleness_probe")
        store.conn.execute(
            "UPDATE memories SET created_at=?, last_access=?, access_count=0 WHERE id=?",
            (now - 250 * DAY, now - 250 * DAY, old_id))
        store.conn.execute(
            "UPDATE memories SET created_at=?, last_access=?, access_count=5 WHERE id=?",
            (now - 1 * DAY, now, new_id))
        topic_ids[topic] = {"old": old_id, "new": new_id}
    store.conn.commit()

    wins = {"plain": 0, "memory": 0}
    for topic, ids in topic_ids.items():
        for mode in ("plain", "memory"):
            r = _make_retriever(store, mode, params)
            hits = r.search(topic, k=1, now=now)
            if hits and hits[0].memory.id == ids["new"]:
                wins[mode] += 1
    n = len(topic_ids)
    # 清理探针，别污染后续
    for ids in topic_ids.values():
        store.forget(ids["old"])
        store.forget(ids["new"])
    return {
        "n_topics": n,
        "new_version_top1_rate": {
            mode: round(wins[mode] / n, 4) if n else 0.0 for mode in ("plain", "memory")
        },
        "note": "同主题旧版 vs 新版，top-1 命中新版的比例；记忆层应显著高于 plain(≈随机)",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/demo.db")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--stale-ratio", type=float, default=0.6)
    args = ap.parse_args()

    # 评测必须无副作用：aging/写入探针都会改库，所以在**副本**上跑，
    # 绝不碰用户的原始记忆库。（曾经的 bug：直接把 aging 写回原库，
    # 导致后续 conflict_eval 在被动过手脚的数据上得出 335 条误取代的假结论。）
    src = Path(args.db)
    if not src.exists():
        print(f"数据库不存在：{src}，先跑 demo.py")
        return
    work = src.with_name(src.stem + ".evalcopy" + src.suffix)
    shutil.copy2(src, work)
    store = MemoryStore(work)
    params = DecayParams()
    now = time.time()
    gold = build_gold(store)
    if not gold:
        print("评测集为空，先跑 demo.py 生成 data/demo.db")
        return

    stale_ids = age_store(store, now, args.stale_ratio)
    result = {
        "n_memories": store.count(),
        "n_queries": len(gold),
        "k": args.k,
        "aged": {"stale_ratio": args.stale_ratio, "n_stale": len(stale_ids)},
        "exp1_recall": exp_recall(store, gold, args.k, params, now),
        "exp1b_weight_sweep": exp_weight_sweep(store, gold, args.k, params, now),
        "exp2_staleness": exp_staleness(store, params, now),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    Path("data/eval_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n-> data/eval_result.json")
    store.conn.close()
    work.unlink(missing_ok=True)  # 清理评测副本，原库始终未被触碰


if __name__ == "__main__":
    main()
