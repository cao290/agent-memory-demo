"""冲突消解的效果评测 —— 回答「消解到底有没有用」和「有没有误伤」。

两个实验：

  实验一 · 污染消除：构造「旧认知 + 新认知」冲突对，测
            (a) 消解前，矛盾记忆同时被召回的比例（污染率）
            (b) 消解后，只召回新认知的比例
  实验二 · 误伤率：在真实 310 条库上跑检测，统计
            (a) 自动取代对数（应极少——同源文档的不同片段不该被当成版本更新）
            (b) 待裁决对数（交上游裁决，保守）

设计原则：**误判比漏判危险**。漏判只是没消解，误判会静默隐藏一条可能正确的记忆。
所以评测格外关注「误伤率」，而不是一味追求「检出越多越好」。
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from .conflict import ConflictParams, resolve
from .retrieval import HybridRetriever
from .store import MemoryStore

DAY = 86400.0


def exp_pollution_removed(store: MemoryStore, now: float) -> dict:
    """构造真实冲突，测消解前后的召回污染率。"""
    # 三组「旧→新」认知冲突，模拟 Agent 长期运行中的知识更新
    groups = [
        [
            "限流处理｜撞到 429 就直接 sleep 60 秒死等。",
            "限流处理｜死等 60 秒太低效，应改用令牌桶预限流 + 指数退避 + 换源。",
        ],
        [
            "下载加速｜大文件直接单线程下载，经常超时。",
            "下载加速｜应分段并行下载：Range 切片 + 偏移写入 + 段内续传。",
        ],
        [
            "水印判定｜用像素法判断水印，直接比对角落像素。",
            "水印判定｜像素法会误判，改用服务端 lr 参数判定。",
        ],
    ]
    topic_ids = []
    for g in groups:
        old_id = store.add(g[0], kind="经验", source="probe", now=now - 60 * DAY)
        new_id = store.add(g[1], kind="经验", source="probe", now=now)
        topic_ids.append((old_id, new_id))

    queries = ["限流处理怎么做", "大文件下载怎么加速", "水印怎么判定"]

    # 消解前：统计每组「新旧同时出现在 top-k」的比例
    before_polluted = 0
    for (old_id, new_id), q in zip(topic_ids, queries):
        r = HybridRetriever(store, only_active=False).build()
        ids = {h.memory.id for h in r.search(q, k=5, now=now)}
        if old_id in ids and new_id in ids:
            before_polluted += 1

    # 执行消解
    report = resolve(store, ConflictParams(), auto_apply=True, only_active=False)

    # 消解后：统计每组「只剩新/不含旧」的比例
    after_clean = 0
    for (old_id, new_id), q in zip(topic_ids, queries):
        r = HybridRetriever(store, only_active=True).build()
        ids = {h.memory.id for h in r.search(q, k=5, now=now)}
        if new_id in ids and old_id not in ids:
            after_clean += 1

    n = len(groups)
    return {
        "n_groups": n,
        "polluted_before": before_polluted,
        "polluted_before_rate": round(before_polluted / n, 4),
        "clean_after": after_clean,
        "clean_after_rate": round(after_clean / n, 4),
        "auto_superseded_in_probe": len(report.superseded),
    }


def exp_false_positive(store: MemoryStore) -> dict:
    """在真实库上统计自动取代数——越少越安全（同源片段不该被误杀）。"""
    r = resolve(store, ConflictParams(), auto_apply=False)
    n = max(r.n_scanned, 1)
    return {
        "n_scanned": r.n_scanned,
        "n_pairs_flagged_total": r.n_pairs,
        "auto_supersede": len(r.superseded),
        "to_adjudicate": len(r.flagged),
        "auto_supersede_rate": round(len(r.superseded) / n, 4),
        "note": "auto_supersede 应为 0 或极低：同源文档的不同片段共享标题，"
                "但时间戳相同、不构成「版本更新」，不该被自动取代。",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/demo.db")
    args = ap.parse_args()

    # 与 eval.py 同一条原则：评测无副作用。冲突评测会写入探针记忆、
    # 并给真实记忆打上 superseded 标记——必须在副本上做，绝不改原库。
    src = Path(args.db)
    if not src.exists():
        print(f"数据库不存在：{src}，先跑 demo.py")
        return
    work = src.with_name(src.stem + ".conflicteval" + src.suffix)
    shutil.copy2(src, work)
    store = MemoryStore(work)
    now = time.time()

    base_n = store.count()
    result = {"base_memories": base_n}
    result["exp2_false_positive"] = exp_false_positive(store)
    result["exp1_pollution_removed"] = exp_pollution_removed(store, now)
    store.conn.execute("DELETE FROM memories WHERE source='probe'")
    store.conn.commit()

    print(json.dumps(result, ensure_ascii=False, indent=2))
    Path("data/conflict_eval.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n-> data/conflict_eval.json")
    store.conn.close()
    work.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
