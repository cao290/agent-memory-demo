"""完整演示：把「写入 → 召回 → 冲突消解」全流程跑一遍，最后出一张可观测性面板。

单跑 demo.py 只展示生命周期，库里没有召回记录也没有冲突关系，面板会是空的。
这个脚本把三件事串起来，让面板上全是真实数据——面试时打开就能讲。

运行：  python dashboard_demo.py                # 用内置样例
        python dashboard_demo.py --source <目录>  # 用你自己的 markdown 知识库
产物：  data/dashboard.html（自包含，可直接用浏览器打开）
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from memory import DecayParams, MemoryStore
from memory.conflict import resolve
from memory.dashboard import export_snapshot, render_html
from memory.ingest import ingest_path
from memory.retrieval import HybridRetriever

DB = "data/dashboard_demo.db"
DAY = 86400.0

# 模拟 Agent 在真实使用中的一批查询（产生召回记录）
QUERIES = [
    "流式下载卡住不动怎么办",
    "下载大文件怎么加速",
    "水印怎么判定",
    "接口返回 429 了怎么处理",
    "公众号阅读量上不去",
    "知乎草稿怎么写进去",
    "浏览器自动化发布踩了什么坑",
    "MCP 检索怎么做",
    "小说分镜怎么写",
    "封面接口怎么调",
    "限流怎么做",
    "看门狗是什么",
]


def rule(t: str):
    print("\n" + "=" * 68)
    print(f"  {t}")
    print("=" * 68)


def main():
    ap = argparse.ArgumentParser(description="全流程演示 + 生成可观测性面板")
    ap.add_argument("--source", default="", help="你自己的 markdown 知识库目录")
    args = ap.parse_args()

    p = Path(DB)
    if p.exists():
        p.unlink()
    store = MemoryStore(DB)
    params = DecayParams()

    # ── 1. 写入 ──────────────────────────────────────────────────────
    rule("① 写入：从知识库抽取记忆")
    source = Path(args.source) if args.source else None
    if source and source.exists():
        n = ingest_path(store, source, kind="经验")
        print(f"从 {source} 写入 {n} 条，库内共 {store.count()} 条")
    else:
        # 复用 demo.py 的样例，避免两处各维护一份（曾经的口径不一致来源）
        from demo import _seed_samples
        _seed_samples(store)
        print(f"使用内置样例，写入 {store.count()} 条")

    # ── 2. 模拟真实使用：多轮召回（产生召回记录 + 强化）─────────────
    rule("② 模拟使用：跑 12 条查询，产生召回记录与强化")
    r = HybridRetriever(store, params=params).build()
    now = time.time()
    total_hits = 0
    for i, q in enumerate(QUERIES):
        # 让查询发生在过去的不同时间点，模拟几周的持续使用
        t = now - (len(QUERIES) - i) * 2 * DAY
        hits = r.recall(q, k=3, now=t, reinforce=True)
        total_hits += len(hits)
    print(f"共召回 {total_hits} 次，落入访问日志")

    # ── 3. 冲突消解：写入「旧认知→新认知」迭代，触发消解 ─────────────
    rule("③ 冲突消解：知识迭代，旧认知被新认知取代")
    iterations = [
        ("限流处理", "撞到 429 就 sleep 60 秒死等。", "429 死等太低效，改成令牌桶 + 指数退避 + 换源。"),
        ("下载加速", "大文件单线程下载，经常超时。", "改成 Range 分段并行 + 偏移写入 + 段内续传。"),
        ("水印判定", "用像素法比对角落像素判断水印。", "像素法会误判，改用服务端 lr 参数。"),
    ]
    for i, (topic, old_t, new_t) in enumerate(iterations):
        store.add(f"{topic}｜{old_t}", kind="经验", source="迭代记录",
                  now=now - (20 - i * 3) * DAY)
        store.add(f"{topic}｜{new_t}", kind="经验", source="迭代记录", now=now - i * DAY)
    report = resolve(store, auto_apply=True, only_active=False)
    print(f"  {report.summary()}")
    for c in report.superseded[:5]:
        print(f"    [取代] #{c.old.id} → #{c.new.id}  ({c.similarity:.2f})")

    # ── 4. 出一张真实数据的面板 ─────────────────────────────────────
    rule("④ 生成可观测性面板")
    snap = export_snapshot(store, params)
    Path("data/dashboard.html").write_text(render_html(snap), encoding="utf-8")
    Path("data/memory_snapshot.json").write_text(
        __import__("json").dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  记忆 {snap['n_memories']} 条  active {snap['n_active']}  "
          f"superseded {snap['n_superseded']}  关系 {len(snap['links'])} 条")
    print(f"  -> data/dashboard.html（用浏览器直接打开，离线可用）")

    rule("完成")
    print("面板上四张图现在都是真实数据：生命周期 / 强度分布 / 冲突关系 / 召回热度。")


if __name__ == "__main__":
    main()
