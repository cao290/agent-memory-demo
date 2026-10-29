"""冲突消解演示 —— 新认知如何推翻旧认知，并且真的不再被召回。

这个故事比「遗忘」更锋利，因为它是**即时**发生的：
  遗忘靠时间慢慢淡出，而冲突消解在新记忆写入的那一刻就动手。

场景：你对同一件事先后有三条认知，后一条推翻了前一条。
  ① 「水印用像素法判断」        （早期认知）
  ② 「像素法不准，要用服务端 lr 参数」 （校准后的认知，推翻①）
  ③ 再写入一条②的近似重复       （应被识别为重复，不产生新冲突）

跑完能看到：
  · 消解前：三条都在库里，互相矛盾，检索时同时被召回 → 模型拿到矛盾的结论
  · 消解后：旧认知 state=superseded，检索只返回最新那条

运行：  python conflict_demo.py
"""
from __future__ import annotations

from pathlib import Path

from memory import MemoryStore
from memory.conflict import ConflictParams, resolve
from memory.retrieval import HybridRetriever

DB = "data/conflict.db"
DAY = 86400.0


def rule(title: str):
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)


def main():
    p = Path(DB)
    if p.exists():
        p.unlink()
    store = MemoryStore(DB)

    rule("场景搭建：同一主题，先后三条记忆，后一条推翻前一条")
    import time
    t0 = time.time()

    m1 = store.add(
        "水印判定｜用像素法判断视频是否带水印：直接比对画面角落像素。",
        kind="经验", source="笔记/水印.md", now=t0,
    )
    m2 = store.add(
        "水印判定｜像素法判断水印不靠谱，会误判；应改用服务端返回的 lr 参数判定。",
        kind="经验", source="笔记/水印.md", now=t0 + 30 * DAY,
    )
    m3 = store.add(
        "水印判定｜判定水印不要用像素法，误判率高，正确做法是读服务端 lr 参数。",
        kind="经验", source="笔记/水印.md", now=t0 + 31 * DAY,
    )
    print(f"  ① #{m1}  {store.get(m1).text[:46]}…")
    print(f"  ② #{m2}  {store.get(m2).text[:46]}…")
    print(f"  ③ #{m3}  {store.get(m3).text[:46]}…  （②的近义重复）")

    # ── 消解前 ───────────────────────────────────────────────────────
    rule("消解前：三条并存，检索会把矛盾结论一起返回")
    r = HybridRetriever(store, only_active=False).build()
    for i, h in enumerate(r.search("水印怎么判定", k=5), 1):
        print(f"  #{i} [{h.memory.state}] {h.memory.text[:52]}…")
    print("\n  ⚠ 模型拿到「用像素法」和「别用像素法」两条矛盾记忆，只能瞎猜。")

    # ── 检测与消解 ───────────────────────────────────────────────────
    rule("检测与消解：相似度 + 时间先后 → 自动取代；证据不足的只标记")
    report = resolve(store, ConflictParams(), auto_apply=True)
    print(f"  {report.summary()}\n")
    for c in report.superseded:
        print(f"  [自动取代] #{c.old.id} → 被 #{c.new.id} 取代")
        print(f"      旧: {c.old.text[:50]}…")
        print(f"      新: {c.new.text[:50]}…")
        print(f"      理由: {c.reason}\n")
    for c in report.flagged:
        print(f"  [待裁决] #{c.old.id} ↔ #{c.new.id}  相似度 {c.similarity:.2f}")
        print(f"      理由: {c.reason}\n")

    # ── 消解后 ───────────────────────────────────────────────────────
    rule("消解后：旧认知被标记 superseded，检索不再返回它")
    r2 = HybridRetriever(store, only_active=True).build()
    for i, h in enumerate(r2.search("水印怎么判定", k=5), 1):
        print(f"  #{i} [{h.memory.state}] {h.memory.text[:52]}…")
    print(f"\n  当前 active 记忆数：{len(store.actives())} / 全库 {store.count()}")
    print("  → 模型现在只会看到最新的正确认知，矛盾消失。")

    # ── 关系图 ───────────────────────────────────────────────────────
    rule("记忆关系：谁取代了谁（可追溯）")
    for link in store.all_links():
        print(
            f"  #{link['src_id']} --{link['relation']}--> #{link['dst_id']}"
            f"  (score={link['score']:.2f})"
        )
        print(f"      {link['reason']}")

    rule("完成")
    print(f"记忆库：{DB}")
    print("可追溯性：任何一条 superseded 记忆都能顺着 memory_links 找到取代它的新记忆，")
    print("         所以「消解」不是删除，是保留历史、隐藏旧版本——出问题能回滚。")


if __name__ == "__main__":
    main()
