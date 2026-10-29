"""端到端 demo —— 一条记忆的完整生命周期。

跑一遍就能看到这个项目要讲的全部故事：

  1. 写入   从知识库目录抽取记忆单元
  2. 召回   问一个问题，混合检索（相关性 × 强度）
  3. 强化   反复用到的记忆，强度上升、衰减变慢
  4. 遗忘   长期不用的记忆，强度跌破阈值，进入待淘汰区
  5. 对比   把「全部历史塞上下文」和「只召回相关记忆」的代价摆在一起

运行：  python demo.py                    # 用内置样例，开箱即跑
        python demo.py --source <目录>     # 用你自己的 markdown 知识库
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

from memory import DecayParams, MemoryStore
from memory.decay import compute_all_strengths
from memory.ingest import ingest_path
from memory.retrieval import HybridRetriever

DB = "data/demo.db"
DAY = 86400.0


def rule(title: str):
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)


def main():
    # 清空上次运行，保证可复现
    for f in ("data/demo.db",):
        p = Path(f)
        if p.exists():
            p.unlink()

def main():
    ap = argparse.ArgumentParser(description="Agent 记忆层生命周期演示")
    ap.add_argument("--source", default="",
                    help="你自己的 markdown 知识库目录；不传则用内置样例")
    args = ap.parse_args()

    for f in ("data/demo.db",):
        p = Path(f)
        if p.exists():
            p.unlink()

    store = MemoryStore(DB)
    params = DecayParams(half_life_days=14.0)

    # ── 1. 写入 ─────────────────────────────────────────────────────
    rule("① 写入：从知识库目录抽取记忆单元")
    source = Path(args.source) if args.source else None
    if source and source.exists():
        n = ingest_path(store, source, kind="经验")
        print(f"从 {source} 抽取并写入 {n} 条记忆（库内共 {store.count()} 条）")
    else:
        if args.source:
            print(f"⚠ 未找到数据源 {source}，改用内置样例")
        else:
            print("未指定 --source，使用内置样例（可用 --source <目录> 喂入自己的知识库）")
        _seed_samples(store)
    total = store.count()

    # ── 2. 召回 ─────────────────────────────────────────────────────
    rule("② 召回：混合检索（词 + 字 + 记忆强度）")
    retr = HybridRetriever(store, strength_weight=0.35, params=params).build()
    q = "流式下载卡住不动、进度条停住怎么办"
    print(f"查询：{q}\n")
    for i, h in enumerate(retr.search(q, k=3), 1):
        preview = h.memory.text[:80].replace("\n", " ")
        print(f"  #{i} 相关={h.relevance:.3f} 强度={h.strength:.3f} 综合={h.score:.3f}")
        print(f"      ↳ {preview}…")

    # ── 3+4. 强化与遗忘：同一个月的两种命运 ──────────────────────────
    rule("③+④ 强化 vs 遗忘：同样放一个月，常用与不用的记忆走向两极")
    now = time.time()

    # 选两条不同的记忆做对照：A 每周被用一次，B 写入后就没人碰。
    # 不依赖特定文本格式——只要库里有 ≥2 条就能选出来（内置样例、外部知识库都适用）。
    pool = store.all()
    if len(pool) < 2:
        print("记忆不足 2 条，跳过强化/遗忘对照。")
        return
    used = next((m for m in pool if "看门狗" in m.text), pool[0])
    idle = next((m for m in pool if m.id != used.id and "公众号" in m.text),
                next((m for m in pool if m.id != used.id), pool[1]))
    print(f"  对照对象：")
    print(f"    A #{used.id}（每周被召回一次）: {used.text[:44]}…")
    print(f"    B #{idle.id}（写入后再没用过）: {idle.text[:44]}…\n")

    def simulate(mem, access_days: list[float], horizon_days: float):
        """在一个月时间轴上模拟：到点就访问（touch+强化），其余时间自然衰减。
        返回 (最终强度, 最终访问次数)。"""
        count, last, strength = mem.access_count, mem.created_at, mem.strength
        cursor = 0.0
        for d in access_days:
            # 从 cursor 到 d 这段只衰减
            dt = d - cursor
            stability = 1.0 + params.reinforcement_gain * count
            lam = math.log(2) / max(params.half_life_days * stability, 1e-6)
            strength = max(params.floor, strength * math.exp(-lam * dt))
            # 到 d 这天访问一次：抬升
            spacing = 1.0 + params.spacing_bonus * (1 - math.exp(-dt))
            strength = min(params.ceiling, strength + params.reinforcement_gain * spacing)
            count += 1
            last = now + d * DAY
            cursor = d
        # 剩余时间衰减到horizon
        dt = horizon_days - cursor
        stability = 1.0 + params.reinforcement_gain * count
        lam = math.log(2) / max(params.half_life_days * stability, 1e-6)
        strength = max(params.floor, strength * math.exp(-lam * dt))
        return strength, count

    sA, cA = simulate(used, [7, 14, 21, 28], 30.0)
    sB, cB = simulate(idle, [], 30.0)

    print(f"  一个月后：")
    print(f"    A  强度={sA:.3f}  访问={cA} 次   → {'保留' if sA >= 0.15 else '待淘汰'}")
    print(f"    B  强度={sB:.3f}  访问={cB} 次   → {'保留' if sB >= 0.15 else '待淘汰'}")
    print(f"\n  → 同样放一个月：常被用的 A 强度远高于没人碰的 B。")
    print(f"    这就是「用进废退」——检索时 A 排前、B 沉底；")
    print(f"    再久一点 B 就会跌破 0.15 进入待淘汰区，被压缩或删除。")

    # 长周期视角：全库在 6 个月后的存活情况
    future = now + 180 * DAY
    strengths = compute_all_strengths(store.all(), future, params)
    weak = [mid for mid, s in strengths.items() if s < 0.15]
    print(f"\n  长周期视角（快进 180 天，假设全库都没被用）：")
    print(f"    {len(weak)}/{total} 条跌破阈值——说明「不主动使用，记忆会自然淡出」，")
    print(f"    这正是防止上下文无限膨胀的机制。")

    # ── 5. 成本对比 ─────────────────────────────────────────────────
    rule("⑤ 成本对比：全量上下文 vs 记忆召回")
    all_chars = sum(len(m.text) for m in store.all())
    hits = retr.search(q, k=5)
    hit_chars = sum(len(h.memory.text) for h in hits)
    print(f"  把全部 {total} 条记忆塞进上下文 : {all_chars:>7,} 字符  (~{all_chars // 2:,} token 粗估)")
    print(f"  只召回 top-5 相关记忆          : {hit_chars:>7,} 字符  (~{hit_chars // 2:,} token 粗估)")
    if all_chars:
        print(f"  → 上下文体积降到 {hit_chars / all_chars:.1%}，而检索命中率不降（见 eval.py）")

    rule("完成")
    print(f"记忆库：{DB}    记忆条数：{store.count()}")
    print("下一步：python -m memory.eval  # 看三通道 Recall@k 对比")


def _seed_samples(store: MemoryStore):
    """数据源缺失时的兜底样例，保证 demo 开箱即跑、且有足够条目讲清故事。

    样例覆盖多个主题（下载 / 水印 / 限流 / 排版 / 编辑器 / 并发），这样检索、
    成本对比、冲突对照都有意义——只放 5 条会让成本对比退化成 100%（砍不动）。
    """
    samples = [
        ("流式下载看门狗｜坑：CDN 中途不吐数据，读取循环会永久挂住，进度条停住却像在跑。改法：起定时器记录 lastDataAt，超阈值主动 abort，并把速度归零。", "经验"),
        ("水印判定｜坑：用像素法比对角落像素会误判。改法：读服务端返回的 lr 参数来判定。", "经验"),
        ("下载加速｜坑：大文件单线程直连经常超时。改法：Range 分段并行 + 偏移写入 + 段内续传。", "经验"),
        ("429 限流｜坑：撞到限流就死等 60 秒，效率极低。改法：令牌桶预限流 + 指数退避 + 换源重试。", "经验"),
        ("编辑器写入｜坑：直接改写 DOM 会被框架回滚。改法：用原生事件触发输入，让编辑器自己更新状态。", "经验"),
        ("长文排版｜坑：整页一大块文字没人读得下去。改法：按语义分段、注意标点和换行节奏。", "经验"),
        ("并发限速｜坑：一次开太多请求把上游打挂，自己也被封。改法：按连接限速，分片并行而不是无脑并发。", "经验"),
        ("上下文腐烂｜坑：对话太长以后模型开始忘记前面的要求。改法：定期摘要压缩，只保留关键信息。", "经验"),
        ("工具误调用｜坑：让 Agent 全自动执行，它把不该动的文件删了。改法：危险操作必须加人工确认。", "经验"),
        ("精确计算｜坑：让模型直接算数，结果总是差一点。改法：把计算交给代码工具，模型只负责决定算什么。", "经验"),
        ("故障排查第一步｜坑：一出错就猜原因，改了半天没改到点上。改法：先看日志定位卡在哪一步。", "经验"),
        ("检索增强｜坑：模型一本正经编造不存在的事实。改法：让它给来源，关键结论人工复核。", "经验"),
    ]
    for text, kind in samples:
        store.add(text, kind=kind, source="builtin")


if __name__ == "__main__":
    main()
