"""词法 vs 语义 的对比评测 —— 回答「什么时候该上语义嵌入」。

单看一个总召回率说明不了问题，而且**评比基准本身会偏向某一方**：
如果 gold 集定义成「含查询关键词的文档」，词法天然占便宜（它就是在匹配关键词），
拿它评语义不公平。所以本评测的核心是**实验二：零词面重叠**——
刻意构造「查询和答案一个字都不重合、但意思相关」的用例，这才是语义嵌入真正的战场。

四个实验：

  实验一 · 同义改写召回：查询用同义词改写，看两边各召回多少（客观基线）
  实验二 · 零词面重叠：查询与答案无任何共同词，词法必然漏，看语义能否补上 ★核心
  实验三 · 中文语义边界：量同义/无关句对的相似度分布，用数字说明模型选型的影响
  实验四 · 融合是否更好：词法+语义混合能否超过任一单通道

运行：  python -m memory.semantic_eval --db data/demo.db
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

from .retrieval import HybridRetriever
from .semantic import SemanticEmbedder
from .store import MemoryStore

# 实验一：同义改写（查询与库内用词部分不重叠）
REWRITE_QUERIES = {
    "下载卡住了不动怎么处理": ["看门狗", "停滞"],
    "接口被拒了怎么解决": ["限流", "429", "403"],
    "怎么判断视频有没有角标": ["水印"],
    "账号数据不涨怎么办": ["吸粉", "完播"],
    "文章怎么排版才好看": ["分段", "标点"],
}

# 实验二：零词面重叠。左=查询（刻意避开答案用词），右=答案记忆（构造的），
# 二者**没有任何共同的实词**，只能靠语义匹配。
ZERO_OVERLAP = [
    ("东西下到一半就不动了，一直没反应", "流式下载需要加数据停滞看门狗，超时未收到数据就主动中断"),
    ("对方服务器老是说我请求太频繁", "撞到 429 要做令牌桶预限流加指数退避，不要死等"),
    ("屏幕上那个小图标怎么确认有没有叠加", "带水印的视频要读服务端 lr 参数，别用像素法比对"),
    ("作品发出去之后看的人多但没人关注我", "吸粉量为零说明缺少关注理由，要先补人格与系列感"),
    ("怎么把我写的东西分成一段一段的", "长文要按语义分段并注意标点，避免整页一大块"),
    ("复制来的代码粘贴上去后页面变成一片空白", "编辑器写入要用原生事件，直接改 DOM 会被回滚"),
    ("我一次开太多请求就被封了", "并发要限速，按连接分片下载，别让单点压力过大"),
    ("表格里数字老是算错", "精确计算交给代码工具，不要让模型硬算"),
    ("聊得时间长了它就开始忘事", "上下文过长会腐烂，要定期摘要压缩只留关键信息"),
    ("怎么知道这个模型行不行", "要准备测试用例做评测，每次改动跑一遍看分数变化"),
    ("它自己乱调用工具把文件删了", "危险操作必须加人工确认，不能让 Agent 全自动执行"),
    ("两个说法对不上该信哪个", "新旧记忆冲突时要检测并消解，晚写入的有效版本优先"),
]

# 实验四：原词查询（回归用，确认融合没有帮倒忙）
ORIGINAL_QUERIES = {
    "流式下载看门狗怎么加": ["看门狗"],
    "限流 429 怎么处理": ["限流", "429"],
    "水印判定方法": ["水印"],
    "公众号运营复盘": ["公众号"],
    "标点符号要注意什么": ["标点"],
}


def _gold_keyword(store: MemoryStore, queries: Dict[str, List[str]]) -> Dict[str, Set[int]]:
    """按关键词构造 gold（用于实验一/四）。"""
    mems = store.all()
    out = {}
    for q, keys in queries.items():
        ids = {m.id for m in mems if any(k in m.text for k in keys)}
        if ids:
            out[q] = ids
    return out


def _recall_at_k(retrieved: List[int], gold: Set[int]) -> float:
    return len(set(retrieved) & gold) / len(gold) if gold else 0.0


def run_recall(store: MemoryStore, gold: Dict[str, Set[int]], backend: str,
               k: int, now: float, semantic_weight: float = 0.5) -> dict:
    r = HybridRetriever(store, backend=backend, strength_weight=0.0,
                        semantic_weight=semantic_weight).build()
    scores = []
    for q, gids in gold.items():
        hits = r.search(q, k=k, now=now)
        scores.append(_recall_at_k([h.memory.id for h in hits], gids))
    return {
        "backend": backend,
        "active_backend": r.active_backend,
        "recall@k": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "n_queries": len(gold),
        "degraded": bool(r.degrade_reason),
    }


def exp_zero_overlap(store: MemoryStore, k: int, now: float) -> dict:
    """★核心实验：查询与答案零词面重叠，词法必漏，看语义能否补上。

    这是唯一能公平评判语义嵌入的场景。关键设计：**探针单独成库**。
    如果把探针混进 310 条真实记忆里检索，真实记忆会抢占 top-k（很多真实记忆语义上
    也沾边），测出来的其实是「在一堆噪声里找一条」，而不是纯粹测语义匹配能力。
    单独成库后，候选只有 12 条答案 + 少量干扰项，能干净地反映语义通道本身强不强。
    """
    # 探针库：12 条答案 + 12 条语义无关的干扰项（模拟「有别的记忆存在」）
    probe_texts = [ans for _, ans in ZERO_OVERLAP]
    distractors = [
        "GMT 时区换算要加 8 小时", "文件命名统一用日期前缀", "配色不超过三种主色",
        "提交前跑一遍格式检查", "备份要先建目录再复制", "长文档分成多章存放",
        "快捷键能省很多时间", "命名避免中文和空格", "注释写清楚约束原因",
        "日志按天轮转别写进同一个文件", "配置项集中管理不要散落", "版本号用语义化写法",
    ]
    tmp = MemoryStore("data/_semantic_probe.db")
    probe_ids = []
    for i, ans in enumerate(probe_texts):
        probe_ids.append(tmp.add(f"答案{i}｜{ans}", kind="探针", source="probe"))
    for d in distractors:
        tmp.add(d, kind="干扰", source="probe")

    out = {}
    for backend in ("lexical", "semantic", "hybrid"):
        r = HybridRetriever(tmp, backend=backend, strength_weight=0.0,
                            semantic_weight=0.5).build()
        hit = 0
        for (q, _), mid in zip(ZERO_OVERLAP, probe_ids):
            got = [h.memory.id for h in r.search(q, k=k)]
            if mid in got:
                hit += 1
        out[backend] = round(hit / len(probe_ids), 4)

    tmp.conn.close()
    Path("data/_semantic_probe.db").unlink(missing_ok=True)
    return {
        "n_cases": len(probe_ids),
        "n_distractors": len(distractors),
        "top_k_hit_rate": out,
        "note": "查询与答案无任何共同实词。词法接近 0；语义应显著更高。"
                "探针单独成库，避免真实记忆抢占 top-k 污染测量。",
    }


def exp_semantic_limits(model_id: str | None = None) -> dict:
    """实验三：量同义 vs 无关的相似度分布，用数字说明模型选型影响。"""
    sem = SemanticEmbedder(model_id) if model_id else SemanticEmbedder()
    if not sem._ensure_loaded():
        return {"available": False, "reason": sem.reason}
    pairs = [
        ("下载卡住了怎么办", "流式下载挂死怎么排查", "同义"),
        ("接口被拒绝", "撞到 429 限流", "同义"),
        ("视频角标判断", "水印判定方法", "同义"),
        ("下载加速", "公众号阅读量", "无关"),
        ("水印判定", "小说分镜怎么写", "无关"),
        ("限流处理", "封面接口怎么调", "无关"),
    ]
    sims = []
    for a, b, kind in pairs:
        va = sem.transform_query(a)  # 查询侧带前缀
        vb = sem.transform_one(b)
        sims.append({"a": a, "b": b, "kind": kind, "sim": round(float(va @ vb), 3)})
    same = [s["sim"] for s in sims if s["kind"] == "同义"]
    diff = [s["sim"] for s in sims if s["kind"] == "无关"]
    return {
        "available": True,
        "model": sem.model_id,
        "pairs": sims,
        "synonym_avg": round(sum(same) / len(same), 3) if same else None,
        "unrelated_avg": round(sum(diff) / len(diff), 3) if diff else None,
        "gap": round((sum(same) / len(same)) - (sum(diff) / len(diff)), 3) if same and diff else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/demo.db")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    src = Path(args.db)
    if not src.exists():
        print(f"数据库不存在：{src}，先跑 demo.py")
        return
    work = src.with_name(src.stem + ".semeval" + src.suffix)
    shutil.copy2(src, work)
    store = MemoryStore(work)
    now = time.time()

    result = {
        "n_memories": store.count(),
        "k": args.k,
        "exp1_rewrite_keyword_gold": [
            run_recall(store, _gold_keyword(store, REWRITE_QUERIES), b, args.k, now)
            for b in ("lexical", "semantic")
        ],
        "exp2_zero_word_overlap": exp_zero_overlap(store, args.k, now),
        "exp3_semantic_limits": exp_semantic_limits(),
        "exp4_original_keyword_gold": [
            run_recall(store, _gold_keyword(store, ORIGINAL_QUERIES), b, args.k, now)
            for b in ("lexical", "hybrid")
        ],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    Path("data/semantic_eval.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n-> data/semantic_eval.json")
    store.conn.close()
    work.unlink(missing_ok=True)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
