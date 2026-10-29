"""可观测性：把记忆库的内部状态导出成人类能看懂的数据 + 自包含 HTML 面板。

为什么 Agent 记忆需要可观测性
--------------------------
记忆系统最危险的地方是「静默出错」：记忆被错误取代了、该召回的被遗忘了、
强度衰减曲线跑偏了——这些在日志里不会报错，只会在几周后表现为「Agent 变笨了」。
可观测性把内部状态摊开，让这些静默故障**可见**。

本模块导出四张图（纯 SVG 自绘，无任何前端依赖，离线可开）：

  1. 生命周期时间线：一条记忆从写入→被召回→强度变化→被取代的完整轨迹
  2. 强度分布直方图：全库记忆的强度分布，直观看「有多少在待淘汰区」
  3. 冲突关系图：谁取代了谁（supersede / contradicts 有向边）
  4. 召回热力图：哪些记忆被召回最多（用进废退的证据）

面试价值：这四个图把「我的记忆层在干什么」从抽象概念变成一张能当场打开的图。
"""
from __future__ import annotations

import argparse
import html
import json
import time
from collections import Counter
from pathlib import Path
from typing import List

from .decay import DecayParams, compute_all_strengths
from .store import Memory, MemoryStore


# ── 数据导出 ─────────────────────────────────────────────────────────

def export_snapshot(store: MemoryStore, params: DecayParams | None = None) -> dict:
    """把库内状态导成一个 JSON 快照，供面板渲染，也可单独喂给别的工具。"""
    params = params or DecayParams()
    now = time.time()
    mems = store.all()
    strengths = compute_all_strengths(mems, now, params)

    # 访问统计
    access = {
        r["memory_id"]: r["c"]
        for r in store.conn.execute(
            "SELECT memory_id, COUNT(*) c FROM access_log GROUP BY memory_id"
        )
    }

    items = []
    for m in mems:
        s = strengths.get(m.id, params.floor)
        items.append({
            "id": m.id,
            "title": m.text.split("｜", 1)[0][:40],
            "kind": m.kind,
            "state": m.state,
            "strength": round(s, 4),
            "access_count": m.access_count,
            "access_logged": int(access.get(m.id, 0)),
            "age_days": round((now - m.created_at) / 86400.0, 1),
            "idle_days": round((now - m.last_access) / 86400.0, 1),
        })

    links = store.all_links()
    return {
        "generated_at": now,
        "n_memories": len(mems),
        "n_active": sum(1 for m in mems if m.state == "active"),
        "n_superseded": sum(1 for m in mems if m.state == "superseded"),
        "params": {"half_life_days": params.half_life_days, "floor": params.floor},
        "weak_threshold": 0.15,
        "items": sorted(items, key=lambda x: x["strength"]),
        "links": [
            {"src": l["src_id"], "dst": l["dst_id"],
             "relation": l["relation"], "score": round(l["score"], 3)}
            for l in links
        ],
        "state_counts": dict(Counter(m.state for m in mems)),
    }


# ── SVG 绘图（纯字符串拼接，零依赖）────────────────────────────────

def _clip(text: str, max_width: float) -> str:
    """按「显示宽度」截断：CJK 字符宽 1.0，其余约 0.55。避免中文标题溢出重叠。"""
    out, w = [], 0.0
    for ch in text:
        cw = 1.0 if ord(ch) > 0x2E80 else 0.55
        if w + cw > max_width:
            out.append("…")
            break
        out.append(ch)
        w += cw
    return "".join(out)

def _svg_strength_histogram(items: List[dict], width: int = 560, height: int = 180) -> str:
    """强度分布直方图，用颜色区分「待淘汰区」和「保留区」。"""
    bins = [0] * 10
    for it in items:
        idx = min(9, int(it["strength"] * 10))
        bins[idx] += 1
    mx = max(bins) or 1
    pad_l, pad_b, pad_t = 34, 26, 12
    w = width - pad_l - 12
    h = height - pad_b - pad_t
    bw = w / 10
    bars = []
    for i, c in enumerate(bins):
        bh = h * c / mx
        x = pad_l + i * bw + 1
        y = pad_t + (h - bh)
        # 0~0.1, 0.1~0.2 属于待淘汰区（弱），用红色系；其余绿色系
        color = "#e2574c" if i < 2 else "#3f9e6b"
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw-2:.1f}" height="{bh:.1f}" '
            f'fill="{color}" rx="2"><title>强度 {i/10:.1f}~{(i+1)/10:.1f}：{c} 条</title></rect>'
        )
        if c:
            bars.append(
                f'<text x="{x+bw/2-1:.1f}" y="{y-3:.1f}" font-size="9" '
                f'fill="#666" text-anchor="middle">{c}</text>'
            )
    # 待淘汰区分界线
    xline = pad_l + 2 * bw
    bars.append(
        f'<line x1="{xline:.1f}" y1="{pad_t}" x2="{xline:.1f}" y2="{pad_t+h}" '
        f'stroke="#e2574c" stroke-dasharray="4 3" stroke-width="1"/>'
        f'<text x="{xline+4:.1f}" y="{pad_t+11}" font-size="9" fill="#e2574c">待淘汰区 ←</text>'
    )
    labels = "".join(
        f'<text x="{pad_l + i*bw + bw/2:.1f}" y="{height-8}" font-size="9" '
        f'fill="#888" text-anchor="middle">{i/10:.1f}</text>'
        for i in range(10)
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px">'
        f'<text x="{pad_l}" y="{pad_t}" font-size="11" fill="#333">'
        f'记忆强度分布（{len(items)} 条）</text>'
        + "".join(bars) + labels + "</svg>"
    )


def _svg_conflict_graph(links: List[dict], items: List[dict],
                        width: int = 560, height: int = 240) -> str:
    """冲突关系图：supersedes 边（新→旧）。

    高度按实际行数算，不写死——否则数据少时会留下大片空白（曾经的问题）。
    """
    if not links:
        return (
            f'<svg viewBox="0 0 {width} 60" width="100%" style="max-width:{width}px">'
            f'<text x="12" y="30" font-size="12" fill="#888">'
            f'当前无冲突关系（库里没有互相取代的记忆）</text></svg>'
        )
    by_id = {it["id"]: it for it in items}
    # 只画 supersedes 边，限制数量保证可读
    sup = [l for l in links if l["relation"] == "supersedes"][:12]
    rows = len(sup)
    lh = 22
    h = 34 + max(rows, 1) * lh + 8  # 动态高度
    out = [f'<text x="12" y="16" font-size="11" fill="#333">'
           f'冲突消解关系（新 → 取代 → 旧），共 {len(links)} 条关系'
           + (f'，图示前 {rows} 条取代' if len(sup) < len(links) else '') + '</text>']
    for i, l in enumerate(sup):
        y = 34 + i * lh
        # 中文字符比等量 ASCII 宽，标题要按「显示宽度」截断，否则会和箭头重叠
        new_t = html.escape(by_id.get(l["src"], {}).get("title", f"#{l['src']}"))
        old_t = html.escape(by_id.get(l["dst"], {}).get("title", f"#{l['dst']}"))
        out.append(f'<text x="8" y="{y+4}" font-size="10" fill="#2b6cb0">'
                   f'#{l["src"]} {_clip(new_t, 16)}</text>')
        out.append(f'<text x="290" y="{y+4}" font-size="10" fill="#c05621">→</text>')
        out.append(f'<text x="310" y="{y+4}" font-size="10" fill="#888">'
                   f'#{l["dst"]} {_clip(old_t, 14)}</text>')
        out.append(f'<text x="{width-46}" y="{y+4}" font-size="9" fill="#aaa">'
                   f'{l["score"]:.2f}</text>')
    return (f'<svg viewBox="0 0 {width} {h}" width="100%" '
            f'style="max-width:{width}px">' + "".join(out) + "</svg>")


def _svg_recall_heat(items: List[dict], width: int = 560, height: int = 150) -> str:
    """召回热力图：横向条形，长度=访问次数，直观展示「用进废退」。"""
    top = sorted(items, key=lambda x: -x["access_count"])[:15]
    if not top or top[0]["access_count"] == 0:
        return (
            f'<svg viewBox="0 0 {width} 60" width="100%" style="max-width:{width}px">'
            f'<text x="12" y="30" font-size="12" fill="#888">'
            f'尚无召回记录（跑一次检索后这里会显示哪些记忆被用得最多）</text></svg>'
        )
    mx = top[0]["access_count"] or 1
    lh = 16
    h = 30 + len(top) * lh
    out = [f'<text x="12" y="16" font-size="11" fill="#333">'
           f'召回热度 Top {len(top)}（用进废退的证据）</text>']
    for i, it in enumerate(top):
        y = 28 + i * lh
        bar_w = (width - 200) * it["access_count"] / mx
        out.append(f'<text x="12" y="{y+10}" font-size="9" fill="#555">'
                   f'#{it["id"]} {html.escape(it["title"][:20])}</text>')
        out.append(f'<rect x="150" y="{y+2}" width="{bar_w:.1f}" height="10" '
                   f'fill="#4a90d9" rx="2"/>')
        out.append(f'<text x="{150+bar_w+4:.1f}" y="{y+11}" font-size="9" fill="#666">'
                   f'{it["access_count"]}</text>')
    return (f'<svg viewBox="0 0 {width} {h}" width="100%" style="max-width:{width}px">'
            + "".join(out) + "</svg>")


def _svg_lifecycle(snapshot: dict, width: int = 560, height: int = 130) -> str:
    """生命周期概览：active / superseded / 待淘汰 三类占比。"""
    n = snapshot["n_memories"] or 1
    act = snapshot["n_active"]
    sup = snapshot["n_superseded"]
    weak = sum(1 for it in snapshot["items"] if it["strength"] < snapshot["weak_threshold"])
    segs = [
        ("保留 (active)", act, "#3f9e6b"),
        ("已取代 (superseded)", sup, "#c05621"),
        ("待淘汰（强度<0.15）", max(0, weak - sup), "#e2574c"),
    ]
    x = 12
    out = [f'<text x="12" y="16" font-size="11" fill="#333">'
           f'记忆库状态构成（共 {snapshot["n_memories"]} 条）</text>']
    for label, cnt, color in segs:
        w = (width - 40) * cnt / n
        if w > 0:
            out.append(f'<rect x="{x:.1f}" y="28" width="{w:.1f}" height="22" fill="{color}">'
                       f'<title>{label}：{cnt}</title></rect>')
        x += w
    ly = 74
    lx = 12
    for label, cnt, color in segs:
        out.append(f'<rect x="{lx}" y="{ly}" width="10" height="10" fill="{color}"/>')
        out.append(f'<text x="{lx+14}" y="{ly+9}" font-size="10" fill="#555">'
                   f'{label}：{cnt}</text>')
        lx += 190
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" '
            f'style="max-width:{width}px">' + "".join(out) + "</svg>")


# ── HTML 组装 ────────────────────────────────────────────────────────

def render_html(snapshot: dict) -> str:
    gen = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(snapshot["generated_at"]))
    return f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agent 记忆层 · 可观测性面板</title>
<style>
  body{{font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
       margin:0;padding:24px;background:#f6f7f9;color:#222}}
  .wrap{{max-width:640px;margin:0 auto}}
  h1{{font-size:19px;margin:0 0 4px}}
  .meta{{color:#888;font-size:12px;margin-bottom:20px}}
  .card{{background:#fff;border:1px solid #e6e8eb;border-radius:10px;
        padding:14px 16px;margin-bottom:14px}}
  .card h2{{font-size:13px;margin:0 0 10px;color:#444;font-weight:600}}
  .kpis{{display:flex;gap:10px;flex-wrap:wrap}}
  .kpi{{flex:1;min-width:110px;background:#fff;border:1px solid #e6e8eb;
       border-radius:10px;padding:12px 14px}}
  .kpi .v{{font-size:22px;font-weight:700;color:#1a202c}}
  .kpi .l{{font-size:11px;color:#888;margin-top:2px}}
  .foot{{color:#aaa;font-size:11px;text-align:center;margin-top:18px}}
</style></head><body><div class="wrap">
<h1>Agent 记忆层 · 可观测性面板</h1>
<div class="meta">生成于 {gen} · 半衰期 {snapshot['params']['half_life_days']} 天 ·
待淘汰阈值 {snapshot['weak_threshold']}</div>

<div class="kpis">
  <div class="kpi"><div class="v">{snapshot['n_memories']}</div><div class="l">记忆总数</div></div>
  <div class="kpi"><div class="v">{snapshot['n_active']}</div><div class="l">检索可见 (active)</div></div>
  <div class="kpi"><div class="v">{snapshot['n_superseded']}</div><div class="l">已被取代</div></div>
  <div class="kpi"><div class="v">{len(snapshot['links'])}</div><div class="l">记忆间关系</div></div>
</div>

<div class="card"><h2>① 记忆生命周期概览</h2>{_svg_lifecycle(snapshot)}</div>
<div class="card"><h2>② 记忆强度分布</h2>{_svg_strength_histogram(snapshot['items'])}</div>
<div class="card"><h2>③ 冲突消解关系</h2>{_svg_conflict_graph(snapshot['links'], snapshot['items'])}</div>
<div class="card"><h2>④ 召回热度</h2>{_svg_recall_heat(snapshot['items'])}</div>

<div class="foot">agent-memory-demo · 纯 SVG 自绘，无前端依赖 · 数据快照 JSON 见 memory_snapshot.json</div>
</div></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/demo.db")
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()
    store = MemoryStore(args.db)
    snap = export_snapshot(store)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "memory_snapshot.json").write_text(
        json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "dashboard.html").write_text(render_html(snap), encoding="utf-8")
    print(f"记忆 {snap['n_memories']} 条 / active {snap['n_active']} / "
          f"superseded {snap['n_superseded']} / 关系 {len(snap['links'])} 条")
    print(f"-> {out/'dashboard.html'}")
    print(f"-> {out/'memory_snapshot.json'}")


if __name__ == "__main__":
    main()
