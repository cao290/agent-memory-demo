"""把知识库文件抽取成记忆单元。

带「坑→改法」结构的笔记，本身就是理想的记忆单元：
一个文件 = 一条可独立召回的经验。这里做两件事：

  1. **抽取**：从 md 里切出标题 + 正文，做成一条带 tags 的记忆；
  2. **去噪**：丢掉过短（无信息量）和纯骨架（只有标题没正文）的条目。

通用性：这个 ingest 不绑定任何特定知识库，任何 md 目录都能喂进来——
`python -m memory.ingest <目录或文件> --kind 经验`
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterator, List

from .store import MemoryStore

# 记忆正文里真正的「坑→改法」骨架，用它判断一条 md 有没有信息量
LESSON_RE = re.compile(r"坑[→\-–:：]|改法|病因|做法|要点|结论|原因")

# 笔记模板句：很多文件都有，抽成记忆会制造大量近似重复，必须洗掉
BOILERPLATE = [
    re.compile(r"^>\s*\d{4}-\d{2}-\d{2}\s*建[。.]?经验条目[:：].*$"),
    re.compile(r"^经验条目[:：]一条一事.*$"),
    re.compile(r"^#+\s*$"),
]


def _strip_boilerplate(text: str) -> str:
    """删掉模板句/空标题行。这些行跨文件重复，留着会污染冲突检测与检索。"""
    out = []
    for ln in text.split("\n"):
        if any(p.match(ln.strip()) for p in BOILERPLATE):
            continue
        out.append(ln)
    return "\n".join(out)


def _clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def iter_files(root: Path) -> Iterator[Path]:
    if root.is_file():
        yield root
        return
    for p in sorted(root.rglob("*.md")):
        if p.name.lower() in ("readme.md",):
            continue
        yield p


def extract_memory(path: Path, kind: str, max_chars: int = 1200) -> list[dict]:
    """从一个 md 文件抽记忆单元。返回 list[dict]（一个文件可能切成多条）。"""
    raw = path.read_text(encoding="utf-8", errors="replace")
    raw = _clean(_strip_boilerplate(raw))
    if not raw:
        return []

    lines = raw.split("\n")
    title = ""
    for ln in lines:
        if ln.startswith("# "):
            title = ln.lstrip("# ").strip()
            break
    if not title:
        title = path.stem

    # 按二级/三级标题切块；没有就整体一块
    blocks: list[str] = []
    cur: list[str] = []
    for ln in lines:
        if re.match(r"^#{1,3} ", ln) and cur:
            blocks.append("\n".join(cur).strip())
            cur = [ln]
        else:
            cur.append(ln)
    if cur:
        blocks.append("\n".join(cur).strip())
    if not blocks:
        blocks = [raw]

    mems: list[dict] = []
    for i, b in enumerate(blocks):
        body = re.sub(r"^#{1,3} .*$", "", b, count=1, flags=re.M).strip()
        if len(body) < 30:
            continue  # 太短，无信息量
        blk_title = title
        m = re.match(r"^#{1,3} (.+)$", b.strip())
        if m and m.group(1).strip() and i > 0:
            blk_title = m.group(1).strip()
        text = f"{blk_title}｜{body}"[:max_chars]
        mems.append(
            {
                "text": text,
                "kind": kind,
                "source": str(path),
                "tags": _tags_from(path, title, bool(LESSON_RE.search(body))),
            }
        )
    return mems


def _tags_from(path: Path, title: str, has_lesson: bool) -> list[str]:
    tags = []
    # 取路径里最近的分类目录名当标签（如「01-踩坑与修复」）
    for part in reversed(path.parts[:-1]):
        if re.match(r"^\d{2}-", part):
            tags.append(part)
            break
    if has_lesson:
        tags.append("坑-改法")
    return tags


def ingest_path(store: MemoryStore, root: str | Path, kind: str = "经验") -> int:
    root = Path(root)
    n = 0
    for f in iter_files(root):
        for m in extract_memory(f, kind):
            store.add(m["text"], kind=m["kind"], source=m["source"], tags=m["tags"])
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="md 文件或目录")
    ap.add_argument("--db", default="data/memory.db")
    ap.add_argument("--kind", default="经验")
    args = ap.parse_args()
    store = MemoryStore(args.db)
    n = ingest_path(store, args.path, args.kind)
    print(f"写入 {n} 条记忆，库内共 {store.count()} 条 -> {args.db}")


if __name__ == "__main__":
    main()
