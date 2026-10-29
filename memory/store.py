"""记忆存储层（SQLite）。

这是本项目与「普通 RAG」的分水岭：普通 RAG 存的是**文档块**，只读、不变；
记忆系统存的是**带生命周期的记忆单元**，每条记忆有自己的状态：

  - created_at   写入时间
  - last_access  最后一次被召回的时间
  - access_count 累计被召回次数
  - strength     记忆强度 = 衰减项 × 强化项（见 decay.py）

有了这三个字段，记忆才能「用进废退」：常被召回的越活越强，久不碰的逐渐淡出。
「没有遗忘机制的记忆系统，比没有记忆更糟糕」——这就是那张表存在的理由。
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

SCHEMA_TABLES = """
CREATE TABLE IF NOT EXISTS memories (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    text         TEXT    NOT NULL,          -- 记忆正文（要喂给模型的那一段）
    kind         TEXT    NOT NULL DEFAULT 'note',   -- 记忆类型：经验/事实/偏好...
    source       TEXT    NOT NULL DEFAULT '',       -- 来源（文件路径 / 会话 id）
    tags         TEXT    NOT NULL DEFAULT '[]',     -- JSON 数组
    created_at   REAL    NOT NULL,
    last_access  REAL    NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    strength     REAL    NOT NULL DEFAULT 1.0,
    state        TEXT    NOT NULL DEFAULT 'active',  -- active / superseded / archived
    meta         TEXT    NOT NULL DEFAULT '{}',     -- 任意附加字段
    UNIQUE(text, source)                            -- 同一来源里同一段话不重复写入
);
CREATE TABLE IF NOT EXISTS access_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id  INTEGER NOT NULL,
    at         REAL    NOT NULL,
    query      TEXT    NOT NULL DEFAULT '',
    score      REAL    NOT NULL DEFAULT 0.0
);
CREATE TABLE IF NOT EXISTS memory_links (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    src_id     INTEGER NOT NULL,   -- 关系起点记忆
    dst_id     INTEGER NOT NULL,   -- 关系终点记忆
    relation   TEXT    NOT NULL,   -- supersedes / contradicts / duplicate_of
    score      REAL    NOT NULL DEFAULT 0.0,  -- 判定为冲突时的相似度
    reason     TEXT    NOT NULL DEFAULT '',
    at         REAL    NOT NULL,
    UNIQUE(src_id, dst_id, relation)
);
"""

# 索引在迁移之后创建，避免老库缺列导致 CREATE INDEX 失败
SCHEMA_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_kind ON memories(kind);
CREATE INDEX IF NOT EXISTS idx_strength ON memories(strength);
CREATE INDEX IF NOT EXISTS idx_state ON memories(state);
CREATE INDEX IF NOT EXISTS idx_log_mem ON access_log(memory_id);
CREATE INDEX IF NOT EXISTS idx_link_src ON memory_links(src_id);
CREATE INDEX IF NOT EXISTS idx_link_dst ON memory_links(dst_id);
"""

# 老库升级：给已存在的 memories 表补 state 列（幂等）
MIGRATIONS = [
    "ALTER TABLE memories ADD COLUMN state TEXT NOT NULL DEFAULT 'active'",
]


@dataclass
class Memory:
    text: str
    kind: str = "note"
    source: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: float = 0.0
    last_access: float = 0.0
    access_count: int = 0
    strength: float = 1.0
    state: str = "active"
    meta: dict = field(default_factory=dict)
    id: Optional[int] = None


class MemoryStore:
    def __init__(self, path: str | Path = "data/memory.db"):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA_TABLES)
        self._migrate()
        self.conn.executescript(SCHEMA_INDEXES)
        self.conn.commit()

    def _migrate(self):
        """给旧库补新列，保证升级后老数据不丢、代码不崩。"""
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(memories)")}
        for stmt in MIGRATIONS:
            col = stmt.split("ADD COLUMN")[1].split()[0]
            if col not in cols:
                try:
                    self.conn.execute(stmt)
                except sqlite3.OperationalError:
                    pass

    # ---- 写 ----------------------------------------------------------
    def add(
        self,
        text: str,
        kind: str = "note",
        source: str = "",
        tags: Optional[List[str]] = None,
        meta: Optional[dict] = None,
        now: Optional[float] = None,
    ) -> int:
        """写入一条记忆。同 (text, source) 已存在则返回已有 id（幂等）。"""
        now = now if now is not None else time.time()
        cur = self.conn.execute(
            """INSERT OR IGNORE INTO memories
               (text, kind, source, tags, created_at, last_access, access_count, strength, meta)
               VALUES (?,?,?,?,?,?,0,1.0,?)""",
            (
                text,
                kind,
                source,
                json.dumps(tags or [], ensure_ascii=False),
                now,
                now,
                json.dumps(meta or {}, ensure_ascii=False),
            ),
        )
        self.conn.commit()
        if cur.lastrowid:
            return int(cur.lastrowid)
        row = self.conn.execute(
            "SELECT id FROM memories WHERE text=? AND source=?", (text, source)
        ).fetchone()
        return int(row["id"])

    def add_many(self, items: Iterable[Memory]) -> List[int]:
        return [self.add(asdict(m)["text"], m.kind, m.source, m.tags, m.meta) for m in items]

    # ---- 读 ----------------------------------------------------------
    def all(self) -> List[Memory]:
        rows = self.conn.execute("SELECT * FROM memories").fetchall()
        return [self._row(r) for r in rows]

    def get(self, mid: int) -> Optional[Memory]:
        r = self.conn.execute("SELECT * FROM memories WHERE id=?", (mid,)).fetchone()
        return self._row(r) if r else None

    def count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) c FROM memories").fetchone()["c"])

    # ---- 状态更新（记忆的生命周期）------------------------------------
    def touch(self, mid: int, query: str = "", score: float = 0.0, now: Optional[float] = None):
        """记一次「被用到」：访问次数 +1，刷新 last_access，并写 access_log。

        这是「强化」的原料——decay.py 会据此把强度抬上去。
        """
        now = now if now is not None else time.time()
        self.conn.execute(
            """UPDATE memories
               SET access_count = access_count + 1, last_access = ?
               WHERE id = ?""",
            (now, mid),
        )
        self.conn.execute(
            "INSERT INTO access_log (memory_id, at, query, score) VALUES (?,?,?,?)",
            (mid, now, query, score),
        )
        self.conn.commit()

    def set_strength(self, mid: int, value: float):
        self.conn.execute("UPDATE memories SET strength=? WHERE id=?", (value, mid))

    def apply_strengths(self, strengths: dict[int, float]):
        """批量落库强度（评分子系统一次算完，一次写回）。"""
        self.conn.executemany(
            "UPDATE memories SET strength=? WHERE id=?",
            [(v, k) for k, v in strengths.items()],
        )
        self.conn.commit()

    def forget(self, mid: int):
        """永久删除一条记忆（硬遗忘）。"""
        self.conn.execute("DELETE FROM memories WHERE id=?", (mid,))
        self.conn.commit()

    def weak_memories(self, threshold: float = 0.15, limit: int = 100) -> List[Memory]:
        rows = self.conn.execute(
            "SELECT * FROM memories WHERE strength < ? ORDER BY strength ASC LIMIT ?",
            (threshold, limit),
        ).fetchall()
        return [self._row(r) for r in rows]

    # ---- 记忆状态（冲突消解用）---------------------------------------
    def set_state(self, mid: int, state: str):
        """标记一条记忆的状态：active / superseded（已被取代）/ archived（已压缩归档）。"""
        assert state in ("active", "superseded", "archived")
        self.conn.execute("UPDATE memories SET state=? WHERE id=?", (state, mid))
        self.conn.commit()

    def actives(self) -> List[Memory]:
        """只取仍生效的记忆——检索默认只看这些，被取代的记忆不再污染结果。"""
        rows = self.conn.execute(
            "SELECT * FROM memories WHERE state='active'"
        ).fetchall()
        return [self._row(r) for r in rows]

    # ---- 记忆关系（谁取代了谁）---------------------------------------
    def link(self, src_id: int, dst_id: int, relation: str, score: float = 0.0, reason: str = ""):
        """建立记忆间关系：src supersedes/contradicts/duplicate_of dst。"""
        self.conn.execute(
            """INSERT OR REPLACE INTO memory_links (src_id, dst_id, relation, score, reason, at)
               VALUES (?,?,?,?,?,?)""",
            (src_id, dst_id, relation, score, reason, time.time()),
        )
        self.conn.commit()

    def links_of(self, mid: int) -> List[dict]:
        rows = self.conn.execute(
            "SELECT * FROM memory_links WHERE src_id=? OR dst_id=? ORDER BY at DESC",
            (mid, mid),
        ).fetchall()
        return [dict(r) for r in rows]

    def all_links(self) -> List[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM memory_links").fetchall()]

    # ---- 序列化 ------------------------------------------------------
    def to_jsonl(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for m in self.all():
                f.write(json.dumps(asdict(m), ensure_ascii=False) + "\n")

    @staticmethod
    def _row(r: sqlite3.Row) -> Memory:
        return Memory(
            id=r["id"],
            text=r["text"],
            kind=r["kind"],
            source=r["source"],
            tags=json.loads(r["tags"]),
            created_at=r["created_at"],
            last_access=r["last_access"],
            access_count=r["access_count"],
            strength=r["strength"],
            state=r["state"],
            meta=json.loads(r["meta"]),
        )
