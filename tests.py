"""单元测试 —— 不依赖任何数据源，纯内存跑，验证核心行为。

覆盖四个最容易出错的点（都是开发中真出过问题的）：
  1. 遗忘衰减的单调性：越久没用强度越低
  2. 强化：用过的记忆强度更高、衰减更慢
  3. 冲突消解：新的取代旧的，且可追溯
  4. 检索的归一化：相关度与强度量纲不同，混合后不应被一方压倒

运行：  python -m pytest -q        （或用 unittest：python -m unittest tests -v）
"""
from __future__ import annotations

import math
import tempfile
import time
import unittest
from pathlib import Path

from memory.conflict import ConflictParams, resolve
from memory.decay import DecayParams, decayed_strength, reinforce
from memory.retrieval import HybridRetriever
from memory.store import MemoryStore

DAY = 86400.0


class TempStoreCase(unittest.TestCase):
    """每个用例用独立的临时数据库，互不干扰，且不碰任何真实数据。"""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.store = MemoryStore(self.tmp / "t.db")

    def tearDown(self):
        self.store.conn.close()


class TestDecay(TempStoreCase):
    def test_strength_decays_over_time(self):
        now = time.time()
        mid = self.store.add("测试记忆", now=now)
        m = self.store.get(mid)
        s0 = decayed_strength(m, now)
        s30 = decayed_strength(m, now + 30 * DAY)
        s90 = decayed_strength(m, now + 90 * DAY)
        self.assertAlmostEqual(s0, 1.0, places=5, msg="刚写入强度应为 1")
        self.assertLess(s30, s0, "30 天后强度应下降")
        self.assertLess(s90, s30, "90 天后应更低（单调衰减）")
        self.assertGreaterEqual(s90, 0.0, "强度不应为负")

    def test_reinforcement_raises_strength(self):
        now = time.time()
        mid = self.store.add("测试记忆", now=now - 10 * DAY)
        m = self.store.get(mid)
        before = decayed_strength(m, now)
        after = reinforce(m, now, hits=1)
        self.assertGreater(after, before, "召回一次后强度应上升")

    def test_reinforced_decays_slower(self):
        """常被用的记忆，30 天后应比没用过的强。"""
        now = time.time()
        fresh = self.store.add("没被用过的记忆", now=now - 30 * DAY)
        used = self.store.add("常被用的记忆", now=now - 30 * DAY)
        m_used = self.store.get(used)
        # 模拟用过 10 次
        self.store.conn.execute(
            "UPDATE memories SET access_count=10, last_access=? WHERE id=?", (now, used))
        self.store.conn.commit()
        m_used = self.store.get(used)
        m_fresh = self.store.get(fresh)
        self.assertGreater(
            decayed_strength(m_used, now), decayed_strength(m_fresh, now),
            "访问次数多的记忆衰减应更慢")


class TestRetrieval(TempStoreCase):
    def test_returns_relevant_first(self):
        self.store.add("流式下载要加停滞看门狗，超时主动中断")
        self.store.add("公众号文章排版要分段")
        self.store.add("配色不超过三种主色")
        r = HybridRetriever(self.store).build()
        hits = r.search("下载停滞怎么办", k=1)
        self.assertTrue(hits)
        self.assertIn("看门狗", hits[0].memory.text)

    def test_only_active_returned(self):
        """被取代的记忆不应出现在检索结果里——这是冲突消解生效的判据。"""
        keep = self.store.add("下载加速｜用分段并行 + Range 切片")
        drop = self.store.add("下载加速｜单线程直连，简单但经常超时")
        self.store.set_state(drop, "superseded")
        r = HybridRetriever(self.store).build()
        ids = [h.memory.id for h in r.search("下载加速", k=10)]
        self.assertIn(keep, ids)
        self.assertNotIn(drop, ids, "superseded 记忆不应被召回")

    def test_scale_normalization(self):
        """相关度(小量纲)与强度(0~1)混合后不应被强度压倒：相关性高的仍应排前。"""
        now = time.time()
        # A 高度相关但"没人用过"（强度低）；B 无关但"刚用过"（强度高）
        a = self.store.add("流式下载停滞看门狗超时中断", now=now - 100 * DAY)
        b = self.store.add("配色不要超过三种主色", now=now)
        self.store.conn.execute(
            "UPDATE memories SET access_count=20, last_access=? WHERE id=?", (now, b))
        self.store.conn.commit()
        r = HybridRetriever(self.store, strength_weight=0.35).build()
        hits = r.search("流式下载停滞看门狗", k=2, now=now)
        self.assertEqual(hits[0].memory.id, a,
                         "相关性极高的记忆不应被高强度的无关记忆压倒（归一化生效）")


class TestConflict(TempStoreCase):
    def test_newer_supersedes_older(self):
        now = time.time()
        old = self.store.add("水印判定｜用像素法比对角落像素",
                             source="note.md", now=now - 60 * DAY)
        new = self.store.add("水印判定｜像素法会误判，改用服务端 lr 参数",
                             source="note.md", now=now)
        report = resolve(self.store, ConflictParams(), auto_apply=True)
        self.assertEqual(self.store.get(old).state, "superseded", "旧记忆应被标记取代")
        self.assertEqual(self.store.get(new).state, "active", "新记忆应保持生效")
        # 关系可追溯
        links = self.store.links_of(new)
        self.assertTrue(any(l["relation"] == "supersedes" for l in links),
                        "应留下 supersedes 关系")

    def test_unrelated_not_resolved(self):
        """无关记忆不应被误判（误判比漏判危险）。"""
        now = time.time()
        a = self.store.add("水印判定方法", now=now - 10 * DAY)
        b = self.store.add("公众号排版注意分段", now=now)
        resolve(self.store, ConflictParams(), auto_apply=True)
        self.assertEqual(self.store.get(a).state, "active", "无关记忆不该被取代")
        self.assertEqual(self.store.get(b).state, "active")


class TestIngest(TempStoreCase):
    def test_boilerplate_stripped(self):
        """模板句应被清洗，否则会制造大量近似重复记忆。"""
        from memory.ingest import extract_memory
        md = self.tmp / "x.md"
        md.write_text(
            "# 标题\n\n> 2026-10-27 建。经验条目：一条一事。\n\n"
            "### 小节\n- **坑→改法**：这是真正的正文内容，足够长以通过长度过滤。\n",
            encoding="utf-8")
        mems = extract_memory(md, "经验")
        self.assertTrue(mems)
        for m in mems:
            self.assertNotIn("经验条目：一条一事", m["text"], "模板句应被洗掉")


if __name__ == "__main__":
    unittest.main(verbosity=2)
