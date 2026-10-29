"""零依赖向量化模块。

设计取舍
--------
真正的语义嵌入需要 sentence-transformers（几百 MB 模型）。为了「任何人在任何机器上
clone 下来都能离线跑出同样的数字」，本项目默认用 **两个互补的词法空间**：

  1. 词通道 (word)：jieba 分词后的词 TF-IDF
  2. 字通道 (char)：字符 2-gram TF-IDF

它们在中文短文本上捕捉的东西不同：词通道抓到「看门狗 / 停滞 / abort」这类术语，
字通道能兜住分词切错、生造词、专有名词。两个通道做混合检索，比单一词法更稳。

embed.py 只负责「把文本变成向量」，不负责相似度语义判断——那是 retrieval.py 的事。
想换成真语义嵌入时，实现同一个 Embedder 接口即可（见 README「如何换成语义嵌入」）。
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, List, Sequence

import numpy as np

try:
    import jieba

    _HAS_JIEBA = True
except ImportError:  # pragma: no cover - 退化路径
    _HAS_JIEBA = False

# 中文标点/空白/常见无意义符号，分词前先清掉，避免它们混进词表
_PUNCT_RE = re.compile(r"[\s\u3000,，。、；;：:!！?？()（）\[\]【】{}<>\"'“”‘’`~@#$%^&*_+=|\\/\-—…·]+")


def tokenize(text: str) -> List[str]:
    """分词：中文用 jieba，英文/数字保持原样（小写化）。"""
    text = _PUNCT_RE.sub(" ", text.lower())
    if _HAS_JIEBA:
        tokens = [t.strip() for t in jieba.lcut(text) if t.strip()]
    else:
        # 退化：按字符切，保证无 jieba 也能跑
        tokens = list(text.replace(" ", ""))
    # 过滤单字符英文噪声，但保留中文单字（中文单字常有意义）
    out = []
    for t in tokens:
        if len(t) == 1 and t.isascii() and not t.isdigit():
            continue
        out.append(t)
    return out


def char_ngrams(text: str, n: int = 2) -> List[str]:
    """字符 n-gram：把连续的中文/字母数字压平后切 n 元。"""
    flat = re.sub(r"[\s\u3000]+", "", text.lower())
    if len(flat) < n:
        return [flat] if flat else []
    return [flat[i : i + n] for i in range(len(flat) - n + 1)]


class TfidfVectorizer:
    """最小可用的 TF-IDF 向量器（sublinear tf + 平滑 idf），输出 L2 归一化向量。

    之所以 L2 归一化：归一化后余弦相似度 = 点积，检索时直接矩阵乘法即可，快。
    """

    def __init__(self, analyzer: str = "word", min_df: int = 1):
        assert analyzer in ("word", "char")
        self.analyzer = analyzer
        self.min_df = min_df
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None

    # ---- 特征抽取 ----------------------------------------------------
    def _tokens(self, text: str) -> List[str]:
        return tokenize(text) if self.analyzer == "word" else char_ngrams(text)

    def fit(self, corpus: Sequence[str]) -> "TfidfVectorizer":
        df = Counter()
        for doc in corpus:
            df.update(set(self._tokens(doc)))
        kept = sorted(t for t, c in df.items() if c >= self.min_df)
        self.vocab = {t: i for i, t in enumerate(kept)}
        n = len(corpus)
        # 平滑 idf：log((1+n)/(1+df)) + 1，避免除零、也给高频词留一点权重
        self.idf = np.array(
            [math.log((1 + n) / (1 + df[t])) + 1.0 for t in kept], dtype=np.float64
        )
        return self

    def transform(self, texts: Iterable[str]) -> np.ndarray:
        assert self.idf is not None, "先 fit()"
        V = len(self.vocab)
        out = np.zeros((0, V), dtype=np.float64)
        rows = []
        for text in texts:
            vec = np.zeros(V, dtype=np.float64)
            tf = Counter(self._tokens(text))
            for tok, cnt in tf.items():
                j = self.vocab.get(tok)
                if j is None:
                    continue
                # sublinear tf：1+log(tf)，抑制长文本里重复词的支配
                vec[j] = (1.0 + math.log(cnt)) * self.idf[j]
            n = np.linalg.norm(vec)
            if n > 0:
                vec /= n
            rows.append(vec)
        return np.vstack(rows) if rows else out


class HybridEmbedder:
    """词通道 + 字通道的封装，对外只暴露 transform。"""

    def __init__(self, min_df: int = 1):
        self.word = TfidfVectorizer("word", min_df=min_df)
        self.char = TfidfVectorizer("char", min_df=min_df)

    def fit(self, corpus: Sequence[str]) -> "HybridEmbedder":
        self.word.fit(corpus)
        self.char.fit(corpus)
        return self

    def transform(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        return self.word.transform(texts), self.char.transform(texts)

    def transform_one(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        w, c = self.transform([text])
        return w[0], c[0]
