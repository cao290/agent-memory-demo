"""可选语义嵌入 —— 用 ONNX 跑真语义向量，无需安装 PyTorch。

背景与取舍
----------
默认检索用词法双通道（TF-IDF 词 + 字 n-gram），因为要「零重依赖、离线可复现」。
但词法有硬伤：**抓不到同义改写**——用户问「下载卡住了」，库里写的是「流式下载挂死」，
词面上几乎没有重叠，词法检索就漏了。

语义嵌入能解决这个问题，但常规方案要装 `sentence-transformers` + PyTorch（几个 GB）。
本项目改用 **ONNX Runtime**（几十 MB）直接跑模型，避开了 PyTorch：

    依赖（全部可选）：onnxruntime + tokenizers
    模型：BAAI/bge-small-zh-v1.5（中文，约 95MB，ONNX 版见 Xenova/bge-small-zh-v1.5）

**选模型踩过的坑**：一开始用了最常见的 all-MiniLM-L6-v2，结果它在中文上几乎无效——
实测「同义」和「无关」句对的相似度均值只差 0.078，无关句对（0.476）甚至高过部分同义对
（0.216），排序完全乱掉，语义召回 0.0 还不如词法的 0.303。换成中文模型 bge-small-zh 后
gap 拉开到 0.197（2.5 倍），才真正可用。**这是个真实教训：英文嵌入模型不能直接拿来跑中文。**

设计原则：**语义是可选加速，词法永远是默认与兜底**。
任何一步失败（没装 onnxruntime、模型文件缺失、加载报错）都优雅降级回词法，
绝不让「可选功能」变成「跑不起来」。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

# 默认模型：中文优先。英文场景可传 model_id="sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL = "Xenova/bge-small-zh-v1.5"
DIM_FALLBACK = 512  # bge-small-zh 输出维度

# BGE 系列建议给查询加指令前缀以提升检索效果（官方推荐，只加在 query 侧）
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


def _hub_cache_dir(model_id: str) -> Optional[Path]:
    """在 HF 缓存里定位模型快照目录（不联网、不依赖 huggingface_hub）。"""
    root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    folder = "models--" + model_id.replace("/", "--")
    snap_root = root / "hub" / folder / "snapshots"
    if not snap_root.exists():
        return None
    # 取任意一个快照（通常只有一个）
    for snap in sorted(snap_root.iterdir(), reverse=True):
        if (snap / "onnx" / "model.onnx").exists():
            return snap
    return None


class SemanticEmbedder:
    """ONNX 语义嵌入。构造时不加载模型（懒加载），首次 transform 才真正初始化。

    提供与 HybridEmbedder 一致的 transform/transform_one 接口，因此检索层可以
    无差别替换。加载失败时 `available` 为 False，调用方据此降级。
    """

    def __init__(self, model_id: str = DEFAULT_MODEL, max_len: int = 256, pooling: str = "cls"):
        """pooling: "cls"（BGE 系列用 CLS token）或 "mean"（MiniLM 等用均值池化）。"""
        assert pooling in ("cls", "mean")
        self.model_id = model_id
        self.max_len = max_len
        self.pooling = pooling
        self.available = False
        self.reason = ""
        self._session = None
        self._tokenizer = None
        self._input_names: List[str] = []
        self._dim = DIM_FALLBACK
        # 是否给查询加 BGE 官方建议的指令前缀（只加 query，不加文档）
        self._use_query_prefix = "bge" in model_id.lower()

    # ---- 懒加载 ------------------------------------------------------
    def _ensure_loaded(self) -> bool:
        if self.available:
            return True
        if self._session is not None:  # 已经试过且失败
            return False
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as e:
            self.reason = f"缺少可选依赖（onnxruntime / tokenizers）：{e}"
            self._session = None
            return False

        snap = _hub_cache_dir(self.model_id)
        if snap is None:
            self.reason = (
                f"未在本地 HF 缓存找到模型 {self.model_id}；联网拉取（约 95MB）："
                f"python -m memory.semantic --download"
            )
            self._session = None
            return False

        try:
            self._tokenizer = Tokenizer.from_file(str(snap / "tokenizer.json"))
            self._tokenizer.enable_truncation(max_length=self.max_len)
            self._tokenizer.enable_padding(length=None)
            sess = ort.InferenceSession(
                str(snap / "onnx" / "model.onnx"), providers=["CPUExecutionProvider"]
            )
            self._session = sess
            self._input_names = [i.name for i in sess.get_inputs()]
            self._dim = int(sess.get_outputs()[0].shape[-1])
            self.available = True
            return True
        except Exception as e:  # 任何加载异常都降级，不让可选功能拖垮主流程
            self.reason = f"模型加载失败，降级到词法：{e}"
            self._session = None
            return False

    # ---- 编码 --------------------------------------------------------
    def _encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        if is_query and self._use_query_prefix:
            texts = [BGE_QUERY_PREFIX + t for t in texts]
        encs = self._tokenizer.encode_batch(list(texts))
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        ttype = np.array([e.type_ids for e in encs], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask, "token_type_ids": ttype}
        feed = {k: v for k, v in feed.items() if k in self._input_names}
        hidden = self._session.run(None, feed)[0]  # [B, T, D]

        if self.pooling == "cls":
            # BGE 等模型用 [CLS]（第一个 token）作为句向量
            emb = hidden[:, 0, :]
        else:
            # mean pooling：只平均有效 token（忽略 padding）
            m = mask[..., None].astype(np.float32)
            emb = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / np.clip(norm, 1e-9, None)

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        """返回 [N, D] 的 L2 归一化语义向量；不可用时抛 RuntimeError。"""
        if not self._ensure_loaded():
            raise RuntimeError(self.reason or "语义嵌入不可用")
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return self._encode(texts)

    def transform_query(self, text: str) -> np.ndarray:
        """查询侧编码：BGE 需要加指令前缀，与文档侧不同。"""
        if not self._ensure_loaded():
            raise RuntimeError(self.reason or "语义嵌入不可用")
        return self._encode([text], is_query=True)[0]

    def transform_one(self, text: str) -> np.ndarray:
        return self.transform([text])[0]

    def status(self) -> str:
        self._ensure_loaded()
        if self.available:
            return f"语义嵌入可用（{self.model_id}, dim={self._dim}, pooling={self.pooling}）"
        return f"语义嵌入不可用 → 降级词法：{self.reason}"


def download_model(model_id: str = DEFAULT_MODEL) -> str:
    """从 HF 拉取模型到本地缓存（仅需 onnx 模型 + tokenizer + config）。"""
    import urllib.request

    root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    dst = root / "hub" / ("models--" + model_id.replace("/", "--")) / "snapshots" / "main"
    base = f"https://huggingface.co/{model_id}/resolve/main/"
    files = {
        "onnx/model.onnx": dst / "onnx" / "model.onnx",
        "tokenizer.json": dst / "tokenizer.json",
        "config.json": dst / "config.json",
    }
    for rel, out in files.items():
        if out.exists() and out.stat().st_size > 1000:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(base + rel, out)
    return str(dst)


if __name__ == "__main__":
    import sys

    if "--download" in sys.argv:
        p = download_model()
        print(f"模型已就绪：{p}")
    else:
        print(SemanticEmbedder().status())
