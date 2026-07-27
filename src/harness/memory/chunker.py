from __future__ import annotations

import importlib
import logging
import re

from semantic_text_splitter import CodeSplitter, MarkdownSplitter, TextSplitter

logger = logging.getLogger("harness.memory.chunker")

# kind → tree-sitter 语言模块（只支持主流几种；其余语言回退递归 TextSplitter）
_LANG_MODULES = {
    "python": "tree_sitter_python",
    "javascript": "tree_sitter_javascript",
    "typescript": "tree_sitter_typescript",
    "java": "tree_sitter_java",
    "go": "tree_sitter_go",
}

# 文件后缀 → 切分 kind
_EXT_KIND = {
    "md": "markdown", "markdown": "markdown", "html": "markdown", "htm": "markdown",
    "py": "python",
    "js": "javascript", "mjs": "javascript", "cjs": "javascript", "jsx": "javascript",
    "ts": "typescript", "tsx": "typescript",
    "java": "java", "go": "go",
    "txt": "text", "pdf": "text", "docx": "text", "doc": "text",
}

_lang_cache: dict = {}
_MD_HEADING = re.compile(r"^#{1,6}\s", re.M)
_MD_TABLE = re.compile(r"^\s*\|.+\|\s*$", re.M)


def kind_for_filename(filename: str | None) -> str:
    """文件名后缀 → 切分 kind；未知/无后缀返回 auto（交内容嗅探）。"""
    name = filename or ""
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    return _EXT_KIND.get(ext, "auto")


def _sniff(text: str) -> str:
    """内容嗅探：有 markdown 标题/表格/代码围栏 → markdown，否则纯文本。"""
    if "```" in text or _MD_HEADING.search(text) or _MD_TABLE.search(text):
        return "markdown"
    return "text"


def _lang_capsule(kind: str):
    """取 tree-sitter 语言 capsule；未支持/加载失败返回 None（回退递归切分）。"""
    if kind not in _LANG_MODULES:
        return None
    if kind in _lang_cache:
        return _lang_cache[kind]
    cap = None
    try:
        mod = importlib.import_module(_LANG_MODULES[kind])
        # tree_sitter_typescript 无 language()，用 language_typescript()
        cap = mod.language_typescript() if kind == "typescript" else mod.language()
    except Exception as e:  # 语言包缺失/加载失败 → 回退 TextSplitter
        logger.warning("代码语言 %s 加载失败，回退递归切分：%s", kind, e)
    _lang_cache[kind] = cap
    return cap


def _make_splitter(kind: str, capacity, overlap: int):
    if kind == "markdown":
        # 结构化：按标题层级切、保表格/代码块/列表完整；不做重叠避免污染结构
        return MarkdownSplitter(capacity)
    lang = _lang_capsule(kind)
    if lang is not None:
        # 代码：tree-sitter 按语法边界切，不切断函数/类
        return CodeSplitter(lang, capacity)
    # 纯文本 / PDF：递归按语义边界（段落→句子→…），带 overlap
    return TextSplitter(capacity, overlap=overlap)


def chunk(text: str, chunk_size: int = 1000, overlap: int = 200,
          kind: str = "auto", hard_max: int | None = None) -> list[str]:
    """结构感知切分（所有 RAG 入库统一入口）。

    kind：``markdown`` | ``text`` | ``auto`` | 语言名(python/javascript/typescript/java/go)。
    ``auto`` 按内容嗅探 markdown / 纯文本；语言名走 tree-sitter 代码切分（不切断函数）。
    capacity=(chunk_size, hard_max) 为硬上限：尽量在语义边界填满，绝不超 hard_max；
    表格 / 代码块在上限内保持完整（切断的表格 embedding 几乎无检索价值）。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正")
    text = (text or "").strip()
    if not text:
        return []
    hi = hard_max if (hard_max and hard_max > chunk_size) else chunk_size * 2
    capacity = (chunk_size, hi)
    ov = max(0, min(overlap, chunk_size - 1))          # overlap 必须小于容量下限
    resolved = _sniff(text) if kind == "auto" else kind
    splitter = _make_splitter(resolved, capacity, ov)
    return [c for c in splitter.chunks(text) if c.strip()]
