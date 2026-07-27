# src/harness/browser/extract.py
from __future__ import annotations

import html as _html
import re

# trafilatura（连同其传递依赖 dateparser）导入约 0.6s，却只在真正抽取网页正文时才用。
# 故不在模块顶层导入——改为在下面两个函数内延迟导入，避免拖慢后端启动（import 缓存，重复无开销）。

_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


def extract_main_text(html: str) -> str:
    """用 trafilatura 去样板（导航/广告/页脚）提取正文；抽不到返回空串。"""
    if not html:
        return ""
    import trafilatura
    try:
        return trafilatura.extract(html) or ""
    except Exception:
        return ""


def _extract_title(html: str) -> str:
    """标题：优先 trafilatura 元数据，回退 <title> 正则；抽不到返回空串。"""
    if not html:
        return ""
    import trafilatura
    try:
        meta = trafilatura.extract_metadata(html)
        if meta and getattr(meta, "title", None):
            return meta.title.strip()
    except Exception:
        pass
    m = _TITLE_RE.search(html)
    if m:
        return _html.unescape(m.group(1)).strip()
    return ""


def extract_title_and_text(html: str) -> tuple[str, str]:
    """返回 (标题, 正文)。正文复用 extract_main_text，标题走 _extract_title；
    抽不到分别返回空串。全程兜底，绝不抛异常。"""
    return _extract_title(html), extract_main_text(html)
