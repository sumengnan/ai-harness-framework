import pytest

from harness.memory.chunker import chunk, kind_for_filename, _sniff


# ---- 基础 ----

def test_short_text_not_split():
    assert chunk("hi", 1000, 200) == ["hi"]


def test_empty_text():
    assert chunk("   ", 1000, 200) == []


def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        chunk("x", 0, 0)


def test_long_text_splits_and_covers():
    text = "。".join(f"第{i}句话到此为止" for i in range(60)) + "。"
    chunks = chunk(text, 100, 20, kind="text")
    assert len(chunks) > 1
    # 不丢内容：每句都还在
    joined = "".join(chunks)
    for i in range(60):
        assert f"第{i}句话" in joined


# ---- 后缀 → kind 映射 ----

def test_kind_for_filename():
    assert kind_for_filename("a.md") == "markdown"
    assert kind_for_filename("page.html") == "markdown"
    assert kind_for_filename("m.py") == "python"
    assert kind_for_filename("s.ts") == "typescript"
    assert kind_for_filename("x.java") == "java"
    assert kind_for_filename("doc.pdf") == "text"
    assert kind_for_filename("notes.txt") == "text"
    assert kind_for_filename("无后缀标题") == "auto"


# ---- auto 嗅探 ----

def test_sniff_markdown_vs_text():
    assert _sniff("# 标题\n正文") == "markdown"
    assert _sniff("普通一段文字没有结构") == "text"
    assert _sniff("| A | B |\n|---|---|\n| 1 | 2 |") == "markdown"
    assert _sniff("```py\ncode\n```") == "markdown"


# ---- Markdown：表格 / 代码块绝不切断 ----

def test_markdown_keeps_table_intact():
    md = (
        "# 大标题\n\n" + "介绍段落。" * 40 + "\n\n"
        "## 数据\n\n"
        "| 姓名 | 分数 |\n|---|---|\n| 甲 | 90 |\n| 乙 | 85 |\n\n"
        "结尾说明。" * 40
    )
    chunks = chunk(md, 200, 0, kind="markdown")
    tbl = [c for c in chunks if "| 姓名 | 分数 |" in c]
    assert tbl, "表格头应完整落在某个 chunk"
    assert "| 甲 | 90 |" in tbl[0] and "| 乙 | 85 |" in tbl[0]   # 整张表未被切断


def test_markdown_keeps_code_fence_intact():
    md = (
        "# 说明\n\n" + "文字。" * 60 + "\n\n"
        "```python\ndef solve():\n    return 42\n```\n\n"
        + "更多文字。" * 60
    )
    chunks = chunk(md, 150, 0, kind="markdown")
    code = [c for c in chunks if "def solve():" in c]
    assert code and "return 42" in code[0] and "```" in code[0]  # 代码块完整


# ---- 代码：不切断函数 ----

def test_code_python_keeps_functions_whole():
    code = "\n\n".join(
        f"def func_{i}(x):\n    y = x + {i}\n    return y" for i in range(12))
    chunks = chunk(code, 120, 0, kind="python")
    assert len(chunks) > 1
    # 含某函数 def 的块也应含其 return（未被从函数中间切断）
    for c in chunks:
        for i in range(12):
            if f"def func_{i}(x):" in c:
                assert "return y" in c


# ---- 向后兼容：旧位置参数调用仍可用 ----

def test_backward_compatible_positional_call():
    out = chunk("一些内容。" * 50, 100, 20)   # 旧 chunk(text, size, overlap)
    assert isinstance(out, list) and all(isinstance(c, str) for c in out)
    assert len("".join(out)) > 0
