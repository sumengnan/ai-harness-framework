from harness.memory.memory import Memory
from harness.memory.sqlite_backend import SqliteVecBackend
from harness.tools.base import ToolRegistry, ToolExecutor
from harness.tools.builtins.memory_search import SearchKnowledgeTool, SearchMemoryTool
from harness.tools.builtins.memory_write import RememberTool
from harness.types import ToolCall


def _mem(mock_embedder):
    return Memory(SqliteVecBackend(":memory:", dimension=64),
                  mock_embedder(dimension=64), chunk_size=1000, overlap=0)


async def test_remember_then_search_roundtrip(mock_embedder):
    """remember 写入必须能被 search_memory 召回——两者默认 scope 成对，改一个必须改另一个。"""
    mem = _mem(mock_embedder)
    reg = ToolRegistry()
    reg.register(RememberTool(mem))
    reg.register(SearchMemoryTool(mem))
    ex = ToolExecutor(reg)

    w = await ex.execute(ToolCall(id="c1", name="remember",
                                  arguments={"text": "the cat sat on the mat"}))
    assert w.is_error is False
    assert "已记住" in w.content

    r = await ex.execute(ToolCall(id="c2", name="search_memory",
                                  arguments={"query": "cat", "k": 3}))
    assert r.is_error is False
    assert "the cat sat on the mat" in r.content


async def test_remember_is_invisible_to_knowledge_search(mock_embedder):
    """记忆与知识库是两个 scope：AI 私记的东西不该混进用户资料的检索结果。"""
    mem = _mem(mock_embedder)
    reg = ToolRegistry()
    reg.register(RememberTool(mem))
    reg.register(SearchKnowledgeTool(mem))
    ex = ToolExecutor(reg)

    await ex.execute(ToolCall(id="w1", name="remember",
                              arguments={"text": "用户偏好简洁的回答"}))
    r = await ex.execute(ToolCall(id="c1", name="search_knowledge",
                                  arguments={"query": "偏好", "k": 3}))
    assert "用户偏好" not in r.content
    assert "未在知识库" in r.content


async def test_knowledge_is_invisible_to_memory_search(mock_embedder):
    """反向：用户上传的资料不该被当成 AI 自己的记忆召回。"""
    mem = _mem(mock_embedder)
    await mem.add_texts(["光合作用把二氧化碳转化为葡萄糖"], "knowledge")
    reg = ToolRegistry()
    reg.register(SearchMemoryTool(mem))
    ex = ToolExecutor(reg)
    r = await ex.execute(ToolCall(id="c1", name="search_memory",
                                  arguments={"query": "光合作用", "k": 3}))
    assert "光合作用" not in r.content
    assert "长期记忆" in r.content


async def test_no_hit_messages_are_distinct(mock_embedder):
    """两个工具的空命中文案必须可区分：知识库那句被 validating/verify 逐字匹配当哨兵。"""
    mem = _mem(mock_embedder)
    reg = ToolRegistry()
    reg.register(SearchKnowledgeTool(mem))
    reg.register(SearchMemoryTool(mem))
    ex = ToolExecutor(reg)

    k = await ex.execute(ToolCall(id="c1", name="search_knowledge",
                                  arguments={"query": "anything", "k": 3}))
    assert k.content == "（未在知识库中检索到相关内容）"

    m = await ex.execute(ToolCall(id="c2", name="search_memory",
                                  arguments={"query": "anything", "k": 3}))
    assert m.content == "（未检索到相关的长期记忆）"


async def test_search_uses_default_k_when_k_omitted(mock_embedder):
    # 不传 k 时应回落到 default_k；default_k=1 → 只返回 1 条
    mem = _mem(mock_embedder)
    reg = ToolRegistry()
    reg.register(SearchMemoryTool(mem, default_k=1))
    reg.register(RememberTool(mem))
    ex = ToolExecutor(reg)
    # 先写入两条含相同词的记忆
    await ex.execute(ToolCall(id="w1", name="remember",
                              arguments={"text": "cat one"}))
    await ex.execute(ToolCall(id="w2", name="remember",
                              arguments={"text": "cat two"}))
    r = await ex.execute(ToolCall(id="c1", name="search_memory",
                                  arguments={"query": "cat"}))
    assert r.is_error is False
    # default_k=1 → 只有一行结果
    assert len([ln for ln in r.content.splitlines() if ln.strip()]) == 1


# ---------- 检索条数下限 ----------

async def test_knowledge_search_raises_tiny_k_to_floor(mock_embedder):
    """回归：模型自作主张传 k=3，几条片段覆盖不住知识库，回答就变成「资料里没提到」。
    光调默认值没用——显式传参会盖掉默认值，故设下限兜底。"""
    mem = _mem(mock_embedder)
    await mem.add_texts([f"AI 资料第 {i} 段" for i in range(30)], "knowledge")
    reg = ToolRegistry(); reg.register(SearchKnowledgeTool(mem))
    ex = ToolExecutor(reg)
    r = await ex.execute(ToolCall(id="c1", name="search_knowledge",
                                  arguments={"query": "AI", "k": 3}))
    assert len([ln for ln in r.content.splitlines() if ln.strip()]) == 10


async def test_knowledge_search_honours_larger_k(mock_embedder):
    mem = _mem(mock_embedder)
    await mem.add_texts([f"AI 资料第 {i} 段" for i in range(60)], "knowledge")
    reg = ToolRegistry(); reg.register(SearchKnowledgeTool(mem))
    ex = ToolExecutor(reg)
    r = await ex.execute(ToolCall(id="c1", name="search_knowledge",
                                  arguments={"query": "AI", "k": 40}))
    assert len([ln for ln in r.content.splitlines() if ln.strip()]) == 40


async def test_knowledge_search_caps_at_fifty(mock_embedder):
    """上限防止把上下文撑爆。"""
    mem = _mem(mock_embedder)
    await mem.add_texts([f"AI 资料第 {i} 段" for i in range(80)], "knowledge")
    reg = ToolRegistry(); reg.register(SearchKnowledgeTool(mem))
    ex = ToolExecutor(reg)
    r = await ex.execute(ToolCall(id="c1", name="search_knowledge",
                                  arguments={"query": "AI", "k": 999}))
    assert len([ln for ln in r.content.splitlines() if ln.strip()]) == 50


async def test_memory_search_keeps_small_k(mock_embedder):
    """记忆检索不设下限：它是 AI 自己记的零散条目，不是作答依据，多取无益。"""
    mem = _mem(mock_embedder)
    await mem.add_texts([f"偏好 {i}" for i in range(20)], "memory")
    reg = ToolRegistry(); reg.register(SearchMemoryTool(mem))
    ex = ToolExecutor(reg)
    r = await ex.execute(ToolCall(id="c1", name="search_memory",
                                  arguments={"query": "偏好", "k": 3}))
    assert len([ln for ln in r.content.splitlines() if ln.strip()]) == 3
