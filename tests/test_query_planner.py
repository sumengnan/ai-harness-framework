import asyncio

from harness.memory.query_planner import EMPTY_PLAN, QueryPlanner


class _Comp:
    """假 LLM completer：记录调用次数，返回预设响应。"""
    def __init__(self, resp):
        self.resp = resp
        self.calls = 0

    async def __call__(self, system, user):
        self.calls += 1
        return self.resp


async def test_disabled_all_off_returns_empty_no_llm():
    c = _Comp("{}")
    p = QueryPlanner(c, multi_query=False, hyde=False, entity=False)
    assert not p.enabled
    assert await p.plan("q") is EMPTY_PLAN and c.calls == 0


async def test_disabled_when_no_completer():
    p = QueryPlanner(None, multi_query=True, hyde=True, entity=True)
    assert not p.enabled
    assert await p.plan("q") is EMPTY_PLAN


async def test_parses_all_fields_and_truncates_variants():
    resp = '{"variants":["a","b","c","d"],"hypothetical":"答案","entity_keys":["user.pref.lang"]}'
    p = QueryPlanner(_Comp(resp), multi_query=True, multi_query_n=3, hyde=True, entity=True)
    plan = await p.plan("q")
    assert plan.variants == ["a", "b", "c"]                # 截到 n=3
    assert plan.hypothetical == "答案"
    assert plan.entity_keys == ["user.pref.lang"]


async def test_only_enabled_fields_are_read():
    resp = '{"variants":["a"],"hypothetical":"x","entity_keys":["e"]}'
    p = QueryPlanner(_Comp(resp), multi_query=True, hyde=False, entity=False)
    plan = await p.plan("q")
    assert plan.variants == ["a"] and plan.hypothetical == "" and plan.entity_keys == []


async def test_code_fence_json_parsed():
    p = QueryPlanner(_Comp('```json\n{"variants":["a"]}\n```'), multi_query=True)
    assert (await p.plan("q")).variants == ["a"]


async def test_bad_json_returns_empty():
    p = QueryPlanner(_Comp("这不是 JSON"), multi_query=True)
    assert await p.plan("q") is EMPTY_PLAN


async def test_timeout_returns_empty():
    class _Slow:
        async def __call__(self, s, u):
            await asyncio.sleep(0.5)
            return '{"variants":["a"]}'
    p = QueryPlanner(_Slow(), multi_query=True, timeout_s=0.01)
    assert await p.plan("q") is EMPTY_PLAN
