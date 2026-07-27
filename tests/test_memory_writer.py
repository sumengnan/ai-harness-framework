# tests/test_memory_writer.py
import json
import logging

from harness.memory.record import MemType
from harness.memory.record import MemoryFilter, MemoryRecord
from harness.memory.reranker import NoOpReranker
from harness.memory.retriever import RetrievalConfig, Retriever
from harness.memory.sqlite_backend import SqliteVecBackend
from harness.memory.writer import ExtractedFact, MemoryWriter, _parse_facts
from harness.memory.writer import MemoryOp, _parse_ops


class ScriptedCompleter:
    """按顺序返回预设响应的假 LLM completer（async (sys, user) -> str）。"""
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
    async def __call__(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        return self._responses.pop(0)


def test_parse_facts_valid():
    raw = '[{"text":"用户偏好中文","mem_type":"semantic","entity_key":"user.pref.lang","importance":0.8}]'
    facts = _parse_facts(raw)
    assert len(facts) == 1
    assert facts[0].text == "用户偏好中文"
    assert facts[0].mem_type == MemType.SEMANTIC
    assert facts[0].entity_key == "user.pref.lang"
    assert facts[0].importance == 0.8


def test_parse_facts_code_fence():
    raw = '```json\n[{"text":"事实","mem_type":"episodic"}]\n```'
    facts = _parse_facts(raw)
    assert len(facts) == 1 and facts[0].mem_type == MemType.EPISODIC
    assert facts[0].importance == 0.5 and facts[0].entity_key == ""


def test_parse_facts_invalid_json_returns_empty():
    assert _parse_facts("对不起我不会") == []
    assert _parse_facts('{"not":"a list"}') == []


def test_parse_facts_bad_memtype_defaults_semantic():
    facts = _parse_facts('[{"text":"x","mem_type":"weird"}]')
    assert facts[0].mem_type == MemType.SEMANTIC


def test_extract_prompt_has_importance_rubric():
    # importance 量表：给 LLM 明确的分档锚点，打分才一致（否则只有"0~1"凭感觉）
    from harness.memory.writer import _EXTRACT_SYS
    assert "量表" in _EXTRACT_SYS
    for anchor in ("0.9~1.0", "0.6~0.8", "0.4~0.5", "0.1~0.3"):
        assert anchor in _EXTRACT_SYS


async def test_extract_calls_llm_and_parses(mock_embedder):
    comp = ScriptedCompleter(['[{"text":"用户在学 Python","mem_type":"semantic","importance":0.7}]'])
    w = MemoryWriter(backend=None, embedder=mock_embedder(dimension=64),
                     retriever=None, complete=comp)
    facts = await w._extract("我最近在学 Python")
    assert len(facts) == 1 and facts[0].text == "用户在学 Python"
    assert len(comp.calls) == 1


async def test_extract_empty_on_llm_garbage(mock_embedder):
    comp = ScriptedCompleter(["这不是 JSON"])
    w = MemoryWriter(backend=None, embedder=mock_embedder(dimension=64),
                     retriever=None, complete=comp)
    assert await w._extract("闲聊") == []


def _facts(*texts):
    return [ExtractedFact(text=t, mem_type=MemType.SEMANTIC) for t in texts]


def test_parse_ops_resolves_fact_index():
    facts = _facts("f0", "f1")
    ops = _parse_ops('[{"op":"REPLACE","fact_index":0,"supersede_ids":["x"]},'
                     '{"op":"NOOP","fact_index":1}]', facts)
    assert ops[0].op == "REPLACE" and ops[0].fact.text == "f0" and ops[0].supersede_ids == ["x"]
    assert ops[1].op == "NOOP" and ops[1].fact.text == "f1"


def test_parse_ops_uncovered_fact_defaults_add():
    facts = _facts("f0", "f1")
    ops = _parse_ops('[{"op":"NOOP","fact_index":0}]', facts)
    adds = [o for o in ops if o.op == "ADD"]
    assert any(o.fact.text == "f1" for o in adds)


def test_parse_ops_invalid_json_degrades_to_all_add():
    facts = _facts("f0", "f1")
    ops = _parse_ops("不是 JSON", facts)
    assert all(o.op == "ADD" for o in ops) and len(ops) == 2


async def _writer(mock_embedder, responses):
    backend = SqliteVecBackend(":memory:", dimension=64)
    emb = mock_embedder(dimension=64)
    retr = Retriever(backend, emb, NoOpReranker(), RetrievalConfig())
    return MemoryWriter(backend, emb, retr, ScriptedCompleter(responses)), backend, emb


async def test_gather_candidates_by_entity_and_semantic(mock_embedder):
    w, backend, emb = await _writer(mock_embedder, [])
    v = (await emb.embed(["用户偏好深色主题"]))[0]
    backend.upsert([MemoryRecord(owner_id="u1", kind="k", mem_type=MemType.SEMANTIC,
                                 text="用户偏好深色主题", embedding=v, id="old1",
                                 entity_key="user.pref.theme")])
    facts = [ExtractedFact(text="用户偏好深色主题", mem_type=MemType.SEMANTIC,
                           entity_key="user.pref.theme")]
    cands = await w._gather_candidates("u1", "k", facts)
    assert any(c.id == "old1" for c in cands)


async def test_reconcile_no_candidates_all_add(mock_embedder):
    w, backend, emb = await _writer(mock_embedder, [])
    facts = _facts("f0", "f1")
    ops = await w._reconcile(facts, [])
    assert all(o.op == "ADD" for o in ops) and len(ops) == 2


async def test_write_add_new_fact(mock_embedder):
    w, backend, emb = await _writer(mock_embedder,
        ['[{"text":"用户在学 Rust","mem_type":"semantic","entity_key":"user.learning","importance":0.7}]'])
    ids = await w.write("u1", "k", "我在学 Rust")
    assert len(ids) == 1
    got = backend.get(ids)
    assert got[0].text == "用户在学 Rust" and got[0].mem_type == MemType.SEMANTIC
    assert got[0].importance == 0.7 and got[0].source == "extract"


async def test_write_empty_extract_noop(mock_embedder):
    w, backend, emb = await _writer(mock_embedder, ["[]"])
    assert await w.write("u1", "k", "哈哈哈") == []


async def test_write_replace_supersedes_old(mock_embedder):
    w, backend, emb = await _writer(mock_embedder, [
        '[{"text":"用户偏好浅色主题","mem_type":"semantic","entity_key":"user.pref.theme"}]',
        '[{"op":"REPLACE","fact_index":0,"supersede_ids":["old1"]}]',
    ])
    v = (await emb.embed(["用户偏好深色主题"]))[0]
    backend.upsert([MemoryRecord(owner_id="u1", kind="k", mem_type=MemType.SEMANTIC,
                                 text="用户偏好深色主题", embedding=v, id="old1",
                                 entity_key="user.pref.theme", version=1)])
    ids = await w.write("u1", "k", "其实我喜欢浅色主题")
    assert backend.get(["old1"])[0].superseded == 1
    new = backend.get(ids)
    assert new[0].text == "用户偏好浅色主题" and new[0].version == 2


async def test_write_noop_dedup(mock_embedder):
    w, backend, emb = await _writer(mock_embedder, [
        '[{"text":"用户偏好深色主题","mem_type":"semantic","entity_key":"user.pref.theme"}]',
        '[{"op":"NOOP","fact_index":0}]',
    ])
    v = (await emb.embed(["用户偏好深色主题"]))[0]
    backend.upsert([MemoryRecord(owner_id="u1", kind="k", mem_type=MemType.SEMANTIC,
                                 text="用户偏好深色主题", embedding=v, id="old1",
                                 entity_key="user.pref.theme")])
    ids = await w.write("u1", "k", "深色主题真好")
    assert ids == []
    assert backend.get(["old1"])[0].superseded == 0


class _FlakyEmbedder:
    """第 fail_on 次（1-based）embed 抛异常，其余正常委托给真实 embedder。"""
    def __init__(self, real, fail_on):
        self._real = real
        self._n = 0
        self._fail_on = fail_on
        self.dimension = real.dimension
    async def embed(self, texts):
        self._n += 1
        if self._n == self._fail_on:
            raise RuntimeError("embed 抖动")
        return await self._real.embed(texts)


async def test_apply_skips_failed_op_keeps_others(mock_embedder):
    from harness.memory.reranker import NoOpReranker
    from harness.memory.retriever import RetrievalConfig, Retriever
    from harness.memory.sqlite_backend import SqliteVecBackend
    backend = SqliteVecBackend(":memory:", dimension=64)
    emb = _FlakyEmbedder(mock_embedder(dimension=64), fail_on=2)   # 第 2 条事实 embed 抛错
    retr = Retriever(backend, mock_embedder(dimension=64), NoOpReranker(), RetrievalConfig())
    # 提炼出两条事实，无候选 → 两条都 ADD；第 2 条 embed 失败应被跳过
    comp = ScriptedCompleter(['[{"text":"事实一","mem_type":"semantic"},'
                              '{"text":"事实二","mem_type":"semantic"}]'])
    w = MemoryWriter(backend, emb, retr, comp)
    ids = await w.write("u1", "k", "输入")           # 不应抛异常
    assert len(ids) == 1                             # 第一条成功写入，第二条被跳过
    assert backend.get(ids)[0].text == "事实一"


async def test_write_survives_gather_candidates_failure(mock_embedder):
    from harness.memory.record import MemoryFilter
    from harness.memory.reranker import NoOpReranker
    from harness.memory.retriever import RetrievalConfig, Retriever
    from harness.memory.sqlite_backend import SqliteVecBackend
    backend = SqliteVecBackend(":memory:", dimension=64)
    emb = mock_embedder(dimension=64)
    retr = Retriever(backend, emb, NoOpReranker(), RetrievalConfig())

    class _BoomRetriever:
        async def retrieve(self, *a, **k):
            raise RuntimeError("retriever 抖动")

    comp = ScriptedCompleter(['[{"text":"事实X","mem_type":"semantic"}]'])
    w = MemoryWriter(backend, emb, _BoomRetriever(), comp)
    ids = await w.write("u1", "k", "输入")           # gather 抛错 → 降级，不崩
    assert len(ids) == 1 and backend.get(ids)[0].text == "事实X"


async def test_writer_sets_ttl_by_type(mock_embedder):
    backend = SqliteVecBackend(":memory:", dimension=64)
    emb = mock_embedder(dimension=64)
    from harness.memory.reranker import NoOpReranker
    from harness.memory.retriever import RetrievalConfig, Retriever
    retr = Retriever(backend, emb, NoOpReranker(), RetrievalConfig())
    comp = ScriptedCompleter(['[{"text":"用户今天做了练习","mem_type":"episodic"}]'])
    w = MemoryWriter(backend, emb, retr, comp,
                     ttl_by_type={"episodic": 86400}, now_fn=lambda: 1000)
    ids = await w.write("u1", "k", "今天练习了")
    rec = backend.get(ids)[0]
    assert rec.expires_at == 1000 + 86400

    comp2 = ScriptedCompleter(['[{"text":"用户是后端工程师","mem_type":"semantic"}]'])
    w2 = MemoryWriter(backend, emb, retr, comp2,
                      ttl_by_type={"episodic": 86400}, now_fn=lambda: 1000)
    ids2 = await w2.write("u1", "k2", "我是后端")
    assert backend.get(ids2)[0].expires_at == 0


async def test_writer_no_ttl_by_default(mock_embedder):
    backend = SqliteVecBackend(":memory:", dimension=64)
    emb = mock_embedder(dimension=64)
    from harness.memory.reranker import NoOpReranker
    from harness.memory.retriever import RetrievalConfig, Retriever
    retr = Retriever(backend, emb, NoOpReranker(), RetrievalConfig())
    comp = ScriptedCompleter(['[{"text":"用户今天做了练习","mem_type":"episodic"}]'])
    w = MemoryWriter(backend, emb, retr, comp)
    ids = await w.write("u1", "k", "今天练习了")
    assert backend.get(ids)[0].expires_at == 0


# ---- 静默退化必须出声（否则「调和在不在工作」不可观测）----

def test_parse_facts_warns_when_output_not_json(caplog):
    """解析失败 = 本轮一条记忆都不写，而写入本就 best-effort、调用方看不出区别。"""
    with caplog.at_level(logging.WARNING, logger="harness.memory.writer"):
        assert _parse_facts("这不是 JSON") == []
    assert "记忆提炼" in caplog.text and "不写入任何记忆" in caplog.text


def test_parse_facts_warns_when_output_not_a_list(caplog):
    with caplog.at_level(logging.WARNING, logger="harness.memory.writer"):
        assert _parse_facts('{"text": "被包成对象了"}') == []
    assert "不是数组" in caplog.text


def test_parse_ops_warns_when_degrading_to_all_add(caplog):
    """退化为全 ADD = 调和等于没跑：不去重、不消矛盾，记忆库会灌满重复与矛盾。"""
    facts = [ExtractedFact(text="a", mem_type=MemType.SEMANTIC),
             ExtractedFact(text="b", mem_type=MemType.SEMANTIC)]
    with caplog.at_level(logging.WARNING, logger="harness.memory.writer"):
        ops = _parse_ops("模型今天不想吐 JSON", facts)
    assert [o.op for o in ops] == ["ADD", "ADD"]          # 行为不变：仍兜底
    assert "记忆调和" in caplog.text and "不去重、不消矛盾" in caplog.text


def test_parse_ops_warns_on_partial_degradation(caplog):
    """部分退化同样是悄悄漏判：这些事实没经调和就直接进库。"""
    facts = [ExtractedFact(text="a", mem_type=MemType.SEMANTIC),
             ExtractedFact(text="b", mem_type=MemType.SEMANTIC)]
    raw = json.dumps([{"op": "NOOP", "fact_index": 0},      # 只判了第 0 条
                      {"op": "BOGUS", "fact_index": 1}])    # 无效判定
    with caplog.at_level(logging.WARNING, logger="harness.memory.writer"):
        ops = _parse_ops(raw, facts)
    assert [o.op for o in ops] == ["NOOP", "ADD"]
    assert "1 项判定无效被跳过" in caplog.text and "1/2 条事实没拿到判定" in caplog.text


def test_parse_ops_silent_when_fully_covered(caplog):
    """全部判到就不该报警——告警要能指示真问题，不能天天喊狼来了。"""
    facts = [ExtractedFact(text="a", mem_type=MemType.SEMANTIC)]
    with caplog.at_level(logging.WARNING, logger="harness.memory.writer"):
        _parse_ops(json.dumps([{"op": "NOOP", "fact_index": 0}]), facts)
    assert caplog.text == ""


def test_parse_failure_logs_no_user_content(caplog):
    """日志只记数量：raw 是从用户对话里提炼的事实，不该进日志（同 log.info 的既有惯例）。"""
    secret = "用户的身份证号是 110101199001011234"
    with caplog.at_level(logging.WARNING, logger="harness.memory.writer"):
        _parse_facts(secret)
        _parse_ops(secret, [ExtractedFact(text="x", mem_type=MemType.SEMANTIC)])
    assert "110101199001011234" not in caplog.text
    assert str(len(secret)) in caplog.text          # 只报长度


# ---- 提炼/调和可用不同模型（提炼走快速档，调和留主模型）----

async def test_extract_uses_extract_completer_when_given(mock_embedder):
    """提炼是机械活 → 可换便宜小模型。"""
    fast = ScriptedCompleter(['[{"text":"用户在学 Python","mem_type":"episodic"}]'])
    main = ScriptedCompleter([])
    w = MemoryWriter(backend=None, embedder=mock_embedder(dimension=64),
                     retriever=None, complete=main, extract_complete=fast)
    facts = await w._extract("我最近在学 Python")
    assert len(facts) == 1
    assert len(fast.calls) == 1 and main.calls == []      # 提炼没打扰主模型


async def test_reconcile_always_uses_main_completer(mock_embedder):
    """调和是判断题、判 REPLACE 会永久作废旧记忆 → 绝不能落到快速档上。"""
    fast = ScriptedCompleter([])
    main = ScriptedCompleter(['[{"op":"NOOP","fact_index":0}]'])
    w = MemoryWriter(backend=None, embedder=mock_embedder(dimension=64),
                     retriever=None, complete=main, extract_complete=fast)
    facts = [ExtractedFact(text="a", mem_type=MemType.SEMANTIC)]
    cand = [MemoryRecord(owner_id="u1", kind="k", mem_type=MemType.SEMANTIC,
                         text="a", embedding=[0.0] * 64)]
    ops = await w._reconcile(facts, cand)
    assert [o.op for o in ops] == ["NOOP"]
    assert len(main.calls) == 1 and fast.calls == []      # 调和没落到快速档


async def test_extract_completer_defaults_to_main(mock_embedder):
    """省略 extract_complete → 二者同源，行为与旧版一致（harness 库调用方不受影响）。"""
    main = ScriptedCompleter(['[{"text":"x","mem_type":"semantic"}]'])
    w = MemoryWriter(backend=None, embedder=mock_embedder(dimension=64),
                     retriever=None, complete=main)
    await w._extract("随便")
    assert len(main.calls) == 1


# ---- 分阶段耗时（用于判断 record_turn 该不该改成 fire-and-forget）----

async def test_write_logs_per_phase_timings(mock_embedder, monkeypatch, caplog):
    """耗时须分到阶段：只有总数的话，看不出该改异步还是该关调和的思考链。"""
    from harness.memory import writer as W
    ticks = iter([
        0.0, 1.5,      # 提炼 1500ms
        1.5, 1.7,      # 找候选 200ms
        1.7, 4.2,      # 调和 2500ms
        4.2, 4.3,      # 落库 100ms
    ])

    class _Clock:
        def monotonic(self): return next(ticks)
        def time(self): return 1_700_000_000.0
    monkeypatch.setattr(W, "time", _Clock())

    w, backend, emb = await _writer(mock_embedder, [
        '[{"text":"用户在学 Rust","mem_type":"semantic"}]',
    ])
    with caplog.at_level(logging.INFO, logger="harness.memory.writer"):
        await w.write("u1", "k", "我在学 Rust")
    assert "耗时=4300ms" in caplog.text
    assert "（提炼1500/候选200/调和2500/落库100）" in caplog.text


async def test_write_timing_survives_reconcile_shortcut(mock_embedder, caplog):
    """无候选时调和会短路（不调 LLM），计时不能因此错位或抛。"""
    w, backend, emb = await _writer(mock_embedder, [
        '[{"text":"全新事实","mem_type":"semantic"}]',
    ])
    with caplog.at_level(logging.INFO, logger="harness.memory.writer"):
        await w.write("u1", "k", "随便说点")
    assert "耗时=" in caplog.text and "提炼" in caplog.text


async def test_no_write_log_when_nothing_extracted(mock_embedder, caplog):
    """提炼为空即早返回：不该记一条「新增=0」的流水账混淆视听。"""
    w, backend, emb = await _writer(mock_embedder, ["[]"])
    with caplog.at_level(logging.INFO, logger="harness.memory.writer"):
        assert await w.write("u1", "k", "哈哈") == []
    assert "记忆写入" not in caplog.text
