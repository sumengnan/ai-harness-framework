from harness.context.manager import ContextManager
from harness.skills.context import SkillContextManager, resolve_loaded_skills
from harness.skills.registry import SkillRegistry
from harness.state import RunState
from harness.types import Message, Role, ToolCall


def _reg(tmp_path):
    for n, desc, body in [("etl", "表格清洗", "ETL步骤正文"), ("pdf", "PDF处理", "PDF步骤正文")]:
        d = tmp_path / n
        d.mkdir()
        (d / "SKILL.md").write_text(
            f"---\nname: {n}\ndescription: {desc}\n---\n{body}", encoding="utf-8")
    return SkillRegistry(str(tmp_path))


def _load(name, cid="1"):
    return Message(role=Role.ASSISTANT,
                   tool_calls=[ToolCall(id=cid, name="load_skill", arguments={"name": name})])


def _unload(name, cid="2"):
    return Message(role=Role.ASSISTANT,
                   tool_calls=[ToolCall(id=cid, name="unload_skill", arguments={"name": name})])


def test_resolve_empty(tmp_path):
    reg = _reg(tmp_path)
    st = RunState(run_id="r")
    st.append(Message(role=Role.USER, content="hi"))
    assert resolve_loaded_skills(st, reg) == []


def test_resolve_load_then_unload(tmp_path):
    reg = _reg(tmp_path)
    st = RunState(run_id="r")
    st.append(_load("etl"))
    st.append(_load("pdf"))
    st.append(_unload("etl"))
    assert resolve_loaded_skills(st, reg) == ["pdf"]


def test_resolve_ignores_unknown(tmp_path):
    reg = _reg(tmp_path)
    st = RunState(run_id="r")
    st.append(_load("nope"))
    assert resolve_loaded_skills(st, reg) == []


def test_resolve_dedup_preserves_order(tmp_path):
    reg = _reg(tmp_path)
    st = RunState(run_id="r")
    st.append(_load("pdf"))
    st.append(_load("etl"))
    st.append(_load("pdf"))
    assert resolve_loaded_skills(st, reg) == ["pdf", "etl"]


def test_context_injects_index_only_when_nothing_loaded(tmp_path):
    cm = SkillContextManager(ContextManager("你是助手"), _reg(tmp_path))
    st = RunState(run_id="r")
    st.append(Message(role=Role.USER, content="hi"))
    built = cm.build(st)
    assert built[0].role == Role.SYSTEM
    assert "你是助手" in built[0].content
    assert "可用技能" in built[0].content
    assert "表格清洗" in built[0].content
    assert "已加载技能详情" not in built[0].content
    assert built[-1].content == "hi"


def test_context_injects_loaded_body(tmp_path):
    cm = SkillContextManager(ContextManager("S"), _reg(tmp_path))
    st = RunState(run_id="r")
    st.append(_load("etl"))
    st.append(Message(role=Role.USER, content="干活"))
    built = cm.build(st)
    assert "已加载技能详情" in built[0].content
    assert "ETL步骤正文" in built[0].content
    assert "PDF步骤正文" not in built[0].content


def test_context_single_system_message(tmp_path):
    cm = SkillContextManager(ContextManager("S"), _reg(tmp_path))
    st = RunState(run_id="r")
    st.append(_load("etl"))
    st.append(Message(role=Role.USER, content="q"))
    built = cm.build(st)
    assert sum(1 for m in built if m.role == Role.SYSTEM) == 1


def test_context_transparent_when_registry_empty(tmp_path):
    reg = SkillRegistry(str(tmp_path / "empty"))
    inner = ContextManager("S")
    cm = SkillContextManager(inner, reg)
    st = RunState(run_id="r")
    st.append(Message(role=Role.USER, content="hi"))
    assert [m.content for m in cm.build(st)] == [m.content for m in inner.build(st)]


def test_context_does_not_mutate_state(tmp_path):
    cm = SkillContextManager(ContextManager("S"), _reg(tmp_path))
    st = RunState(run_id="r")
    st.append(_load("etl"))
    n = len(st.messages)
    cm.build(st)
    assert len(st.messages) == n
