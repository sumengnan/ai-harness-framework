from harness.persistence.serialize import (
    message_to_dict, message_from_dict, runstate_to_dict, runstate_from_dict, event_to_dict)
from harness.types import Message, Role, ToolCall, ToolResult
from harness.state import RunState
from harness.events import RunFinished, TextDelta, ToolFinished, ModelUsage, Progress
from harness.usage import Usage


def test_message_roundtrip():
    m = Message(role=Role.ASSISTANT, content=None,
                tool_calls=[ToolCall(id="c1", name="calc", arguments={"x": 1})])
    m2 = message_from_dict(message_to_dict(m))
    assert m2.role == Role.ASSISTANT and m2.content is None
    assert m2.tool_calls[0].id == "c1" and m2.tool_calls[0].arguments == {"x": 1}


def test_runstate_roundtrip():
    st = RunState(run_id="r1"); st.step = 2
    st.append(Message(role=Role.USER, content="hi"))
    st.append(Message(role=Role.ASSISTANT, content=None,
                      tool_calls=[ToolCall(id="c1", name="t", arguments={})]))
    st.append(Message(role=Role.TOOL, content="42", tool_call_id="c1"))
    st2 = runstate_from_dict(runstate_to_dict(st))
    assert st2.run_id == "r1" and st2.step == 2
    assert [m.role for m in st2.messages] == [Role.USER, Role.ASSISTANT, Role.TOOL]
    assert st2.messages[1].tool_calls[0].id == "c1"
    assert st2.messages[2].tool_call_id == "c1"


def test_roundtrip_unicode_nested_and_none():
    st = RunState(run_id="r1"); st.step = 5
    # 含中文、嵌套 dict/list 的工具参数
    st.append(Message(role=Role.ASSISTANT, content=None,
                      tool_calls=[ToolCall(id="c1", name="搜索",
                                           arguments={"查询": "北京天气",
                                                      "过滤": {"城市": ["北京", "上海"], "天数": 3}})]))
    # content=None / tool_call_id=None 的消息
    st.append(Message(role=Role.USER, content="你好，世界", tool_call_id=None))
    st.append(Message(role=Role.TOOL, content="42", tool_call_id="c1"))
    st2 = runstate_from_dict(runstate_to_dict(st))
    assert st2.run_id == "r1" and st2.step == 5
    assert st2.messages[0].content is None
    assert st2.messages[0].tool_calls[0].name == "搜索"
    assert st2.messages[0].tool_calls[0].arguments == {
        "查询": "北京天气", "过滤": {"城市": ["北京", "上海"], "天数": 3}}
    assert st2.messages[1].content == "你好，世界"
    assert st2.messages[1].tool_call_id is None
    assert st2.messages[2].tool_call_id == "c1"


def test_event_to_dict_variants():
    assert event_to_dict(TextDelta(text="hi")) == {"type": "TextDelta", "data": {"text": "hi"}}
    tf = event_to_dict(ToolFinished(result=ToolResult("c1", "ok", False)))
    assert tf["type"] == "ToolFinished" and tf["data"]["result"]["content"] == "ok"
    rf = event_to_dict(RunFinished(message=Message(role=Role.ASSISTANT, content="done")))
    assert rf["data"]["message"]["content"] == "done"
    mu = event_to_dict(ModelUsage(usage=Usage(1, 2, 3), cost_usd=0.5, attempts=1, latency_ms=10.0))
    assert mu["data"]["usage"]["total"] == 3 and mu["data"]["cost_usd"] == 0.5


def test_progress_detail_serialized():
    ev = Progress(scope="subagent:executor:s1", text="调用工具 x",
                  status="ok", key="c1",
                  detail={"tool": "x", "args": {"q": "a"}, "result": "r", "is_error": False})
    d = event_to_dict(ev)
    assert d["type"] == "Progress"
    assert d["data"]["detail"] == {"tool": "x", "args": {"q": "a"}, "result": "r", "is_error": False}


def test_progress_detail_defaults_none():
    d = event_to_dict(Progress(scope="sandbox", text="启动…"))
    assert d["data"]["detail"] is None
