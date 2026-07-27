from __future__ import annotations

from pydantic import BaseModel

from ..context.manager import ContextManager
from ..events import Progress, RunError, RunFinished, ToolFinished, ToolStarted
from ..loop.agent_loop import AgentLoop
from ..progress import emit, reset_current_agent, set_current_agent
from ..tools.base import Tool, ToolRegistry
from .spec import AgentRoster, AgentSpec


class DispatchTool(Tool):
    """把子任务派给专职子 agent 的工具。

    深度语义：顶层 DispatchTool 由调用方直接构造并注册到主 loop，不受 max_depth
    约束；`depth` 从调用方传入的值起计数。仅当 `depth + 1 < max_depth` 时才向子
    registry 注入下一层 dispatch，从而限制 agent 树的最大层数。
    """

    name = "dispatch"

    class Params(BaseModel):
        agent: str
        task: str

    def __init__(self, roster: AgentRoster, tool_pool: dict, client, budget=None,
                 tracer=None, depth: int = 0, max_depth: int = 2,
                 sub_max_steps: int = 10, model_name: str = "", price_map=None,
                 loop_detect_window: int = 0) -> None:
        self._roster = roster
        self._tool_pool = tool_pool
        self._client = client
        self._budget = budget
        self._tracer = tracer
        self._depth = depth
        self._max_depth = max_depth
        self._sub_max_steps = sub_max_steps
        self._model_name = model_name
        self._price_map = price_map
        self._loop_detect_window = loop_detect_window
        self.description = (
            "把一个子任务派发给专职子 agent 执行，返回其最终结果。\n"
            + roster.describe())

    def _build_sub_registry(self, spec: AgentSpec) -> ToolRegistry:
        reg = ToolRegistry()
        for tname in spec.tool_names:
            tool = self._tool_pool.get(tname)
            if tool is not None:
                reg.register(tool)
        if self._depth + 1 < self._max_depth:   # 未达深度上限才给下一层派发能力
            reg.register(DispatchTool(
                self._roster, self._tool_pool, self._client, self._budget, self._tracer,
                depth=self._depth + 1, max_depth=self._max_depth,
                sub_max_steps=self._sub_max_steps, model_name=self._model_name,
                price_map=self._price_map, loop_detect_window=self._loop_detect_window))
        return reg

    async def run(self, params: "DispatchTool.Params") -> str:
        spec = self._roster.get(params.agent)
        if spec is None:
            raise ValueError(f"未知角色：{params.agent}。可用：{self._roster.names()}")
        sub_loop = AgentLoop(
            client=self._client, registry=self._build_sub_registry(spec),
            context=ContextManager(spec.system_prompt),
            max_steps=self._sub_max_steps, budget=self._budget,
            tracer=self._tracer, model_name=self._model_name, price_map=self._price_map,
            loop_detect_window=self._loop_detect_window)
        final = None
        error = None
        scope = f"subagent:{params.agent}"
        emit(Progress(scope, f"开始任务：{params.task}"))
        # 记录每个工具调用 id 对应的工具名/入参，以便在 ToolFinished 时回填名称、状态与明细
        tool_names: dict[str, str] = {}
        tool_args: dict[str, object] = {}
        # 标记归属：子 agent 执行期间深层沙箱进度由 emit() 自动打上本 agent 名
        agent_token = set_current_agent(params.agent)
        try:
            async for ev in sub_loop.run(params.task):
                if isinstance(ev, ToolStarted):
                    tc = ev.tool_call
                    tool_names[tc.id] = tc.name
                    tool_args[tc.id] = tc.arguments
                    # status=running + key=工具调用 id：前端把「开始/完成」折叠成同一行并更新状态
                    emit(Progress(scope, f"调用工具 {tc.name}",
                                  status="running", key=tc.id,
                                  detail={"tool": tc.name, "args": tc.arguments}))
                elif isinstance(ev, ToolFinished):
                    r = ev.result
                    name = tool_names.get(r.tool_call_id, "工具")
                    emit(Progress(scope, f"调用工具 {name}",
                                  status="error" if r.is_error else "ok", key=r.tool_call_id,
                                  detail={"tool": name, "args": tool_args.get(r.tool_call_id),
                                          "result": r.content, "is_error": r.is_error,
                                          "image": (r.meta or {}).get("image")}))
                elif isinstance(ev, RunFinished):
                    final = ev.message.content
                elif isinstance(ev, RunError):
                    error = ev.error
        finally:
            reset_current_agent(agent_token)
        if final is None:
            emit(Progress(scope, f"未产出结果：{error or '未知'}", status="error"))
            raise RuntimeError(f"子 agent[{params.agent}] 未产出结果：{error or '未知'}")
        emit(Progress(scope, "任务完成", status="ok"))
        return final
