from __future__ import annotations

from pydantic import BaseModel

from ..events import Progress
from ..progress import emit
from ..tools.base import Tool
from .registry import SkillRegistry


class LoadSkillTool(Tool):
    name = "load_skill"
    description = (
        "加载一个技能，使其详细步骤/说明注入到你的上下文中。只在当前任务确实需要该技能时调用。"
        "加载后其正文会出现在系统提示的「已加载技能详情」里，可直接按步骤执行。")

    class Params(BaseModel):
        name: str

    def __init__(self, registry: SkillRegistry) -> None:
        self._reg = registry

    async def run(self, params: "LoadSkillTool.Params") -> str:
        if not self._reg.has(params.name):
            raise ValueError(f"未知技能：{params.name}。可用技能：{self._reg.names()}")
        # 上报到前端：技能被引用（加载）。每次动作各成一行，故不设 key（不折叠）
        skill = self._reg.get(params.name)
        desc = f"：{skill.description}" if skill and skill.description else ""
        emit(Progress("skill", f"引用技能「{params.name}」{desc}", status="ok"))
        return f"已加载技能「{params.name}」，其详细说明已注入上下文，可直接按其步骤执行。"


class UnloadSkillTool(Tool):
    name = "unload_skill"
    description = "释放一个已加载的技能，把它的正文从上下文移除以腾出空间。仅在确认不再需要该技能时调用。"

    class Params(BaseModel):
        name: str

    def __init__(self, registry: SkillRegistry) -> None:
        self._reg = registry

    async def run(self, params: "UnloadSkillTool.Params") -> str:
        if not self._reg.has(params.name):
            raise ValueError(f"未知技能：{params.name}。可用技能：{self._reg.names()}")
        # 上报到前端：技能被卸载（从上下文移除）
        emit(Progress("skill", f"卸载技能「{params.name}」"))
        return f"已释放技能「{params.name}」，其正文已从上下文移除。"


class ReadSkillResourceTool(Tool):
    name = "read_skill_resource"
    description = (
        "读取某个技能目录内的资源文件（如 scripts/ 下的脚本、refs/ 下的参考资料）。"
        "先 load_skill 看正文提到哪些资源，再用本工具按需读取；要在沙箱运行脚本时，"
        "读到内容后写入沙箱再用 run_python/run_shell 执行。")

    class Params(BaseModel):
        skill: str
        path: str

    def __init__(self, registry: SkillRegistry) -> None:
        self._reg = registry

    async def run(self, params: "ReadSkillResourceTool.Params") -> str:
        return self._reg.read_resource(params.skill, params.path)
