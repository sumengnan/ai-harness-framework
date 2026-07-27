from __future__ import annotations

import os
from dataclasses import dataclass

import yaml


@dataclass(frozen=True)
class Skill:
    name: str
    description: str   # 给主 agent 看的一句话能力说明（进元数据索引）
    body: str          # SKILL.md 正文（load 后注入上下文）
    dir: str           # 技能目录绝对路径（read_skill_resource 的沙盒根）
    triggers: tuple[str, ...] = ()   # 触发关键词（frontmatter 逗号分隔）；供路由层确定性匹配挂载


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """解析 SKILL.md 头部的 `---` YAML 块 + 正文。

    无合法 frontmatter（无 `---` 前缀或未闭合）时返回 ({}, 原文)。frontmatter 解析
    失败或非映射时 meta 取空 dict；值统一转为字符串。
    """
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return {}, text
    try:
        data = yaml.safe_load("\n".join(lines[1:end]))
    except yaml.YAMLError:
        data = None
    if not isinstance(data, dict):
        data = {}
    meta = {str(k): ("" if v is None else str(v)) for k, v in data.items()}
    body = "\n".join(lines[end + 1:]).lstrip("\n")
    return meta, body


class SkillRegistry:
    """启动时扫描 skills 目录一次，之后只读、进程内单例、并发安全。

    技能 = `<skills_dir>/<name>/SKILL.md`（frontmatter: name + description）
    及可选的 scripts/、refs/ 等资源文件。
    """

    def __init__(self, skills_dir: str, resource_max_chars: int = 8000) -> None:
        self._dir = skills_dir
        self._resource_max_chars = resource_max_chars
        self._skills: dict[str, Skill] = {}
        self._warnings: list[str] = []
        self._scan()

    def _scan(self) -> None:
        if not os.path.isdir(self._dir):
            return
        for name in sorted(os.listdir(self._dir)):
            sub = os.path.join(self._dir, name)
            md = os.path.join(sub, "SKILL.md")
            if not os.path.isdir(sub) or not os.path.isfile(md):
                continue
            try:
                with open(md, encoding="utf-8") as f:
                    text = f.read()
            except OSError as e:
                self._warnings.append(f"读取失败 {md}：{e}")
                continue
            meta, body = parse_frontmatter(text)
            sname = meta.get("name", "").strip()
            desc = meta.get("description", "").strip()
            if not sname or not desc:
                self._warnings.append(f"跳过 {sub}：SKILL.md 缺少 name 或 description")
                continue
            if sname in self._skills:
                self._warnings.append(f"跳过 {sub}：技能名重复 {sname}")
                continue
            triggers = tuple(
                t.strip() for t in meta.get("triggers", "").split(",") if t.strip())
            self._skills[sname] = Skill(
                name=sname, description=desc, body=body.strip(),
                dir=os.path.abspath(sub), triggers=triggers)

    # --- 查询 ---
    def names(self) -> list[str]:
        return list(self._skills.keys())

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def has(self, name: str) -> bool:
        return name in self._skills

    def body(self, name: str) -> str | None:
        s = self._skills.get(name)
        return s.body if s else None

    def is_empty(self) -> bool:
        return not self._skills

    @property
    def warnings(self) -> list[str]:
        return list(self._warnings)

    def index_text(self) -> str:
        """常驻系统提示的技能元数据索引（便宜：仅名+描述+用法）。"""
        if not self._skills:
            return ""
        lines = [
            "# 可用技能（Skills）",
            "下面是你可以按需加载的技能。仅当当前任务确实需要时才加载，避免占用上下文。",
            "- load_skill(name)：加载技能，其详细步骤会追加到系统提示末尾供你参照。",
            "- read_skill_resource(skill, path)：读取技能目录内的脚本/资源文件（如 scripts/、refs/）。",
            "- unload_skill(name)：释放不再需要的技能，腾出上下文空间。",
            "",
        ]
        lines += [f"- {s.name}：{s.description}" for s in self._skills.values()]
        return "\n".join(lines)

    def read_resource(self, name: str, rel_path: str) -> str:
        """读取技能内的资源文件，带路径穿越防护与截断。越界/不存在抛 ValueError。"""
        s = self._skills.get(name)
        if s is None:
            raise ValueError(f"未知技能：{name}。可用：{self.names()}")
        base = s.dir
        target = os.path.abspath(os.path.join(base, rel_path))
        if target != base and not target.startswith(base + os.sep):
            raise ValueError(f"非法资源路径（越界）：{rel_path}")
        if not os.path.isfile(target):
            raise ValueError(f"资源不存在：{rel_path}")
        with open(target, encoding="utf-8", errors="replace") as f:
            content = f.read()
        if len(content) > self._resource_max_chars:
            content = content[: self._resource_max_chars] + "…(已截断)"
        return content
