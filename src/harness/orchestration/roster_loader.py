from __future__ import annotations

import os

import yaml

from .spec import AgentSpec


def load_roster(agents_dir: str) -> tuple[list[AgentSpec], list[str]]:
    """从 agents_dir 下的 *.yaml / *.yml 每文件加载一个 AgentSpec。

    纯解析 + 结构校验，**不涉及工具池过滤**（与工具池的耦合留在装配层）。
    畸形文件跳过并记 warning，绝不抛异常，保证启动不因配置错误而崩。
    返回 (specs, warnings)：specs 按文件名排序、重名保留先出现者。
    """
    specs: list[AgentSpec] = []
    warnings: list[str] = []
    if not os.path.isdir(agents_dir):
        return specs, warnings

    seen: set[str] = set()
    for fname in sorted(os.listdir(agents_dir)):
        if not fname.endswith((".yaml", ".yml")):
            continue
        path = os.path.join(agents_dir, fname)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except (OSError, yaml.YAMLError) as e:
            warnings.append(f"跳过 {path}：YAML 解析失败：{e}")
            continue
        if not isinstance(data, dict):
            warnings.append(f"跳过 {path}：内容不是映射（key: value）结构")
            continue

        name = str(data.get("name") or "").strip()
        desc = str(data.get("description") or "").strip()
        prompt = str(data.get("system_prompt") or "").strip()
        tools = data.get("tool_names") or []
        if not name or not desc or not prompt:
            warnings.append(f"跳过 {path}：缺少 name / description / system_prompt")
            continue
        if not isinstance(tools, list) or not all(isinstance(t, str) for t in tools):
            warnings.append(f"跳过 {path}：tool_names 必须是字符串列表")
            continue
        if name in seen:
            warnings.append(f"跳过 {path}：角色名重复 {name}")
            continue

        seen.add(name)
        specs.append(AgentSpec(name=name, description=desc,
                               system_prompt=prompt, tool_names=list(tools)))
    return specs, warnings
