import pytest

from harness.skills.registry import SkillRegistry
from harness.skills.tools import LoadSkillTool, ReadSkillResourceTool, UnloadSkillTool


def _reg(tmp_path):
    d = tmp_path / "etl"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\nname: etl\ndescription: 表格清洗\n---\n正文", encoding="utf-8")
    (d / "refs").mkdir()
    (d / "refs" / "t.md").write_text("贴士", encoding="utf-8")
    return SkillRegistry(str(tmp_path))


async def test_load_skill_ok(tmp_path):
    t = LoadSkillTool(_reg(tmp_path))
    out = await t.run(t.Params(name="etl"))
    assert "etl" in out


async def test_load_skill_unknown_raises(tmp_path):
    t = LoadSkillTool(_reg(tmp_path))
    with pytest.raises(ValueError):
        await t.run(t.Params(name="nope"))


async def test_unload_skill_ok(tmp_path):
    t = UnloadSkillTool(_reg(tmp_path))
    out = await t.run(t.Params(name="etl"))
    assert "etl" in out


async def test_unload_skill_unknown_raises(tmp_path):
    t = UnloadSkillTool(_reg(tmp_path))
    with pytest.raises(ValueError):
        await t.run(t.Params(name="nope"))


async def test_read_resource_ok(tmp_path):
    t = ReadSkillResourceTool(_reg(tmp_path))
    out = await t.run(t.Params(skill="etl", path="refs/t.md"))
    assert out == "贴士"


async def test_read_resource_traversal_raises(tmp_path):
    t = ReadSkillResourceTool(_reg(tmp_path))
    with pytest.raises(ValueError):
        await t.run(t.Params(skill="etl", path="../../etc/passwd"))
