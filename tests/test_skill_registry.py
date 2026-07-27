import pytest

from harness.skills.registry import SkillRegistry, parse_frontmatter


def _mk(tmp_path, name, meta, body="正文内容", resources=None):
    d = tmp_path / name
    d.mkdir()
    fm = "---\n" + "".join(f"{k}: {v}\n" for k, v in meta.items()) + "---\n" + body
    (d / "SKILL.md").write_text(fm, encoding="utf-8")
    for rel, content in (resources or {}).items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return d


def test_parse_frontmatter_basic():
    meta, body = parse_frontmatter("---\nname: a\ndescription: 描述\n---\n# 正文\nx")
    assert meta["name"] == "a"
    assert meta["description"] == "描述"
    assert body.strip().startswith("# 正文")


def test_parse_frontmatter_no_frontmatter():
    meta, body = parse_frontmatter("# 没有 frontmatter")
    assert meta == {}
    assert body == "# 没有 frontmatter"


def test_parse_frontmatter_strips_quotes():
    meta, _ = parse_frontmatter('---\nname: "a"\ndescription: \'描述\'\n---\n正文')
    assert meta["name"] == "a"
    assert meta["description"] == "描述"


def test_scan_finds_valid_skill(tmp_path):
    _mk(tmp_path, "etl", {"name": "etl", "description": "表格清洗"}, "# 步骤\n1. 读CSV")
    reg = SkillRegistry(str(tmp_path))
    assert reg.names() == ["etl"]
    assert reg.has("etl")
    assert "步骤" in reg.body("etl")
    assert reg.get("etl").description == "表格清洗"


def test_index_text_lists_skills(tmp_path):
    _mk(tmp_path, "etl", {"name": "etl", "description": "表格清洗"})
    idx = SkillRegistry(str(tmp_path)).index_text()
    assert "etl" in idx and "表格清洗" in idx
    assert "load_skill" in idx


def test_missing_dir_is_empty(tmp_path):
    reg = SkillRegistry(str(tmp_path / "nope"))
    assert reg.is_empty()
    assert reg.index_text() == ""


def test_empty_dir_is_empty(tmp_path):
    reg = SkillRegistry(str(tmp_path))
    assert reg.is_empty()


def test_malformed_skill_skipped_not_crash(tmp_path):
    _mk(tmp_path, "bad", {"name": "bad"})  # 缺 description
    _mk(tmp_path, "good", {"name": "good", "description": "ok"})
    reg = SkillRegistry(str(tmp_path))
    assert reg.names() == ["good"]
    assert any("bad" in w for w in reg.warnings)


def test_no_skillmd_dir_skipped(tmp_path):
    (tmp_path / "notaskill").mkdir()
    _mk(tmp_path, "ok", {"name": "ok", "description": "d"})
    assert SkillRegistry(str(tmp_path)).names() == ["ok"]


def test_read_resource_ok(tmp_path):
    _mk(tmp_path, "etl", {"name": "etl", "description": "d"},
        resources={"refs/tips.md": "小贴士", "scripts/x.py": "print(1)"})
    reg = SkillRegistry(str(tmp_path))
    assert reg.read_resource("etl", "refs/tips.md") == "小贴士"
    assert reg.read_resource("etl", "scripts/x.py") == "print(1)"


def test_read_resource_path_traversal_rejected(tmp_path):
    _mk(tmp_path, "etl", {"name": "etl", "description": "d"})
    (tmp_path / "secret.txt").write_text("机密", encoding="utf-8")
    reg = SkillRegistry(str(tmp_path))
    with pytest.raises(ValueError):
        reg.read_resource("etl", "../secret.txt")
    with pytest.raises(ValueError):
        reg.read_resource("etl", "/etc/passwd")


def test_read_resource_unknown_skill_or_file(tmp_path):
    _mk(tmp_path, "etl", {"name": "etl", "description": "d"})
    reg = SkillRegistry(str(tmp_path))
    with pytest.raises(ValueError):
        reg.read_resource("nope", "a.txt")
    with pytest.raises(ValueError):
        reg.read_resource("etl", "missing.txt")


def test_read_resource_truncates(tmp_path):
    _mk(tmp_path, "etl", {"name": "etl", "description": "d"}, resources={"big.txt": "x" * 100})
    reg = SkillRegistry(str(tmp_path), resource_max_chars=10)
    out = reg.read_resource("etl", "big.txt")
    assert out.startswith("x" * 10)
    assert "截断" in out
