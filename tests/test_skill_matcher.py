from harness.skills.registry import SkillRegistry
from harness.skills.matcher import SkillMatcher


def _mk(tmp_path, name, meta, body="正文"):
    d = tmp_path / name
    d.mkdir()
    fm = "---\n" + "".join(f"{k}: {v}\n" for k, v in meta.items()) + "---\n" + body
    (d / "SKILL.md").write_text(fm, encoding="utf-8")


def _matcher(tmp_path):
    return SkillMatcher(SkillRegistry(str(tmp_path)))


def test_triggers_parsed_from_frontmatter(tmp_path):
    _mk(tmp_path, "s1", {"name": "s1", "description": "d", "triggers": "错题, 薄弱"})
    reg = SkillRegistry(str(tmp_path))
    assert reg.get("s1").triggers == ("错题", "薄弱")


def test_no_triggers_defaults_empty(tmp_path):
    _mk(tmp_path, "s1", {"name": "s1", "description": "d"})
    reg = SkillRegistry(str(tmp_path))
    assert reg.get("s1").triggers == ()


def test_match_by_trigger(tmp_path):
    _mk(tmp_path, "exam", {"name": "exam", "description": "考试", "triggers": "模考, 考前复习"})
    m = _matcher(tmp_path)
    assert m.match("帮我考前复习一下").name == "exam"
    assert m.match("今天天气不错") is None


def test_match_picks_most_hits(tmp_path):
    _mk(tmp_path, "a", {"name": "a", "description": "d", "triggers": "复习"})
    _mk(tmp_path, "b", {"name": "b", "description": "d", "triggers": "复习, 错题, 薄弱"})
    m = _matcher(tmp_path)
    # 消息命中 b 的 3 个词、a 的 1 个 → 取命中多的 b
    assert m.match("复习错题看看薄弱点").name == "b"


def test_no_triggers_never_matches(tmp_path):
    _mk(tmp_path, "s1", {"name": "s1", "description": "d"})   # 无 triggers 的技能不参与匹配
    m = _matcher(tmp_path)
    assert m.match("讲讲错题") is None


def test_empty_message(tmp_path):
    _mk(tmp_path, "s1", {"name": "s1", "description": "d", "triggers": "错题"})
    m = _matcher(tmp_path)
    assert m.match("") is None
    assert m.match("   ") is None
