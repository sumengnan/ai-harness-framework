import textwrap

from harness.orchestration.roster_loader import load_roster


def _write(tmp_path, fname, content):
    (tmp_path / fname).write_text(textwrap.dedent(content), encoding="utf-8")


def test_load_valid_agent_multiline_and_list(tmp_path):
    _write(tmp_path, "researcher.yaml", """
        name: researcher
        description: 检索
        system_prompt: |
          你是研究员。
          第二行。
        tool_names:
          - search_knowledge
          - http_request
    """)
    specs, warns = load_roster(str(tmp_path))
    assert warns == []
    assert len(specs) == 1
    s = specs[0]
    assert s.name == "researcher"
    assert s.description == "检索"
    assert "第二行" in s.system_prompt
    assert s.tool_names == ["search_knowledge", "http_request"]


def test_missing_dir_returns_empty(tmp_path):
    specs, warns = load_roster(str(tmp_path / "nope"))
    assert specs == [] and warns == []


def test_empty_dir_returns_empty(tmp_path):
    specs, warns = load_roster(str(tmp_path))
    assert specs == [] and warns == []


def test_skip_malformed_yaml(tmp_path):
    _write(tmp_path, "bad.yaml", "name: [unclosed\n")
    _write(tmp_path, "ok.yaml", """
        name: ok
        description: d
        system_prompt: p
        tool_names: [calculator]
    """)
    specs, warns = load_roster(str(tmp_path))
    assert [s.name for s in specs] == ["ok"]
    assert any("bad.yaml" in w for w in warns)


def test_skip_missing_field(tmp_path):
    _write(tmp_path, "nodesc.yaml", "name: x\nsystem_prompt: p\ntool_names: []\n")
    specs, warns = load_roster(str(tmp_path))
    assert specs == []
    assert any("nodesc.yaml" in w for w in warns)


def test_tool_names_must_be_str_list(tmp_path):
    _write(tmp_path, "bad.yaml", """
        name: x
        description: d
        system_prompt: p
        tool_names:
          - 123
    """)
    specs, warns = load_roster(str(tmp_path))
    assert specs == []
    assert any("tool_names" in w for w in warns)


def test_duplicate_name_keeps_first(tmp_path):
    _write(tmp_path, "a_first.yaml",
           "name: dup\ndescription: d1\nsystem_prompt: p\ntool_names: [calculator]\n")
    _write(tmp_path, "b_second.yaml",
           "name: dup\ndescription: d2\nsystem_prompt: p\ntool_names: [calculator]\n")
    specs, warns = load_roster(str(tmp_path))
    assert len(specs) == 1
    assert specs[0].description == "d1"        # 按文件名排序，a_first 在前
    assert any("重复" in w for w in warns)


def test_non_mapping_skipped(tmp_path):
    _write(tmp_path, "list.yaml", "- a\n- b\n")
    specs, warns = load_roster(str(tmp_path))
    assert specs == []
    assert any("list.yaml" in w for w in warns)


def test_only_yaml_files_considered(tmp_path):
    _write(tmp_path, "readme.txt", "name: x")
    _write(tmp_path, "a.yaml",
           "name: a\ndescription: d\nsystem_prompt: p\ntool_names: [calculator]\n")
    specs, _ = load_roster(str(tmp_path))
    assert [s.name for s in specs] == ["a"]


def test_yml_extension_also_loaded(tmp_path):
    _write(tmp_path, "a.yml",
           "name: a\ndescription: d\nsystem_prompt: p\ntool_names: [calculator]\n")
    specs, _ = load_roster(str(tmp_path))
    assert [s.name for s in specs] == ["a"]
