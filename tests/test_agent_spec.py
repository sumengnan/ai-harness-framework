from harness.orchestration.spec import AgentSpec, AgentRoster


def test_roster_get_and_names():
    r = AgentRoster([
        AgentSpec("a", "descA", "pA", ["t1"]),
        AgentSpec("b", "descB", "pB", []),
    ])
    assert r.get("a").system_prompt == "pA"
    assert r.get("missing") is None
    assert set(r.names()) == {"a", "b"}


def test_describe_contains_roles():
    r = AgentRoster([AgentSpec("researcher", "擅长检索", "p", ["browse"])])
    d = r.describe()
    assert "researcher" in d and "擅长检索" in d
