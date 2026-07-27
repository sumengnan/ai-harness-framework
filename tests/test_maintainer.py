# tests/test_maintainer.py
from harness.memory.record import MemType, MemoryFilter, MemoryRecord
from harness.memory.maintainer import ConsolidationConfig, MemoryMaintainer
from harness.memory.sqlite_backend import SqliteVecBackend


class ScriptedCompleter:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
    async def __call__(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        return self._responses.pop(0)


def _epi(rid, vec):
    return MemoryRecord(owner_id="u1", kind="k", mem_type=MemType.EPISODIC,
                        text=f"episodic {rid}", embedding=list(vec), id=rid)


def _maintainer(backend, mock_embedder, responses, **cfg):
    return MemoryMaintainer(backend, mock_embedder(dimension=3),
                            ScriptedCompleter(responses),
                            ConsolidationConfig(**cfg), now_fn=lambda: 1000)


async def test_consolidate_merges_similar(mock_embedder):
    b = SqliteVecBackend(":memory:", dimension=3)
    b.upsert([_epi("a", [1.0, 0.0, 0.0]), _epi("b", [1.0, 0.0, 0.0]),
              _epi("c", [0.0, 1.0, 0.0])])
    m = _maintainer(b, mock_embedder, ["用户偏好已固化"], min_cluster=2)
    out = await m.consolidate("u1", "k")
    assert out["clusters"] == 1 and out["merged"] == 2 and out["created"] == 1
    assert b.get(["a"])[0].superseded == 1 and b.get(["b"])[0].superseded == 1
    assert b.get(["c"])[0].superseded == 0
    incl = b.vector_search([1.0, 0.0, 0.0],
                           filters=MemoryFilter(owner_id="u1", mem_type="semantic"), k=5)
    assert any(h.record.source == "consolidate" for h in incl)


async def test_consolidate_skips_isolated(mock_embedder):
    b = SqliteVecBackend(":memory:", dimension=3)
    b.upsert([_epi("a", [1.0, 0.0, 0.0]), _epi("b", [0.0, 1.0, 0.0])])
    m = _maintainer(b, mock_embedder, [], min_cluster=2)
    out = await m.consolidate("u1", "k")
    assert out["clusters"] == 0
    assert b.get(["a"])[0].superseded == 0 and b.get(["b"])[0].superseded == 0


async def test_consolidate_llm_failure_skips_cluster(mock_embedder):
    b = SqliteVecBackend(":memory:", dimension=3)
    b.upsert([_epi("a", [1.0, 0.0, 0.0]), _epi("b", [1.0, 0.0, 0.0])])

    class Boom:
        async def __call__(self, s, u):
            raise RuntimeError("蒸馏失败")

    m = MemoryMaintainer(b, mock_embedder(dimension=3), Boom(),
                         ConsolidationConfig(min_cluster=2), now_fn=lambda: 1000)
    out = await m.consolidate("u1", "k")
    assert out["created"] == 0
    assert b.get(["a"])[0].superseded == 0


async def test_consolidate_supersede_failure_rolls_back(mock_embedder):
    b = SqliteVecBackend(":memory:", dimension=3)
    b.upsert([_epi("a", [1.0, 0.0, 0.0]), _epi("b", [1.0, 0.0, 0.0])])

    class _FailSupersede:
        def __init__(self, inner):
            self._inner = inner
        def __getattr__(self, n):
            return getattr(self._inner, n)
        def set_superseded(self, ids):
            raise RuntimeError("supersede 失败")

    wrapped = _FailSupersede(b)
    m = MemoryMaintainer(wrapped, mock_embedder(dimension=3),
                         ScriptedCompleter(["蒸馏结果"]),
                         ConsolidationConfig(min_cluster=2), now_fn=lambda: 1000)
    out = await m.consolidate("u1", "k")
    assert out["created"] == 0                        # 未计入
    assert b.get(["a"])[0].superseded == 0            # 源未被 supersede
    from harness.memory.record import MemoryFilter
    incl = b.vector_search([1.0, 0.0, 0.0],
                           filters=MemoryFilter(owner_id="u1", mem_type="semantic"), k=5)
    assert not any(h.record.source == "consolidate" for h in incl)   # 新记录已回滚


async def test_maintain_purges_and_consolidates(mock_embedder):
    b = SqliteVecBackend(":memory:", dimension=3, now_fn=lambda: 1000)
    exp = _epi("old", [0.0, 0.0, 1.0]); exp.expires_at = 100
    b.upsert([exp, _epi("a", [1.0, 0.0, 0.0]), _epi("b", [1.0, 0.0, 0.0])])
    m = _maintainer(b, mock_embedder, ["固化事实"], min_cluster=2)
    out = await m.maintain("u1", "k")
    assert out["purged"] == 1 and out["created"] == 1
    assert b.get(["old"]) == []


def _sem(rid, vec):
    return MemoryRecord(owner_id="u1", kind="k", mem_type=MemType.SEMANTIC,
                        text=f"preference {rid}", embedding=list(vec), id=rid)


async def test_consolidate_semantic_merges_preferences(mock_embedder):
    # C1：把同主题的多条 semantic 偏好合并成一条（手动「整理相似偏好」）
    b = SqliteVecBackend(":memory:", dimension=3)
    b.upsert([_sem("a", [1.0, 0.0, 0.0]), _sem("b", [1.0, 0.0, 0.0]),
              _sem("c", [0.0, 1.0, 0.0])])
    comp = ScriptedCompleter(["合并后的偏好"])
    m = MemoryMaintainer(b, mock_embedder(dimension=3), comp,
                         ConsolidationConfig(min_cluster=2), now_fn=lambda: 1000)
    out = await m.consolidate_semantic("u1", "k")
    assert out["clusters"] == 1 and out["merged"] == 2 and out["created"] == 1
    assert b.get(["a"])[0].superseded == 1 and b.get(["b"])[0].superseded == 1
    assert b.get(["c"])[0].superseded == 0             # 不相似的偏好保留
    assert "偏好" in comp.calls[0][0]                    # 用的是偏好蒸馏提示词，非情景
    incl = b.vector_search([1.0, 0.0, 0.0],
                           filters=MemoryFilter(owner_id="u1", mem_type="semantic"), k=5)
    assert any(h.record.source == "consolidate" for h in incl)


async def test_consolidate_default_leaves_semantic_untouched(mock_embedder):
    # 向后兼容：默认 consolidate（episodic 档）不得动 semantic 偏好
    b = SqliteVecBackend(":memory:", dimension=3)
    b.upsert([_sem("a", [1.0, 0.0, 0.0]), _sem("b", [1.0, 0.0, 0.0])])
    m = _maintainer(b, mock_embedder, [], min_cluster=2)
    out = await m.consolidate("u1", "k")
    assert out["clusters"] == 0
    assert b.get(["a"])[0].superseded == 0 and b.get(["b"])[0].superseded == 0
