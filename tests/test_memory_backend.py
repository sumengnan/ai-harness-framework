# tests/test_memory_backend.py
from harness.memory.backend import MemoryBackend


def test_protocol_has_expected_methods():
    for name in ("upsert", "delete", "get",
                 "vector_search", "keyword_search", "list_by_entity"):
        assert hasattr(MemoryBackend, name)
