from harness.config import HarnessConfig


def test_defaults():
    # _env_file=None：断言源码默认值，不受开发机本地 .env 影响
    cfg = HarnessConfig(api_key="k", _env_file=None)
    assert cfg.model == "gpt-4o-mini"
    assert cfg.max_steps == 100
    assert cfg.base_url.endswith("/v1")


def test_env_override(monkeypatch):
    monkeypatch.setenv("HARNESS_MODEL", "deepseek-chat")
    monkeypatch.setenv("HARNESS_MAX_STEPS", "3")
    cfg = HarnessConfig(api_key="k", _env_file=None)
    assert cfg.model == "deepseek-chat"
    assert cfg.max_steps == 3


def test_reliability_defaults():
    cfg = HarnessConfig(api_key="k", _env_file=None)
    assert cfg.max_retries == 2
    assert cfg.retry_base_delay == 0.5
    assert cfg.max_tokens_budget is None
    assert cfg.max_wall_seconds is None
    assert cfg.tool_result_max_chars == 1_000_000
    assert cfg.otel_enabled is False
    assert cfg.otel_exporter == "console"
    assert cfg.price_map == {}


def test_memory_defaults():
    cfg = HarnessConfig(api_key="k", _env_file=None)
    assert cfg.embedding_base_url.endswith("/v1")
    assert cfg.embedding_model == "text-embedding-3-small"
    assert cfg.embedding_dimension == 1536
    assert cfg.memory_db_path == "memory.db"
    assert cfg.chunk_size == 1000
    assert cfg.chunk_overlap == 200
    assert cfg.search_top_k == 10
    assert cfg.memory_collection == "knowledge"


def test_sandbox_defaults():
    cfg = HarnessConfig(api_key="k", _env_file=None)
    assert cfg.sandbox_backend == "local"
    assert cfg.sandbox_shell_image == "quay.io/centos/centos:stream9"   # run_shell / 缺省 fs / 沙箱内 curl+getent
    assert cfg.sandbox_lang_images["python"] == "python:3.12-slim"
    assert cfg.sandbox_network == "bridge"               # 所有语言容器统一网络
    assert cfg.sandbox_idle_timeout == 3600.0            # 语言容器空闲驱逐 1h
    assert cfg.sandbox_cpus == 1.0                       # 资源上限：CPU 1 核
    assert cfg.sandbox_mem_limit == "200m"               # 内存 200m（容纳 tmpfs 工作区）
    assert cfg.sandbox_disk_limit == "500m"              # 工作区磁盘（tmpfs）500m
    assert cfg.sandbox_exec_timeout == 600.0             # 单次执行超时 10min（罩住 pip 子进程）
    assert cfg.sandbox_output_max_chars == 1_000_000     # read_file 读大文件用
    assert cfg.http_allowed_domains == []
    assert cfg.http_block_private is True
    assert cfg.http_max_response_bytes == 10_000_000
    assert cfg.http_max_redirects == 5


def test_browser_defaults():
    cfg = HarnessConfig(api_key="k", _env_file=None)
    assert cfg.browser_headless is True
    assert cfg.browser_nav_timeout == 30.0
    assert cfg.browser_wait_until == "networkidle"
    assert cfg.browser_output_max_chars == 8000
    assert cfg.browser_user_agent == ""


def test_multiagent_defaults():
    cfg = HarnessConfig(api_key="k", _env_file=None)
    assert cfg.max_dispatch_depth == 2
    assert cfg.sub_agent_max_steps == 100


def test_episodic_defaults():
    cfg = HarnessConfig(api_key="k", _env_file=None)
    assert cfg.episode_collection == "episodes"
    assert cfg.episode_recall_k == 3


def test_persistence_defaults():
    cfg = HarnessConfig(api_key="k", _env_file=None)
    assert cfg.persistence_db_path == "harness.db"
