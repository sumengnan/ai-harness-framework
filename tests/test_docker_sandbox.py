import os
from unittest.mock import Mock

import pytest

from harness.config import HarnessConfig
from harness.sandbox.base import SandboxError
from harness.sandbox.docker import DockerSandbox
from harness.sandbox.factory import build_sandbox
from harness.sandbox.local import LocalSandbox


def _mock_docker_sandbox():
    sb = DockerSandbox(docker_host="tcp://h:2376", image="python:3.12-slim")
    container = Mock()
    exec_res = Mock()
    exec_res.output = (b"", b"")
    exec_res.exit_code = 0
    container.exec_run.return_value = exec_res
    sb._container = container
    sb._client = Mock()
    return sb, container


def test_factory_local_default():
    cfg = HarnessConfig(api_key="k", _env_file=None)   # 断言源码默认，不受本地 .env 影响
    assert isinstance(build_sandbox(cfg), LocalSandbox)


async def test_start_builds_tls_client(monkeypatch):
    # 直连 daemon TLS 端口：start() 应以配置的证书构造 TLSConfig 并传给 DockerClient
    import docker
    import docker.tls
    captured = {}

    class FakeTLS:
        def __init__(self, **kw):
            captured["tls_kwargs"] = kw

    def fake_docker_client(base_url, tls):
        captured["base_url"] = base_url
        captured["tls"] = tls
        client = Mock()
        client.containers.run.return_value = Mock()
        return client

    monkeypatch.setattr(docker.tls, "TLSConfig", FakeTLS)
    monkeypatch.setattr(docker, "DockerClient", fake_docker_client)

    sb = DockerSandbox(
        docker_host="tcp://h:2376", image="python:3.12-slim",
        tls_ca_cert="/certs/ca.pem", tls_client_cert="/certs/cert.pem",
        tls_client_key="/certs/key.pem", tls_verify=True)
    await sb.start()

    assert captured["base_url"] == "tcp://h:2376"
    assert isinstance(captured["tls"], FakeTLS)
    assert captured["tls_kwargs"]["client_cert"] == ("/certs/cert.pem", "/certs/key.pem")
    assert captured["tls_kwargs"]["ca_cert"] == "/certs/ca.pem"
    assert captured["tls_kwargs"]["verify"] is True


def test_tmpfs_owner_opts_numeric_and_named():
    # 数字 uid:gid → 内核层设属主，让非 root 沙箱用户可写（cap_drop=ALL 下无法事后 chown）
    sb = DockerSandbox(docker_host="tcp://h:2376", image="img", user="1000:1000")
    assert sb._tmpfs_owner_opts() == "uid=1000,gid=1000,mode=0700"
    # 单值 uid 复用为 gid
    assert DockerSandbox("tcp://h", "img", user="1001")._tmpfs_owner_opts() == "uid=1001,gid=1001,mode=0700"
    # 非数字用户名无法作 uid= 挂载选项 → 退回人人可写
    assert DockerSandbox("tcp://h", "img", user="appuser")._tmpfs_owner_opts() == "mode=0777"


async def test_start_tmpfs_is_writable_by_sandbox_user(monkeypatch):
    # 回归护栏：工作区 tmpfs 必须带 uid/gid/mode 属主选项，否则非 root 用户无法写、
    # 且 docker cp 写不进 tmpfs——会导致 run_java 等「file not found」失败。
    import docker
    captured = {}

    def fake_docker_client(base_url, tls):
        client = Mock()
        def run(image, **kw):
            captured.update(kw)
            return Mock()
        client.containers.run.side_effect = run
        client.images.get.return_value = Mock()
        return client

    monkeypatch.setattr(docker, "DockerClient", fake_docker_client)
    sb = DockerSandbox(docker_host="tcp://h:2376", image="img", user="1000:1000",
                       workspace="/workspace", disk_limit="500m",
                       mem_limit="200m", cpus=2.0)
    await sb.start()
    tmpfs = captured["tmpfs"]["/workspace"]
    assert "uid=1000" in tmpfs and "gid=1000" in tmpfs and "mode=07" in tmpfs
    assert "size=500m" in tmpfs                       # 工作区磁盘上限来自 disk_limit
    assert captured["mem_limit"] == "200m"            # 内存上限
    assert captured["nano_cpus"] == 2_000_000_000     # CPU 上限（核→nano_cpus）


async def test_staging_transfer_emits_no_progress_noise(monkeypatch):
    # /tmp 中转的 mkdir/cp/rm 走 _exec_raw，不得 emit Progress——否则前端沙箱日志被刷屏
    import harness.sandbox.docker as dockermod
    events = []
    monkeypatch.setattr(dockermod, "emit", lambda ev: events.append(ev))
    sb, container = _mock_docker_sandbox()
    await sb.write_file("Main.java", "class Main{}")   # 内部走 mkdir→put→cp→rm
    assert events == [], f"中转不应产生 Progress 事件，却有：{events}"
    # 对照：公有 exec 仍上报（真实执行要可见）
    await sb.exec(["echo", "hi"], 5)
    assert events, "公有 exec 应照常 emit Progress"


async def test_list_files_enforces_path_constraint():
    # #2：list_files 必须先 resolve_in_workspace，逃逸路径抛 SandboxError
    sb, container = _mock_docker_sandbox()
    with pytest.raises(SandboxError):
        await sb.list_files("/etc")
    with pytest.raises(SandboxError):
        await sb.list_files("../..")
    container.exec_run.assert_not_called()


async def test_exec_timeout_ceil_not_truncated():
    # #6：timeout<1 不应被 int() 截成 0（=不限时），应向上取整为至少 1
    sb, container = _mock_docker_sandbox()
    await sb.exec(["echo", "hi"], timeout=0.5)
    wrapped = container.exec_run.call_args.args[0]
    assert wrapped[:2] == ["timeout", "1"]


async def test_close_still_closes_client_on_remove_error():
    # #8：remove 抛异常时仍要 client.close()，且两者状态都置空
    sb, container = _mock_docker_sandbox()
    client = sb._client
    container.remove.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError):
        await sb.close()
    client.close.assert_called_once()
    assert sb._container is None
    assert sb._client is None


@pytest.mark.skipif(not os.getenv("HARNESS_SANDBOX_DOCKER_HOST"),
                    reason="需要真实远程 docker（HARNESS_SANDBOX_DOCKER_HOST）")
async def test_docker_exec_roundtrip():
    cfg = HarnessConfig(
        api_key="k", sandbox_backend="docker",
        sandbox_docker_host=os.environ["HARNESS_SANDBOX_DOCKER_HOST"],
        sandbox_docker_tls_ca_cert=os.environ.get("HARNESS_SANDBOX_DOCKER_TLS_CA_CERT", ""),
        sandbox_docker_tls_client_cert=os.environ.get("HARNESS_SANDBOX_DOCKER_TLS_CLIENT_CERT", ""),
        sandbox_docker_tls_client_key=os.environ.get("HARNESS_SANDBOX_DOCKER_TLS_CLIENT_KEY", ""))
    sb = build_sandbox(cfg)
    await sb.start()
    try:
        await sb.write_file("a.txt", "hi")
        assert await sb.read_file("a.txt") == "hi"
        r = await sb.exec(["python3", "-c", "print(6*7)"], timeout=15)
        assert "42" in r.stdout
    finally:
        await sb.close()


async def test_exec_quiet_suppresses_sandbox_progress():
    # quiet=True（如 DNS getent 解析）不应产出「执行 …」沙箱活动进度；普通 exec 仍产出
    from harness.events import Progress
    from harness.progress import reset_emitter, set_emitter
    sb, _ = _mock_docker_sandbox()
    events: list = []
    tok = set_emitter(events.append)
    try:
        await sb.exec(["getent", "ahosts", "example.com"], 5, quiet=True)
        exec_progress = [e for e in events
                         if isinstance(e, Progress) and e.text.startswith("执行")]
        assert exec_progress == []                       # 静默：无「执行」进度
        await sb.exec(["echo", "hi"], 5)                 # 普通 exec
        assert any(e.text.startswith("执行") for e in events
                   if isinstance(e, Progress))           # 仍上报
    finally:
        reset_emitter(tok)
