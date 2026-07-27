from unittest.mock import Mock

from harness import progress
from harness.events import Progress
from harness.sandbox.docker import DockerSandbox


def test_emitter_set_and_reset():
    got = []
    token = progress.set_emitter(got.append)
    progress.emit(Progress("sandbox", "hi"))
    progress.reset_emitter(token)
    progress.emit(Progress("sandbox", "after"))   # 已 reset：无 emitter，忽略
    assert [(e.scope, e.text) for e in got] == [("sandbox", "hi")]


async def test_sandbox_start_emits_progress(monkeypatch):
    import docker
    import docker.tls

    def fake_client(base_url, tls):
        c = Mock()
        c.images.get.return_value = Mock()        # 镜像已存在，不走 pull 分支
        c.containers.run.return_value = Mock()
        return c

    monkeypatch.setattr(docker.tls, "TLSConfig", lambda **k: object())
    monkeypatch.setattr(docker, "DockerClient", fake_client)

    got = []
    token = progress.set_emitter(got.append)
    try:
        sb = DockerSandbox(docker_host="tcp://h:2376", image="python:3.12")
        await sb.start()
    finally:
        progress.reset_emitter(token)

    events = [e for e in got if isinstance(e, Progress)]
    assert {e.scope for e in events} == {"sandbox"}
    texts = [e.text for e in events]
    assert any("启动" in t and "沙箱" in t for t in texts)   # 「启动 沙箱…」进度
    # 成功由 status=ok 表达（不再单独发「已就绪」文本）
    assert any(getattr(e, "status", None) == "ok" for e in events)


async def test_sandbox_exec_emits_progress_even_when_reused():
    # 容器已就绪（复用）：start() 提前返回不产进度，但每次 exec 仍应上报沙箱执行
    sb = DockerSandbox(docker_host="tcp://h:2376", image="python:3.12")
    container = Mock()
    exec_res = Mock()
    exec_res.output = (b"ok", b"")
    exec_res.exit_code = 0
    container.exec_run.return_value = exec_res
    sb._container = container

    got = []
    token = progress.set_emitter(got.append)
    try:
        await sb.exec(["sh", "-c", "ls"], timeout=5)
    finally:
        progress.reset_emitter(token)

    events = [e for e in got if isinstance(e, Progress)]
    assert events and all(e.scope == "sandbox" for e in events)
    # 命令行以 running→ok 折叠（同一 key），成功不再另起「执行完成」结果文字
    run = next(e for e in events if e.text.startswith("执行 ") and e.status == "running")
    ok = [e for e in events if e.text.startswith("执行 ") and e.status == "ok"]
    assert ok and ok[0].key == run.key
    assert not any(e.text == "执行完成" for e in events)


async def test_sandbox_exec_emits_failure_on_nonzero_exit():
    sb = DockerSandbox(docker_host="tcp://h:2376", image="python:3.12")
    container = Mock()
    exec_res = Mock()
    exec_res.output = (b"", b"not found")
    exec_res.exit_code = 127
    container.exec_run.return_value = exec_res
    sb._container = container

    got = []
    token = progress.set_emitter(got.append)
    try:
        await sb.exec(["sh", "-c", "nope"], timeout=5)
    finally:
        progress.reset_emitter(token)

    events = [e for e in got if isinstance(e, Progress)]
    # 命令行标记为失败状态
    assert any(e.text.startswith("执行 ") and e.status == "error" for e in events)
    # 失败仍输出失败原因：退出码 + stderr 摘要
    assert any(e.status == "error" and "失败原因" in e.text and "127" in e.text for e in events)
    assert any("not found" in e.text for e in events)
