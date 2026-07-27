# src/harness/sandbox/docker.py
from __future__ import annotations

import asyncio
import io
import math
import os
import tarfile
import uuid

from .base import ExecResult, SandboxError, resolve_in_workspace
from ..events import Progress
from ..progress import emit


class DockerSandbox:
    """远程 Linux 云服务器的 Docker 容器沙箱（docker SDK 直连 daemon 的 TLS 端口）。真正的安全边界。

    已知限制（容器内符号链接）：resolve_in_workspace 的 os.path.realpath 符号链接
    检测运行在宿主机侧，对容器内的路径无效；因此 /workspace 的路径限制是尽力而为的
    （能拦住 ../、绝对路径这类显式逃逸）。真正的安全边界是容器隔离本身——read_only
    根文件系统 + cap_drop=ALL + 非 root 用户 + network=none + 用后一次性销毁；即便在
    容器内借符号链接逃出 /workspace，也只能读到同一个一次性容器内的文件。
    """

    def __init__(self, docker_host: str, image: str, workspace: str = "/workspace",
                 user: str = "1000:1000", network: str = "none", mem_limit: str = "512m",
                 cpus: float = 1.0, pids_limit: int = 128, read_only: bool = False,
                 disk_limit: str = "500m",
                 tls_ca_cert: str = "", tls_client_cert: str = "",
                 tls_client_key: str = "", tls_verify: bool = True,
                 labels: dict | None = None, display_name: str = "沙箱") -> None:
        self.workspace = workspace
        self.image = image                  # 公开：供工具在结果 meta 里标注「用的哪个镜像」
        self._display_name = display_name   # 前端进度里区分基础沙箱/子沙箱
        self._docker_host = docker_host
        self._image = image
        self._user = user
        self._network = network
        self._mem_limit = mem_limit
        self._cpus = cpus
        self._pids_limit = pids_limit
        self._disk_limit = disk_limit
        self._read_only = read_only
        self._tls_ca_cert = tls_ca_cert
        self._tls_client_cert = tls_client_cert
        self._tls_client_key = tls_client_key
        self._tls_verify = tls_verify
        self._labels = labels or None   # 打到容器上，供重启后按标签回收孤儿容器
        self._client = None
        self._container = None

    def _tls_config(self):
        """按配置构造 docker TLS 客户端配置：双向 TLS（客户端证书/私钥 + CA）。"""
        from docker.tls import TLSConfig
        client_cert = ((self._tls_client_cert, self._tls_client_key)
                       if self._tls_client_cert and self._tls_client_key else None)
        return TLSConfig(
            client_cert=client_cert,
            ca_cert=self._tls_ca_cert or None,
            verify=self._tls_verify,
        )

    async def start(self) -> None:
        if self._container is not None:
            return
        import docker
        # 「启动 X…」running→ok 同 key 折叠成一行：成功即变绿勾，不再单独发「已就绪」；
        # 连接 daemon 是实现细节，不上报。
        start_key = uuid.uuid4().hex
        emit(Progress("sandbox", f"启动 {self._display_name}…",
                      status="running", key=start_key))
        self._client = await asyncio.to_thread(
            docker.DockerClient, base_url=self._docker_host, tls=self._tls_config())
        # 镜像缺失则显式拉取，让"拉取镜像"这一步在前端可见
        try:
            await asyncio.to_thread(self._client.images.get, self._image)
        except docker.errors.ImageNotFound:
            emit(Progress("sandbox", f"拉取镜像 {self._image}…（首次较慢）"))
            await asyncio.to_thread(self._client.images.pull, self._image)
        # 工作区用 tmpfs（临时、限容量、支持只读根）。关键：带 uid/gid/mode 挂载选项让
        # 非 root 沙箱用户可写——cap_drop=ALL 下连 root 都没 CAP_CHOWN，无法事后 chown，
        # 只能在挂载时由内核设属主。
        self._container = await asyncio.to_thread(
            self._client.containers.run,
            self._image, command="sleep infinity", detach=True,
            working_dir=self.workspace, user=self._user, network_mode=self._network,
            read_only=self._read_only,
            tmpfs={self.workspace: f"rw,size={self._disk_limit},{self._tmpfs_owner_opts()}"},
            mem_limit=self._mem_limit, nano_cpus=int(self._cpus * 1e9),
            pids_limit=self._pids_limit, cap_drop=["ALL"],
            security_opt=["no-new-privileges"], auto_remove=False,
            labels=self._labels)
        emit(Progress("sandbox", f"启动 {self._display_name}…",
                      status="ok", key=start_key))

    async def close(self) -> None:
        try:
            if self._container is not None:
                await asyncio.to_thread(self._container.remove, force=True)
        finally:
            self._container = None
            if self._client is not None:
                await asyncio.to_thread(self._client.close)
                self._client = None

    async def for_language(self, language: str | None = None,
                           version: str | None = None) -> "DockerSandbox":
        # 单容器直连（测试/无会话管理时）：无按语言路由，就是这一个容器。start 幂等。
        await self.start()
        return self

    def _tmpfs_owner_opts(self) -> str:
        """从 self._user（"uid:gid" 或 "uid"）解析 tmpfs 的 uid/gid/mode 挂载选项。

        非数字用户名无法作为 tmpfs uid= 选项，此时退回仅 mode=0777（人人可写），
        保证沙箱进程仍可写工作区。
        """
        parts = str(self._user).split(":")
        uid = parts[0]
        gid = parts[1] if len(parts) > 1 else parts[0]
        if uid.isdigit() and gid.isdigit():
            return f"uid={uid},gid={gid},mode=0700"
        return "mode=0777"

    # docker cp（put_archive/get_archive）无法读写 tmpfs 挂载点——文件会被静默丢弃。
    # 工作区是 tmpfs，故所有传输经非 tmpfs 的 /tmp 暂存中转：写=put 到 /tmp 再容器内
    # cp 进工作区；读=容器内 cp 到 /tmp 再 get。cp 用 argv（非 shell），受控路径无注入。
    async def _put_via_stage(self, dest_dir: str, tar_bytes: bytes) -> None:
        stage = f"/tmp/.mcpx_{uuid.uuid4().hex}"
        r = await self._exec_raw(["mkdir", "-p", stage], 15)
        if r.exit_code != 0:
            raise SandboxError(f"创建暂存目录失败：{(r.stderr or r.stdout).strip()}")
        await asyncio.to_thread(self._container.put_archive, stage, tar_bytes)
        r = await self._exec_raw(["cp", "-a", f"{stage}/.", f"{dest_dir}/"], 60)
        await self._exec_raw(["rm", "-rf", stage], 15)
        if r.exit_code != 0:
            raise SandboxError(f"写入工作区失败：{(r.stderr or r.stdout).strip()}")

    async def _get_via_stage(self, real: str) -> bytes:
        stage = f"/tmp/.mcpx_{uuid.uuid4().hex}"
        r = await self._exec_raw(["mkdir", "-p", stage], 15)
        if r.exit_code != 0:
            raise SandboxError(f"创建暂存目录失败：{(r.stderr or r.stdout).strip()}")
        # real 已被 resolve_in_workspace 约束在工作区内；用 argv 传参避免 shell 注入
        r = await self._exec_raw(["cp", "-a", real, f"{stage}/"], 60)
        if r.exit_code != 0:
            await self._exec_raw(["rm", "-rf", stage], 15)
            raise SandboxError(f"读取失败：{(r.stderr or r.stdout).strip()}")
        bits, _ = await asyncio.to_thread(
            self._container.get_archive, f"{stage}/{os.path.basename(real)}")
        data = b"".join(bits)
        await self._exec_raw(["rm", "-rf", stage], 15)
        return data

    async def _exec_raw(self, command: list[str], timeout: float) -> ExecResult:
        """在容器内跑命令并返回结果，但**不发 Progress 事件**。

        供内部 housekeeping（/tmp 中转的 mkdir/cp/rm）复用，避免把这些实现细节刷到
        前端沙箱活动日志——那些是修复 tmpfs 无法 docker cp 的搬运动作，用户不关心。
        """
        await self.start()
        wrapped = ["timeout", str(max(1, math.ceil(timeout))), *command]
        res = await asyncio.to_thread(
            self._container.exec_run, wrapped, workdir=self.workspace, demux=True)
        out, err = res.output if isinstance(res.output, tuple) else (res.output, b"")
        return ExecResult((out or b"").decode(errors="replace"),
                          (err or b"").decode(errors="replace"),
                          res.exit_code, timed_out=(res.exit_code == 124))

    async def exec(self, command: list[str], timeout: float,
                   *, quiet: bool = False) -> ExecResult:
        # quiet=True：内部基础设施命令（如 DNS 解析 getent ahosts），走静默路径不刷沙箱活动日志
        if quiet:
            return await self._exec_raw(command, timeout)
        await self.start()   # 幂等；确保容器启动进度先于本次“执行”进度
        # 容器复用时 start() 不再产出进度；每次执行仍上报，让沙箱活动可见
        desc = " ".join(command).replace("\n", " ")
        if len(desc) > 60:
            desc = desc[:60]
        # running→ok/error 用同一 key 折叠成一行：命令前只显示状态标识，成功不再另起结果文字
        exec_key = uuid.uuid4().hex
        emit(Progress("sandbox", f"执行 {desc}", status="running", key=exec_key))
        res = await self._exec_raw(command, timeout)
        if res.exit_code == 0:
            emit(Progress("sandbox", f"执行 {desc}", status="ok", key=exec_key))
        else:
            emit(Progress("sandbox", f"执行 {desc}", status="error", key=exec_key))
            # 失败仍输出失败原因：退出码 + 错误输出摘要
            snippet = (res.stderr or res.stdout).strip().replace("\n", " ")
            reason = f"失败原因：exit_code={res.exit_code}"
            if snippet:
                reason += f"，{snippet[:120]}"
            emit(Progress("sandbox", reason, status="error"))
        return res

    def _member_tar(self, member: str, data: bytes) -> bytes:
        """把单个文件打成 tar 字节，成员名为相对工作区根的路径（可含子目录）。"""
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as tar:
            info = tarfile.TarInfo(name=member)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        return stream.getvalue()

    async def write_file(self, path: str, content: str) -> None:
        await self.start()
        real = resolve_in_workspace(self.workspace, path)
        member = os.path.relpath(real, os.path.realpath(self.workspace))
        await self._put_via_stage(self.workspace, self._member_tar(member, content.encode()))

    async def write_bytes(self, path: str, data: bytes) -> None:
        """写裸字节到工作区（支持子目录，如 uploads/foo.png——tar 成员用相对路径，
        cp -a 进工作区时自动建子目录）。"""
        await self.start()
        real = resolve_in_workspace(self.workspace, path)
        member = os.path.relpath(real, os.path.realpath(self.workspace))
        await self._put_via_stage(self.workspace, self._member_tar(member, data))

    async def read_file(self, path: str) -> str:
        await self.start()
        real = resolve_in_workspace(self.workspace, path)
        stream = io.BytesIO(await self._get_via_stage(real))
        with tarfile.open(fileobj=stream) as tar:
            member = tar.next()
            return tar.extractfile(member).read().decode(errors="replace")

    async def list_files(self, path: str = ".") -> list[str]:
        await self.start()
        real = resolve_in_workspace(self.workspace, path)  # #2：先约束路径再执行
        # 用 _exec_raw：ls 是轻量文件操作（与 read_file/write_file 同类），不刷沙箱进度；
        # 作为 list_files 工具被调用时另有「工具调用」面板展示。
        res = await self._exec_raw(["ls", "-1", real], timeout=10)
        return [ln for ln in res.stdout.splitlines() if ln]

    async def archive_workspace(self) -> bytes:
        """把整个工作区打成 tar 字节（含 workspace 目录名），供跨容器整目录迁移。

        工作区是 tmpfs、无法直接 get_archive，故先在容器内 cp 到 /tmp 暂存再打包；暂存副本
        保留 workspace 目录名，使产出的 tar 顶层成员仍是 "workspace/…"，与 extract_workspace 对称。
        """
        await self.start()
        base = os.path.basename(self.workspace)
        stage = f"/tmp/.mcpws_{uuid.uuid4().hex}"
        r = await self._exec_raw(["mkdir", "-p", stage], 15)
        if r.exit_code != 0:
            raise SandboxError(f"创建暂存目录失败：{(r.stderr or r.stdout).strip()}")
        # cp -a /workspace /tmp/stage/ → /tmp/stage/workspace
        r = await self._exec_raw(["cp", "-a", self.workspace, f"{stage}/"], 120)
        if r.exit_code != 0:
            await self._exec_raw(["rm", "-rf", stage], 15)
            raise SandboxError(f"归档工作区失败：{(r.stderr or r.stdout).strip()}")
        bits, _ = await asyncio.to_thread(self._container.get_archive, f"{stage}/{base}")
        data = b"".join(bits)
        await self._exec_raw(["rm", "-rf", stage], 15)
        return data

    async def extract_workspace(self, data: bytes) -> None:
        """把 archive_workspace() 产出的 tar（顶层 "workspace/…"）解回工作区。

        工作区是 tmpfs、put_archive 写不进去，故先解到 /tmp 暂存再容器内 cp 进工作区。
        """
        await self.start()
        base = os.path.basename(self.workspace)
        stage = f"/tmp/.mcpws_{uuid.uuid4().hex}"
        r = await self._exec_raw(["mkdir", "-p", stage], 15)
        if r.exit_code != 0:
            raise SandboxError(f"创建暂存目录失败：{(r.stderr or r.stdout).strip()}")
        await asyncio.to_thread(self._container.put_archive, stage, data)  # → stage/workspace/…
        # 把 stage/workspace 的内容 cp 进真正的工作区
        r = await self._exec_raw(["cp", "-a", f"{stage}/{base}/.", f"{self.workspace}/"], 120)
        await self._exec_raw(["rm", "-rf", stage], 15)
        if r.exit_code != 0:
            raise SandboxError(f"还原工作区失败：{(r.stderr or r.stdout).strip()}")
