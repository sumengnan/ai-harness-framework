# src/harness/sandbox/factory.py
from __future__ import annotations

from .local import LocalSandbox


def _docker_for(config, image: str, labels: dict | None = None,
                network: str | None = None, display_name: str = "沙箱",
                mem_limit: str | None = None):
    """按 config 造一个指定镜像的 DockerSandbox（除 image/labels/network/mem 外参数完全一致）。

    network 为 None 时取 config.sandbox_network（所有语言容器统一的网络）；浏览器容器可传更大
    mem_limit（config.browser_sandbox_mem_limit）避免 Chromium 被 OOM。display_name 区分前端进度里的容器。
    """
    from .docker import DockerSandbox
    return DockerSandbox(
        docker_host=config.sandbox_docker_host, image=image,
        workspace=config.sandbox_workspace, user=config.sandbox_user,
        network=network if network is not None else config.sandbox_network,
        mem_limit=mem_limit if mem_limit is not None else config.sandbox_mem_limit,
        cpus=config.sandbox_cpus, pids_limit=config.sandbox_pids_limit,
        disk_limit=getattr(config, "sandbox_disk_limit", "500m"),
        read_only=config.sandbox_read_only,
        tls_ca_cert=config.sandbox_docker_tls_ca_cert,
        tls_client_cert=config.sandbox_docker_tls_client_cert,
        tls_client_key=config.sandbox_docker_tls_client_key,
        tls_verify=config.sandbox_docker_tls_verify, labels=labels,
        display_name=display_name)


def build_sandbox(config, labels: dict | None = None, *, image: str | None = None,
                  network: str | None = None, display_name: str = "沙箱",
                  mem_limit: str | None = None):
    """造一个 Sandbox：docker 后端 → 指定 image 的 DockerSandbox；否则 → LocalSandbox。

    不再有「基础镜像 / 按语言路由」之分——每个容器就是「某镜像的一个 DockerSandbox」，
    按语言起哪个镜像由 SandboxManager 决定（见 get_lang）。labels 打到容器上供孤儿回收。
    """
    if config.sandbox_backend == "docker":
        return _docker_for(config, image, labels=labels, network=network,
                           display_name=display_name, mem_limit=mem_limit)
    return LocalSandbox()
