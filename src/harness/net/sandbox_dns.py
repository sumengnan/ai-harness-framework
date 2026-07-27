# src/harness/net/sandbox_dns.py
from __future__ import annotations

# 在沙箱容器内解析主机名 → IP 列表。抓取（HTTP/浏览器）真正出网都发生在沙箱，故 DNS 也在
# 沙箱内解析，避免宿主与沙箱两侧 DNS 视图不一致带来的 SSRF 绕过。
# 用 glibc 自带的 getent（各基础镜像都有：ubuntu/debian/python 均可），不依赖容器内有
# python3 或 curl——基础镜像可保持精简（如纯 ubuntu:24.04）。


async def resolve_in_sandbox(sandbox, host: str, timeout: float = 30.0) -> list[str]:
    """在沙箱容器内解析 host，返回去重后的 IP 列表（供 net.policy.check_url 的 resolve 使用）。

    `getent ahosts <host>` 每行形如 "<ip>  STREAM <host>"，取首列即 IP（含 A/AAAA）。
    """
    # quiet=True：DNS 解析是内部基础设施动作，不刷到前端沙箱活动日志（用户不关心 getent ahosts）
    res = await sandbox.exec(["getent", "ahosts", host], timeout + 5, quiet=True)
    ips: list[str] = []
    for line in res.stdout.splitlines():
        parts = line.split()
        if parts:
            ips.append(parts[0])
    return sorted(set(ips))
