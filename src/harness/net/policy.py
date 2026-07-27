# src/harness/net/policy.py
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse


class PolicyError(Exception):
    ...


def default_resolve(host: str) -> list[str]:
    infos = socket.getaddrinfo(host, None)
    return list({info[4][0] for info in infos})


def _is_blocked_ip(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True  # 无法识别的地址一律视为不安全
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    # is_global 覆盖私有/回环/链路本地/保留/CGNAT(RFC6598) 等全部非公网段；
    # multicast 归类为 is_global=False 之外的情况另行拦截。
    return (not ip.is_global) or ip.is_multicast


def _host_allowed(host: str, allowed_domains: list[str]) -> bool:
    host = host.lower()
    for d in allowed_domains:
        d = d.lower()
        if host == d or host.endswith("." + d):
            return True
    return False


def check_url(url: str, allowed_domains: list[str], block_private: bool,
              resolve=default_resolve) -> None:
    """URL 不合规则抛 PolicyError。

    已知限制（DNS rebinding TOCTOU）：此处校验用的 DNS 解析与 httpx 实际发起
    连接时的解析是两次独立解析，二者之间存在 rebinding 时间窗口——攻击者控制的
    域名可在校验后把记录改指向内网。对不可信域名应使用 http_allowed_domains
    白名单模式作为强控制，仅靠 block_private 无法完全杜绝此类攻击。
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise PolicyError(f"仅允许 http/https：{parsed.scheme or '(空)'}")
    host = parsed.hostname
    if not host:
        raise PolicyError(f"无效 URL：{url}")
    if allowed_domains and not _host_allowed(host, allowed_domains):
        raise PolicyError(f"域名不在白名单：{host}")
    if block_private:
        ips = resolve(host)
        if not ips:
            raise PolicyError(f"无法解析主机：{host}")
        for ip in ips:
            if _is_blocked_ip(ip):
                raise PolicyError(f"目标为内网/保留地址，已拦截：{host} → {ip}")
