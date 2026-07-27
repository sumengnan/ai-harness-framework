# src/harness/shell/policy.py
"""危险 shell 命令判定（精选黑名单）。

这是**面向人工的启发式绊线**，不是安全边界——真正的边界是容器隔离
（cap_drop=ALL、network=none、非 root、tmpfs 工作区、用后销毁）。这里只对已知
危险的命令模式返回一个 Danger，交由上层弹窗让人工确认；安全命令返回 None 直接放行。
判定有意偏向多弹窗（宁可多问一次），因此对引号/转义等边角不追求精确。
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Danger:
    pattern: str   # 命中的规则名（简短标识）
    reason: str    # 面向用户的中文说明


# (正则, 规则名, 说明) —— 命中任一即需人工确认
_RULES: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"\brm\s+(?:-\w*[rf]\w*\s+)+"), "rm -rf", "递归/强制删除文件"),
    (re.compile(r"\brm\s+[^|;&]*(?:\s/\s*$|\s/\s|~|\*)"), "rm 危险目标", "删除根目录/家目录/通配目标"),
    # 兜底：任何 rm 调用都要人工确认，不止上面那两种危险形态——`rm 某文件` 一样是不可逆的
    # 删除。放在两条专用规则之后，让 `rm -rf /` 仍报更具体的理由。
    # 前缀 [;&|/] 是为了堵两个绕过：管道/串联后的 `... ; rm x`，以及绝对路径 `/bin/rm x`。
    # 刻意不匹配 `-` 前缀，故 `docker run --rm` 不会误报；结尾用 (?:\s|$) 才能接住
    # `find . | xargs rm` 这种 rm 位于命令末尾的写法。
    (re.compile(r"(?:^|[;&|/]|\s)rm(?:\s|$)"), "rm", "删除文件"),
    (re.compile(r"\bmkfs(?:\.\w+)?\b"), "mkfs", "格式化文件系统"),
    (re.compile(r"\bdd\b[^|;&]*\bof=/dev/"), "dd->设备", "直写块设备"),
    (re.compile(r">\s*/dev/(?:sd|nvme|hd|xvd)"), "写块设备", "覆盖磁盘设备"),
    (re.compile(r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"), "fork炸弹", "fork 炸弹"),
    (re.compile(r"\bchmod\s+-R\s+0*777\s+/"), "chmod 777 /", "对根目录放开全部权限"),
    (re.compile(r"\bchown\s+-R\b[^|;&]*\s/\s*$"), "chown -R /", "递归改根目录属主"),
    (re.compile(r"\b(?:shred|wipe)\b"), "shred", "不可恢复地擦除数据"),
    (re.compile(r"\bfind\b[^|;&]*-(?:delete|exec\s+rm)\b"), "find 批量删除", "批量删除文件"),
    (re.compile(r"\b(?:curl|wget)\b[^|]*\|\s*(?:sudo\s+)?(?:sh|bash|zsh)\b"), "管道执行", "下载脚本直接执行"),
    (re.compile(r"\bsudo\b"), "sudo", "提权执行"),
    (re.compile(r"\bkill(?:all)?\b\s+-9\s+-?1\b"), "kill -1", "杀死所有进程"),
    (re.compile(r"\bgit\s+clean\s+-\w*f\w*d"), "git clean -fd", "删除未跟踪文件/目录"),
    (re.compile(r">\s*/etc/"), "写 /etc", "覆盖系统配置文件"),
]


def classify_command(command: str) -> Danger | None:
    """安全返回 None；命中黑名单返回首个 Danger。command 为 run_shell 的原始命令串。"""
    for rx, name, reason in _RULES:
        if rx.search(command):
            return Danger(name, reason)
    return None
