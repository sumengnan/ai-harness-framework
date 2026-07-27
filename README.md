# ai-harness-framework

[![PyPI](https://img.shields.io/pypi/v/ai-harness-framework.svg)](https://pypi.org/project/ai-harness-framework/)
[![Python](https://img.shields.io/pypi/pyversions/ai-harness-framework.svg)](https://pypi.org/project/ai-harness-framework/)
[![CI](https://github.com/sumengnan/ai-harness-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/sumengnan/ai-harness-framework/actions/workflows/ci.yml)
[![License](https://img.shields.io/pypi/l/ai-harness-framework.svg)](#license)

```bash
pip install ai-harness-framework
```

最小 Agent 运行时内核。把"调模型 → 调工具 → 再调模型"这个循环，连同它在真实场景里必然会撞上的那些失败模式，一起做成一个可复用的包。

不是 agent 框架，不替你写 prompt，不预设你的业务形状。它只负责一件事：**让一个随机的模型，在一个确定性的壳子里可靠地干活。**

```
消息 ──▶ ModelClient ──▶ tool_calls ──▶ ToolExecutor ──▶ 结果回填 ──┐
          ▲                                                        │
          └────────────── AgentLoop（预算 / 打转检测 / 快照）◀───────┘
```

## 为什么是这些模块

Agent 循环本身只有几十行，难的是它必须扛住的事。下面每一条都对应一次真实的翻车：

| 失败模式 | 对策 | 位置 |
|---|---|---|
| 模型返回的工具参数不符合 schema | 校验错误原样喂回，让模型自纠正，**不崩** | `tools/base.py` |
| 工具调用参数不是合法 JSON | 同上，回填明确错误让它重发 | `loop/agent_loop.py` |
| 模型不肯停，一直调工具 | `max_steps` 硬上限 | `loop/agent_loop.py` |
| 模型原地打转，反复同参调同一工具 | 签名窗口检测 → 注入纠偏 **并同时升温** → 仍重复则中止 | `loop/agent_loop.py` |
| token / 墙钟花超 | 步边界检查，超限即中止 | `reliability/budget.py` |
| 工具超时或抛异常 | 兜成 `is_error` 结果，不让单个工具搞崩整个 run | `tools/base.py` |
| 端点 200 但空产出（内容安全拦截 / 截断） | 记 warning 留证，避免静默失败无从定位 | `llm/openai_compat.py` |
| 跑到一半崩了 | 步边界存快照，`resume(run_id)` 续跑 | `persistence/checkpoint.py` |
| 危险命令（`rm -rf` 之类） | 分类拦截 → 请求人工确认 → 拒绝后**强制模型如实交代** | `shell/policy.py` + `approval.py` |

> 关于"打转纠偏要连采样一起换"：只改措辞不改温度时，模型很容易照着同一条采样路径再走一遍——"请换个思路"这句话本身也是低概率才被听进去的。这类细节是这个包存在的理由。

## 安装

需要 Python ≥ 3.11。核心只有 4 个依赖（`openai` / `pydantic` / `pydantic-settings` / `opentelemetry-api`），重依赖全部下放到 extras，按需装：

```bash
pip install ai-harness-framework                       # 内核：模型 + 工具 + 循环 + 预算 + 事件
pip install "ai-harness-framework[memory,mcp]"         # 只要某几个特性
pip install "ai-harness-framework[all]"                # 全部特性
```

用 uv 的话：

```bash
uv add ai-harness-framework
uv add "ai-harness-framework[all]"
```

| Extra | 装什么 | 用到的模块 |
|---|---|---|
| `telemetry` | opentelemetry-sdk + OTLP exporter | 真正导出 span（不装则退化为 no-op，内核照常运行） |
| `tokenizer` | tiktoken | 精确 token 计数（不装则走估算回退） |
| `sandbox-docker` | docker[ssh] | `harness.sandbox.docker`（`LocalSandbox` 不需要） |
| `http` | httpx | `harness.tools.builtins.http_tool` |
| `browser` | trafilatura | `harness.browser` 正文提取 |
| `mcp` | mcp | `harness.mcp` |
| `skills` / `orchestration` | pyyaml | 技能注册表 / 多 agent 名册 |
| `memory` | sqlite-vec、semantic-text-splitter、tree-sitter-* | `harness.memory` 向量检索 |

判定依据是**模块级 import**：惰性 import（函数体内 / try 块内）一律下放。只用内核的消费者不会被 docker、sqlite-vec 这类重依赖拖累。

## 快速上手

```python
import asyncio

from pydantic import BaseModel

from harness.config import HarnessConfig
from harness.context.manager import ContextManager
from harness.events import RunError, RunFinished, TextDelta, ToolStarted
from harness.llm.openai_compat import OpenAICompatibleClient
from harness.loop.agent_loop import AgentLoop
from harness.reliability.budget import BudgetTracker
from harness.tools.base import Tool, ToolRegistry


class GetWeather(Tool):
    name = "get_weather"
    description = "查询某城市当前天气。"

    class Params(BaseModel):          # schema 从这里自动生成，也在这里自动校验
        city: str

    async def run(self, params: "GetWeather.Params") -> str:
        return f"{params.city}：晴，24°C"


async def main() -> None:
    config = HarnessConfig(
        api_key="sk-...",
        base_url="https://api.deepseek.com/v1",   # 任意 OpenAI 兼容端点
        model="deepseek-chat",
    )

    registry = ToolRegistry()
    registry.register(GetWeather())

    loop = AgentLoop(
        client=OpenAICompatibleClient(config),
        registry=registry,
        context=ContextManager("你是一个乐于助人的助手。"),
        model_name=config.model,
        max_steps=10,
        budget=BudgetTracker(max_tokens=100_000, max_wall_seconds=120),
        loop_detect_window=3,        # 连续 3 步同参调同一工具 → 判打转
    )

    async for ev in loop.run("北京天气怎么样？"):
        if isinstance(ev, TextDelta):
            print(ev.text, end="", flush=True)
        elif isinstance(ev, ToolStarted):
            print(f"\n[调用 {ev.tool_call.name}]")
        elif isinstance(ev, RunFinished):
            print("\n[完成]")
        elif isinstance(ev, RunError):
            print(f"\n[出错] {ev.error}")


asyncio.run(main())
```

`AgentLoop.run()` 是一个异步事件流。事件类型见 `harness/events.py`——`TextDelta` / `ReasoningDelta` / `ToolCallRequested` / `ToolStarted` / `ToolFinished` / `ModelUsage` / `ApprovalRequired` / `RunFinished` / `RunError` 等。消费方决定怎么渲染，内核不假设你有没有 UI。

## 厂商适配

`OpenAICompatibleClient` 指向任意 OpenAI 兼容端点（DeepSeek、Kimi、智谱 GLM、通义 Qwen、vLLM、Ollama…），并处理了各家的实际差异：

- **思考模式参数不统一**——上游只表达厂商中立的 `enable_thinking` 意图，落成哪家的参数由适配层翻译（Qwen/百炼用 `enable_thinking`，DeepSeek 用 `thinking={"type": ...}`）。
- **厂商识别按模型名优先、再看 base_url**——统一网关下同一个 `base_url` 转发多家模型时，靠 URL 判不出厂商。
- **不认某参数的模型直接剔除**，避免端点因"未知参数"报错。
- **`reasoning_content` 单独走 `ReasoningDelta` 事件**，不混进正文。

## 模块地图

| 模块 | 职责 |
|---|---|
| `types` / `events` / `state` | 核心数据类型与事件定义 |
| `llm/` | `ModelClient` 协议、OpenAI 兼容实现、采样参数管理 |
| `tools/` | `Tool` 基类、注册表、执行器（校验 + 兜底 + 截断） |
| `loop/` | `AgentLoop`——步循环、打转检测、快照、span |
| `context/` | 每轮发什么消息：窗口、裁剪、预算 |
| `reliability/` | 预算追踪、重试退避 |
| `persistence/` | checkpoint、序列化、轨迹落库 |
| `telemetry/` | OpenTelemetry tracer |
| `approval.py` | 人工审批闸门 |
| `sandbox/` | 执行隔离：Docker / Local |
| `shell/` `net/` | 危险命令分类、SSRF 防护 |
| `skills/` `orchestration/` `mcp/` `memory/` `browser/` | 可选能力 |

## 分层契约

`harness` 是内核，**不得反向依赖任何上层应用**。这条边界由 `tests/test_architecture_layering.py` 用 AST 逐文件钉死——注释里提到 `app` 不算违规，只有真正的 import 语句才算。

随手写一句 `from app.config import AppConfig` 就能把内核焊死在某个产品上，而且不会有任何报错。所以它需要一个测试来守，而不是靠自觉。

## 消费者

- **ai-learning-helper**——学习助手。第一个消费者，本包即从其 `src/harness` 抽出。
- **ai-fix-code-loop**——自动修 bug 的 agentic loop：测试失败进去，验证过的补丁出来。第二个消费者。

抽包时的验收标准就是：ai-learning-helper 的**完整测试套件（1906 项）在 `harness` 来自本包而非其本地 `src/` 的情况下全绿**。一个只服务过一个应用的抽象，没人知道它是不是真的抽象。

## 开发

```bash
uv sync --all-extras --group dev
uv run pytest                # 657 passed, 3 skipped
```

Python ≥ 3.11。

## License

MIT
