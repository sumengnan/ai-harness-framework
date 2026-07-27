# src/harness/telemetry/tracer.py
from __future__ import annotations

from opentelemetry import trace


def get_tracer(name: str = "harness"):
    """返回 tracer。未配置全局 provider 时 OTel 默认返回 no-op tracer，零开销、不报错。"""
    return trace.get_tracer(name)


def setup_telemetry(config) -> None:
    """按配置安装 OTel provider；otel_enabled=False 时什么都不做（保持 no-op）。"""
    if not getattr(config, "otel_enabled", False):
        return

    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    provider = TracerProvider()
    if config.otel_exporter == "otlp":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter(endpoint=config.otel_endpoint or None)
    else:
        exporter = ConsoleSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
