from harness.telemetry.tracer import get_tracer, setup_telemetry


def test_get_tracer_noop_by_default_does_not_raise():
    tracer = get_tracer()
    with tracer.start_as_current_span("x") as span:
        span.set_attribute("k", "v")   # no-op tracer 也不报错


def test_setup_telemetry_disabled_is_noop():
    class Cfg:
        otel_enabled = False
        otel_exporter = "console"
        otel_endpoint = ""

    # 不抛异常即可
    setup_telemetry(Cfg())
