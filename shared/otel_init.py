# otel_init.py  — import this at the top of every LENS service
# Usage:  from otel_init import init_otel, get_tracer
import os
import logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.b3 import B3MultiFormat
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.propagators.textmap import TraceContextTextMapPropagator

log = logging.getLogger("otel_init")

_tracer: trace.Tracer | None = None


def init_otel(service_name: str) -> trace.Tracer:
    """
    Initialise OpenTelemetry SDK for a LENS microservice.
    - Exports spans via OTLP gRPC to the collector sidecar / service.
    - Auto-instruments: requests (→ TorchServe), psycopg2 (PG), redis.
    - Sets W3C TraceContext + B3 composite propagator so Kafka headers carry context.
    """
    global _tracer
    if _tracer:
        return _tracer

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT",
                         "http://lens-otel-opentelemetry-collector.monitoring.svc.cluster.local:4317")

    resource = Resource.create({SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(endpoint=endpoint, insecure=True)
        )
    )
    trace.set_tracer_provider(provider)

    # W3C + B3 composite propagator — needed for Kafka header injection
    set_global_textmap(CompositePropagator([
        TraceContextTextMapPropagator(),
        B3MultiFormat(),
    ]))

    # Auto-instrument common libraries
    RequestsInstrumentor().instrument()   # TorchServe + MinIO boto3 HTTP
    Psycopg2Instrumentor().instrument()  # PostgreSQL queries
    RedisInstrumentor().instrument()      # Redis get/set/setex

    _tracer = trace.get_tracer(service_name)
    log.info(f"OTel SDK initialised — service={service_name} endpoint={endpoint}")
    return _tracer


def get_tracer() -> trace.Tracer:
    if _tracer is None:
        raise RuntimeError("call init_otel(service_name) before get_tracer()")
    return _tracer


def inject_kafka_headers(headers: dict) -> dict:
    """Inject current W3C trace context into a Kafka message header dict."""
    from opentelemetry.propagate import inject
    carrier: dict = {}
    inject(carrier)
    headers.update({k: v.encode() for k, v in carrier.items()})
    return headers


def extract_kafka_context(headers: list) -> object:
    """Extract W3C trace context from Kafka message headers (list of (key, bytes) tuples)."""
    from opentelemetry.propagate import extract
    carrier = {k: v.decode() for k, v in (headers or []) if isinstance(v, bytes)}
    return extract(carrier)
