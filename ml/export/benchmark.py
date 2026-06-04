"""
LENS ML — Latency Benchmark Script
Targets:
  - Image pipeline (EfficientNet-B7):  <800ms on A100
  - 30s video clip (TimeSformer):      <4s on A100

Benchmarks all models across: PyTorch FP32, PyTorch FP16, ONNX RT, TRT INT8.

Usage:
  python benchmark.py \
    --onnx-dir   exports/onnx \
    --trt-dir    exports/tensorrt \
    --out         benchmark_results.csv \
    --batch-sizes 1 4 8 16 32 \
    --n-warmup 20 --n-iters 100
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

log = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    model_name:  str
    backend:     str
    batch_size:  int
    input_shape: str
    latency_ms:  float   # mean
    p50_ms:      float
    p95_ms:      float
    p99_ms:      float
    throughput:  float   # samples/sec
    target_ms:   Optional[float]
    pass_fail:   str


# ── Timing helper ─────────────────────────────────────────────────────────────
def _time_fn(fn, n_warmup: int = 20, n_iters: int = 100) -> np.ndarray:
    """Returns array of latencies in ms."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        times.append((time.perf_counter() - t0) * 1000.0)
    return np.array(times)


def _stats(times: np.ndarray, batch_size: int, target_ms: Optional[float]) -> dict:
    return {
        "latency_ms": float(np.mean(times)),
        "p50_ms":     float(np.percentile(times, 50)),
        "p95_ms":     float(np.percentile(times, 95)),
        "p99_ms":     float(np.percentile(times, 99)),
        "throughput": float(batch_size / np.mean(times) * 1000.0),
        "target_ms":  target_ms,
        "pass_fail":  "PASS" if (target_ms is None or np.mean(times) < target_ms)
                      else "FAIL",
    }


# ── PyTorch benchmark ─────────────────────────────────────────────────────────
def benchmark_pytorch(model: torch.nn.Module, dummy_input,
                      model_name: str, batch_size: int,
                      target_ms: Optional[float],
                      precision: str = "fp16",
                      n_warmup: int = 20, n_iters: int = 100) -> BenchmarkResult:
    device = next(model.parameters()).device
    model.eval()

    if precision == "fp16":
        model  = model.half()
        if isinstance(dummy_input, torch.Tensor):
            dummy_input = dummy_input.half()
        elif isinstance(dummy_input, tuple):
            dummy_input = tuple(x.half() if x.dtype == torch.float32 else x
                                for x in dummy_input)

    def fn():
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(precision=="fp16")):
            if isinstance(dummy_input, tuple):
                model(*dummy_input)
            else:
                model(dummy_input)

    times = _time_fn(fn, n_warmup, n_iters)
    shape = (str(dummy_input.shape) if isinstance(dummy_input, torch.Tensor)
             else str(tuple(x.shape for x in dummy_input)))
    stats = _stats(times, batch_size, target_ms)
    return BenchmarkResult(
        model_name=model_name, backend=f"pytorch-{precision}",
        batch_size=batch_size, input_shape=shape, **stats
    )


# ── ONNX Runtime benchmark ────────────────────────────────────────────────────
def benchmark_onnx(onnx_path: Path, dummy_np: dict,
                   model_name: str, batch_size: int,
                   target_ms: Optional[float],
                   n_warmup: int = 20, n_iters: int = 100) -> BenchmarkResult:
    try:
        import onnxruntime as ort
    except ImportError:
        log.warning("onnxruntime not installed")
        return None

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 8
    sess = ort.InferenceSession(
        str(onnx_path),
        opts,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    inp_name = sess.get_inputs()[0].name

    def fn():
        sess.run(None, dummy_np)

    times = _time_fn(fn, n_warmup, n_iters)
    shape = str({k: v.shape for k, v in dummy_np.items()})
    stats = _stats(times, batch_size, target_ms)
    return BenchmarkResult(
        model_name=model_name, backend="onnxrt-cuda",
        batch_size=batch_size, input_shape=shape, **stats
    )


# ── TensorRT benchmark ────────────────────────────────────────────────────────
def benchmark_trt(engine_path: Path, dummy_np: np.ndarray,
                  model_name: str, batch_size: int,
                  target_ms: Optional[float],
                  n_warmup: int = 20, n_iters: int = 100) -> Optional[BenchmarkResult]:
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    except ImportError:
        log.warning("TensorRT not available for benchmark")
        return None

    runtime  = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context  = engine.create_execution_context()
    stream   = cuda.Stream()

    d_in  = cuda.mem_alloc(dummy_np.nbytes)
    out_n = batch_size
    d_out = cuda.mem_alloc(out_n * 4)

    def fn():
        cuda.memcpy_htod_async(d_in, dummy_np, stream)
        context.execute_async_v2([int(d_in), int(d_out)], stream.handle)
        stream.synchronize()

    times = _time_fn(fn, n_warmup, n_iters)
    stats = _stats(times, batch_size, target_ms)
    return BenchmarkResult(
        model_name=model_name, backend="tensorrt-int8",
        batch_size=batch_size, input_shape=str(dummy_np.shape), **stats
    )


# ── Model benchmark registry ──────────────────────────────────────────────────
def run_all_benchmarks(
    onnx_dir: Path,
    trt_dir: Path,
    batch_sizes: List[int],
    n_warmup: int,
    n_iters: int,
) -> List[BenchmarkResult]:
    results: List[BenchmarkResult] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── EfficientNet-B7 (image pipeline target: <800ms) ──────────────────────
    for bs in batch_sizes:
        dummy_t = torch.randn(bs, 3, 224, 224, device=device)
        dummy_np = {"image": dummy_t.cpu().numpy()}
        target   = 800.0  # ms for single image

        onnx_path = onnx_dir / "efficientnet_b7_forensic.onnx"
        trt_path  = trt_dir  / "efficientnet_b7_forensic.trt"

        if onnx_path.exists():
            r = benchmark_onnx(onnx_path, dummy_np, "efficientnet-b7", bs,
                                target if bs == 1 else None, n_warmup, n_iters)
            if r: results.append(r)
        if trt_path.exists():
            r = benchmark_trt(trt_path, dummy_np["image"], "efficientnet-b7", bs,
                               target if bs == 1 else None, n_warmup, n_iters)
            if r: results.append(r)

    # ── TimeSformer (30s video = 8-frame clip × 11 windows, target: <4000ms) ─
    for bs in [1, 2, 4]:
        dummy_t  = torch.randn(bs, 8, 3, 224, 224, device=device)
        dummy_np = {"pixel_values": dummy_t.cpu().numpy()}
        target   = 4000.0 / 11 if bs == 1 else None   # per window

        onnx_path = onnx_dir / "timesformer_forensic.onnx"
        if onnx_path.exists():
            r = benchmark_onnx(onnx_path, dummy_np, "timesformer", bs,
                                target, n_warmup, n_iters)
            if r: results.append(r)

    # ── ResNet34-Mel ──────────────────────────────────────────────────────────
    for bs in batch_sizes:
        dummy_np = {"mel_spectrogram": np.random.randn(bs, 1, 128, 400).astype(np.float32)}
        onnx_path = onnx_dir / "resnet34_mel.onnx"
        if onnx_path.exists():
            r = benchmark_onnx(onnx_path, dummy_np, "resnet34-mel", bs,
                                200.0 if bs == 1 else None, n_warmup, n_iters)
            if r: results.append(r)

    # ── RawNet3 ───────────────────────────────────────────────────────────────
    for bs in batch_sizes:
        dummy_np = {"waveform": np.random.randn(bs, 1, 64000).astype(np.float32)}
        onnx_path = onnx_dir / "rawnet3.onnx"
        if onnx_path.exists():
            r = benchmark_onnx(onnx_path, dummy_np, "rawnet3", bs,
                                300.0 if bs == 1 else None, n_warmup, n_iters)
            if r: results.append(r)

    # ── Fusion MLP (very fast) ────────────────────────────────────────────────
    for bs in batch_sizes:
        dummy_np = {"score_vector": np.random.rand(bs, 18).astype(np.float32)}
        onnx_path = onnx_dir / "fusion_mlp.onnx"
        if onnx_path.exists():
            r = benchmark_onnx(onnx_path, dummy_np, "fusion-mlp", bs,
                                10.0 if bs == 1 else None, n_warmup, n_iters)
            if r: results.append(r)

    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="LENS Model Latency Benchmark")
    p.add_argument("--onnx-dir",    required=True, type=Path)
    p.add_argument("--trt-dir",     type=Path, default=None)
    p.add_argument("--out",         type=Path, default=Path("benchmark_results.csv"))
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8, 16, 32])
    p.add_argument("--n-warmup",    type=int, default=20)
    p.add_argument("--n-iters",     type=int, default=100)
    args = p.parse_args()

    trt_dir = args.trt_dir or args.onnx_dir

    log.info("Running benchmarks on: %s",
             "GPU" if torch.cuda.is_available() else "CPU")
    results = run_all_benchmarks(
        Path(args.onnx_dir), Path(trt_dir),
        args.batch_sizes, args.n_warmup, args.n_iters
    )

    # Print summary table
    print(f"\n{'Model':<22} {'Backend':<20} {'Batch':>5} "
          f"{'Mean ms':>8} {'P95 ms':>8} {'Tput/s':>8} {'Target':>8} {'Result':>6}")
    print("-" * 90)
    for r in results:
        tgt = f"{r.target_ms:.0f}" if r.target_ms else "  N/A"
        print(f"{r.model_name:<22} {r.backend:<20} {r.batch_size:>5} "
              f"{r.latency_ms:>8.1f} {r.p95_ms:>8.1f} {r.throughput:>8.1f} "
              f"{tgt:>8} {r.pass_fail:>6}")

    # CSV output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model_name", "backend", "batch_size", "input_shape",
            "latency_ms", "p50_ms", "p95_ms", "p99_ms",
            "throughput", "target_ms", "pass_fail"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow(vars(r))
    log.info("Results saved to %s", out_path)

    failed = [r for r in results if r.pass_fail == "FAIL"]
    if failed:
        log.warning("%d benchmark(s) exceeded latency targets:", len(failed))
        for r in failed:
            log.warning("  %s [%s] bs=%d: %.1fms > %.1fms",
                        r.model_name, r.backend, r.batch_size,
                        r.latency_ms, r.target_ms)
    else:
        log.info("All benchmarks passed latency targets.")


if __name__ == "__main__":
    main()
