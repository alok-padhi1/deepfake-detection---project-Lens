"""
LENS ML — TensorRT INT8 Quantization Script
Target: <0.3% AUC drop from FP32 baseline.
Uses TensorRT calibration with a representative calibration dataset.

Requirements:
  - tensorrt >= 10.1
  - torch2trt >= 0.4
  - pycuda
  - NVIDIA GPU (A100 recommended)

Usage:
  python tensorrt_quantize.py \
    --onnx-dir   exports/onnx \
    --out-dir    exports/tensorrt \
    --calib-dir  /data/calibration_images \
    --batch-size 64 --max-workspace-gb 8
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Iterator, List, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    log.warning("TensorRT or pycuda not installed — quantization unavailable")
    TRT_AVAILABLE = False

TRT_LOGGER = trt.Logger(trt.Logger.WARNING) if TRT_AVAILABLE else None


# ── INT8 Calibrator ───────────────────────────────────────────────────────────
class ImageCalibrator(trt.IInt8EntropyCalibrator2 if TRT_AVAILABLE else object):
    """
    INT8 calibrator using representative image dataset.
    Feeds normalized (B, 3, H, W) float32 batches to TensorRT calibration.
    """
    def __init__(
        self,
        calib_images: List[Path],
        batch_size: int = 64,
        image_size: int = 224,
        cache_file: str = "calibration.cache",
    ) -> None:
        if TRT_AVAILABLE:
            super().__init__()
        self.image_paths = calib_images
        self.batch_size  = batch_size
        self.image_size  = image_size
        self.cache_file  = cache_file
        self._current    = 0

        # Allocate CUDA buffer
        self._buf_size = batch_size * 3 * image_size * image_size * 4   # float32
        if TRT_AVAILABLE:
            self._device_input = cuda.mem_alloc(self._buf_size)

    def _load_batch(self) -> Optional[np.ndarray]:
        paths = self.image_paths[self._current:self._current + self.batch_size]
        if not paths:
            return None
        imgs = []
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.image_size, self.image_size),
                             interpolation=cv2.INTER_LANCZOS4)
            imgs.append(img)
        # Pad to batch_size
        while len(imgs) < self.batch_size:
            imgs.append(imgs[-1])
        arr = np.stack(imgs, axis=0).astype(np.float32) / 127.5 - 1.0
        return arr.transpose(0, 3, 1, 2)   # (B, C, H, W)

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names):
        batch = self._load_batch()
        if batch is None:
            return None
        self._current += self.batch_size
        cuda.memcpy_htod(self._device_input, batch.tobytes())
        return [int(self._device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        log.info("Calibration cache written: %s", self.cache_file)


# ── ONNX → TensorRT builder ───────────────────────────────────────────────────
def build_engine(
    onnx_path: Path,
    out_path: Path,
    precision: str = "int8",   # "fp32" | "fp16" | "int8"
    max_workspace_gb: int = 8,
    calibrator=None,
    min_batch: int = 1,
    opt_batch: int = 16,
    max_batch: int = 64,
) -> None:
    if not TRT_AVAILABLE:
        log.error("TensorRT not available — skipping %s", onnx_path.name)
        return

    log.info("Building TRT engine: %s [%s]", onnx_path.name, precision)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser  = trt.OnnxParser(network, TRT_LOGGER)
    config  = builder.create_builder_config()

    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        max_workspace_gb * (1 << 30)
    )

    if precision == "fp16" and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8" and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        if calibrator:
            config.int8_calibrator = calibrator
    # FP32 = default

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                log.error("ONNX parse error: %s", parser.get_error(i))
            raise RuntimeError(f"ONNX parse failed: {onnx_path}")

    # Dynamic shape profile
    profile = builder.create_optimization_profile()
    inp     = network.get_input(0)
    shape   = inp.shape
    # Replace batch dimension (-1 or 1) with dynamic range
    min_shape = (min_batch, *shape[1:])
    opt_shape = (opt_batch, *shape[1:])
    max_shape = (max_batch, *shape[1:])
    profile.set_shape(inp.name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"Engine build failed for {onnx_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(serialized)
    size_mb = out_path.stat().st_size / 1e6
    log.info("Engine saved: %s  %.1f MB", out_path.name, size_mb)


# ── AUC validation after quantization ─────────────────────────────────────────
def validate_engine_auc(
    engine_path: Path,
    onnx_path: Path,
    val_images: List[Path],
    val_labels: List[int],
    image_size: int = 224,
) -> dict:
    """
    Compare FP32 ONNX vs INT8 TRT engine AUC on a validation set.
    Returns {"onnx_auc": ..., "trt_auc": ..., "auc_drop": ...}
    """
    import onnxruntime as ort
    from sklearn.metrics import roc_auc_score

    def _run_onnx(imgs: np.ndarray) -> np.ndarray:
        sess = ort.InferenceSession(str(onnx_path),
                                    providers=["CUDAExecutionProvider"])
        inp_name = sess.get_inputs()[0].name
        return sess.run(None, {inp_name: imgs})[0]

    def _run_trt(imgs: np.ndarray) -> np.ndarray:
        if not TRT_AVAILABLE:
            return np.zeros(len(imgs))
        runtime  = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        import pycuda.driver as cuda
        stream  = cuda.Stream()
        d_in    = cuda.mem_alloc(imgs.nbytes)
        d_out   = cuda.mem_alloc(len(imgs) * 4)
        cuda.memcpy_htod_async(d_in, imgs.tobytes(), stream)
        context.execute_async_v2([int(d_in), int(d_out)], stream.handle)
        stream.synchronize()
        out = np.empty(len(imgs), dtype=np.float32)
        cuda.memcpy_dtoh(out, d_out)
        return out

    # Preprocess
    imgs = []
    for p in val_images:
        img = cv2.imread(str(p))
        if img is None:
            img = np.zeros((image_size, image_size, 3), np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (image_size, image_size), cv2.INTER_LANCZOS4)
        imgs.append(img)
    arr = np.stack(imgs).astype(np.float32) / 127.5 - 1.0
    arr = arr.transpose(0, 3, 1, 2)   # (N, 3, H, W)
    labels = np.array(val_labels)

    onnx_logits = _run_onnx(arr).flatten()
    trt_logits  = _run_trt(arr).flatten()

    from scipy.special import expit as sigmoid
    onnx_auc = roc_auc_score(labels, sigmoid(onnx_logits))
    trt_auc  = roc_auc_score(labels, sigmoid(trt_logits))
    drop     = onnx_auc - trt_auc

    result = {"onnx_auc": onnx_auc, "trt_auc": trt_auc, "auc_drop": drop}
    log.info("AUC validation — ONNX:%.4f  TRT:%.4f  DROP:%.4f",
             onnx_auc, trt_auc, drop)
    if drop > 0.003:
        log.warning("AUC drop %.4f exceeds target of 0.003 (0.3%%)", drop)
    else:
        log.info("AUC drop within target (<0.3%%)")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="LENS TensorRT INT8 Quantization")
    p.add_argument("--onnx-dir",         required=True, type=Path)
    p.add_argument("--out-dir",          required=True, type=Path)
    p.add_argument("--calib-dir",        type=Path, default=None,
                   help="Directory of calibration images for INT8")
    p.add_argument("--precision",        choices=["fp32","fp16","int8"],
                   default="int8")
    p.add_argument("--batch-size",       type=int, default=64)
    p.add_argument("--max-workspace-gb", type=int, default=8)
    p.add_argument("--calib-count",      type=int, default=5000,
                   help="Number of calibration images")
    args = p.parse_args()

    onnx_dir = Path(args.onnx_dir)
    out_dir  = Path(args.out_dir)
    onnx_files = sorted(onnx_dir.glob("*.onnx"))
    if not onnx_files:
        log.error("No ONNX files found in %s", onnx_dir)
        return

    # Calibration images
    calib_images: List[Path] = []
    if args.calib_dir and Path(args.calib_dir).exists():
        calib_images = sorted(Path(args.calib_dir).rglob("*.jpg"))[:args.calib_count]
    log.info("Calibration images: %d", len(calib_images))

    for onnx_path in onnx_files:
        out_path = out_dir / onnx_path.with_suffix(".trt").name
        cache_file = str(out_dir / f"{onnx_path.stem}_int8.cache")

        calibrator = None
        if args.precision == "int8" and calib_images:
            calibrator = ImageCalibrator(
                calib_images, args.batch_size,
                cache_file=cache_file
            )

        try:
            build_engine(
                onnx_path, out_path,
                precision=args.precision,
                max_workspace_gb=args.max_workspace_gb,
                calibrator=calibrator,
            )
        except Exception as e:
            log.error("Failed to build engine for %s: %s", onnx_path.name, e)

    log.info("\n=== TensorRT build complete ===")
    for f in sorted(out_dir.glob("*.trt")):
        log.info("  %s  %.1f MB", f.name, f.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
