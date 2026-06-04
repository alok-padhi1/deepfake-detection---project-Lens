"""
LENS ML — TorchServe .mar Packaging Script
Packages all models with custom handlers into .mar archives.

Usage:
  python torchserve_package.py \
    --checkpoints-dir checkpoints \
    --handlers-dir    handlers \
    --out-dir         exports/mar \
    --version         1.0
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

MAR_SPECS = [
    {
        "model_name":    "lens_efficientnet",
        "handler":       "handlers/image_handler.py",
        "checkpoint":    "checkpoints/efficientnet/efficientnet_best.pth",
        "extra_files":   ["models/efficientnet_forensic.py",
                          "datasets/augmentation.py"],
        "requirements":  "requirements.txt",
    },
    {
        "model_name":    "lens_timesformer",
        "handler":       "handlers/video_handler.py",
        "checkpoint":    "checkpoints/timesformer/timesformer_best.pth",
        "extra_files":   ["models/timesformer_video.py",
                          "datasets/augmentation.py"],
        "requirements":  "requirements.txt",
    },
    {
        "model_name":    "lens_audio_mel",
        "handler":       "handlers/audio_handler.py",
        "checkpoint":    "checkpoints/audio_dual/audio_dual_best.pth",
        "extra_files":   ["models/audio_dual.py",
                          "datasets/augmentation.py"],
        "requirements":  "requirements.txt",
    },
    {
        "model_name":    "lens_rawnet3",
        "handler":       "handlers/audio_handler.py",
        "checkpoint":    "checkpoints/audio_dual/audio_dual_best.pth",
        "extra_files":   ["models/audio_dual.py"],
        "requirements":  "requirements.txt",
    },
    {
        "model_name":    "lens_syncnet",
        "handler":       "handlers/video_handler.py",
        "checkpoint":    "checkpoints/syncnet/syncnet_best.pth",
        "extra_files":   ["models/syncnet_crossmodal.py"],
        "requirements":  "requirements.txt",
    },
    {
        "model_name":    "lens_fusion",
        "handler":       "handlers/fusion_handler.py",
        "checkpoint":    "checkpoints/fusion/fusion_best.pth",
        "extra_files":   ["models/bayesian_fusion.py",
                          "checkpoints/fusion/isotonic_calibrators.pkl"],
        "requirements":  "requirements.txt",
    },
]


def package_model(spec: dict, out_dir: Path, version: str,
                  base_dir: Path) -> bool:
    model_name = spec["model_name"]
    out_path   = out_dir / f"{model_name}.mar"

    # Resolve paths relative to base_dir
    handler    = base_dir / spec["handler"]
    ckpt       = base_dir / spec["checkpoint"]
    extra_files= [str(base_dir / f) for f in spec.get("extra_files", [])]
    req_file   = base_dir / spec.get("requirements", "requirements.txt")

    if not handler.exists():
        log.warning("Handler not found: %s — skipping %s", handler, model_name)
        return False
    if not ckpt.exists():
        log.warning("Checkpoint not found: %s — packaging without weights",ckpt)
        ckpt_arg = []
    else:
        ckpt_arg = ["--serialized-file", str(ckpt)]

    # Filter missing extra files
    existing_extras = [f for f in extra_files if Path(f).exists()]
    if len(existing_extras) < len(extra_files):
        missing = set(extra_files) - set(existing_extras)
        log.warning("Missing extra files for %s: %s", model_name, missing)

    cmd = [
        sys.executable, "-m", "torch_model_archiver",
        "--model-name",      model_name,
        "--version",         version,
        "--handler",         str(handler),
        *ckpt_arg,
        "--export-path",     str(out_dir),
        "--force",
    ]
    if existing_extras:
        cmd.extend(["--extra-files", ",".join(existing_extras)])
    if req_file.exists():
        cmd.extend(["--requirements-file", str(req_file)])

    log.info("Packaging %s ...", model_name)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("torch-model-archiver failed for %s:\n%s",
                  model_name, result.stderr)
        return False

    size_mb = out_path.stat().st_size / 1e6 if out_path.exists() else 0
    log.info("  → %s  %.1f MB", out_path.name, size_mb)
    return True


def generate_config_yaml(mar_names: list, out_dir: Path) -> None:
    """Generate TorchServe config.properties and model-store setup."""
    config_path = out_dir / "config.properties"
    config = f"""
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082
grpc_inference_port=7070
grpc_management_port=7071
model_store={out_dir}
load_models=all
number_of_netty_threads=4
netty_client_threads=0
job_queue_size=1000
model_server_home=/tmp/torchserve
default_workers_per_model=1
default_response_timeout=600
unregister_model_timeout=120
max_request_size=67108864
max_response_size=67108864
"""
    config_path.write_text(config.strip())
    log.info("TorchServe config written: %s", config_path)

    # docker-compose snippet
    compose_snippet = out_dir / "torchserve-compose.yml"
    compose_snippet.write_text(f"""
version: '3.8'
services:
  torchserve:
    image: pytorch/torchserve:latest-gpu
    command: >
      torchserve --start
      --ncs
      --model-store /mnt/models
      --ts-config /mnt/config/config.properties
    volumes:
      - {out_dir}:/mnt/models
      - {out_dir}:/mnt/config
    ports:
      - "8080:8080"
      - "8081:8081"
      - "8082:8082"
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
""".strip())
    log.info("Docker Compose snippet written: %s", compose_snippet)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="LENS TorchServe .mar Packager")
    p.add_argument("--base-dir",   type=Path, default=Path("."),
                   help="Root directory containing checkpoints/, models/, etc.")
    p.add_argument("--out-dir",    required=True, type=Path)
    p.add_argument("--version",    default="1.0")
    p.add_argument("--models",     nargs="*", default=None,
                   help="Subset of model names to package (default: all)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = MAR_SPECS
    if args.models:
        specs = [s for s in specs if s["model_name"] in args.models]

    packed = []
    for spec in specs:
        ok = package_model(spec, out_dir, args.version, Path(args.base_dir))
        if ok:
            packed.append(spec["model_name"])

    generate_config_yaml(packed, out_dir)
    log.info("\n=== Packaging complete: %d/%d models ===", len(packed), len(specs))
    for name in packed:
        mar = out_dir / f"{name}.mar"
        if mar.exists():
            log.info("  ✓ %s  %.1f MB", mar.name, mar.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
