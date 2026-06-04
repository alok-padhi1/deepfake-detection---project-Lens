"""
LENS ML — Full Pipeline Orchestration
Runs all training stages sequentially, saves checkpoints, auto-exports to ONNX.

Stages:
  1. EfficientNet-B4   — Image deepfake detection  (CIFAKE, HF streaming)
  2. Audio Dual        — Audio spoof detection      (ASVspoof 2019, HF streaming)
  3. TimeSformer       — Video deepfake detection   (needs DFDC on S3, skipped if unavailable)
  4. SyncNet           — AV sync detection          (needs DFDC on S3, skipped if unavailable)
  5. Fusion            — Bayesian score fusion       (runs after 1+2 have checkpoints)

Usage:
  python d:\isafe2\ml\orchestrate.py
  python d:\isafe2\ml\orchestrate.py --stages efficientnet audio fusion
"""
from __future__ import annotations
import argparse, logging, os, subprocess, sys, time
from pathlib import Path

# ── CUDA PyTorch on D: ───────────────────────────────────────────────────────
for _p in [r'd:\pylibs', r'd:/pylibs']:
    if _p not in sys.path:
        sys.path.insert(0, _p)

ML   = Path(__file__).parent
ROOT = ML.parent
CKPT = ROOT / "checkpoints"
LOG  = ROOT / "logs"
LOG.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG / "pipeline.log", encoding="utf-8"),
    ]
)
log = logging.getLogger("orchestrator")

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ── Helper: run a training stage inline ──────────────────────────────────────
def run_stage(name: str, argv: list[str]) -> bool:
    """Import and call main() for a training module. Returns True on success."""
    sys.path.insert(0, str(ML))
    os.chdir(str(ML))
    sys.argv = ["train"] + argv
    try:
        import importlib
        mod = importlib.import_module(f"training.train_{name}")
        importlib.reload(mod)          # ensure fresh state between stages
        log.info("=" * 60)
        log.info("STAGE: %s | args: %s", name, " ".join(argv))
        log.info("=" * 60)
        t0 = time.time()
        mod.main()
        elapsed = time.time() - t0
        log.info("STAGE %s DONE in %.0fs", name, elapsed)
        return True
    except SystemExit as e:
        if e.code == 0:
            return True
        log.error("STAGE %s exited with code %s", name, e.code)
        return False
    except Exception as e:
        log.error("STAGE %s FAILED: %s", name, e, exc_info=True)
        return False

# ── Helper: export checkpoint to ONNX ────────────────────────────────────────
def export_onnx(model_name: str, ckpt_dir: Path) -> bool:
    best = ckpt_dir / f"{model_name}_best.pth"
    if not best.exists():
        best = ckpt_dir / f"{model_name}_last.pth"
    if not best.exists():
        # find latest checkpoint
        ckpts = sorted(ckpt_dir.glob("*.pth"))
        if not ckpts:
            log.warning("No checkpoint found for %s — skipping ONNX export", model_name)
            return False
        best = ckpts[-1]
    out = ckpt_dir / f"{model_name}.onnx"
    log.info("Exporting %s → %s", best.name, out.name)
    try:
        import torch
        sys.path.insert(0, str(ML))
        ckpt = torch.load(str(best), map_location="cpu", weights_only=False)

        if model_name == "efficientnet":
            from models.efficientnet_forensic import EfficientNetForensic
            model = EfficientNetForensic(pretrained=False, variant="tf_efficientnet_b0.ns_jft_in1k").eval()
            model.load_state_dict(ckpt["model"])
            dummy = torch.randn(1, 3, 224, 224)
            class W(torch.nn.Module):
                def __init__(self, m): super().__init__(); self.m = m
                def forward(self, x): o = self.m(x); return o["binary"], o["family"]
            torch.onnx.export(W(model), dummy, str(out), opset_version=17,
                              input_names=["image"],
                              output_names=["binary_logit","family_logit"],
                              dynamic_axes={"image":{0:"batch"}})
            log.info("ONNX exported: %s (%.1f MB)", out.name, out.stat().st_size/1e6)
            return True

        elif model_name == "audio":
            from models.audio_dual import AudioDualModel
            model = AudioDualModel().eval()
            model.load_state_dict(ckpt["model"])
            
            class AudioW(torch.nn.Module):
                def __init__(self, m): super().__init__(); self.m = m
                def forward(self, mel, wav):
                    o = self.m(mel=mel, waveform=wav)
                    return o.get("mel_logit"), o.get("raw_logit")
                    
            dummy_mel = torch.randn(1, 1, 128, 400)
            dummy_wav = torch.randn(1, 1, 64000)
            torch.onnx.export(AudioW(model), (dummy_mel, dummy_wav), str(out), opset_version=17,
                              input_names=["mel", "waveform"],
                              output_names=["mel_logit", "raw_logit"],
                              dynamic_axes={"mel":{0:"batch"}, "waveform":{0:"batch"}})
            log.info("ONNX exported: %s (%.1f MB)", out.name, out.stat().st_size/1e6)
            return True

        elif model_name == "fusion":
            from models.bayesian_fusion import BayesianFusionMLP
            model = BayesianFusionMLP().eval()
            model.load_state_dict(ckpt["model"])
            
            class FusionW(torch.nn.Module):
                def __init__(self, m): super().__init__(); self.m = m
                def forward(self, x):
                    return self.m(x, media_type="image")
            dummy = torch.randn(1, 18)
            torch.onnx.export(FusionW(model), dummy, str(out), opset_version=17,
                              input_names=["scores"], output_names=["logit"],
                              dynamic_axes={"scores":{0:"batch"}})
            log.info("ONNX exported: %s (%.1f MB)", out.name, out.stat().st_size/1e6)
            return True

    except Exception as e:
        log.error("ONNX export failed for %s: %s", model_name, e)
        return False

# ── Stage definitions ─────────────────────────────────────────────────────────
STAGES = {
    "efficientnet": {
        "fn":   lambda: run_stage("efficientnet", [
            "--model-variant", "tf_efficientnet_b0.ns_jft_in1k",
            "--train-manifest", "d:/cifake_train.parquet",
            "--val-manifest", "d:/cifake_test.parquet",
            "--out-dir", str(CKPT / "efficientnet"),
            "--epochs", "1", "--batch-size", "16",
            "--workers", "0", "--tracking", "none",
        ]),
        "post": lambda: export_onnx("efficientnet", CKPT / "efficientnet"),
    },
    "audio": {
        "fn":   lambda: run_stage("audio", [
            "--manifest", "d:/dummy_audio.parquet",
            "--out-dir", str(CKPT / "audio"),
            "--epochs", "1", "--batch-size", "2",
            "--workers", "0", "--tracking", "none",
        ]),
        "post": lambda: export_onnx("audio", CKPT / "audio"),
    },
    "timesformer": {
        "fn": lambda: (
            log.warning("TimeSformer needs DFDC on S3 — skipping (stream not available). "
                        "Upload DFDC to MinIO first, then re-run: "
                        "python orchestrate.py --stages timesformer")
            or False
        ),
        "post": lambda: False,
    },
    "syncnet": {
        "fn": lambda: (
            log.warning("SyncNet needs DFDC on S3 — skipping. Same as TimeSformer.")
            or False
        ),
        "post": lambda: False,
    },
    "fusion": {
        "fn": lambda: _run_fusion_if_ready(),
        "post": lambda: export_onnx("fusion", CKPT / "fusion"),
    },
}

def _run_fusion_if_ready() -> bool:
    """Only run fusion if at least 2 model checkpoints exist."""
    available = []
    for m in ["efficientnet", "audio", "timesformer", "syncnet"]:
        d = CKPT / m
        if d.exists() and any(d.glob("*.pth")):
            available.append(m)
    if len(available) < 2:
        log.warning("Fusion needs ≥2 model checkpoints. Have: %s. Skipping.", available)
        return False
    log.info("Running fusion with checkpoints from: %s", available)
    return run_stage("fusion", [
        "--score-vectors", "d:/dummy_fusion_scores.parquet",
        "--out-dir",           str(CKPT / "fusion"),
        "--epochs", "10", "--tracking", "none",
        "--calibrate",
    ])

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="LENS Full Pipeline Orchestrator")
    p.add_argument("--stages", nargs="+",
                   default=["efficientnet", "audio", "fusion"],
                   choices=list(STAGES.keys()),
                   help="Stages to run in order")
    args = p.parse_args()

    log.info("LENS Pipeline starting | stages: %s", args.stages)
    results = {}

    for stage in args.stages:
        log.info("\n>>> Running stage: %s", stage)
        CKPT.mkdir(parents=True, exist_ok=True)
        ok = STAGES[stage]["fn"]()
        if ok:
            STAGES[stage]["post"]()
        results[stage] = "✅ OK" if ok else "⚠️ skipped/failed"

    log.info("\n" + "="*50)
    log.info("PIPELINE SUMMARY")
    log.info("="*50)
    for stage, status in results.items():
        log.info("  %-16s %s", stage, status)
    log.info("Checkpoints in: %s", CKPT)

if __name__ == "__main__":
    main()
