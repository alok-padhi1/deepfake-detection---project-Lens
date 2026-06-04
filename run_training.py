"""
LENS ML — Universal Training Launcher
Imports and calls main() directly from each training script.

Usage:
  python d:\isafe2\run_training.py efficientnet --use-hf-streaming --hf-dataset cifake --out-dir d:\isafe2\checkpoints\efficientnet --epochs 20 --batch-size 16 --tracking none
"""
import sys, os
from pathlib import Path

# ── CUDA PyTorch on D: drive ──────────────────────────────────────────────────
for _p in [r'd:\pylibs', r'd:/pylibs']:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── ML package root ───────────────────────────────────────────────────────────
ML_DIR = Path(__file__).parent / "ml"
sys.path.insert(0, str(ML_DIR))
os.chdir(str(ML_DIR))

MODULES = {
    "efficientnet": "training.train_efficientnet",
    "timesformer":  "training.train_timesformer",
    "audio":        "training.train_audio",
    "syncnet":      "training.train_syncnet",
    "fusion":       "training.train_fusion",
}

if len(sys.argv) < 2 or sys.argv[1] not in MODULES:
    print("Usage: python run_training.py <model> [args...]")
    print("Models:", ", ".join(MODULES.keys()))
    sys.exit(1)

model = sys.argv[1]
# Remove the model name so argparse in the training script sees clean args
sys.argv = [sys.argv[0]] + sys.argv[2:]

# Import and call main() directly
import importlib
mod = importlib.import_module(MODULES[model])
mod.main()
