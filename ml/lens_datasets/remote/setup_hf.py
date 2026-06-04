"""
LENS ML — HuggingFace Token Setup & Dataset Access Test
Run this ONCE to configure your HF token and verify dataset access.

Usage:
  python ml/datasets/remote/setup_hf.py --token hf_xxxxxxxxxxxx

This saves the token to ~/.huggingface/token (standard HF location).
Never hardcode the token in source files.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def save_token(token: str) -> None:
    """Save token to ~/.huggingface/token (standard HuggingFace location)."""
    token_dir  = Path.home() / ".huggingface"
    token_file = token_dir / "token"
    token_dir.mkdir(exist_ok=True)
    token_file.write_text(token.strip())
    token_file.chmod(0o600)   # owner read-only
    print(f"✅ Token saved to {token_file}")


def test_login(token: str) -> bool:
    try:
        from huggingface_hub import login, whoami
        login(token=token, add_to_git_credential=False)
        info = whoami()
        print(f"✅ Logged in as: {info['name']}")
        return True
    except Exception as e:
        print(f"❌ Login failed: {e}")
        return False


def test_genimage_access(token: str) -> bool:
    """Check streaming access with CIFAKE (no terms required) then GenImage."""
    print("\nTesting CIFAKE dataset (60k real + 60k AI-generated)...")
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "dragonintelligence/CIFAKE-image-dataset",
            split="train",
            streaming=True,
            token=token,
        )
        sample = next(iter(ds))
        keys = list(sample.keys())
        print(f"\u2705 CIFAKE accessible \u2014 keys: {keys}")
        label = sample.get('label', sample.get('labels', 'N/A'))
        print(f"   Label field: {label}")
        return True
    except Exception as e:
        print(f"\u274c CIFAKE failed: {e}")

    print("\nTrying deepfake face dataset...")
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "itsLeen/deepfake_vs_real_image_detection",
            split="train",
            streaming=True,
            token=token,
        )
        sample = next(iter(ds))
        print(f"\u2705 Deepfake face dataset accessible \u2014 keys: {list(sample.keys())}")
        return True
    except Exception as e:
        print(f"\u274c Deepfake face dataset failed: {e}")
        return False


def test_asvspoof_access(token: str) -> bool:
    """ASVspoof is not on HF Hub — always returns False with guidance."""
    print("\nASVspoof: not available on HuggingFace Hub")
    print("   \u2192 Use local download from https://www.asvspoof.org")
    return False


def print_next_steps(genimage_ok: bool, asvspoof_ok: bool) -> None:
    print("\n" + "=" * 60)
    print("LENS ML - Remote Dataset Status")
    print("=" * 60)

    img_status = "Ready to stream" if genimage_ok else "Access issue"
    print("\nDataset Availability:")
    print(f"  CIFAKE/Deepfake faces (HF stream):   {img_status}")
    print("  ASVspoof (audio):                     Download from asvspoof.org (25GB)")
    print("  DFDC (128K video clips):              Needs S3/MinIO upload")
    print("  FaceForensics++ (1M frames):          Needs S3/MinIO upload")

    print("\nTraining you can do RIGHT NOW:")
    if genimage_ok:
        print("  [OK] Step 1: Train EfficientNet-B7 on CIFAKE (streams from HF, 0 GB local)")
        print("  [OK] Step 2: Upgrade to full GenImage once you accept HF terms")
        print("  [!]  Step 3: Audio + Video models need dataset downloads")
        print("\n  Run this to start training NOW:")
        print("    python d:\\isafe2\\ml\\training\\train_efficientnet.py")
        print("      --use-hf-streaming --hf-dataset cifake")
        print("      --out-dir checkpoints/efficientnet --tracking none")
    else:
        print("  Fix dataset access above, then re-run this script")


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(description="LENS HuggingFace Setup")
    p.add_argument("--token", required=True,
                   help="HuggingFace token (hf_xxxxxxxxxxxx)")
    p.add_argument("--skip-tests", action="store_true")
    args = p.parse_args()

    token = args.token.strip()
    if not token.startswith("hf_"):
        print("❌ Token should start with 'hf_' — check you copied it correctly")
        sys.exit(1)

    save_token(token)
    if not test_login(token):
        sys.exit(1)

    if args.skip_tests:
        print("\n✅ Token saved. Run without --skip-tests to test dataset access.")
        return

    genimage_ok  = test_genimage_access(token)
    asvspoof_ok  = test_asvspoof_access(token)
    print_next_steps(genimage_ok, asvspoof_ok)


if __name__ == "__main__":
    main()
