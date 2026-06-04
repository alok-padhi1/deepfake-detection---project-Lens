"""
LENS ML — ONNX Export Script
Exports all 5 models to ONNX with correct input shapes, opset 17, dynamic batch.

Usage:
  python export_onnx.py \
    --efficientnet checkpoints/efficientnet/efficientnet_best.pth \
    --timesformer  checkpoints/timesformer/timesformer_best.pth \
    --audio-mel    checkpoints/audio_dual/audio_dual_best.pth \
    --syncnet      checkpoints/syncnet/syncnet_best.pth \
    --fusion       checkpoints/fusion/fusion_best.pth \
    --out-dir      exports/onnx
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import onnx
import onnxsim

from models.efficientnet_forensic import EfficientNetForensic
from models.timesformer_video      import TimeSformerForensic
from models.audio_dual             import AudioDualModel, ResNet34Mel, RawNet3
from models.syncnet_crossmodal     import SyncNetCrossModal
from models.bayesian_fusion        import BayesianFusionMLP, INPUT_DIM

log = logging.getLogger(__name__)
OPSET = 17


def _load_model(model: nn.Module, ckpt_path: Optional[Path]) -> nn.Module:
    if ckpt_path and Path(ckpt_path).exists():
        state = torch.load(ckpt_path, map_location="cpu")
        model_state = state.get("model", state)
        model.load_state_dict(model_state, strict=False)
        log.info("Loaded weights from %s", ckpt_path)
    model.eval()
    return model


def _export(model: nn.Module, dummy_input, out_path: Path,
            input_names: list, output_names: list,
            dynamic_axes: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Exporting → %s", out_path)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(out_path),
            opset_version=OPSET,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_params=True,
            do_constant_folding=True,
            verbose=False,
        )

    # Validate
    model_onnx = onnx.load(str(out_path))
    onnx.checker.check_model(model_onnx)
    log.info("ONNX validation passed: %s", out_path.name)

    # Simplify
    try:
        simplified, ok = onnxsim.simplify(model_onnx)
        if ok:
            onnx.save(simplified, str(out_path))
            log.info("ONNX simplified successfully")
    except Exception as e:
        log.warning("ONNX simplification failed (non-fatal): %s", e)


# ── Model 1: EfficientNet-B7 ──────────────────────────────────────────────────
def export_efficientnet(ckpt: Optional[Path], out_dir: Path) -> None:
    log.info("=== Exporting EfficientNet-B7 ===")

    class EfficientNetWrapper(nn.Module):
        """Wraps model to return only the binary logit for ONNX."""
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            out = self.model(x)
            return out["binary"], out["family"], out["patch"]

    model = EfficientNetWrapper(_load_model(EfficientNetForensic(pretrained=False), ckpt))
    dummy = torch.randn(1, 3, 224, 224)
    _export(
        model, dummy,
        out_dir / "efficientnet_b7_forensic.onnx",
        input_names=["image"],
        output_names=["binary_logit", "family_logit", "patch_logit"],
        dynamic_axes={
            "image":        {0: "batch"},
            "binary_logit": {0: "batch"},
            "family_logit": {0: "batch"},
            "patch_logit":  {0: "batch"},
        }
    )


# ── Model 2: TimeSformer ──────────────────────────────────────────────────────
def export_timesformer(ckpt: Optional[Path], out_dir: Path) -> None:
    log.info("=== Exporting TimeSformer ===")
    model = _load_model(TimeSformerForensic(), ckpt)
    dummy = torch.randn(1, 8, 3, 224, 224)   # (B, T, C, H, W)
    _export(
        model, dummy,
        out_dir / "timesformer_forensic.onnx",
        input_names=["pixel_values"],
        output_names=["binary_logit"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "binary_logit": {0: "batch"},
        }
    )


# ── Model 3a: ResNet34 Mel ────────────────────────────────────────────────────
def export_audio_mel(ckpt: Optional[Path], out_dir: Path) -> None:
    log.info("=== Exporting ResNet34-Mel ===")

    class MelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, mel):
            out = self.model(mel)
            return out["logit"], out["embedding"]

    base = AudioDualModel()
    if ckpt and Path(ckpt).exists():
        state = torch.load(ckpt, map_location="cpu")
        base.load_state_dict(state.get("model", state), strict=False)
    model = MelWrapper(base.mel_model).eval()
    dummy = torch.randn(1, 1, 128, 400)
    _export(
        model, dummy,
        out_dir / "resnet34_mel.onnx",
        input_names=["mel_spectrogram"],
        output_names=["logit", "embedding"],
        dynamic_axes={
            "mel_spectrogram": {0: "batch", 3: "time"},
            "logit":           {0: "batch"},
            "embedding":       {0: "batch"},
        }
    )


# ── Model 3b: RawNet3 ─────────────────────────────────────────────────────────
def export_audio_raw(ckpt: Optional[Path], out_dir: Path) -> None:
    log.info("=== Exporting RawNet3 ===")

    class RawWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, wav):
            out = self.model(wav)
            return out["logit"], out["embedding"]

    base = AudioDualModel()
    if ckpt and Path(ckpt).exists():
        state = torch.load(ckpt, map_location="cpu")
        base.load_state_dict(state.get("model", state), strict=False)
    model = RawWrapper(base.raw_model).eval()
    dummy = torch.randn(1, 1, 64000)   # 4s at 16kHz
    _export(
        model, dummy,
        out_dir / "rawnet3.onnx",
        input_names=["waveform"],
        output_names=["logit", "embedding"],
        dynamic_axes={
            "waveform":  {0: "batch", 2: "time"},
            "logit":     {0: "batch"},
            "embedding": {0: "batch"},
        }
    )


# ── Model 4: SyncNet ──────────────────────────────────────────────────────────
def export_syncnet(ckpt: Optional[Path], out_dir: Path) -> None:
    log.info("=== Exporting SyncNet (single 1s window) ===")

    class SyncNetWrapper(nn.Module):
        """Fixed 1-second window for ONNX export."""
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, vid, aud):
            # vid: (B, 30, 3, 224, 224)  aud: (B, 50, 320)
            v_tok = self.model.vid_tokenizer(vid)
            a_tok = self.model.aud_tokenizer(aud)
            for layer in self.model.cross_attn_layers:
                v_tok, a_tok = layer(v_tok, a_tok)
            v_vec = self.model.vid_pool(v_tok.permute(0, 2, 1)).squeeze(-1)
            a_vec = self.model.aud_pool(a_tok.permute(0, 2, 1)).squeeze(-1)
            fused = torch.cat([v_vec, a_vec], dim=-1)
            return self.model.classifier(fused)

    model = SyncNetWrapper(_load_model(SyncNetCrossModal(), ckpt))
    vid   = torch.randn(1, 30, 3, 224, 224)
    aud   = torch.randn(1, 50, 320)
    _export(
        model, (vid, aud),
        out_dir / "syncnet_crossmodal.onnx",
        input_names=["video_frames", "audio_frames"],
        output_names=["binary_logit"],
        dynamic_axes={
            "video_frames": {0: "batch"},
            "audio_frames": {0: "batch"},
            "binary_logit": {0: "batch"},
        }
    )


# ── Model 5: Fusion MLP ───────────────────────────────────────────────────────
def export_fusion(ckpt: Optional[Path], out_dir: Path) -> None:
    log.info("=== Exporting Fusion MLP ===")

    class FusionWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, scores):
            # Default to image type for ONNX; media_type handled at inference
            return torch.sigmoid(self.model(scores, "image"))

    model = FusionWrapper(_load_model(BayesianFusionMLP(), ckpt))
    dummy = torch.randn(1, INPUT_DIM)
    _export(
        model, dummy,
        out_dir / "fusion_mlp.onnx",
        input_names=["score_vector"],
        output_names=["fake_probability"],
        dynamic_axes={
            "score_vector":    {0: "batch"},
            "fake_probability":{0: "batch"},
        }
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="LENS ONNX Export")
    p.add_argument("--efficientnet", type=Path, default=None)
    p.add_argument("--timesformer",  type=Path, default=None)
    p.add_argument("--audio-dual",   type=Path, default=None)
    p.add_argument("--syncnet",      type=Path, default=None)
    p.add_argument("--fusion",       type=Path, default=None)
    p.add_argument("--out-dir",      required=True, type=Path)
    args = p.parse_args()

    export_efficientnet(args.efficientnet, args.out_dir)
    export_timesformer(args.timesformer,   args.out_dir)
    export_audio_mel(args.audio_dual,      args.out_dir)
    export_audio_raw(args.audio_dual,      args.out_dir)
    export_syncnet(args.syncnet,           args.out_dir)
    export_fusion(args.fusion,             args.out_dir)

    log.info("\n=== All exports complete ===")
    for f in sorted(args.out_dir.glob("*.onnx")):
        size_mb = f.stat().st_size / 1e6
        log.info("  %s  %.1f MB", f.name, size_mb)


if __name__ == "__main__":
    main()
