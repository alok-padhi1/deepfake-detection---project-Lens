"""
LENS ML — Synthetic Dataset Loader
800K AI-generated images from:
  - Stable Diffusion 3.x (SD3)
  - Midjourney v6 (MJ6)
  - DALL-E 3 (DALLE3)
  - Flux (FLUX)
Each source has its own loader class with configurable metadata parsing.
All sources are unified into a single manifest.

Expected root layout per source:
  $SYNTH_ROOT/
    sd3/
      images/  *.{jpg,png,webp}
      metadata.jsonl   (one JSON per line: {"filename": ..., "prompt": ..., ...})
    midjourney_v6/
      images/
      metadata.jsonl
    dalle3/
      images/
      metadata.jsonl
    flux/
      images/
      metadata.jsonl
"""

from __future__ import annotations

import json
import logging
import shutil
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd
from PIL import Image
from tqdm import tqdm

log = logging.getLogger(__name__)

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class SynthConfig:
    synth_root: Path
    out_root: Path
    target_size: Tuple[int, int] = (224, 224)
    quality: int = 95
    num_workers: int = 16
    max_per_source: Optional[int] = None   # None = all


# ── Base loader ───────────────────────────────────────────────────────────────
class SyntheticSourceLoader(ABC):
    source_name: str
    generator_family: str   # "Diffusion" / "GAN"

    def __init__(self, src_dir: Path, cfg: SynthConfig) -> None:
        self.src_dir = src_dir
        self.cfg = cfg
        self._meta: Dict[str, dict] = {}

    def _load_metadata_jsonl(self) -> None:
        jsonl = self.src_dir / "metadata.jsonl"
        if not jsonl.exists():
            return
        with open(jsonl) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    fname = obj.get("filename") or obj.get("file_name") or ""
                    self._meta[fname] = obj
                except json.JSONDecodeError:
                    pass

    def _image_paths(self) -> List[Path]:
        img_dir = self.src_dir / "images"
        if not img_dir.exists():
            img_dir = self.src_dir
        paths = [p for p in img_dir.rglob("*")
                 if p.suffix.lower() in SUPPORTED_EXT]
        if self.cfg.max_per_source:
            paths = paths[:self.cfg.max_per_source]
        return sorted(paths)

    def _out_path(self, src: Path, split: str) -> Path:
        out = (Path(self.cfg.out_root) / split
               / self.source_name / src.stem).with_suffix(".jpg")
        return out

    @abstractmethod
    def extra_meta(self, src: Path) -> dict:
        """Return source-specific metadata fields for this image."""

    def build_tasks(self, split: str) -> List[tuple]:
        self._load_metadata_jsonl()
        paths = self._image_paths()
        return [
            (str(p), str(self._out_path(p, split)),
             1, "Diffusion", self.source_name,
             split, self.cfg.target_size, self.cfg.quality,
             self.extra_meta(p))
            for p in paths
        ]


class SD3Loader(SyntheticSourceLoader):
    source_name = "SD3"
    generator_family = "Diffusion"

    def extra_meta(self, src: Path) -> dict:
        m = self._meta.get(src.name, {})
        return {
            "prompt":     m.get("prompt", ""),
            "cfg_scale":  m.get("cfg_scale", ""),
            "steps":      m.get("steps", ""),
            "model":      m.get("model_version", "sd3"),
        }


class MidjourneyV6Loader(SyntheticSourceLoader):
    source_name = "MJ6"
    generator_family = "Diffusion"

    def extra_meta(self, src: Path) -> dict:
        m = self._meta.get(src.name, {})
        return {
            "prompt":    m.get("prompt", ""),
            "aspect":    m.get("aspect_ratio", ""),
            "version":   m.get("version", "6"),
            "style":     m.get("style", ""),
        }


class DALLE3Loader(SyntheticSourceLoader):
    source_name = "DALLE3"
    generator_family = "Diffusion"

    def extra_meta(self, src: Path) -> dict:
        m = self._meta.get(src.name, {})
        return {
            "prompt":    m.get("prompt", ""),
            "quality":   m.get("quality", "hd"),
            "size":      m.get("size", ""),
            "style":     m.get("style", "vivid"),
        }


class FluxLoader(SyntheticSourceLoader):
    source_name = "FLUX"
    generator_family = "Diffusion"

    def extra_meta(self, src: Path) -> dict:
        m = self._meta.get(src.name, {})
        return {
            "prompt":        m.get("prompt", ""),
            "guidance":      m.get("guidance_scale", ""),
            "num_steps":     m.get("num_inference_steps", ""),
            "model_variant": m.get("model", "flux"),
        }


SOURCE_REGISTRY: Dict[str, type] = {
    "sd3":           SD3Loader,
    "midjourney_v6": MidjourneyV6Loader,
    "dalle3":        DALLE3Loader,
    "flux":          FluxLoader,
}


# ── Image processor ───────────────────────────────────────────────────────────
def _process_image(args: tuple) -> Optional[dict]:
    (src, dst, label, family, generator,
     split, target_size, quality, extra) = args
    src = Path(src); dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        img = Image.open(src).convert("RGB")
        img = img.resize(target_size, Image.LANCZOS)
        img.save(str(dst), "JPEG", quality=quality)
    except Exception as e:
        log.debug("Skip %s: %s", src, e)
        return None

    record = {
        "src_path":  str(src),
        "dst_path":  str(dst),
        "label":     label,
        "family":    family,
        "generator": generator,
        "split":     split,
    }
    record.update(extra)
    return record


# ── Stratified split assignment ───────────────────────────────────────────────
def _assign_split(paths: List[Path], seed: int = 42) -> Dict[str, str]:
    """80/10/10 deterministic split per-source."""
    import random
    rng = random.Random(seed)
    idxs = list(range(len(paths)))
    rng.shuffle(idxs)
    n = len(idxs)
    train_end = int(0.8 * n)
    val_end   = int(0.9 * n)
    split_map: Dict[str, str] = {}
    for i, idx in enumerate(idxs):
        if i < train_end:
            split_map[str(paths[idx])] = "train"
        elif i < val_end:
            split_map[str(paths[idx])] = "val"
        else:
            split_map[str(paths[idx])] = "test"
    return split_map


# ── Main ──────────────────────────────────────────────────────────────────────
def build_synthetic(cfg: SynthConfig) -> pd.DataFrame:
    out_root = Path(cfg.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    synth_root = Path(cfg.synth_root)

    all_tasks: List[tuple] = []

    for dir_name, loader_cls in SOURCE_REGISTRY.items():
        src_dir = synth_root / dir_name
        if not src_dir.exists():
            log.warning("Source dir not found: %s — skipping", src_dir)
            continue

        loader = loader_cls(src_dir, cfg)
        loader._load_metadata_jsonl()
        paths = loader._image_paths()
        split_map = _assign_split(paths)

        for p in paths:
            split = split_map[str(p)]
            out_path = loader._out_path(p, split)
            extra = loader.extra_meta(p)
            all_tasks.append((
                str(p), str(out_path),
                1, "Diffusion", dir_name,
                split, cfg.target_size, cfg.quality, extra
            ))

        log.info("[%s] %d images", dir_name, len(paths))

    log.info("Synthetic total tasks: %d", len(all_tasks))

    records: List[dict] = []
    with ThreadPoolExecutor(max_workers=cfg.num_workers) as ex:
        futs = {ex.submit(_process_image, t): t for t in all_tasks}
        with tqdm(total=len(all_tasks), desc="Synthetic") as bar:
            for fut in as_completed(futs):
                rec = fut.result()
                if rec:
                    records.append(rec)
                bar.update(1)

    df = pd.DataFrame(records)
    manifest = out_root / "synthetic_manifest.parquet"
    df.to_parquet(manifest, index=False)
    log.info("Synthetic done: %d images → %s", len(df), manifest)
    return df


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="LENS Synthetic Loader")
    p.add_argument("--synth-root", required=True, type=Path)
    p.add_argument("--out-root",   required=True, type=Path)
    p.add_argument("--size",       type=int, default=224)
    p.add_argument("--workers",    type=int, default=16)
    p.add_argument("--max-per-source", type=int, default=None)
    args = p.parse_args()

    cfg = SynthConfig(
        synth_root=args.synth_root,
        out_root=args.out_root,
        target_size=(args.size, args.size),
        num_workers=args.workers,
        max_per_source=args.max_per_source,
    )
    df = build_synthetic(cfg)
    print(df.groupby(["split", "generator"]).size().to_string())
