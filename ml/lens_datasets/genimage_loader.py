"""
LENS ML — GenImage Dataset Loader
1.35M AI-generated images from 8 generators vs real ImageNet images.
Generators: Midjourney, SD-1.4, SD-1.5, VQDM, Wukong, GLIDE, ADM, BigGAN
Expected layout:
  $GENIMAGE_ROOT/
    imagenet_ai_0419_biggan/  (or similar per-generator dirs)
    imagenet_ai_0419_vqdm/
    imagenet_ai_0424_sdv5/
    imagenet_ai_0424_wukong/
    imagenet_glide/
    imagenet_midjourney/
    imagenet_ai_0508_adm/
    imagenet_stablediffusion/
    imagenet_real/            (real ImageNet validation images)
    README.md
Each dir contains: train/ and val/ subdirectories with class folders.
"""

from __future__ import annotations

import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from PIL import Image
from tqdm import tqdm

log = logging.getLogger(__name__)

# Generator directory name → canonical family tag
GENERATOR_MAP: Dict[str, dict] = {
    "imagenet_ai_0419_biggan":         {"family": "GAN",       "generator": "BigGAN"},
    "imagenet_ai_0419_vqdm":           {"family": "Diffusion", "generator": "VQDM"},
    "imagenet_ai_0424_sdv5":           {"family": "Diffusion", "generator": "SD-1.5"},
    "imagenet_ai_0424_wukong":         {"family": "GAN",       "generator": "Wukong"},
    "imagenet_glide":                  {"family": "Diffusion", "generator": "GLIDE"},
    "imagenet_midjourney":             {"family": "Diffusion", "generator": "Midjourney"},
    "imagenet_ai_0508_adm":            {"family": "Diffusion", "generator": "ADM"},
    "imagenet_stablediffusion":        {"family": "Diffusion", "generator": "SD-1.4"},
}
REAL_DIR = "imagenet_real"
SUPPORTED_EXT: Set[str] = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class GenImage_Config:
    genimage_root: Path
    out_root: Path
    target_size: Tuple[int, int] = (224, 224)
    quality: int = 95
    num_workers: int = 16
    copy_mode: bool = False          # True=copy files, False=symlink (saves disk)
    max_per_class: Optional[int] = None  # cap per ImageNet class


@dataclass
class GenImageRecord:
    src_path: str
    dst_path: str
    label: int            # 0=real, 1=fake
    family: str
    generator: str
    imagenet_class: str   # synset e.g. "n01440764"
    split: str


# ── Process single image ──────────────────────────────────────────────────────
def _process_image(args: tuple) -> Optional[dict]:
    (src, dst, label, family, generator, cls, split,
     target_size, quality, copy_mode) = args
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        if copy_mode:
            img = Image.open(src).convert("RGB")
            img = img.resize(target_size, Image.LANCZOS)
            img.save(dst, "JPEG", quality=quality)
        else:
            if not dst.exists():
                dst.symlink_to(src.resolve())
    except Exception as e:
        log.debug("Skip %s: %s", src, e)
        return None

    return {
        "src_path":       str(src),
        "dst_path":       str(dst),
        "label":          label,
        "family":         family,
        "generator":      generator,
        "imagenet_class": cls,
        "split":          split,
    }


# ── Collect tasks ─────────────────────────────────────────────────────────────
def _collect_tasks(cfg: GenImage_Config) -> List[tuple]:
    root = Path(cfg.genimage_root)
    out  = Path(cfg.out_root)
    tasks: List[tuple] = []

    def _scan_dir(src_dir: Path, label: int, family: str,
                  generator: str, split: str) -> None:
        counts: Dict[str, int] = {}
        for img_path in src_dir.rglob("*"):
            if img_path.suffix.lower() not in SUPPORTED_EXT:
                continue
            # imagenet_class = parent folder of image (synset)
            cls = img_path.parent.name
            if cfg.max_per_class:
                if counts.get(cls, 0) >= cfg.max_per_class:
                    continue
                counts[cls] = counts.get(cls, 0) + 1

            ext = ".jpg" if cfg.copy_mode else img_path.suffix
            rel = img_path.relative_to(src_dir)
            dst = out / split / generator / str(rel).replace(img_path.suffix, ext)
            tasks.append((
                str(img_path), str(dst), label, family, generator,
                cls, split,
                cfg.target_size, cfg.quality, cfg.copy_mode
            ))

    # Real images
    for split in ("train", "val"):
        real_dir = root / REAL_DIR / split
        if real_dir.exists():
            _scan_dir(real_dir, 0, "REAL", "Real-ImageNet", split)
        else:
            log.warning("Real %s dir not found: %s", split, real_dir)

    # Generated images
    for dir_name, meta in GENERATOR_MAP.items():
        gen_dir = root / dir_name
        if not gen_dir.exists():
            log.warning("Generator dir not found: %s", gen_dir)
            continue
        for split in ("train", "val"):
            split_dir = gen_dir / split
            if split_dir.exists():
                _scan_dir(split_dir, 1, meta["family"],
                          meta["generator"], split)

    # Add test split heuristic: rename val → test for generated images
    # (GenImage doesn't have a separate test; we handle splits externally)
    log.info("GenImage tasks collected: %d", len(tasks))
    return tasks


# ── Main ──────────────────────────────────────────────────────────────────────
def build_genimage(cfg: GenImage_Config) -> pd.DataFrame:
    out_root = Path(cfg.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    tasks = _collect_tasks(cfg)

    records: List[dict] = []
    with ThreadPoolExecutor(max_workers=cfg.num_workers) as ex:
        futs = {ex.submit(_process_image, t): t for t in tasks}
        with tqdm(total=len(tasks), desc="GenImage") as bar:
            for fut in as_completed(futs):
                rec = fut.result()
                if rec:
                    records.append(rec)
                bar.update(1)

    df = pd.DataFrame(records)
    manifest = out_root / "genimage_manifest.parquet"
    df.to_parquet(manifest, index=False)
    log.info("GenImage done: %d images → %s", len(df), manifest)
    return df


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="LENS GenImage Loader")
    p.add_argument("--genimage-root", required=True, type=Path)
    p.add_argument("--out-root",      required=True, type=Path)
    p.add_argument("--size",          type=int,  default=224)
    p.add_argument("--workers",       type=int,  default=16)
    p.add_argument("--copy",          action="store_true",
                   help="Copy+resize images (uses more disk but avoids symlinks)")
    p.add_argument("--max-per-class", type=int, default=None)
    args = p.parse_args()

    cfg = GenImage_Config(
        genimage_root=args.genimage_root,
        out_root=args.out_root,
        target_size=(args.size, args.size),
        num_workers=args.workers,
        copy_mode=args.copy,
        max_per_class=args.max_per_class,
    )
    df = build_genimage(cfg)
    print(df.groupby(["split", "generator"]).size().to_string())
