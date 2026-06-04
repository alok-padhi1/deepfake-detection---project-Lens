"""
LENS ML — DFDC (DeepFake Detection Challenge) Dataset Loader
128,154 video clips across train/test partitions.
Extracts frames at 30fps with face detection.
Expected layout:
  $DFDC_ROOT/
    dfdc_train_part_{0..49}/
      metadata.json
      *.mp4
    dfdc_test/
      labels.csv  (optional — test labels not public)
      *.mp4
Outputs: $OUT_ROOT/<split>/<label_str>/<video_id>_<frame_idx>.jpg
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

log = logging.getLogger(__name__)

LABEL_MAP = {"REAL": 0, "FAKE": 1}


@dataclass
class DFDC_Config:
    dfdc_root: Path
    out_root: Path
    fps: int = 30
    max_frames_per_video: int = 150   # 5s at 30fps
    face_crop: bool = True
    face_pad: float = 0.25
    target_size: Tuple[int, int] = (224, 224)
    quality: int = 92
    num_workers: int = 12
    val_fraction: float = 0.10        # fraction of train parts used as val


# ── Part metadata scanner ─────────────────────────────────────────────────────
def _collect_tasks(cfg: DFDC_Config) -> List[dict]:
    """Scan all dfdc_train_part_* dirs and build task list with splits."""
    root = Path(cfg.dfdc_root)
    parts = sorted(root.glob("dfdc_train_part_*"))
    if not parts:
        raise FileNotFoundError(f"No dfdc_train_part_* dirs found in {root}")

    # Val: last ceil(val_fraction * num_parts) parts
    n_val = max(1, int(len(parts) * cfg.val_fraction))
    val_parts = set(p.name for p in parts[-n_val:])

    tasks: List[dict] = []
    for part_dir in parts:
        split = "val" if part_dir.name in val_parts else "train"
        meta_file = part_dir / "metadata.json"
        if not meta_file.exists():
            log.warning("No metadata.json in %s", part_dir)
            continue
        with open(meta_file) as f:
            metadata: Dict[str, dict] = json.load(f)

        for fname, info in metadata.items():
            video_path = part_dir / fname
            if not video_path.exists():
                continue
            label_str = info.get("label", "FAKE").upper()
            label = LABEL_MAP.get(label_str, 1)
            tasks.append({
                "video_path": str(video_path),
                "video_id":   video_path.stem,
                "label":      label,
                "label_str":  label_str,
                "split":      split,
                "part":       part_dir.name,
                "original":   info.get("original", None),
            })

    # Test set (no labels available publicly)
    test_dir = root / "dfdc_test"
    if test_dir.exists():
        labels_csv = test_dir / "labels.csv"
        label_lookup: Dict[str, int] = {}
        if labels_csv.exists():
            ldf = pd.read_csv(labels_csv)
            label_lookup = dict(zip(ldf["filename"], ldf["label"].astype(int)))
        for vp in sorted(test_dir.glob("*.mp4")):
            label = label_lookup.get(vp.name, -1)   # -1 = unknown
            tasks.append({
                "video_path": str(vp),
                "video_id":   vp.stem,
                "label":      label,
                "label_str":  "UNKNOWN" if label == -1 else ("REAL" if label == 0 else "FAKE"),
                "split":      "test",
                "part":       "dfdc_test",
                "original":   None,
            })

    log.info("DFDC tasks: %d total (train+val=%d, test=%d)",
             len(tasks),
             sum(1 for t in tasks if t["split"] in ("train", "val")),
             sum(1 for t in tasks if t["split"] == "test"))
    return tasks


# ── Frame extractor (runs in subprocess) ─────────────────────────────────────
def _extract_dfdc(task: dict, cfg_dict: dict) -> List[dict]:
    video_path = task["video_path"]
    video_id   = task["video_id"]
    label      = task["label"]
    label_str  = task["label_str"]
    split      = task["split"]

    fps         = cfg_dict["fps"]
    max_frames  = cfg_dict["max_frames_per_video"]
    face_crop   = cfg_dict["face_crop"]
    pad         = cfg_dict["face_pad"]
    size        = tuple(cfg_dict["target_size"])
    quality     = cfg_dict["quality"]
    out_root    = Path(cfg_dict["out_root"])

    out_dir = out_root / split / label_str
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(native_fps / fps)))

    # Simple face detector
    proto = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(proto)

    records: List[dict] = []
    total = 0
    saved = 0

    while saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if total % step == 0:
            if face_crop:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(gray, 1.1, 4)
                if len(faces) > 0:
                    areas = [w * h for (_, _, w, h) in faces]
                    x, y, w, h = faces[np.argmax(areas)]
                    ih, iw = frame.shape[:2]
                    pw, ph = int(w * pad), int(h * pad)
                    x1 = max(0, x - pw); y1 = max(0, y - ph)
                    x2 = min(iw, x + w + pw); y2 = min(ih, y + h + ph)
                    frame = frame[y1:y2, x1:x2]
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_LANCZOS4)
            fname = f"{video_id}_{saved:05d}.jpg"
            out_path = out_dir / fname
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            records.append({
                "video_id":  video_id,
                "frame_idx": saved,
                "out_path":  str(out_path),
                "label":     label,
                "label_str": label_str,
                "split":     split,
                "part":      task["part"],
                "original":  task["original"],
            })
            saved += 1
        total += 1

    cap.release()
    return records


def _worker(args: tuple) -> List[dict]:
    task, cfg_dict = args
    return _extract_dfdc(task, cfg_dict)


# ── Main ──────────────────────────────────────────────────────────────────────
def build_dfdc(cfg: DFDC_Config) -> pd.DataFrame:
    out_root = Path(cfg.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    tasks = _collect_tasks(cfg)
    cfg_dict = {
        "fps":                   cfg.fps,
        "max_frames_per_video":  cfg.max_frames_per_video,
        "face_crop":             cfg.face_crop,
        "face_pad":              cfg.face_pad,
        "target_size":           list(cfg.target_size),
        "quality":               cfg.quality,
        "out_root":              str(out_root),
    }

    all_records: List[dict] = []
    with ProcessPoolExecutor(max_workers=cfg.num_workers) as ex:
        futs = {ex.submit(_worker, (t, cfg_dict)): t for t in tasks}
        with tqdm(total=len(tasks), desc="DFDC videos") as bar:
            for fut in as_completed(futs):
                records = fut.result()
                all_records.extend(records)
                bar.update(1)
                bar.set_postfix(frames=len(all_records))

    df = pd.DataFrame(all_records)
    manifest = out_root / "dfdc_manifest.parquet"
    df.to_parquet(manifest, index=False)
    log.info("DFDC done: %d frames → %s", len(df), manifest)
    return df


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="LENS DFDC Loader")
    p.add_argument("--dfdc-root",  required=True, type=Path)
    p.add_argument("--out-root",   required=True, type=Path)
    p.add_argument("--fps",        type=int,   default=30)
    p.add_argument("--max-frames", type=int,   default=150)
    p.add_argument("--workers",    type=int,   default=12)
    p.add_argument("--no-face",    action="store_true")
    args = p.parse_args()

    cfg = DFDC_Config(
        dfdc_root=args.dfdc_root,
        out_root=args.out_root,
        fps=args.fps,
        max_frames_per_video=args.max_frames,
        face_crop=not args.no_face,
        num_workers=args.workers,
    )
    df = build_dfdc(cfg)
    print(df.groupby(["split", "label_str"]).size().to_string())
