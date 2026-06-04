"""
LENS ML — FaceForensics++ Dataset Loader
Handles all 4 manipulation types: Deepfakes (DF), Face2Face (F2F),
FaceShifter (FS), NeuralTextures (NT).
Extracts frames using FFmpeg, assigns binary + fine-grained labels.
Expected directory structure:
  $FF_ROOT/
    original_sequences/actors/c23/videos/*.mp4
    manipulated_sequences/{Deepfakes,Face2Face,FaceShifter,NeuralTextures}/c23/videos/*.mp4
    splits/{train,val,test}.json
Output per frame: {video_id}_{frame_idx:05d}.jpg saved to $OUT_ROOT/<split>/<label>/
"""

from __future__ import annotations

import json
import os
import subprocess
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MANIPULATION_TYPES = {
    "Deepfakes":        {"label": 1, "family": "GAN",      "code": "DF"},
    "Face2Face":        {"label": 1, "family": "GAN",      "code": "F2F"},
    "FaceShifter":      {"label": 1, "family": "GAN",      "code": "FS"},
    "NeuralTextures":   {"label": 1, "family": "Neural",   "code": "NT"},
}
REAL_CODE = "REAL"
COMPRESSION = "c23"         # c0=raw, c23=HQ, c40=LQ


@dataclass
class FF_Config:
    ff_root: Path
    out_root: Path
    fps: int = 30
    max_frames_per_video: int = 300   # ~10s at 30fps  → ≈1M total across corpus
    face_crop: bool = True
    face_pad: float = 0.3             # fractional padding around detected face
    num_workers: int = 8
    splits_file: Optional[Path] = None  # path to splits/{train,val,test}.json
    target_size: Tuple[int, int] = (224, 224)
    quality: int = 95                 # JPEG save quality


@dataclass
class FrameRecord:
    video_id: str
    frame_idx: int
    out_path: Path
    label: int        # 0=real, 1=fake
    family: str       # REAL / GAN / Neural
    manip_code: str   # REAL / DF / F2F / FS / NT
    split: str        # train / val / test


# ── Face detector (lightweight, fast) ─────────────────────────────────────────
class FaceDetector:
    _model = None

    @classmethod
    def get(cls) -> "FaceDetector":
        if cls._model is None:
            cls._model = cls()
        return cls._model

    def __init__(self) -> None:
        proto = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(proto)
        # Prefer DNN face detector if OpenCV contrib is available
        try:
            model_dir = Path(__file__).parent / "assets"
            pb   = str(model_dir / "opencv_face_detector_uint8.pb")
            pbtxt= str(model_dir / "opencv_face_detector.pbtxt")
            self._net = cv2.dnn.readNetFromTensorflow(pb, pbtxt)
            self._use_dnn = True
        except Exception:
            self._use_dnn = False

    def detect(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Return (x, y, w, h) of largest face, or None."""
        h, w = frame_bgr.shape[:2]
        if self._use_dnn:
            blob = cv2.dnn.blobFromImage(frame_bgr, 1.0, (300, 300),
                                         (104, 117, 123), swapRB=False)
            self._net.setInput(blob)
            detections = self._net.forward()
            best_conf, best_box = 0.0, None
            for i in range(detections.shape[2]):
                conf = float(detections[0, 0, i, 2])
                if conf > 0.5 and conf > best_conf:
                    best_conf = conf
                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)
                    best_box = (x1, y1, x2 - x1, y2 - y1)
            return best_box
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0:
            return None
        # return largest
        areas = [w_ * h_ for (_, _, w_, h_) in faces]
        return tuple(faces[np.argmax(areas)])  # type: ignore


def crop_face(frame: np.ndarray, box: Tuple[int, int, int, int],
              pad: float, size: Tuple[int, int]) -> np.ndarray:
    x, y, w, h = box
    ih, iw = frame.shape[:2]
    pw, ph = int(w * pad), int(h * pad)
    x1 = max(0, x - pw);  y1 = max(0, y - ph)
    x2 = min(iw, x + w + pw); y2 = min(ih, y + h + ph)
    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, size, interpolation=cv2.INTER_LANCZOS4)


# ── Per-video extraction worker ───────────────────────────────────────────────
def _extract_video(args: Tuple) -> List[FrameRecord]:
    (video_path, video_id, label, family, manip_code,
     split, out_root, fps, max_frames, face_crop, pad, size, quality) = args

    out_dir = Path(out_root) / split / manip_code
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = FaceDetector.get() if face_crop else None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning("Cannot open %s", video_path)
        return []

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(native_fps / fps)))
    records: List[FrameRecord] = []
    frame_idx_in_video = 0
    saved = 0

    while saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx_in_video % step == 0:
            if face_crop and detector:
                box = detector.detect(frame)
                if box:
                    frame = crop_face(frame, box, pad, size)
                else:
                    frame = cv2.resize(frame, size, interpolation=cv2.INTER_LANCZOS4)
            else:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_LANCZOS4)

            fname = f"{video_id}_{saved:05d}.jpg"
            out_path = out_dir / fname
            cv2.imwrite(str(out_path), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, quality])
            records.append(FrameRecord(
                video_id=video_id, frame_idx=saved,
                out_path=out_path, label=label,
                family=family, manip_code=manip_code, split=split
            ))
            saved += 1
        frame_idx_in_video += 1

    cap.release()
    return records


# ── Split assignment ──────────────────────────────────────────────────────────
def _load_splits(splits_file: Path) -> dict[str, str]:
    """Returns {video_id: split} from official FF++ splits JSON."""
    mapping: dict[str, str] = {}
    with open(splits_file) as f:
        data = json.load(f)
    # Official format: {"train": [["id1","id2"], ...], "val": [...], "test": [...]}
    for split_name, pairs in data.items():
        for pair in pairs:
            for vid_id in pair:
                mapping[vid_id] = split_name
    return mapping


# ── Main loader ───────────────────────────────────────────────────────────────
def build_faceforensics(cfg: FF_Config) -> pd.DataFrame:
    ff_root = Path(cfg.ff_root)
    out_root = Path(cfg.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    split_map: dict[str, str] = {}
    if cfg.splits_file and Path(cfg.splits_file).exists():
        split_map = _load_splits(Path(cfg.splits_file))
    else:
        log.warning("No splits file provided — all videos assigned to 'train'")

    tasks = []

    # ── Real videos ──────────────────────────────────────────────────────────
    real_root = ff_root / "original_sequences" / "actors" / COMPRESSION / "videos"
    for vp in sorted(real_root.glob("*.mp4")):
        vid_id = vp.stem
        split  = split_map.get(vid_id, "train")
        tasks.append((str(vp), vid_id, 0, "REAL", REAL_CODE,
                       split, str(out_root), cfg.fps, cfg.max_frames_per_video,
                       cfg.face_crop, cfg.face_pad, cfg.target_size, cfg.quality))

    # ── Manipulated videos ────────────────────────────────────────────────────
    for manip_name, meta in MANIPULATION_TYPES.items():
        manip_root = (ff_root / "manipulated_sequences"
                      / manip_name / COMPRESSION / "videos")
        for vp in sorted(manip_root.glob("*.mp4")):
            vid_id = vp.stem
            split  = split_map.get(vid_id.split("_")[0], "train")
            tasks.append((str(vp), vid_id, meta["label"], meta["family"],
                          meta["code"], split, str(out_root), cfg.fps,
                          cfg.max_frames_per_video, cfg.face_crop, cfg.face_pad,
                          cfg.target_size, cfg.quality))

    log.info("FF++ extraction: %d videos, %d workers", len(tasks), cfg.num_workers)
    all_records: List[FrameRecord] = []

    with ProcessPoolExecutor(max_workers=cfg.num_workers) as ex:
        futs = {ex.submit(_extract_video, t): t for t in tasks}
        with tqdm(total=len(tasks), desc="FF++ videos") as bar:
            for fut in as_completed(futs):
                records = fut.result()
                all_records.extend(records)
                bar.update(1)
                bar.set_postfix(frames=len(all_records))

    df = pd.DataFrame([vars(r) for r in all_records])
    manifest = out_root / "faceforensics_manifest.parquet"
    df.to_parquet(manifest, index=False)
    log.info("FF++ done: %d frames → %s", len(df), manifest)
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="LENS FaceForensics++ Loader")
    p.add_argument("--ff-root",    required=True, type=Path)
    p.add_argument("--out-root",   required=True, type=Path)
    p.add_argument("--fps",        type=int,  default=30)
    p.add_argument("--max-frames", type=int,  default=300)
    p.add_argument("--no-face-crop", action="store_true")
    p.add_argument("--workers",    type=int,  default=8)
    p.add_argument("--splits",     type=Path, default=None)
    p.add_argument("--size",       type=int,  default=224)
    args = p.parse_args()

    cfg = FF_Config(
        ff_root=args.ff_root,
        out_root=args.out_root,
        fps=args.fps,
        max_frames_per_video=args.max_frames,
        face_crop=not args.no_face_crop,
        num_workers=args.workers,
        splits_file=args.splits,
        target_size=(args.size, args.size),
    )
    df = build_faceforensics(cfg)
    print(df.groupby(["split", "manip_code"]).size().to_string())
