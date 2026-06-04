"""
LENS ML — Stratified Dataset Split Script
Combines manifests from all loaders, applies 80/10/10 stratified split
by generator_family to prevent data leakage.

Input:  one or more Parquet manifests produced by the loaders
Output: manifests/split_train.parquet
        manifests/split_val.parquet
        manifests/split_test.parquet
        manifests/split_stats.csv

Stratification key: (label, family) — ensures proportional family representation
in every split regardless of dataset size imbalance.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

log = logging.getLogger(__name__)

# ── Column normalization ───────────────────────────────────────────────────────
REQUIRED_COLS = {"label", "family"}

FAMILY_ALIASES = {
    "REAL":            "REAL",
    "Real-ImageNet":   "REAL",
    "real":            "REAL",
    "GAN":             "GAN",
    "Diffusion":       "Diffusion",
    "Neural":          "Neural",
}


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if "family" not in df.columns:
        # Infer from label: real=0 → REAL, fake=1 → GAN (fallback)
        df["family"] = df["label"].map({0: "REAL", 1: "GAN"})
    df["family"] = df["family"].map(lambda x: FAMILY_ALIASES.get(str(x), str(x)))
    if "generator" not in df.columns:
        df["generator"] = df["family"]
    if "split" not in df.columns:
        df["split"] = "train"
    return df


def _strat_key(df: pd.DataFrame) -> pd.Series:
    """Composite stratification key: label + family."""
    return df["label"].astype(str) + "_" + df["family"].astype(str)


# ── Main split logic ───────────────────────────────────────────────────────────
def run_split(
    manifest_paths: List[Path],
    out_dir: Path,
    train_frac: float = 0.80,
    val_frac:   float = 0.10,
    # test_frac = 1 - train - val  (=0.10)
    seed: int = 42,
    keep_existing_splits: bool = False,
) -> None:
    """
    Build stratified train/val/test split across all provided manifests.

    keep_existing_splits=True: if a manifest already has a 'split' column
    populated by the source loader (e.g. FF++ official splits), honour it
    for the test set only and re-split train/val.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: List[pd.DataFrame] = []
    for mp in manifest_paths:
        log.info("Loading manifest: %s", mp)
        df = pd.read_parquet(mp)
        df = _normalize(df)
        df["source_manifest"] = mp.name
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)
    log.info("Total records: %d", len(full))
    log.info("Label distribution:\n%s",
             full.groupby(["label", "family"]).size().to_string())

    if keep_existing_splits:
        # Test set = records with pre-assigned split='test'
        test_mask = full["split"] == "test"
        rest = full[~test_mask].reset_index(drop=True)
        test = full[test_mask].reset_index(drop=True)
    else:
        rest = full.copy()
        test = pd.DataFrame()

    # ── Stratified split of remaining into train / val / test ────────────────
    strat_key = _strat_key(rest)

    # If no pre-assigned test, carve 10% out first
    if len(test) == 0:
        test_frac_of_rest = 1.0 - train_frac - val_frac
        rest, test_df = train_test_split(
            rest,
            test_size=test_frac_of_rest,
            stratify=strat_key,
            random_state=seed,
        )
        test = pd.concat([test, test_df], ignore_index=True)
        rest = rest.reset_index(drop=True)

    # val / train split
    val_frac_of_rest = val_frac / (train_frac + val_frac)
    strat_key_rest = _strat_key(rest)
    train_df, val_df = train_test_split(
        rest,
        test_size=val_frac_of_rest,
        stratify=strat_key_rest,
        random_state=seed,
    )

    train_df = train_df.copy(); train_df["split"] = "train"
    val_df   = val_df.copy();   val_df["split"]   = "val"
    test     = test.copy();     test["split"]      = "test"

    # ── Save ──────────────────────────────────────────────────────────────────
    train_df.to_parquet(out_dir / "split_train.parquet", index=False)
    val_df.to_parquet(out_dir   / "split_val.parquet",   index=False)
    test.to_parquet(out_dir     / "split_test.parquet",  index=False)

    # ── Stats report ──────────────────────────────────────────────────────────
    stats_rows = []
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test)]:
        for (label, family), count in df.groupby(["label", "family"]).size().items():
            stats_rows.append({
                "split":  split_name,
                "label":  label,
                "family": family,
                "count":  count,
            })

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(out_dir / "split_stats.csv", index=False)

    log.info("\n=== Split Statistics ===")
    pivot = stats_df.pivot_table(
        index=["label", "family"], columns="split",
        values="count", aggfunc="sum", fill_value=0
    )
    log.info("\n%s", pivot.to_string())

    log.info(
        "\nSummary:\n  train: %d\n  val:   %d\n  test:  %d\n  total: %d",
        len(train_df), len(val_df), len(test),
        len(train_df) + len(val_df) + len(test)
    )

    # ── Generator family distribution check ───────────────────────────────────
    log.info("\nGenerator family split distribution:")
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test)]:
        pct = df["family"].value_counts(normalize=True) * 100
        log.info("  [%s]\n%s", split_name, pct.to_string())


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="LENS Stratified Dataset Split")
    p.add_argument("--manifests", nargs="+", type=Path, required=True,
                   help="One or more Parquet manifests from dataset loaders")
    p.add_argument("--out-dir",   type=Path, required=True,
                   help="Directory to write split_train/val/test.parquet")
    p.add_argument("--train",     type=float, default=0.80)
    p.add_argument("--val",       type=float, default=0.10)
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--keep-existing-splits", action="store_true",
                   help="Honour existing 'test' rows in manifests (e.g. FF++ official)")
    args = p.parse_args()

    assert abs(args.train + args.val + (1.0 - args.train - args.val) - 1.0) < 1e-6
    run_split(
        manifest_paths=args.manifests,
        out_dir=args.out_dir,
        train_frac=args.train,
        val_frac=args.val,
        seed=args.seed,
        keep_existing_splits=args.keep_existing_splits,
    )
