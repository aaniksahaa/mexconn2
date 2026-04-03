#!/usr/bin/env python3
"""
Collect per-run test CSVs into one compiled CSV for MExConn and one for SINGLE.

Expected source layout:
    results/
      mexconn/<domain>/seed_<seed>/test_results.csv
      single/<domain>/seed_<seed>/test_results.csv

The compiled CSVs omit the per-run `model` and `test_patches` columns.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


DEFAULT_KEEP_COLUMNS = [
    "model",
    "domain",
    "organelle",
    "seed",
    "dice_mean",
    "dice_std",
    "iou_mean",
    "iou_std",
    "precision_mean",
    "precision_std",
    "recall_mean",
    "recall_std",
    "f1_mean",
    "f1_std",
    "voi_mean",
    "voi_std",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile result CSVs for mexconn and single runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results_root",
        default="results",
        help="Root directory containing mexconn/ and single/ result folders.",
    )
    parser.add_argument(
        "--output_dir",
        default="results/compiled",
        help="Directory where the compiled CSVs will be written.",
    )
    return parser.parse_args()


def infer_seed_from_path(csv_path: Path) -> int | None:
    for part in csv_path.parts:
        match = re.fullmatch(r"seed_(\d+)", part)
        if match:
            return int(match.group(1))
    return None


def collect_one_mode(results_root: Path, mode: str, output_dir: Path) -> Path:
    pattern = results_root / mode
    csv_paths = sorted(pattern.glob("*/*/test_results.csv"))

    frames: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        df["model"] = mode

        if "seed" not in df.columns:
            inferred_seed = infer_seed_from_path(csv_path)
            if inferred_seed is not None:
                df["seed"] = inferred_seed

        drop_columns = [col for col in ("test_patches",) if col in df.columns]
        df = df.drop(columns=drop_columns)

        keep_columns = [col for col in DEFAULT_KEEP_COLUMNS if col in df.columns]
        remaining_columns = [col for col in df.columns if col not in keep_columns]
        df = df[keep_columns + remaining_columns]

        frames.append(df)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{mode}_compiled.csv"

    if frames:
        compiled = pd.concat(frames, ignore_index=True)
        sort_columns = [col for col in ("domain", "organelle", "seed") if col in compiled.columns]
        if sort_columns:
            compiled = compiled.sort_values(sort_columns).reset_index(drop=True)
    else:
        compiled = pd.DataFrame(columns=DEFAULT_KEEP_COLUMNS)

    compiled.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)

    mexconn_out = collect_one_mode(results_root, "mexconn", output_dir)
    single_out = collect_one_mode(results_root, "single", output_dir)

    print(f"mexconn compiled CSV: {mexconn_out}")
    print(f"single compiled CSV:  {single_out}")


if __name__ == "__main__":
    main()
