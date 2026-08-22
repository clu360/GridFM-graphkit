"""Convert large generated wildfire CSV artifacts to compressed Parquet.

This is intended for experiment traces and provenance tables, not source data
that must remain CSV for a runner. By default it scans the wildfire experiment
tree and tmp/ for CSV files at least 50 MiB, writes .parquet next to each CSV,
and leaves the CSV in place unless --remove-csv is passed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_ROOTS = (
    Path("experiments/test/wildfire_tests"),
    Path("tmp"),
)


def iter_large_csvs(roots: list[Path], min_bytes: int) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.csv"):
            if ".git" in path.parts:
                continue
            try:
                if path.stat().st_size >= min_bytes:
                    paths.append(path)
            except OSError:
                continue
    return sorted(paths, key=lambda p: p.stat().st_size, reverse=True)


def convert_csv(path: Path, remove_csv: bool) -> tuple[int, int, int, int]:
    csv_size = path.stat().st_size
    out_path = path.with_suffix(".parquet")
    df = pd.read_csv(path)
    df.to_parquet(out_path, index=False, compression="zstd")
    parquet_size = out_path.stat().st_size
    if remove_csv:
        path.unlink()
    return len(df), len(df.columns), csv_size, parquet_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=list(DEFAULT_ROOTS),
        help="Directories to scan. Defaults to wildfire experiment outputs and tmp/.",
    )
    parser.add_argument(
        "--min-mb",
        type=float,
        default=50.0,
        help="Only convert CSV files at or above this size in MiB.",
    )
    parser.add_argument(
        "--remove-csv",
        action="store_true",
        help="Delete each CSV after a successful Parquet write.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list matching CSV files.",
    )
    args = parser.parse_args()

    min_bytes = int(args.min_mb * 1024 * 1024)
    paths = iter_large_csvs(args.roots, min_bytes)
    if not paths:
        print("No matching large CSV files found.")
        return

    for path in paths:
        if args.dry_run:
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"{path}\tcsv_mib={size_mb:.2f}")
            continue
        rows, cols, csv_size, parquet_size = convert_csv(path, args.remove_csv)
        ratio = parquet_size / csv_size if csv_size else 0.0
        print(
            f"{path}\trows={rows}\tcols={cols}\t"
            f"csv={csv_size}\tparquet={parquet_size}\tratio={ratio:.4f}"
        )


if __name__ == "__main__":
    main()
