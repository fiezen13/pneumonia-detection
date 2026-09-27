"""Read-only dataset audit utilities for chest X-ray images."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val", "test")
CLASSES = ("NORMAL", "PNEUMONIA")


def discover_images(data_dir: Path | str) -> pd.DataFrame:
    """List all image files under the expected split/class folders."""
    root = Path(data_dir)
    rows: list[dict] = []

    for split in SPLITS:
        for label in CLASSES:
            folder = root / split / label
            if not folder.is_dir():
                continue
            for path in sorted(folder.iterdir()):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                rows.append(
                    {
                        "path": path,
                        "rel_path": str(path.relative_to(root)),
                        "split": split,
                        "label": label,
                        "filename": path.name,
                        "suffix": path.suffix.lower(),
                        "size_bytes": path.stat().st_size,
                    }
                )

    return pd.DataFrame(rows)


def count_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot split × class counts with totals."""
    counts = (
        df.groupby(["split", "label"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=list(SPLITS), columns=list(CLASSES), fill_value=0)
    )
    counts["TOTAL"] = counts.sum(axis=1)
    totals = counts.sum(axis=0).to_frame().T
    totals.index = ["ALL"]
    return pd.concat([counts, totals])


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Per-split class counts and pneumonia ratio."""
    rows = []
    for split in SPLITS:
        subset = df[df["split"] == split]
        n_normal = int((subset["label"] == "NORMAL").sum())
        n_pneumonia = int((subset["label"] == "PNEUMONIA").sum())
        total = n_normal + n_pneumonia
        rows.append(
            {
                "split": split,
                "NORMAL": n_normal,
                "PNEUMONIA": n_pneumonia,
                "total": total,
                "pneumonia_ratio": (n_pneumonia / total) if total else 0.0,
                "normal_ratio": (n_normal / total) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def inspect_image(path: Path) -> dict:
    """Read width/height/mode for one image. Does not modify files."""
    try:
        with Image.open(path) as img:
            img.load()
            width, height = img.size
            mode = img.mode
        return {
            "width": width,
            "height": height,
            "aspect_ratio": width / height if height else None,
            "mode": mode,
            "readable": True,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - collect any read failure
        return {
            "width": None,
            "height": None,
            "aspect_ratio": None,
            "mode": None,
            "readable": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def inspect_images(df: pd.DataFrame) -> pd.DataFrame:
    """Attach image metadata columns to the inventory dataframe."""
    meta = df["path"].map(inspect_image)
    meta_df = pd.DataFrame(list(meta))
    return pd.concat([df.reset_index(drop=True), meta_df], axis=1)


def numeric_summary(series: pd.Series) -> dict:
    """Deterministic numeric summary for a series."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
        }
    return {
        "count": int(clean.count()),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "std": float(clean.std(ddof=0)),
    }


def mode_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Count image modes overall and by split."""
    overall = (
        df["mode"]
        .fillna("UNREADABLE")
        .value_counts()
        .rename_axis("mode")
        .reset_index(name="count")
    )
    overall["split"] = "ALL"
    by_split = (
        df.assign(mode=df["mode"].fillna("UNREADABLE"))
        .groupby(["split", "mode"], observed=True)
        .size()
        .reset_index(name="count")
    )
    return pd.concat([overall, by_split], ignore_index=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 of a file in binary chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def hash_images(df: pd.DataFrame) -> pd.DataFrame:
    """Add sha256 column for every image path."""
    out = df.copy()
    out["sha256"] = [sha256_file(path) for path in out["path"]]
    return out


def duplicate_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per file that shares its SHA-256 with another file."""
    if "sha256" not in df.columns:
        raise ValueError("dataframe must include sha256 column")

    counts = df["sha256"].value_counts()
    dup_hashes = counts[counts > 1].index
    dups = df[df["sha256"].isin(dup_hashes)].copy()
    dups["group_size"] = dups["sha256"].map(counts)
    return dups.sort_values(["sha256", "split", "label", "rel_path"]).reset_index(drop=True)


def within_split_duplicates(dup_df: pd.DataFrame) -> pd.DataFrame:
    """Exact duplicates where all copies live in the same split."""
    if dup_df.empty:
        return dup_df.copy()

    rows = []
    for sha, group in dup_df.groupby("sha256", sort=True):
        splits = sorted(group["split"].unique())
        if len(splits) == 1:
            rows.append(
                {
                    "sha256": sha,
                    "split": splits[0],
                    "n_files": len(group),
                    "labels": ",".join(sorted(group["label"].unique())),
                    "files": " | ".join(group["rel_path"].tolist()),
                }
            )
    return pd.DataFrame(rows)


def cross_split_leakage(dup_df: pd.DataFrame) -> pd.DataFrame:
    """Exact duplicates whose content appears in more than one split."""
    if dup_df.empty:
        return pd.DataFrame(
            columns=["sha256", "splits", "n_files", "labels", "files"]
        )

    rows = []
    for sha, group in dup_df.groupby("sha256", sort=True):
        splits = sorted(group["split"].unique())
        if len(splits) > 1:
            rows.append(
                {
                    "sha256": sha,
                    "splits": ",".join(splits),
                    "n_files": len(group),
                    "labels": ",".join(sorted(group["label"].unique())),
                    "files": " | ".join(group["rel_path"].tolist()),
                }
            )
    return pd.DataFrame(rows)


def leakage_summary(leakage_df: pd.DataFrame) -> dict:
    """Compact summary of cross-split leakage."""
    if leakage_df.empty:
        return {
            "n_leaked_hashes": 0,
            "n_leaked_files": 0,
            "split_pair_counts": {},
        }

    pair_counter: Counter[str] = Counter()
    for splits in leakage_df["splits"]:
        pair_counter[splits] += 1

    return {
        "n_leaked_hashes": int(len(leakage_df)),
        "n_leaked_files": int(leakage_df["n_files"].sum()),
        "split_pair_counts": dict(sorted(pair_counter.items())),
    }


def corrupted_images(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that failed image open/load."""
    cols = [c for c in ["rel_path", "split", "label", "error"] if c in df.columns]
    return df.loc[~df["readable"], cols].reset_index(drop=True)
