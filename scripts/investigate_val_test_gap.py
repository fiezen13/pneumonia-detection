"""Investigate validation vs test score-gap at the locked threshold.

Read-only: no retraining, no threshold/model changes, no reselection.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.data.audit import discover_images, hash_images
from src.data.preprocessing import build_eval_transforms, build_train_transforms
from src.data.split import load_split
from src.evaluation.threshold import evaluate_threshold

LOCKED_THRESHOLD = 0.30
PERSON_RE = re.compile(r"person(\d+)_", re.IGNORECASE)
NORMAL_IM_RE = re.compile(r"(?:NORMAL\d+-)?IM-(\d+)-", re.IGNORECASE)
TYPE_RE = re.compile(r"_(bacteria|virus)_", re.IGNORECASE)


def summarize_scores(scores: list[float]) -> dict[str, Any]:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    qs = np.quantile(arr, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist()
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "q05": float(qs[0]),
        "q25": float(qs[1]),
        "q50": float(qs[2]),
        "q75": float(qs[3]),
        "q95": float(qs[4]),
        "frac_ge_0_30": float((arr >= LOCKED_THRESHOLD).mean()),
        "frac_ge_0_50": float((arr >= 0.5).mean()),
    }


def extract_person_id(rel_path: str) -> str | None:
    m = PERSON_RE.search(Path(rel_path).name)
    return m.group(1) if m else None


def extract_normal_im_id(rel_path: str) -> str | None:
    m = NORMAL_IM_RE.search(Path(rel_path).name)
    return m.group(1) if m else None


def extract_infection_type(rel_path: str) -> str | None:
    m = TYPE_RE.search(Path(rel_path).name)
    return m.group(1).lower() if m else None


def load_val_records(path: Path) -> list[dict]:
    rows = json.loads(path.read_text())
    out = []
    for r in rows:
        out.append(
            {
                "rel_path": r["rel_path"],
                "true_label": r["true_label"],
                "prob_pneumonia": float(r["prob_pneumonia"]),
            }
        )
    return out


def load_test_records(path: Path) -> list[dict]:
    rows = json.loads(path.read_text())
    out = []
    for r in rows:
        out.append(
            {
                "rel_path": r["rel_path"],
                "true_label": r["true_label"],
                "prob_pneumonia": float(r["prob_pneumonia"]),
                "threshold_pred_label": r["threshold_pred_label"],
            }
        )
    return out


def plot_score_distributions(
    val_rows: list[dict],
    test_rows: list[dict],
    output_path: Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    bins = np.linspace(0, 1, 41)

    for ax, label in zip(axes, ["NORMAL", "PNEUMONIA"]):
        v = [r["prob_pneumonia"] for r in val_rows if r["true_label"] == label]
        t = [r["prob_pneumonia"] for r in test_rows if r["true_label"] == label]
        ax.hist(v, bins=bins, alpha=0.55, density=True, label=f"val n={len(v)}")
        ax.hist(t, bins=bins, alpha=0.55, density=True, label=f"test n={len(t)}")
        ax.axvline(LOCKED_THRESHOLD, color="black", linestyle="--", label="thr=0.30")
        ax.axvline(0.5, color="gray", linestyle=":", label="thr=0.50")
        ax.set_title(f"True {label}: P(PNEUMONIA)")
        ax.set_xlabel("P(PNEUMONIA)")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Score distributions — locked model (resnet18_finetune)\n"
        "Investigation only; not used to change the locked threshold."
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_fp_score_hist(test_fps: list[dict], output_path: Path) -> Path:
    scores = [r["prob_pneumonia"] for r in test_fps]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(scores, bins=np.linspace(0.3, 1.0, 29), color="#4C78A8", edgecolor="white")
    ax.axvline(LOCKED_THRESHOLD, color="black", linestyle="--", label="thr=0.30")
    ax.set_title(f"Test FALSE POSITIVES at thr=0.30 (n={len(scores)})")
    ax.set_xlabel("P(PNEUMONIA)")
    ax.set_ylabel("count")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output_path


def analyze_test_false_positives(test_rows: list[dict]) -> dict[str, Any]:
    fps = [
        r
        for r in test_rows
        if r["true_label"] == "NORMAL"
        and r["prob_pneumonia"] >= LOCKED_THRESHOLD
    ]
    # Naming patterns available from filenames only.
    prefixes = Counter()
    im_ids = Counter()
    for r in fps:
        name = Path(r["rel_path"]).name
        if name.startswith("NORMAL2-"):
            prefixes["NORMAL2-IM-*"] += 1
        elif name.startswith("IM-"):
            prefixes["IM-*"] += 1
        else:
            prefixes["other"] += 1
        im = extract_normal_im_id(r["rel_path"])
        if im:
            im_ids[im] += 1

    # Compare FP rate within NORMAL naming subgroups on test.
    normal_test = [r for r in test_rows if r["true_label"] == "NORMAL"]
    subgroup_stats = {}
    for key, pred in [
        ("IM-*", lambda n: n.startswith("IM-")),
        ("NORMAL2-IM-*", lambda n: n.startswith("NORMAL2-")),
    ]:
        subset = [r for r in normal_test if pred(Path(r["rel_path"]).name)]
        fp_sub = [r for r in subset if r["prob_pneumonia"] >= LOCKED_THRESHOLD]
        subgroup_stats[key] = {
            "n_normal": len(subset),
            "n_fp": len(fp_sub),
            "fp_rate": (len(fp_sub) / len(subset)) if subset else None,
            "score_summary": summarize_scores(
                [r["prob_pneumonia"] for r in subset]
            ),
        }

    return {
        "n_fp": len(fps),
        "score_summary": summarize_scores([r["prob_pneumonia"] for r in fps]),
        "filename_prefix_counts": dict(prefixes),
        "unique_im_ids_in_fps": len(im_ids),
        "repeated_im_ids_in_fps": {
            k: v for k, v in im_ids.items() if v > 1
        },
        "normal_subgroup_stats": subgroup_stats,
        "top_confident_fps": sorted(
            fps, key=lambda r: r["prob_pneumonia"], reverse=True
        )[:15],
    }


def patient_id_overlap(split_paths: dict[str, list[str]]) -> dict[str, Any]:
    """Overlap using pneumonia person IDs and NORMAL IM IDs from filenames.

    Limitation: these IDs are filename-derived proxies, not verified patient IDs.
    """
    ids_by_split: dict[str, dict[str, set[str]]] = {
        "train": {"person": set(), "normal_im": set()},
        "val": {"person": set(), "normal_im": set()},
        "test": {"person": set(), "normal_im": set()},
    }
    for split, paths in split_paths.items():
        for p in paths:
            person = extract_person_id(p)
            if person:
                ids_by_split[split]["person"].add(person)
            nim = extract_normal_im_id(p)
            if nim and "NORMAL" in p:
                ids_by_split[split]["normal_im"].add(nim)

    def _overlap(a: str, b: str, kind: str) -> dict[str, Any]:
        inter = sorted(ids_by_split[a][kind] & ids_by_split[b][kind])
        return {
            "n_overlap": len(inter),
            "examples": inter[:20],
        }

    return {
        "counts": {
            split: {
                "n_person_ids": len(ids_by_split[split]["person"]),
                "n_normal_im_ids": len(ids_by_split[split]["normal_im"]),
            }
            for split in ("train", "val", "test")
        },
        "person_id_overlap": {
            "train_val": _overlap("train", "val", "person"),
            "train_test": _overlap("train", "test", "person"),
            "val_test": _overlap("val", "test", "person"),
        },
        "normal_im_id_overlap": {
            "train_val": _overlap("train", "val", "normal_im"),
            "train_test": _overlap("train", "test", "normal_im"),
            "val_test": _overlap("val", "test", "normal_im"),
        },
        "limitation": (
            "IDs parsed from filenames (personNNNN / IM-NNNN). "
            "These are proxies only; they are not confirmed unique patient IDs."
        ),
    }


def exact_content_overlap(data_dir: Path, split: dict) -> dict[str, Any]:
    """SHA-256 exact duplicate check across train/val/test path lists."""
    # Build inventory from split paths only (read-only hashing).
    rows = []
    root = Path(data_dir)
    for split_name in ("train", "val", "test"):
        for rel in split[split_name]:
            rows.append(
                {
                    "path": root / rel,
                    "rel_path": rel,
                    "split": split_name,
                    "label": Path(rel).parts[1],
                }
            )
    import pandas as pd

    df = pd.DataFrame(rows)
    hashed = hash_images(df)
    # Map hash -> splits
    hash_to_splits: dict[str, set[str]] = defaultdict(set)
    hash_to_paths: dict[str, list[str]] = defaultdict(list)
    for _, row in hashed.iterrows():
        hash_to_splits[row["sha256"]].add(row["split"])
        hash_to_paths[row["sha256"]].append(row["rel_path"])

    cross = {
        sha: {
            "splits": sorted(splits),
            "files": hash_to_paths[sha],
        }
        for sha, splits in hash_to_splits.items()
        if len(splits) > 1
    }
    return {
        "n_files_hashed": int(len(hashed)),
        "n_unique_hashes": int(hashed["sha256"].nunique()),
        "n_cross_split_duplicate_hashes": len(cross),
        "cross_split_duplicates": list(cross.values())[:20],
        "val_test_shared_hashes": sum(
            1 for v in cross.values() if set(v["splits"]) >= {"val", "test"}
        ),
    }


def verify_preprocessing() -> dict[str, Any]:
    """Confirm val/test use the same deterministic eval transforms."""
    eval_a = build_eval_transforms(224)
    eval_b = build_eval_transforms(224)
    train_tf = build_train_transforms(224)
    # Structural equality via string repr of compose.
    eval_same = str(eval_a) == str(eval_b)
    return {
        "eval_transform_identical_across_calls": eval_same,
        "eval_transform": str(eval_a),
        "train_transform_differs_from_eval": str(train_tf) != str(eval_a),
        "note": (
            "Both validation and test evaluation use build_eval_transforms "
            "(Resize+ToTensor+Normalize only). Training augmentation is not "
            "applied at evaluation."
        ),
        "evidence": "code inspection of create_datasets / final_test_evaluation / error_analysis",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Investigate val/test gap at locked threshold 0.30."
    )
    parser.add_argument(
        "--val-predictions",
        type=Path,
        default=PROJECT_ROOT
        / "experiments"
        / "error_analysis"
        / "resnet18_finetune"
        / "predictions.json",
    )
    parser.add_argument(
        "--test-predictions",
        type=Path,
        default=PROJECT_ROOT
        / "experiments"
        / "final_evaluation"
        / "final_test_predictions.json",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "data_split.json",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "chest_xray",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "final_evaluation",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "figures" / "final_evaluation",
    )
    args = parser.parse_args()

    val_rows = load_val_records(args.val_predictions)
    test_rows = load_test_records(args.test_predictions)
    split = load_split(args.split_file)

    # 5) Counts
    val_counts = Counter(r["true_label"] for r in val_rows)
    test_counts = Counter(r["true_label"] for r in test_rows)
    counts = {
        "validation": {
            "n": len(val_rows),
            "NORMAL": val_counts["NORMAL"],
            "PNEUMONIA": val_counts["PNEUMONIA"],
            "expected": {"n": 784, "NORMAL": 201, "PNEUMONIA": 583},
            "matches_expected": (
                len(val_rows) == 784
                and val_counts["NORMAL"] == 201
                and val_counts["PNEUMONIA"] == 583
            ),
        },
        "test": {
            "n": len(test_rows),
            "NORMAL": test_counts["NORMAL"],
            "PNEUMONIA": test_counts["PNEUMONIA"],
            "expected": {"n": 624, "NORMAL": 234, "PNEUMONIA": 390},
            "matches_expected": (
                len(test_rows) == 624
                and test_counts["NORMAL"] == 234
                and test_counts["PNEUMONIA"] == 390
            ),
        },
        "class_prior_pneumonia": {
            "validation": val_counts["PNEUMONIA"] / len(val_rows),
            "test": test_counts["PNEUMONIA"] / len(test_rows),
        },
    }

    # 1) Score distributions
    score_dist = {}
    for split_name, rows in [("validation", val_rows), ("test", test_rows)]:
        score_dist[split_name] = {}
        for label in ("NORMAL", "PNEUMONIA"):
            scores = [
                r["prob_pneumonia"] for r in rows if r["true_label"] == label
            ]
            score_dist[split_name][label] = summarize_scores(scores)

    # Metrics at locked threshold for both splits (from saved probs).
    def _arrays(rows):
        y_true = [0 if r["true_label"] == "NORMAL" else 1 for r in rows]
        y_prob = [r["prob_pneumonia"] for r in rows]
        return y_true, y_prob

    val_metrics = evaluate_threshold(*_arrays(val_rows), LOCKED_THRESHOLD)
    test_metrics = evaluate_threshold(*_arrays(test_rows), LOCKED_THRESHOLD)

    # 3) Test FP analysis
    fp_analysis = analyze_test_false_positives(test_rows)

    # Infection-type composition (pneumonia filenames only)
    infection_comp = {}
    for split_name, rows in [("validation", val_rows), ("test", test_rows)]:
        types = Counter(
            extract_infection_type(r["rel_path"]) or "unknown"
            for r in rows
            if r["true_label"] == "PNEUMONIA"
        )
        infection_comp[split_name] = dict(types)

    # 4) Preprocessing
    preprocessing = verify_preprocessing()

    # 6) Overlap checks
    split_paths = {
        "train": split["train"],
        "val": split["val"],
        "test": split["test"],
    }
    id_overlap = patient_id_overlap(split_paths)
    print("Hashing split files for exact-duplicate check...")
    content_overlap = exact_content_overlap(args.data_dir, split)

    # Plots
    dist_fig = plot_score_distributions(
        val_rows,
        test_rows,
        args.figure_dir / "val_test_score_distributions.png",
    )
    fp_fig = plot_fp_score_hist(
        [
            r
            for r in test_rows
            if r["true_label"] == "NORMAL"
            and r["prob_pneumonia"] >= LOCKED_THRESHOLD
        ],
        args.figure_dir / "test_false_positive_scores.png",
    )

    # Evidence vs speculation sections
    evidence = [
        {
            "finding": (
                "At thr=0.30, validation NORMAL FP rate is much lower than test "
                f"NORMAL FP rate "
                f"({score_dist['validation']['NORMAL']['frac_ge_0_30']:.3f} vs "
                f"{score_dist['test']['NORMAL']['frac_ge_0_30']:.3f})."
            ),
            "type": "evidence",
            "support": "score_distribution summaries from saved model probabilities",
        },
        {
            "finding": (
                "True-PNEUMONIA scores are high on both splits "
                f"(val mean={score_dist['validation']['PNEUMONIA']['mean']:.3f}, "
                f"test mean={score_dist['test']['PNEUMONIA']['mean']:.3f}); "
                f"FN remain low (val FN={val_metrics['false_negatives']}, "
                f"test FN={test_metrics['false_negatives']})."
            ),
            "type": "evidence",
            "support": "score summaries + threshold metrics",
        },
        {
            "finding": (
                "Validation/test evaluation transforms are the same deterministic "
                "eval pipeline (no train-time augmentation)."
            ),
            "type": "evidence",
            "support": "preprocessing code inspection",
        },
        {
            "finding": (
                f"Exact SHA-256 cross-split duplicates: "
                f"{content_overlap['n_cross_split_duplicate_hashes']} "
                f"(val-test shared hashes: {content_overlap['val_test_shared_hashes']})."
            ),
            "type": "evidence",
            "support": "content hashing of split file lists",
        },
        {
            "finding": (
                "Filename-derived person-ID overlap train∩test / val∩test is "
                f"{id_overlap['person_id_overlap']['train_test']['n_overlap']} / "
                f"{id_overlap['person_id_overlap']['val_test']['n_overlap']}."
            ),
            "type": "evidence_with_limitation",
            "support": "regex parse of personNNNN from pneumonia filenames",
            "limitation": id_overlap["limitation"],
        },
    ]

    speculation = [
        {
            "hypothesis": (
                "The locked low threshold (0.30), chosen to maximize validation "
                "pneumonia F1, may be poorly calibrated to the test NORMAL score "
                "distribution, inflating test FP."
            ),
            "type": "speculation",
            "note": "Threshold remains locked; this does not justify changing it.",
        },
        {
            "hypothesis": (
                "Validation NORMAL images (drawn from original train) may be "
                "easier/more separable for this model than original test NORMAL images."
            ),
            "type": "speculation",
            "note": (
                "Score shifts are observed; underlying acquisition/population "
                "differences are not directly measured here."
            ),
        },
    ]

    report = {
        "investigation": "val_vs_test_gap_at_locked_threshold",
        "locked_model": "resnet18_finetune",
        "locked_threshold": LOCKED_THRESHOLD,
        "model_or_threshold_changed": False,
        "counts": counts,
        "metrics_at_locked_threshold": {
            "validation": {
                "accuracy": val_metrics["accuracy"],
                "recall_pneumonia": val_metrics["recall_pneumonia"],
                "precision_pneumonia": val_metrics["precision_pneumonia"],
                "f1_pneumonia": val_metrics["f1_pneumonia"],
                "false_negatives": val_metrics["false_negatives"],
                "false_positives": val_metrics["false_positives"],
                "confusion_matrix": val_metrics["confusion_matrix"],
            },
            "test": {
                "accuracy": test_metrics["accuracy"],
                "recall_pneumonia": test_metrics["recall_pneumonia"],
                "precision_pneumonia": test_metrics["precision_pneumonia"],
                "f1_pneumonia": test_metrics["f1_pneumonia"],
                "false_negatives": test_metrics["false_negatives"],
                "false_positives": test_metrics["false_positives"],
                "confusion_matrix": test_metrics["confusion_matrix"],
            },
        },
        "score_distributions": score_dist,
        "test_false_positive_analysis": {
            **{
                k: v
                for k, v in fp_analysis.items()
                if k != "top_confident_fps"
            },
            "top_confident_fps": [
                {
                    "rel_path": r["rel_path"],
                    "prob_pneumonia": r["prob_pneumonia"],
                }
                for r in fp_analysis["top_confident_fps"]
            ],
        },
        "pneumonia_filename_infection_type_counts": infection_comp,
        "preprocessing_check": preprocessing,
        "filename_id_overlap": id_overlap,
        "exact_content_overlap": content_overlap,
        "evidence": evidence,
        "speculation": speculation,
        "limitations": [
            "No DICOM metadata/hospital site/age/sex available in this repo.",
            "Filename IDs are proxies only.",
            "Investigation uses already-saved probabilities; no new model fitting.",
            "Does not prove causal distribution shift.",
        ],
        "artifacts": {
            "score_distribution_figure": str(
                dist_fig.relative_to(PROJECT_ROOT)
            ),
            "test_fp_score_figure": str(fp_fig.relative_to(PROJECT_ROOT)),
        },
    }

    out_path = args.output_dir / "val_test_gap_investigation.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")

    # Compact markdown-friendly stdout summary
    print("\n=== Val vs Test gap investigation (thr=0.30) ===")
    print("COUNTS", json.dumps(counts, indent=2))
    print("\nSCORE DIST")
    for split_name in ("validation", "test"):
        for label in ("NORMAL", "PNEUMONIA"):
            s = score_dist[split_name][label]
            print(
                f"  {split_name}/{label}: mean={s['mean']:.3f} "
                f"median={s['q50']:.3f} frac>=0.30={s['frac_ge_0_30']:.3f} "
                f"frac>=0.50={s['frac_ge_0_50']:.3f}"
            )
    print("\nMETRICS @0.30")
    print("  val", report["metrics_at_locked_threshold"]["validation"])
    print("  test", report["metrics_at_locked_threshold"]["test"])
    print("\nTEST FP prefixes", fp_analysis["filename_prefix_counts"])
    print("  subgroup", fp_analysis["normal_subgroup_stats"])
    print("\nCONTENT OVERLAP cross-split hashes", content_overlap["n_cross_split_duplicate_hashes"])
    print("PERSON overlap val-test", id_overlap["person_id_overlap"]["val_test"])
    print("PERSON overlap train-test", id_overlap["person_id_overlap"]["train_test"])
    print(f"\nWrote {out_path}")
    print(f"Wrote {dist_fig}")
    print(f"Wrote {fp_fig}")


if __name__ == "__main__":
    main()
