"""Run evaluation end-to-end (TODO)."""
from __future__ import annotations

import argparse
import json
import ground_truth
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from eval.ground_truth import build_ground_truth
from eval.metrics import average_precision, pr_curve, precision_recall_f1, roc_curve
from eval.sweeps import run_threshold_sweep_from_scores
from src.config import PLOTS_DIR, RESULTS_DIR
from src.dataset.loader import iter_scans, load_scan
from src.dataset.preprocess import downsample, filter_points, normalize
from src.descriptors.your_descriptor import YourDescriptor
from src.matching.similarity import cosine_similarity

def _load_pose(pose_entry: str, fallback: np.ndarray) -> np.ndarray:
    if not pose_entry:
        return fallback
    tokens = pose_entry.replace(",", " ").split()
    if len(tokens) >= 2:
        values = [float(value) for value in tokens[:2]]
        if len(values) == 2:
            values.append(0.0)
        return np.array(values, dtype=float)
    pose_path = Path(pose_entry)
    if pose_path.exists():
        pose_values = np.loadtxt(pose_path)
        pose_values = np.asarray(pose_values).reshape(-1)
        if pose_values.size >= 3:
            return pose_values[:3]
        if pose_values.size == 2:
            return np.array([pose_values[0], pose_values[1], 0.0], dtype=float)
    return fallback

def _generate_synthetic_dataset(num_scans: int, seed: int):
    rng = np.random.default_rng(seed)
    poses = np.cumsum(rng.normal(scale = 2.0, size =(num_scans, 3)), axis=0)
    scans = []
    for pose in poses:
        points = rng.normal(size =(1024, 3)) + pose[:3] * 0.05
        scans.append(points)
        poses.append(pose)
    return scans, np.vstack(poses)

def _compute_descriptors(scans, descriptor: YourDescriptor):
    descriptors = []
    for points in scans:
        filtered = filter_points(points)
        sampled = downsample(filtered)
        normalized = normalize(sampled)
        descriptors.append(descriptor.compute(normalized))
    return np.vstack(descriptors)

def _score_pairs(descriptors, similarity_func):
    scores = []
    labels = []
    num = descriptors.shape[0]
    for i in range(num):
        for j in range(num):
            if i == j:
                continue
            labels.append(int(ground_truth[i, j]))
            scores.append(cosine_similarity(descriptors[i], descriptors[j]))
    return np.array(labels), np.array(scores)

def _load_dataset(split, num_scans, seed):
    return _generate_synthetic_dataset(num_scans, seed)

def main():
    """Entry point for evaluation script."""
    parser = argparse.ArgumentParser(description="Run place recognition evaluation.")
    parser.add_argument("--split", default="train", help="Dataset split name.")
    parser.add_argument("--distance-threshold", type=int, default=200)
    parser.add_argument("--num-scans", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    scans, poses = _load_dataset(args.split, args.num_scans, args.seed)
    descriptor = YourDescriptor()
    descriptors = _compute_descriptors(scans, descriptor)

    ground_truth = build_ground_truth(poses, agrs.distance_threshold)
    y_true, y_scores = _score_pairs(descriptors, ground_truth)
    y_pred = (y_scores >= args.threshold).astype(int)

    precision, recall, f1 = precision_recall_f1(y_true, y_pred)
    ap = average_precision(y_true, y_scores)
    recalls, precisions, _ = pr_curve(y_true, y_scores)
    fprs, tprs, _ = roc_curve(y_true, y_scores)
    sweep_metrics = run_threshold_sweep_from_scores(y_true, y_scores)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = RESULTS_DIR / "eval_metrics.json"
    metrics_payload = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "average_precision": ap,
        "threshold_sweep": sweep_metrics,
        "num_pairs": int(y_true.size),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    pr_path = PLOTS_DIR / "pr_curve.png"
    plt.figure()
    plt.plot(recalls, precisions, marker="o")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()

    print(f"Saved metrics to {metrics_path}")
    print(f"Saved PR curve to {pr_path}")
    print(f"Saved ROC curve to {roc_path}")


if __name__ == "__main__":
    main()
