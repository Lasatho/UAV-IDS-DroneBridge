"""
Evaluation entry point.

    python evaluate.py --checkpoint checkpoints/<run>/best.pt [overrides...]

Two modes, selected automatically:

1. **Unlabeled (current state)** — no attack windows exist yet. Computes
   anomaly scores on the normal test split, reports the score
   distribution, and calibrates candidate thresholds (percentiles of
   normal scores). The resulting threshold corresponds directly to an
   expected FPR on normal data (e.g. 99th percentile ≈ 1% FPR).

2. **Labeled (after WP1-WP4)** — window labels available. Computes the
   full metric set: precision, recall, F1, AUC-ROC, AUC-PR, FPR at the
   calibrated threshold, and detection latency.

Label integration (attack data) requires mapping phase_labels timestamps
(epoch seconds) onto telemetry timestamps; see `load_window_labels`.

Outputs land next to the checkpoint: scores.npz, eval_report.json.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from data import build_datasets, get_num_features
from models import build_model
from config import set_seed

logger = logging.getLogger("evaluate")


# ---------------------------------------------------------------------------
# Metrics (numpy-only, no sklearn dependency)
# ---------------------------------------------------------------------------

def auc_roc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC-ROC via the rank-sum (Mann-Whitney U) formulation."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    u = ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def auc_pr(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the precision-recall curve (step-wise integration)."""
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    tp = np.cumsum(labels_sorted)
    fp = np.cumsum(1 - labels_sorted)
    precision = tp / (tp + fp)
    recall = tp / labels.sum()
    # Prepend (recall=0, precision=1) anchor
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.sum(np.diff(recall) * precision[1:]))


def threshold_metrics(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> dict[str, float]:
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "threshold": float(threshold),
        "precision": precision, "recall": recall, "f1": f1, "fpr": fpr,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def detection_latency(
    scores: np.ndarray,
    labels: np.ndarray,
    window_starts: np.ndarray,
    mission_ids: np.ndarray,
    threshold: float,
) -> float:
    """Mean latency (in timesteps) from attack onset to first detection.

    Per mission: onset = earliest window start with label 1; detection =
    earliest window start with label 1 AND score >= threshold. Missions
    where the attack is never detected are excluded (recall covers that).
    """
    latencies = []
    for mid in np.unique(mission_ids):
        m = mission_ids == mid
        attacked = m & (labels == 1)
        if not attacked.any():
            continue
        onset = window_starts[attacked].min()
        detected = attacked & (scores >= threshold)
        if detected.any():
            latencies.append(float(window_starts[detected].min() - onset))
    return float(np.mean(latencies)) if latencies else float("nan")


# ---------------------------------------------------------------------------
# Label loading (attack data integration point)
# ---------------------------------------------------------------------------

def load_window_labels(dataset, phase_labels_dir: Path) -> np.ndarray | None:
    """Window-level attack labels from phase_labels JSONs.

    Returns None if no mission in the split contains an attack — the
    unlabeled evaluation mode is used then.

    NOTE (WP1-WP4 integration): phase_labels store attack_start/end as
    epoch seconds, telemetry uses a monotonic host clock (ns). The
    timebase mapping must be defined when the attacker node lands; until
    then missions with attack_type != "none" raise to prevent silently
    wrong labels.
    """
    any_attack = False
    for mid in dataset.mission_ids:
        meta_path = phase_labels_dir / f"{mid}.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get("attack_type", "none") != "none":
            any_attack = True
            break
    if not any_attack:
        return None
    raise NotImplementedError(
        "Attack missions present, but the phase_labels->telemetry timebase "
        "mapping is not implemented yet (epoch seconds vs. monotonic ns). "
        "Define it before evaluating on attack data."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_scores(
    model, dataset, batch_size: int, device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Anomaly scores for every window of a dataset.

    Returns (scores, mission_ids, window_starts), all aligned (N,).
    """
    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )
    scores = []
    for batch in loader:
        scores.append(model.anomaly_score(batch.to(device)).cpu().numpy())
    scores = np.concatenate(scores)

    meta = [dataset.window_meta(i) for i in range(len(dataset))]
    mission_ids = np.array([m[0] for m in meta])
    window_starts = np.array([m[1] for m in meta], dtype=np.int64)
    return scores, mission_ids, window_starts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    set_seed(int(cfg["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rebuild data exactly as in training (same splits via persisted seed).
    datasets, _ = build_datasets(cfg["data"])
    dataset = datasets[args.split]

    name = cfg["model"]["name"]
    model = build_model(
        name,
        num_features=get_num_features(cfg["data"].get("features")),
        window_length=int(cfg["data"]["window_length"]),
        model_cfg=cfg["model"][name],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    logger.info("Loaded %s from %s (epoch %d, val_loss %.5f)",
                name, ckpt_path, ckpt["epoch"], ckpt["val_loss"])

    scores, mission_ids, window_starts = compute_scores(
        model, dataset, args.batch_size, device
    )
    logger.info("%d windows scored on '%s' split", len(scores), args.split)

    # Threshold calibration on normal-score percentiles.
    percentiles = [90.0, 95.0, 99.0, 99.5, 99.9]
    thresholds = {f"p{p:g}": float(np.percentile(scores, p)) for p in percentiles}

    report: dict = {
        "checkpoint": str(ckpt_path),
        "model": name,
        "split": args.split,
        "n_windows": int(len(scores)),
        "score_stats": {
            "mean": float(scores.mean()), "std": float(scores.std()),
            "min": float(scores.min()), "max": float(scores.max()),
            "median": float(np.median(scores)),
        },
        "calibrated_thresholds": thresholds,
    }

    phase_labels_dir = Path(cfg["data"]["dataset_dir"]) / "phase_labels"
    labels = load_window_labels(dataset, phase_labels_dir)

    if labels is None:
        logger.info("No attack missions in split — unlabeled mode "
                    "(score distribution + threshold calibration only).")
        report["mode"] = "unlabeled"
    else:
        report["mode"] = "labeled"
        report["auc_roc"] = auc_roc(scores, labels)
        report["auc_pr"] = auc_pr(scores, labels)
        thr = thresholds["p99"]
        report["metrics_at_p99"] = threshold_metrics(scores, labels, thr)
        report["detection_latency_steps"] = detection_latency(
            scores, labels, window_starts, mission_ids, thr
        )

    out_dir = ckpt_path.parent
    np.savez(
        out_dir / f"scores_{args.split}.npz",
        scores=scores, mission_ids=mission_ids, window_starts=window_starts,
    )
    (out_dir / f"eval_report_{args.split}.json").write_text(
        json.dumps(report, indent=2)
    )
    logger.info("Report written: %s", out_dir / f"eval_report_{args.split}.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()