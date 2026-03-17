#!/usr/bin/env python3
"""
Build an offline skill_id map by clustering ego future trajectories.

This script is intentionally independent from training/inference code.
It reads the existing training list JSON (e.g. diffusion_planner_training.json),
loads each corresponding .npz sample, extracts lightweight trajectory features
from ego future trajectory, runs KMeans, and outputs a skill mapping JSON:

  key:  sample relative path (exact string from data_list[idx])
  value: integer skill_id (cluster index)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm


@dataclass
class SkillFeatureConfig:
    dt: float = 0.1  # seconds between future points (nuPlan default is 10Hz)
    eps: float = 1e-6


def _safe_unit(v: np.ndarray, eps: float) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + eps)


def _angle_wrap(theta: np.ndarray) -> np.ndarray:
    """Wrap angle to [-pi, pi]."""
    return (theta + np.pi) % (2 * np.pi) - np.pi


def extract_ego_future_features(
    ego_future: np.ndarray,
    cfg: SkillFeatureConfig,
) -> np.ndarray:
    """
    ego_future: (T, 3) where columns are [x, y, heading] in ego-centric frame.
    Returns: (F,) float feature vector.
    """
    if ego_future.ndim != 2 or ego_future.shape[1] < 3:
        raise ValueError(f"Unexpected ego_future shape: {ego_future.shape}, expected (T, 3+)")

    traj = ego_future[:, :3].astype(np.float64, copy=False)
    x = traj[:, 0]
    y = traj[:, 1]
    heading = traj[:, 2]

    # Handle degenerate trajectories (all zeros or very short)
    T = traj.shape[0]
    if T < 2:
        return np.zeros((18,), dtype=np.float32)

    # Displacements and distances
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx * dx + dy * dy)
    speed = ds / max(cfg.dt, cfg.eps)  # (T-1,)

    # Heading changes
    dtheta = _angle_wrap(np.diff(heading))  # (T-1,)
    total_dtheta = _angle_wrap(heading[-1] - heading[0])

    # Curvature proxy: kappa ~ dtheta / ds
    kappa = dtheta / (ds + cfg.eps)

    # Acceleration proxy from speed
    accel = np.diff(speed) / max(cfg.dt, cfg.eps) if speed.shape[0] >= 2 else np.zeros((0,), dtype=np.float64)

    # Basic geometric stats
    end_dx = float(x[-1] - x[0])
    end_dy = float(y[-1] - y[0])
    end_disp = float(np.hypot(end_dx, end_dy))

    max_lat = float(np.max(np.abs(y)))  # lateral deviation magnitude
    signed_lat_end = float(y[-1])  # sign encodes left/right bias

    # Speed / accel stats
    mean_speed = float(np.mean(speed)) if speed.size else 0.0
    max_speed = float(np.max(speed)) if speed.size else 0.0
    std_speed = float(np.std(speed)) if speed.size else 0.0

    mean_abs_accel = float(np.mean(np.abs(accel))) if accel.size else 0.0
    max_abs_accel = float(np.max(np.abs(accel))) if accel.size else 0.0

    # Turning / curvature stats
    mean_abs_dtheta = float(np.mean(np.abs(dtheta))) if dtheta.size else 0.0
    max_abs_dtheta = float(np.max(np.abs(dtheta))) if dtheta.size else 0.0

    mean_abs_kappa = float(np.mean(np.abs(kappa))) if kappa.size else 0.0
    max_abs_kappa = float(np.max(np.abs(kappa))) if kappa.size else 0.0

    # Direction of travel stability: average direction changes
    vel = np.stack([dx, dy], axis=-1)
    vel_u = _safe_unit(vel, cfg.eps)
    # cosine similarity between consecutive direction vectors
    if vel_u.shape[0] >= 2:
        dir_cos = np.sum(vel_u[1:] * vel_u[:-1], axis=-1)
        mean_dir_cos = float(np.mean(dir_cos))
        min_dir_cos = float(np.min(dir_cos))
    else:
        mean_dir_cos = 1.0
        min_dir_cos = 1.0

    feats = np.array(
        [
            end_disp,
            end_dx,
            end_dy,
            max_lat,
            signed_lat_end,
            float(total_dtheta),
            float(np.sum(np.abs(dtheta))) if dtheta.size else 0.0,
            mean_abs_dtheta,
            max_abs_dtheta,
            mean_speed,
            max_speed,
            std_speed,
            mean_abs_accel,
            max_abs_accel,
            mean_abs_kappa,
            max_abs_kappa,
            mean_dir_cos,
            min_dir_cos,
        ],
        dtype=np.float32,
    )
    return feats


def load_features_from_list(
    data_dir: str,
    data_list_json: str,
    ego_future_key: str,
    cfg: SkillFeatureConfig,
    limit: Optional[int] = None,
) -> Tuple[List[str], np.ndarray]:
    with open(data_list_json, "r", encoding="utf-8") as f:
        rel_paths = json.load(f)
    if not isinstance(rel_paths, list):
        raise ValueError(f"Expected JSON list in {data_list_json}, got {type(rel_paths)}")

    if limit is not None:
        rel_paths = rel_paths[: int(limit)]

    feats: List[np.ndarray] = []
    kept_paths: List[str] = []

    for rel in tqdm(rel_paths, desc="Extract features", unit="sample"):
        npz_path = os.path.join(data_dir, rel)
        try:
            with np.load(npz_path) as npz:
                ego_future = npz[ego_future_key]
            f = extract_ego_future_features(ego_future, cfg)
            feats.append(f)
            kept_paths.append(rel)
        except KeyError as e:
            with np.load(npz_path) as npz:
                keys = list(npz.keys())
            raise KeyError(f"Missing key {e} in {npz_path}. Available keys: {keys}") from e
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Cannot find npz: {npz_path} (from list entry: {rel})") from e

    if not feats:
        raise RuntimeError("No features extracted (empty dataset?)")

    X = np.stack(feats, axis=0)  # (N, F)
    return kept_paths, X


def _kmeanspp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ initialization. Returns centers (k, d)."""
    n, d = X.shape
    centers = np.empty((k, d), dtype=X.dtype)

    # choose first center uniformly
    idx = rng.integers(0, n)
    centers[0] = X[idx]

    # squared distances to nearest center
    dist2 = np.sum((X - centers[0]) ** 2, axis=1)
    for i in range(1, k):
        dist2_sum = float(dist2.sum())
        if dist2_sum <= 1e-12:
            idx = rng.integers(0, n)
        else:
            probs = dist2 / dist2_sum
            idx = rng.choice(n, p=probs)
        centers[i] = X[idx]
        new_dist2 = np.sum((X - centers[i]) ** 2, axis=1)
        dist2 = np.minimum(dist2, new_dist2)

    return centers


def run_kmeans(
    X: np.ndarray,
    num_skills: int,
    seed: int,
    max_iter: int = 200,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    A lightweight numpy KMeans (Lloyd) implementation with k-means++ init.

    Returns:
      labels: (N,)
      centers: (K, F) in the same space as X.
    """
    if num_skills <= 1:
        raise ValueError(f"num_skills must be >= 2, got {num_skills}")
    n, d = X.shape
    if n < num_skills:
        raise ValueError(f"num_samples ({n}) < num_skills ({num_skills})")

    rng = np.random.default_rng(seed)
    centers = _kmeanspp_init(X, num_skills, rng)

    prev_inertia = None
    labels = np.zeros((n,), dtype=np.int64)

    for _ in range(max_iter):
        # assign
        # distances (n, k)
        dist2 = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dist2, axis=1)

        # update
        new_centers = np.empty_like(centers)
        for k in range(num_skills):
            mask = labels == k
            if not np.any(mask):
                # empty cluster: re-seed to a random point
                new_centers[k] = X[rng.integers(0, n)]
            else:
                new_centers[k] = X[mask].mean(axis=0)

        centers = new_centers
        inertia = float(np.sum(dist2[np.arange(n), labels]))

        if prev_inertia is not None:
            rel_improve = (prev_inertia - inertia) / (abs(prev_inertia) + 1e-12)
            if rel_improve < tol:
                break
        prev_inertia = inertia

    return labels.astype(int), centers.astype(np.float32)


def standardize_features(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    Xn = (X - mean) / std
    return Xn.astype(np.float32), mean.squeeze(0).astype(np.float32), std.squeeze(0).astype(np.float32)


def save_outputs(
    out_dir: str,
    mapping_name: str,
    rel_paths: List[str],
    labels: np.ndarray,
    centers: Optional[np.ndarray] = None,
    stats: Optional[Dict[str, int]] = None,
    meta: Optional[dict] = None,
) -> str:
    os.makedirs(out_dir, exist_ok=True)

    mapping: Dict[str, int] = {p: int(l) for p, l in zip(rel_paths, labels)}
    mapping_path = os.path.join(out_dir, mapping_name)
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2, sort_keys=True)

    if centers is not None:
        centers_path = os.path.join(out_dir, "skill_cluster_centers.json")
        with open(centers_path, "w", encoding="utf-8") as f:
            json.dump({"centers": centers.tolist()}, f, ensure_ascii=False, indent=2)

    if stats is not None:
        stats_path = os.path.join(out_dir, "skill_cluster_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, sort_keys=True)

    if meta is not None:
        meta_path = os.path.join(out_dir, "skill_map_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2, sort_keys=True)

    return mapping_path


def main() -> None:
    p = argparse.ArgumentParser(description="Build offline skill_id mapping by clustering ego future trajectories.")
    p.add_argument("--data_dir", type=str, required=True, help="Directory containing .npz samples (same as training --train_set).")
    p.add_argument("--data_list", type=str, required=True, help="JSON list of sample file names/relative paths (e.g. diffusion_planner_training.json).")
    p.add_argument("--out_dir", type=str, default="preprocess_outputs", help="Output directory for skill_map.json and optional files.")
    p.add_argument("--out_name", type=str, default="skill_map.json", help="Output mapping filename.")
    p.add_argument("--ego_future_key", type=str, default="ego_agent_future", help="Key name in npz for ego future trajectory.")
    p.add_argument("--num_skills", type=int, default=8, help="Number of clusters (skills).")
    p.add_argument("--seed", type=int, default=3407, help="Random seed for clustering.")
    p.add_argument("--dt", type=float, default=0.1, help="Seconds between future trajectory points (default 0.1 for 10Hz).")
    p.add_argument("--limit", type=int, default=None, help="Optionally limit number of samples for quick runs.")
    p.add_argument("--save_centers", action="store_true", help="Save cluster centers to JSON.")
    p.add_argument("--save_stats", action="store_true", help="Save per-cluster counts to JSON.")
    p.add_argument("--no_standardize", action="store_true", help="Disable feature standardization before KMeans.")
    args = p.parse_args()

    cfg = SkillFeatureConfig(dt=float(args.dt))
    rel_paths, X = load_features_from_list(
        data_dir=args.data_dir,
        data_list_json=args.data_list,
        ego_future_key=args.ego_future_key,
        cfg=cfg,
        limit=args.limit,
    )

    if args.no_standardize:
        Xn = X
        feat_mean = None
        feat_std = None
    else:
        Xn, feat_mean, feat_std = standardize_features(X)

    labels, centers = run_kmeans(Xn, num_skills=int(args.num_skills), seed=int(args.seed))

    stats = None
    if args.save_stats:
        uniq, cnt = np.unique(labels, return_counts=True)
        stats = {str(int(k)): int(v) for k, v in zip(uniq, cnt)}

    meta = {
        "data_dir": os.path.abspath(args.data_dir),
        "data_list": os.path.abspath(args.data_list),
        "ego_future_key": args.ego_future_key,
        "num_skills": int(args.num_skills),
        "seed": int(args.seed),
        "dt": float(args.dt),
        "standardized": (not args.no_standardize),
        "feature_dim": int(X.shape[1]),
        "num_samples": int(X.shape[0]),
        "feature_mean": feat_mean.tolist() if feat_mean is not None else None,
        "feature_std": feat_std.tolist() if feat_std is not None else None,
    }

    mapping_path = save_outputs(
        out_dir=args.out_dir,
        mapping_name=args.out_name,
        rel_paths=rel_paths,
        labels=labels,
        centers=centers if args.save_centers else None,
        stats=stats,
        meta=meta,
    )

    print(f"Saved skill map to: {mapping_path}")
    if args.save_centers:
        print(f"Saved centers to: {os.path.join(args.out_dir, 'skill_cluster_centers.json')}")
    if args.save_stats:
        print(f"Saved stats to: {os.path.join(args.out_dir, 'skill_cluster_stats.json')}")


if __name__ == "__main__":
    main()

