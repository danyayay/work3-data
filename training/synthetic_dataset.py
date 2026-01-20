"""Synthetic egocentric dataset for quick experiments and tests.

Creates class-conditioned sequences for visual features, head pose and gaze
so the model can be exercised end-to-end without external data.
"""

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticEgocentricDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 1000,
        T: int = 8,
        visual_dim: int = 512,
        pose_dim: int = 6,
        gaze_dim: int = 2,
        motion_dim: int = 2,
        goal_dim: int = 3,
        n_classes: int = 5,
        seed: int = 0,
        pose_scaler: dict = None,
        gaze_scaler: dict = None,
        motion_scaler: dict = None,
        goal_dist_scaler: dict = None,
    ):
        rng = np.random.RandomState(seed)
        self.num_samples = int(num_samples)
        self.T = int(T)
        self.visual_dim = int(visual_dim)
        self.pose_dim = int(pose_dim)
        self.gaze_dim = int(gaze_dim)
        self.n_classes = int(n_classes)

        # class centroids used to generate correlated sequences
        self.visual_centroids = rng.normal(size=(self.n_classes, self.visual_dim)).astype(np.float32)
        self.pose_centroids = (
            rng.normal(size=(self.n_classes, self.pose_dim)).astype(np.float32)
            if self.pose_dim > 0
            else None
        )
        self.gaze_centroids = (
            rng.normal(size=(self.n_classes, self.gaze_dim)).astype(np.float32)
            if self.gaze_dim > 0
            else None
        )
        self.motion_dim = int(motion_dim)
        self.motion_centroids = (
            rng.normal(size=(self.n_classes, self.motion_dim)).astype(np.float32)
            if self.motion_dim > 0
            else None
        )
        self.goal_dim = int(goal_dim)
        if self.goal_dim > 0:
            # goal: distance and angle per-class centroids
            self.goal_angle_centroids = rng.uniform(low=-np.pi, high=np.pi, size=(self.n_classes,)).astype(np.float32)
            self.goal_dist_centroids = rng.uniform(low=0.5, high=2.0, size=(self.n_classes,)).astype(np.float32)
        else:
            self.goal_angle_centroids = None
            self.goal_dist_centroids = None

        # pre-sample labels
        self.labels = rng.randint(0, self.n_classes, size=(self.num_samples,))
        # optional scalers (dicts with 'mean' and 'std' numpy arrays)
        self.pose_scaler = pose_scaler
        self.gaze_scaler = gaze_scaler
        self.motion_scaler = motion_scaler
        self.goal_dist_scaler = goal_dist_scaler

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        label = int(self.labels[idx])
        
        # visual: centroid + per-timestep gaussian noise
        base_v = self.visual_centroids[label]
        visual = base_v[None, :].repeat(self.T, axis=0) + np.random.normal(scale=0.5, size=(self.T, self.visual_dim))
        visual = visual.astype(np.float32)

        if self.pose_dim > 0:
            base_p = self.pose_centroids[label]
            pose = base_p[None, :].repeat(self.T, axis=0) + np.random.normal(scale=0.1, size=(self.T, self.pose_dim))
            pose = pose.astype(np.float32)
        else:
            pose = torch.empty(self.T, 0, dtype=torch.float32)

        if self.gaze_dim > 0:
            base_g = self.gaze_centroids[label]
            gaze = base_g[None, :].repeat(self.T, axis=0) + np.random.normal(scale=0.05, size=(self.T, self.gaze_dim))
            gaze = gaze.astype(np.float32)
        else:
            gaze = torch.empty(self.T, 0, dtype=torch.float32)

        if self.motion_dim > 0:
            base_m = self.motion_centroids[label]
            motion = base_m[None, :].repeat(self.T, axis=0) + np.random.normal(scale=0.02, size=(self.T, self.motion_dim))
            motion = motion.astype(np.float32)
        else:
            motion = torch.empty(self.T, 0, dtype=torch.float32)

        if self.goal_dim > 0:
            base_angle = self.goal_angle_centroids[label]
            base_d = self.goal_dist_centroids[label]
            angles = base_angle + np.random.normal(scale=0.05, size=(self.T,))
            dvals = base_d + np.random.normal(scale=0.1, size=(self.T,))
            sinb = np.sin(angles).astype(np.float32)
            cosb = np.cos(angles).astype(np.float32)
            goal = np.stack([dvals.astype(np.float32), sinb, cosb], axis=1)
            goal = goal.astype(np.float32)
        else:
            goal = torch.empty(self.T, 0, dtype=torch.float32)

        # apply optional scalers
        def _apply_scaler(arr, scaler):
            if arr is None or scaler is None:
                return arr
            mean = scaler.get('mean')
            std = scaler.get('std')
            if mean is None or std is None:
                return arr
            return (arr - mean) / (std + 1e-6)

        if isinstance(pose, np.ndarray) and self.pose_scaler is not None:
            try:
                pose = _apply_scaler(pose, self.pose_scaler)
            except Exception:
                pass
        if isinstance(gaze, np.ndarray) and self.gaze_scaler is not None:
            try:
                gaze = _apply_scaler(gaze, self.gaze_scaler)
            except Exception:
                pass
        if isinstance(motion, np.ndarray) and self.motion_scaler is not None:
            try:
                motion = _apply_scaler(motion, self.motion_scaler)
            except Exception:
                pass
        if isinstance(goal, np.ndarray) and self.goal_dist_scaler is not None:
            try:
                g = goal.copy()
                g[:, 0:1] = _apply_scaler(g[:, 0:1], self.goal_dist_scaler)
                goal = g
            except Exception:
                pass

        # convert to torch
        visual = torch.from_numpy(visual)
        pose = torch.from_numpy(pose) if isinstance(pose, np.ndarray) else torch.empty(self.T, 0, dtype=torch.float32)
        gaze = torch.from_numpy(gaze) if isinstance(gaze, np.ndarray) else torch.empty(self.T, 0, dtype=torch.float32)
        motion = torch.from_numpy(motion) if isinstance(motion, np.ndarray) else torch.empty(self.T, 0, dtype=torch.float32)
        goal = torch.from_numpy(goal) if isinstance(goal, np.ndarray) else torch.empty(self.T, 0, dtype=torch.float32)

        return visual, pose, gaze, motion, goal, label


def collate_fn(batch):
    vis, pose, gaze, motion, goal, labels = zip(*batch)
    vis = torch.stack(vis, dim=0)
    pose = torch.stack(pose, dim=0) if pose[0].numel() > 0 else None
    gaze = torch.stack(gaze, dim=0) if gaze[0].numel() > 0 else None
    motion = torch.stack(motion, dim=0) if motion[0].numel() > 0 else None
    goal = torch.stack(goal, dim=0) if goal[0].numel() > 0 else None
    labels = torch.tensor(labels, dtype=torch.long)
    return vis, pose, gaze, motion, goal, labels
