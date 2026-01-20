"""Evaluation script for saved fusion model checkpoints."""

import argparse
import torch
from torch.utils.data import DataLoader

from models.fusion import EgocentricFusionTransformer
from training.synthetic_dataset import SyntheticEgocentricDataset, collate_fn


def load_and_eval(ckpt_path: str, dataset_args: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = SyntheticEgocentricDataset(**dataset_args)
    loader = DataLoader(ds, batch_size=64, collate_fn=collate_fn)

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    model = EgocentricFusionTransformer(
        visual_dim=cfg.get("visual_dim", dataset_args.get("visual_dim", 512)),
        pose_dim=cfg.get("pose_dim", dataset_args.get("pose_dim", 6)),
        gaze_dim=cfg.get("gaze_dim", dataset_args.get("gaze_dim", 2)),
        motion_dim=cfg.get("motion_dim", dataset_args.get("motion_dim", 2)),
        goal_dim=cfg.get("goal_dim", dataset_args.get("goal_dim", 3)),
        d_model=cfg.get("d_model", 256),
        n_classes=cfg.get("n_classes", dataset_args.get("n_classes", 5)),
    )
    model.load_state_dict(ckpt["state_dict"]) if "state_dict" in ckpt else model.load_state_dict(ckpt)
    model.to(device)

    # simple evaluation loop
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for vis, pose, gaze, motion, goal, labels in loader:
            vis = vis.to(device)
            labels = labels.to(device)
            pose = pose.to(device) if pose is not None else None
            gaze = gaze.to(device) if gaze is not None else None
            motion = motion.to(device) if motion is not None else None
            goal = goal.to(device) if goal is not None else None
            logits = model(vis, head_pose=pose, gaze=gaze, motion=motion, goal=goal)
            preds = logits.argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total += labels.size(0)

    acc = correct / total
    print(f"Eval accuracy: {acc:.4f}")
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()
    load_and_eval(args.ckpt, {"num_samples": 200, "T": 8, "visual_dim": 512, "pose_dim": 6, "gaze_dim": 2, "n_classes": 5})
