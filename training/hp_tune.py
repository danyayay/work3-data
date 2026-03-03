"""Small hyperparameter tuning / grid search helper.

Runs the train() function from train_fusion with a tiny grid to find a
good LR / d_model pair. Designed for quick local experiments.
"""

from typing import List, Dict, Any

from training.train_fusion import train


def grid_search(lrs: List[float], d_models: List[int], base_config: Dict[str, Any]):
    best = {"acc": 0.0, "config": None}
    for lr in lrs:
        for d in d_models:
            cfg = base_config.copy()
            cfg.update({"lr": lr, "d_model": d, "epochs": cfg.get("epochs", 2)})
            print(f"Running config lr={lr} d_model={d}")
            res = train(cfg, work_dir=None)
            acc = res.get("best_val_acc", 0.0)
            print(f"Result acc={acc:.4f}")
            if acc > best["acc"]:
                best = {"acc": acc, "config": cfg}
    print("Best:", best)
    return best


if __name__ == "__main__":
    base = {"num_samples": 300, "batch_size": 32, "visual_dim": 512, "pose_dim": 6, "gaze_dim": 2, "n_classes": 5, "epochs": 2}
    grid_search([1e-3, 5e-4], [64, 128], base)
