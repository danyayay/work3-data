"""Lightweight pipeline test that runs a tiny end-to-end training.

This test is intended as a smoke test. It trains for a couple of
epochs on synthetic data and asserts the pipeline runs and returns a
reasonable validation accuracy (> random chance).
"""

from training.train_fusion import train


def test_pipeline_smoke():
    cfg = {
        "num_samples": 200,
        "T": 8,
        "visual_dim": 128,
        "pose_dim": 6,
        "gaze_dim": 2,
        "n_classes": 4,
        "batch_size": 32,
        "epochs": 2,
        "d_model": 64,
        "lr": 1e-3,
    }
    result = train(cfg, work_dir=None)
    acc = result.get("best_val_acc", 0.0)
    # random chance is 1/n_classes; require modest improvement
    assert acc >= (1.0 / cfg["n_classes"]) + 0.05


if __name__ == "__main__":
    test_pipeline_smoke()
    print("Smoke test passed")
