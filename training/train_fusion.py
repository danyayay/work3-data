"""Training script for the EgocentricFusionTransformer.

Provides an importable train() for programmatic experiments and a CLI
for quick runs. Uses the synthetic dataset by default.
"""

import argparse
import os
from typing import Dict, Any, Optional, Tuple

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from models.fusion import EgocentricFusionTransformer, SimpleCNNBackbone
from models.fusion_slim import SlimFusionTransformer
from training.synthetic_dataset import SyntheticEgocentricDataset, collate_fn as synth_collate
from training.scalers import compute_stats_from_dataloader, save_stats
from torch.utils.data import Subset

# optional VR dataset import (may require opencv/pandas); import lazily below
try:
    from training.vr_dataset import VRDataset, collate_fn as vr_collate  # type: ignore
except Exception:
    VRDataset = None
    vr_collate = None


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, backbone: Optional[nn.Module] = None):
    """Run evaluation and return (avg_loss, accuracy, class_metrics)

    class_metrics is a dict with keys 'precision','recall','f1' mapping to lists indexed by class id.
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for vis, pose, gaze, motion, goal, labels in loader:
            vis = vis.to(device)
            labels = labels.to(device)
            pose = pose.to(device) if pose is not None else None
            gaze = gaze.to(device) if gaze is not None else None
            motion = motion.to(device) if motion is not None else None
            goal = goal.to(device) if goal is not None else None
            if backbone is not None:
                # vis expected (B,T,H,W,3) -> backbone returns (B,T,feat)
                vis = backbone(vis)
            logits = model(vis, head_pose=pose, gaze=gaze, motion=motion, goal=goal)
            loss = loss_fn(logits, labels)
            total_loss += float(loss.item()) * labels.size(0)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            total += labels.size(0)

    if total == 0:
        raise RuntimeError("No samples in evaluation loader")

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    accuracy = float((all_preds == all_labels).sum()) / all_labels.shape[0]
    avg_loss = total_loss / total

    # determine number of classes from data
    if all_labels.size == 0:
        n_classes = 0
    else:
        n_classes = int(max(all_labels.max(), all_preds.max()) + 1)

    precision_list = []
    recall_list = []
    f1_list = []
    support_list = []
    sum_tp = 0
    sum_fp = 0
    sum_fn = 0
    for c in range(n_classes):
        tp = int(((all_preds == c) & (all_labels == c)).sum())
        fp = int(((all_preds == c) & (all_labels != c)).sum())
        fn = int(((all_preds != c) & (all_labels == c)).sum())
        support = int((all_labels == c).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precision_list.append(prec)
        recall_list.append(rec)
        f1_list.append(f1)
        support_list.append(support)
        sum_tp += tp
        sum_fp += fp
        sum_fn += fn

    # macro averages (unweighted)
    macro_precision = float(np.mean(precision_list)) if len(precision_list) > 0 else 0.0
    macro_recall = float(np.mean(recall_list)) if len(recall_list) > 0 else 0.0
    macro_f1 = float(np.mean(f1_list)) if len(f1_list) > 0 else 0.0

    # micro averages (global counts)
    micro_precision = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0.0
    micro_recall = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

    class_metrics = {
        "precision": precision_list,
        "recall": recall_list,
        "f1": f1_list,
        "support": support_list,
        "macro": {"precision": macro_precision, "recall": macro_recall, "f1": macro_f1},
        "micro": {"precision": micro_precision, "recall": micro_recall, "f1": micro_f1},
    }
    return avg_loss, accuracy, class_metrics


def train(
    config: Dict[str, Any],
    work_dir: Optional[str] = None,
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset selection and visual encoder
    dataset_type = config.get("dataset_type", "synthetic")
    visual_encoder = config.get("visual_encoder", config.get("use_backbone", "pixels"))
    # visual_encoder in {"pixels","simple","clip"}
    if visual_encoder not in ("pixels", "simple", "clip"):
        raise ValueError("visual_encoder must be one of 'pixels','simple','clip'")
    if dataset_type == "vr":
        if VRDataset is None:
            raise RuntimeError("VRDataset not available - missing optional dependencies (cv2/pandas) or import error")
        # import split_person_groups lazily so scripts that only use synthetic data
        # don't require OpenCV/pandas at module import time
        from training.vr_dataset import split_person_groups
        data_split = split_person_groups(config["annotations_path"], config.get('seed'))

        train_ds = VRDataset(
            annotations_path=config["annotations_path"],
            videos_dir=config["videos_dir"],
            vrdata_dir=config.get("vrdata_dir", None),
            T=config.get("T", 8),
            frame_size=tuple(config.get("frame_size", (32, 32))),
            pose_cols=config.get("pose_cols", None),
            gaze_cols=config.get("gaze_cols", None),
            motion_cols=config.get("motion_cols", None),
            return_raw_frames=(visual_encoder != "pixels"),
            person_ids=data_split["train_persons"],
        )
        val_ds = VRDataset(
            annotations_path=config["annotations_path"],
            videos_dir=config["videos_dir"],
            vrdata_dir=config.get("vrdata_dir", None),
            T=config.get("T", 8),
            frame_size=tuple(config.get("frame_size", (32, 32))),
            pose_cols=config.get("pose_cols", None),
            gaze_cols=config.get("gaze_cols", None),
            motion_cols=config.get("motion_cols", None),
            return_raw_frames=(visual_encoder != "pixels"),
            person_ids=data_split["val_persons"],
        )
        test_ds = VRDataset(
            annotations_path=config["annotations_path"],
            videos_dir=config["videos_dir"],
            vrdata_dir=config.get("vrdata_dir", None),
            T=config.get("T", 8),
            frame_size=tuple(config.get("frame_size", (32, 32))),
            pose_cols=config.get("pose_cols", None),
            gaze_cols=config.get("gaze_cols", None),
            motion_cols=config.get("motion_cols", None),
            return_raw_frames=(visual_encoder != "pixels"),
            person_ids=data_split["test_persons"],
        )
        collate = vr_collate
        # visual_dim is flattened frame size (H*W*3)
        if visual_encoder == "pixels":
            visual_dim = train_ds.frame_size[0] * train_ds.frame_size[1] * 3
        else:
            visual_dim = config.get("backbone_out_dim", 256)
        # train_ds[0] -> (vis, pose, gaze, motion, label)
        pose_dim = config.get("pose_dim", train_ds[0][1].shape[1] if train_ds[0][1].numel() > 0 else 0)
        gaze_dim = config.get("gaze_dim", train_ds[0][2].shape[1] if train_ds[0][2].numel() > 0 else 0)
        motion_dim = config.get("motion_dim", train_ds[0][3].shape[1] if train_ds[0][3].numel() > 0 else 0)
        goal_dim = config.get("goal_dim", train_ds[0][4].shape[1] if train_ds[0][4].numel() > 0 else 0)
    else:
        dataset = SyntheticEgocentricDataset(
            num_samples=config.get("num_samples", 500),
            T=config.get("T", 8),
            visual_dim=config.get("visual_dim", 512),
            pose_dim=config.get("pose_dim", 6),
            gaze_dim=config.get("gaze_dim", 2),
            motion_dim=config.get("motion_dim", 2),
            goal_dim=config.get("goal_dim", 3),
            n_classes=config.get("n_classes", 5),
            seed=config.get("seed", 0),
        )
        collate = synth_collate

        visual_dim = config.get("visual_dim", 512)
        pose_dim = config.get("pose_dim", 6)
        gaze_dim = config.get("gaze_dim", 2)
        motion_dim = config.get("motion_dim", 2)
        goal_dim = config.get("goal_dim", 3)

        val_frac = config.get("val_frac", 0.1)
        test_frac = config.get("test_frac", 0.3)
        n_val = int(len(dataset) * val_frac)
        n_test = int(len(dataset) * test_frac)
        n_train = len(dataset) - n_val - n_test
        train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])
    
    # data loaders
    train_loader = DataLoader(train_ds, batch_size=config.get("batch_size", 32), shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=config.get("batch_size", 64), shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=config.get("batch_size", 64), shuffle=False, collate_fn=collate)

    # Fit per-modality scalers on the training set and save them. This avoids data leakage
    # since we compute statistics only from train_loader. The returned stats is a dict
    # like {'pose': {'mean':..., 'std':...}, 'gaze':..., 'motion':..., 'goal_dist':...}
    work_dir = work_dir or config.get("work_dir", None)
    try:
        stats = compute_stats_from_dataloader(train_loader)
        if work_dir is not None:
            save_stats(stats, work_dir)

        # helper to attach stats to dataset or Subset.dataset
        def _attach_stats(ds_obj, stats_dict):
            base = ds_obj.dataset if isinstance(ds_obj, Subset) else ds_obj
            if 'pose' in stats_dict:
                setattr(base, 'pose_scaler', stats_dict['pose'])
            if 'gaze' in stats_dict:
                setattr(base, 'gaze_scaler', stats_dict['gaze'])
            if 'motion' in stats_dict:
                setattr(base, 'motion_scaler', stats_dict['motion'])
            if 'goal_dist' in stats_dict:
                setattr(base, 'goal_dist_scaler', stats_dict['goal_dist'])

        _attach_stats(train_ds, stats)
        _attach_stats(val_ds, stats)
        _attach_stats(test_ds, stats)
    except Exception:
        # if stats computation fails, continue without scalers
        stats = {}

    # model selection: support the original fusion and a slim variant
    model_type = config.get("model", "fusion")
    if model_type == "fusion":
        model = EgocentricFusionTransformer(
            visual_dim=visual_dim,
            pose_dim=pose_dim,
            gaze_dim=gaze_dim,
            motion_dim=locals().get('motion_dim', 0),
            goal_dim=locals().get('goal_dim', 0),
            d_model=config.get("d_model", 256),
            nhead=config.get("nhead", 8),
            num_layers=config.get("num_layers", 4),
            dim_feedforward=config.get("dim_feedforward", 512),
            dropout=config.get("dropout", 0.1),
            n_classes=config.get("n_classes", 5),
            max_time_steps=config.get("max_time_steps", 64),
        ).to(device)
    elif model_type == "slim":
        model = SlimFusionTransformer(
            visual_dim=visual_dim,
            pose_dim=pose_dim,
            motion_dim=motion_dim,
            goal_dim=goal_dim,
            d_model=config.get("d_model", 128),
            nhead=config.get("nhead", 4),
            num_layers=config.get("num_layers", 3),
            dim_feedforward=config.get("dim_feedforward", 256),
            dropout=config.get("dropout", 0.1),
            n_classes=config.get("n_classes", 5),
            max_time_steps=config.get("max_time_steps", 64),
            predict_future_steps=config.get("predict_future_steps", 0),
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # optional backbone for raw frames: SimpleCNN or CLIP
    backbone = None
    if visual_encoder == "simple":
        backbone = SimpleCNNBackbone(in_ch=3, out_dim=config.get("backbone_out_dim", 256), hidden=config.get("backbone_hidden", 64)).to(device)
    elif visual_encoder == "clip":
        # lazy import CLIPBackbone from models.fusion
        try:
            from models.fusion import CLIPBackbone
        except Exception:
            raise RuntimeError("CLIPBackbone not available; ensure 'clip' package is installed or CLIP wrapper is present")
        backbone = CLIPBackbone(model_name=config.get("clip_model", "ViT-B/32"), device=device, freeze=config.get("freeze_clip", False))

    if backbone is not None:
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(backbone.parameters()), lr=config.get("lr", 1e-4), weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 1e-4), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    epochs = int(config.get("epochs", 3))

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_samples = 0
        for vis, pose, gaze, motion, goal, labels in train_loader:
            vis = vis.to(device)
            labels = labels.to(device)
            pose = pose.to(device) if pose is not None else None
            gaze = gaze.to(device) if gaze is not None else None
            motion = motion.to(device) if motion is not None else None
            goal = goal.to(device) if goal is not None else None

            optimizer.zero_grad()
            if backbone is not None:
                # backbone expects (B,T,H,W,3)
                vis_emb = backbone(vis)
            else:
                vis_emb = vis
            logits = model(vis_emb, head_pose=pose, gaze=gaze, motion=motion, goal=goal)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * labels.size(0)
            running_samples += labels.size(0)

        scheduler.step()
        train_loss = running_loss / running_samples
        val_loss, val_acc, val_metrics = evaluate(model, val_loader, device, backbone=backbone)
        print(f"Epoch {epoch}/{epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    print("Finished training. **best_val_acc**=", best_val_acc)

    # save best model
    if work_dir is not None and best_state is not None:
        os.makedirs(work_dir, exist_ok=True)
        ckpt = os.path.join(work_dir, "best_fusion.pt")
        torch.save({"state_dict": best_state, "config": config}, ckpt)

    # Evaluate on test set. If we have a saved best_state, load it for final evaluation.
    if best_state is not None:
        try:
            model.load_state_dict(best_state)
        except Exception:
            # in case of mismatch, ignore and use final weights
            pass

    test_loss, test_acc, test_metrics = evaluate(model, test_loader, device, backbone=backbone)
    print(f"\nTest: test_loss={test_loss:.4f}, test_acc={test_acc:.4f}, test_macrof1={np.mean(test_metrics['f1']):.4f}")

    # Nicely print per-class metrics
    precs = test_metrics.get("precision", [])
    recs = test_metrics.get("recall", [])
    f1s = test_metrics.get("f1", [])
    supports = test_metrics.get("support", [])
    ncls = len(precs)
    if ncls > 0:
        header = f"{'class':>5} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>8}"
        print("\nPer-class metrics:")
        print(header)
        print("-" * len(header))
        for c in range(ncls):
            print(f"{c:5d} {precs[c]:10.4f} {recs[c]:10.4f} {f1s[c]:10.4f} {supports[c]:8d}")
        # macro & micro
        macro = test_metrics.get("macro", {})
        micro = test_metrics.get("micro", {})
        print("\nAverages:")
        print(f"macro - precision: {macro.get('precision',0.0):.4f}, recall: {macro.get('recall',0.0):.4f}, f1: {macro.get('f1',0.0):.4f}")
        print(f"micro - precision: {micro.get('precision',0.0):.4f}, recall: {micro.get('recall',0.0):.4f}, f1: {micro.get('f1',0.0):.4f}")

    # save test metrics to work_dir if provided
    if work_dir is not None:
        os.makedirs(work_dir, exist_ok=True)
        metrics_path = os.path.join(work_dir, "test_metrics.txt")
        try:
            with open(metrics_path, "w") as f:
                f.write(f"test_loss {test_loss}\n")
                f.write(f"test_acc {test_acc}\n")
                f.write("class,precision,recall,f1\n")
                ncls = len(test_metrics.get("precision", []))
                for c in range(ncls):
                    p = test_metrics["precision"][c]
                    r = test_metrics["recall"][c]
                    f1 = test_metrics["f1"][c]
                    f.write(f"{c},{p:.6f},{r:.6f},{f1:.6f}\n")
        except Exception:
            pass

    return {"best_val_acc": best_val_acc, "test_loss": test_loss, "test_acc": test_acc, "test_metrics": test_metrics, "config": config}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--T", type=int, default=60)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--work-dir", type=str, default="./outputs")
    parser.add_argument("--dataset-type", type=str, default="vr", choices=["synthetic", "vr"]) 
    parser.add_argument("--annotations", type=str, default='../data/features/annotations.VRmulti_test_close.json', help="Path to annotations JSON for VR dataset")
    parser.add_argument("--videos-dir", type=str, default='../data/vrvideo_256/clips', help="Root directory of video clips for VR dataset")
    parser.add_argument("--vrdata-dir", type=str, default='../data/vrdata', help="Directory with per-video CSVs for head pose/gaze")
    parser.add_argument("--frame-size", type=int, nargs=2, default=(32, 32), help="H W frame resize for VR dataset")
    parser.add_argument("--visual-encoder", type=str, default="simple", choices=["pixels", "simple", "clip"], help="Which visual encoder to use: flattened pixels, simple CNN, or pretrained CLIP")
    parser.add_argument("--backbone-out-dim", type=int, default=256, help="Output embedding dim of backbone per frame")
    parser.add_argument("--backbone-hidden", type=int, default=64, help="Hidden channels for small CNN backbone")
    parser.add_argument("--clip-model", type=str, default="ViT-B/32", help="CLIP model name to load if --visual-encoder clip")
    parser.add_argument("--freeze-clip", action="store_true", help="Freeze CLIP backbone weights")
    parser.add_argument("--pose-cols", type=str, nargs="*", default=["rot_x","rot_y","rot_z"], help="CSV columns for head pose")
    parser.add_argument("--gaze-cols", type=str, nargs="*", default=["gaze_x","gaze_y"], help="CSV columns for gaze")
    parser.add_argument("--motion-cols", type=str, nargs="*", default=["Ped_Location_x_smoothed","Ped_Location_y_smoothed"], help="CSV columns for past body-frame motion")
    parser.add_argument("--motion-dim", type=int, default=2, help="Dimensionality of motion features per timestep")
    parser.add_argument("--model", type=str, default="fusion", choices=["fusion", "slim"], help="Which model to use: 'fusion' (original) or 'slim' (lightweight)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = {
        "seed": args.seed,
        "epochs": args.epochs,
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "T": args.T,
        "d_model": args.d_model,
        "lr": args.lr,
        "dataset_type": args.dataset_type,
        "annotations_path": args.annotations,
        "videos_dir": args.videos_dir,
        "vrdata_dir": args.vrdata_dir,
        "frame_size": tuple(args.frame_size),
        "visual_encoder": args.visual_encoder,
        "backbone_out_dim": args.backbone_out_dim,
        "backbone_hidden": args.backbone_hidden,
        "clip_model": args.clip_model,
        "freeze_clip": args.freeze_clip,
        "pose_cols": args.pose_cols,
        "gaze_cols": args.gaze_cols,
        "motion_cols": args.motion_cols,
        "motion_dim": args.motion_dim,
    "model": args.model,
        
    }
    result = train(cfg, work_dir=args.work_dir)
