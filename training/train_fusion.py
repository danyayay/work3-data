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
import datetime
import tempfile
import shutil
import json
import copy 
import subprocess

# optional optuna import for hyperparameter tuning
try:
    import optuna
    from optuna.exceptions import TrialPruned
except Exception:
    optuna = None

from models.fusion import EgocentricFusionTransformer, SimpleCNNBackbone
from models.fusion_slim import SlimFusionTransformer
from training.synthetic_dataset import SyntheticEgocentricDataset, collate_fn as synth_collate
from training.scalers import compute_stats_from_dataloader, save_stats
from torch.utils.data import Subset
from utils.util import seed_everything

# optional VR dataset import (may require opencv/pandas); import lazily below
try:
    from training.vr_dataset import VRDataset, collate_fn as vr_collate  # type: ignore
except Exception:
    VRDataset = None
    vr_collate = None


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, backbone: Optional[nn.Module] = None, loss_fn: Optional[nn.Module] = None):
    """Run evaluation and return (avg_loss, accuracy, class_metrics)

    class_metrics is a dict with keys 'precision','recall','f1' mapping to lists indexed by class id.
    """
    model.eval()
    # use provided loss_fn (with class weights) when available, else default
    if loss_fn is None:
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
                # Only apply backbone if we received raw frames with 5 dims (B,T,H,W,3).
                # Synthetic dataset provides per-frame embeddings (B,T,feat) so skip
                # the backbone in that case.
                if vis.dim() == 5:
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
    return avg_loss, accuracy, macro_f1, class_metrics


def train(
    config: Dict[str, Any],
    work_dir: Optional[str] = None,
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.cuda.set_per_process_memory_fraction(0.8, device=0)

    # timestamp for this run (used for saving model + logs)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_lines = []
    log_fh = None
    seed_everything(args.seed)

    n_classes = int(config.get('n_classes', 5))

    def _log(msg: str):
        # print and also stream to the run log file (if available) in real time
        try:
            s = str(msg)
        except Exception:
            s = repr(msg)
        print(s)
        try:
            log_lines.append(s)
        except Exception:
            pass
        try:
            if log_fh is not None:
                log_fh.write(s + "\n")
                log_fh.flush()
        except Exception:
            pass
    
    def _save_ckpt(best_state):
        if run_dir is not None and best_state is not None:
            ckpt = os.path.join(run_dir, f"best_fusion.pt")
            torch.save({"state_dict": best_state, "config": config, "timestamp": timestamp}, ckpt)
            _log(f"  Saved best model to {ckpt}")

    _log("Starting training with config:")
    # _log(json.dumps(config, indent=4))
    _log(json.dumps(config, separators=(",", ":"), sort_keys=True))

    # determine run directory for artifacts. If the caller sets
    # config['_no_timestamp_dir']=True then we will use the provided
    # work_dir as-is (useful when run by Optuna where trial dirs are
    # already timestamped). Otherwise create a timestamped subfolder
    # under work_dir (when provided).
    run_dir = None
    if work_dir is not None:
        if config.get('_no_timestamp_dir', False):
            run_dir = work_dir
        else:
            run_dir = os.path.join(work_dir, timestamp)
        os.makedirs(run_dir, exist_ok=True)
        # save the config for reproducibility
        try:
            with open(os.path.join(run_dir, 'config.json'), 'w') as f:
                json.dump(config, f, indent=4)
        except Exception:
            pass
        # open a streaming log file so _log can write in real time
        try:
            log_path = os.path.join(run_dir, 'training_log.txt')
            # open in append mode so repeated runs don't clobber previous logs
            log_fh = open(log_path, 'a', encoding='utf-8', buffering=1)
        except Exception:
            log_fh = None

    # dataset selection and visual encoder
    dataset_type = config.get("dataset_type", "synthetic")
    visual_encoder = config.get("visual_encoder", -1)
    if visual_encoder not in ("pixels", "simple", "clip"):
        raise ValueError("visual_encoder must be one of 'pixels','simple','clip'")
    if dataset_type == "vr":
        if VRDataset is None:
            raise RuntimeError("VRDataset not available - missing optional dependencies (cv2/pandas) or import error")
        # import split_person_groups lazily so scripts that only use synthetic data
        # don't require OpenCV/pandas at module import time
        from training.vr_dataset import split_person_groups, split_person_testing
        if args.debugging: 
            data_split = split_person_testing(config["annotations_path"])
        else:
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
            goal_cols=config.get("goal_cols", None),
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
            goal_cols=config.get("goal_cols", None),
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
            goal_cols=config.get("goal_cols", None),
            return_raw_frames=(visual_encoder != "pixels"),
            person_ids=data_split["test_persons"],
        )
        collate = vr_collate
        # visual_dim is flattened frame size (H*W*3)
        if visual_encoder == "pixels":
            visual_dim = train_ds.frame_size[0] * train_ds.frame_size[1] * 3
        elif visual_encoder == 'simple':
            visual_dim = config.get("backbone_out_dim", 256)
        elif visual_encoder == 'clip':
            # CLIP output dim depends on model; default ViT-B/32 has 512-dim embeddings
            visual_dim = config.get("backbone_out_dim", 512)
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
            n_classes=n_classes,
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
        if run_dir is not None:
            save_stats(stats, run_dir)

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
            dropout=config.get("dropout", 0),
            n_classes=n_classes,
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
            n_classes=n_classes,
            max_time_steps=config.get("max_time_steps", 64),
            predict_future_steps=config.get("predict_future_steps", 0),
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    _log(model)

    # Print parameter counts for the model (trainable vs total) before training
    try:
        def _count_params(module):
            return int(sum(p.numel() for p in module.parameters() if p.requires_grad))

        def _count_total_params(module):
            return int(sum(p.numel() for p in module.parameters()))

        model_trainable = _count_params(model)
        model_total = _count_total_params(model)
        _log(f"Model parameters: {model_trainable} trainable / {model_total} total")
    except Exception:
        # don't fail training if counting params throws for some custom module
        _log("Model parameter counting failed")

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

    # Log backbone and combined parameter counts (if backbone exists)
    try:
        if backbone is not None:
            def _count_params(module):
                return int(sum(p.numel() for p in module.parameters() if p.requires_grad))

            def _count_total_params(module):
                return int(sum(p.numel() for p in module.parameters()))

            backbone_trainable = _count_params(backbone)
            backbone_total = _count_total_params(backbone)
            model_trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
            combined_trainable = model_trainable + backbone_trainable
            _log(f"Backbone parameters: {backbone_trainable} trainable / {backbone_total} total")
            _log(f"Combined trainable parameters (model + backbone): {combined_trainable}")
    except Exception:
        _log("Backbone parameter counting failed")

    if backbone is not None:
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(backbone.parameters()), lr=config.get("lr", 1e-4), weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 1e-4), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # compute class weights from the training set to handle class imbalance
    counts = np.zeros(n_classes, dtype=np.int64)
    try:
        for _vis, _pose, _gaze, _motion, _goal, labels in train_loader:
            # labels can be a tensor
            lbl = labels.cpu().numpy()
            binc = np.bincount(lbl, minlength=n_classes)
            counts[: len(binc)] += binc
    except Exception:
        # fallback to uniform counts if anything goes wrong
        counts = np.ones(n_classes, dtype=np.int64)

    total = int(counts.sum()) if counts.sum() > 0 else 0
    if total == 0:
        weights = np.ones(n_classes, dtype=np.float32)
    else:
        # inverse frequency, normalized by number of classes so mean weight ~=1
        counts_safe = counts.copy().astype(np.float32)
        counts_safe[counts_safe == 0] = 1.0
        weights = total / (n_classes * counts_safe)
        weights[counts == 0] = 0.0
    # weights = [3.4, 1.0]
    _log(f"weights: {weights}")
    _log(f'Labeling map: {train_ds.label_map}' )
    weights = torch.tensor(weights, dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))

    # track bests for multiple metrics but monitor a single chosen eval metric
    best_val_acc = 0.0
    best_val_macrof1 = 0.0
    best_epoch = 0
    best_state = None
    # the metric to monitor for model selection / early stopping: 'acc' or 'macro_f1'
    monitored_metric = str(config.get('eval_metric', 'acc'))
    # initialize best monitored value to a very small number so any real score is larger
    best_monitored_value = -1.0

    epochs = int(config.get("epochs", 3))
    no_improvement_patience = int(config.get('early_stop_patience', 20))
    epochs_since_improve = 0
    # additional stopper: if neither val_acc nor val_macrof1 change for this many epochs, stop
    no_change_patience = no_improvement_patience // 2 + 1
    epochs_since_no_change = 0

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
            if backbone is not None and vis.dim() == 5:
                # backbone expects (B,T,H,W,3)
                vis_emb = backbone(vis)
            else:
                # either no backbone configured or input is already embeddings
                vis_emb = vis
            logits = model(vis_emb, head_pose=pose, gaze=gaze, motion=motion, goal=goal)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * labels.size(0)
            running_samples += labels.size(0)

            scheduler.step()
            train_loss = running_loss / running_samples

        val_loss, val_acc, val_macrof1, val_metrics = evaluate(model, val_loader, device, backbone=backbone, loss_fn=loss_fn)
        _log(f"Epoch {epoch}/{epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macrof1={val_macrof1:.4f}")

        # choose which metric to use for model selection and early stopping
        if monitored_metric in ("acc", "accuracy"):
            current_monitored = float(val_acc)
        elif monitored_metric in ("macrof1", "macro_f1"):
            current_monitored = float(val_macrof1)
        else:
            raise ValueError(f"Unknown monitored metric: {monitored_metric}")

        # always update secondary tracked metrics for reporting
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        if val_macrof1 > best_val_macrof1:
            best_val_macrof1 = val_macrof1

        # update best according to the chosen monitored metric
        if current_monitored > best_monitored_value:
            best_monitored_value = current_monitored
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            _save_ckpt(best_state)
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
        
        if current_monitored != best_monitored_value:
            epochs_since_no_change = 0
        else:
            epochs_since_no_change += 1

        if epochs_since_improve >= no_improvement_patience:
            _log(f"Early stopping: no improvement for {no_improvement_patience} epochs (stopping at epoch {epoch})")
            break
        if epochs_since_no_change >= no_change_patience:
            _log(f"Early stopping: no change in val_acc or val_macrof1 for {no_change_patience} epochs (stopping at epoch {epoch})")
            break

    _log(f"\nFinished training. \nbest_val_{monitored_metric}= {best_monitored_value:.4f} at epoch {best_epoch}")

    # Evaluate on test set. If we have a saved best_state, load it for final evaluation.
    if best_state is not None:
        try:
            model.load_state_dict(best_state)
        except Exception:
            # in case of mismatch, ignore and use final weights
            pass

    # evaluate test metrics
    val_loss, val_acc, val_macrof1, val_metrics = evaluate(model, val_loader, device, backbone=backbone, loss_fn=loss_fn)
    test_loss, test_acc, test_macrof1, test_metrics = evaluate(model, test_loader, device, backbone=backbone, loss_fn=loss_fn)
    _log(f"\nTest: test_loss={test_loss:.4f}, test_acc={test_acc:.4f}, test_macrof1={test_macrof1:.4f}")

    # Nicely print per-class metrics
    precs = test_metrics.get("precision", [])
    recs = test_metrics.get("recall", [])
    f1s = test_metrics.get("f1", [])
    supports = test_metrics.get("support", [])
    ncls = len(precs)
    if ncls > 0:
        header = f"{'class':>5} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>8}"
        _log("\nPer-class metrics:")
        _log(header)
        _log("-" * len(header))
        for c in range(ncls):
            _log(f"{c:5d} {precs[c]:10.4f} {recs[c]:10.4f} {f1s[c]:10.4f} {supports[c]:8d}")
        # macro & micro
        macro = test_metrics.get("macro", {})
        micro = test_metrics.get("micro", {})
        _log("\nAverages:")
        _log(f"macro - precision: {macro.get('precision',0.0):.4f}, recall: {macro.get('recall',0.0):.4f}, f1: {macro.get('f1',0.0):.4f}")
        _log(f"micro - precision: {micro.get('precision',0.0):.4f}, recall: {micro.get('recall',0.0):.4f}, f1: {micro.get('f1',0.0):.4f}")

    # save test metrics to run_dir if provided
    if run_dir is not None:
        metrics_path = os.path.join(run_dir, "test_metrics.txt")
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

    # save training log
    if run_dir is not None:
        try:
            # if we have an open streaming log file, close it before writing the final copy
            try:
                if log_fh is not None:
                    log_fh.close()
            except Exception:
                pass
            log_path = os.path.join(run_dir, f"training_log.txt")
            with open(log_path, "w") as f:
                for l in log_lines:
                    f.write(str(l) + "\n")
            _log(f"Saved training log to {log_path}")
        except Exception:
            pass

    return {
        "best_val_acc": best_val_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_macrof1": val_macrof1,
        "val_metrics": val_metrics,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_macrof1": test_macrof1,
        "test_metrics": test_metrics,
        # include monitored metric and its name so callers (e.g., Optuna) can use it
        "monitored_metric": monitored_metric,
        "best_monitored": best_monitored_value,
        "config": config,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--T", type=int, default=30)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--work-dir", type=str, default="./outputs")
    parser.add_argument("--dataset-type", type=str, default="vr", choices=["synthetic", "vr"]) 
    parser.add_argument("--debugging", type=bool, default=True, help="If set, run a quick sample training for debugging")
    parser.add_argument("--annotations_path", type=str, default='../data/features/annotations.VRmulti_test_close.json', help="Path to annotations JSON for VR dataset")
    parser.add_argument("--videos-dir", type=str, default='../data/vrvideo_256/clips', help="Root directory of video clips for VR dataset")
    parser.add_argument("--vrdata-dir", type=str, default='../data/vrdata', help="Directory with per-video CSVs for head pose/gaze")
    
    parser.add_argument("--frame-size", type=int, nargs=2, default=(32, 32), help="H W frame resize for VR dataset")
    parser.add_argument("--visual-encoder", type=str, default="simple", choices=["pixels", "simple", "clip"], help="Which visual encoder to use: flattened pixels, simple CNN, or pretrained CLIP")
    
    parser.add_argument("--backbone-out-dim", type=int, default=256, help="Output embedding dim of backbone per frame")
    parser.add_argument("--backbone-hidden", type=int, default=64, help="Hidden channels for small CNN backbone")
    parser.add_argument("--clip-model", type=str, default="ViT-B/32", help="CLIP model name to load if --visual-encoder clip")
    parser.add_argument("--freeze-clip", action="store_true", default=True, help="Freeze CLIP backbone weights")
    
    parser.add_argument("--no-extra-modalities", action="store_true", default=False, help="Disable pose/gaze/motion/goal modalities and use only egocentric video") 
    parser.add_argument("--motion-cols", type=str, nargs="*", default=["Ped_Location_x_smoothed","Ped_Location_y_smoothed"], help="CSV columns for past body-frame motion")
    parser.add_argument("--pose-cols", type=str, nargs="*", default=["rot_x","rot_y","rot_z"], help="CSV columns for head pose")
    parser.add_argument("--gaze-cols", type=str, nargs="*", default=["px_u","px_v"], help="CSV columns for gaze")
    parser.add_argument("--goal-cols", type=str, nargs="*", default=["goal"], help="CSV columns for goal location") 
    parser.add_argument("--motion-dim", type=int, default=2, help="Dimensionality of motion features per timestep")
    
    parser.add_argument("--model", type=str, default="slim", choices=["fusion", "slim"], help="Which model to use: 'fusion' (original) or 'slim' (lightweight)")
    parser.add_argument("--early-stop-patience", type=int, default=20, help="Early stopping patience in epochs (default 20)")
    parser.add_argument("--eval-metric", type=str, default="macro_f1", choices=["acc", "macro_f1"], help="Evaluation metric to monitor for model selection and early stopping")
    
    # Optuna hyperparameter tuning flags
    parser.add_argument("--hp-tune", action="store_true", default=False, help="Run Optuna hyperparameter tuning instead of a single training run")
    parser.add_argument("--hp-trials", type=int, default=100, help="Number of Optuna trials to run")
    parser.add_argument("--hp-keep-checkpoints", action="store_true", help="Keep per-trial checkpoints/artifacts (default: remove them)")
    parser.add_argument("--optuna_url", type=str, default="sqlite:///db.sqlite2", help="Optuna storage URL (default: local SQLite file db.sqlite)")
    
    return parser.parse_args()


def run_optuna(base_cfg: Dict[str, Any], args) -> Optional[object]:
    """Run an Optuna study using the training function as the objective.

    Each trial runs train(...) for a small number of epochs in a temporary
    work directory. The best trial params are written to base_cfg['work_dir']
    (if provided).
    """
    if optuna is None:
        raise RuntimeError("Optuna is not installed. Install it with 'pip install optuna' to use --hp-tune.")

    # create a timestamped study directory to collect all artifacts
    study_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_work = base_cfg.get('work_dir') or args.work_dir or os.getcwd()
    study_dir = os.path.join(base_work, f"{study_ts}_optuna")
    os.makedirs(study_dir, exist_ok=True)
    # streaming optuna log
    optuna_log_fh = None
    try:
        optuna_log_path = os.path.join(study_dir, 'optuna_log.txt')
        optuna_log_fh = open(optuna_log_path, 'a', encoding='utf-8', buffering=1)
    except Exception:
        optuna_log_fh = None

    def _optuna_log(msg: str):
        try:
            s = str(msg)
        except Exception:
            s = repr(msg)
        print(s)
        try:
            if optuna_log_fh is not None:
                optuna_log_fh.write(s + "\n")
                optuna_log_fh.flush()
        except Exception:
            pass

    def _remove_log_file(log_dir):
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
            print(f"Removed log file: {log_dir}")


    def _objective(trial) -> float:
        cfg = dict(base_cfg)
        # sample hyperparameters
        cfg['lr'] = trial.suggest_float('lr', 1e-5, 1e-3)
        cfg['d_model'] = int(trial.suggest_categorical('d_model', [64, 128, 192, 256]))
        # choose nhead that divides d_model
        possible_heads = [1, 2, 4, 8, 16]
        valid_heads = [h for h in possible_heads if cfg['d_model'] % h == 0 and h <= cfg['d_model']]
        if len(valid_heads) == 0:
            valid_heads = [1]
        cfg['nhead'] = int(trial.suggest_categorical('nhead', valid_heads))
        cfg['num_layers'] = int(trial.suggest_int('num_layers', 1, 4))
        # sample feedforward size via a fixed categorical choice to keep the
        # distribution static across trials (avoids Optuna storage compatibility errors)
        ff_choice = trial.suggest_categorical('ff_choice', ['d2', 'd4', '512'])
        if ff_choice == '512':
            cfg['dim_feedforward'] = 512
        elif ff_choice == 'd2':
            cfg['dim_feedforward'] = int(cfg['d_model'] * 2)
        else:
            cfg['dim_feedforward'] = int(cfg['d_model'] * 4)
        cfg['dropout'] = float(trial.suggest_float('dropout', 0.0, 0.3))
        cfg['batch_size'] = int(trial.suggest_categorical('batch_size', [8, 16, 32, 64]))

        cfg['T'] = trial.suggest_categorical('T', [8, 16, 30, 60])
        cfg['model'] = trial.suggest_categorical('model', ['fusion', 'slim'])
        cfg['visual_encoder'] = args.visual_encoder

        if args.visual_encoder != 'clip':
            resize_size = int(trial.suggest_categorical('resize_size', [32, 64, 128, 224]))
        else: 
            resize_size = 224  # CLIP default
        cfg['frame_size'] = (resize_size, resize_size)

        if args.visual_encoder != 'pixels':
            if args.visual_encoder == 'simple':
                cfg['backbone_out_dim'] = int(trial.suggest_int('backbone_out_dim', 64, 256))
                cfg['backbone_hidden'] = int(trial.suggest_int('backbone_hidden', 32, 128))
            elif args.visual_encoder == 'clip':
                cfg['clip_model'] = args.clip_model
                cfg['freeze_clip'] = args.freeze_clip
                cfg['backbone_out_dim'] = 512  # assuming ViT-B/32
                cfg['backbone_hidden'] = int(trial.suggest_int('backbone_hidden', 32, 128))
        else:
            cfg['backbone_out_dim'] = None
            cfg['backbone_hidden'] = None

        # create a per-trial directory under the study dir so that all logs
        # and checkpoints from trials are retained in a single timestamped
        # folder. We set a flag so train() will not create an extra timestamp
        # subfolder.
        trial_dir = os.path.join(study_dir, 'trials', f"trial_{trial.number}")
        os.makedirs(trial_dir, exist_ok=True)
        cfg['_no_timestamp_dir'] = True

        max_attempts = 2  # 最多尝试两次
        error = None
        for _ in range(max_attempts):
            try:
                result = train(cfg, work_dir=trial_dir)
                # capture metrics for Optuna reporting and later analysis
                # prefer the new 'best_monitored' key (monitored metric), fall back to best_val_acc
                val_score = float(result.get('best_monitored', -1))
                # attach these to trial user attrs
                # record the metric that was monitored for this trial
                trial.set_user_attr('monitored_metric', result.get('monitored_metric', -1))
                trial.set_user_attr('val_acc', float(result.get('val_acc', -1)))
                trial.set_user_attr('val_macrof1', result.get('val_macrof1', -1))
                trial.set_user_attr('test_acc', float(result.get('test_acc', -1)))
                trial.set_user_attr('test_macrof1', result.get('test_macrof1', -1))
                
                return val_score
            except Exception as e:  # pragma: no cover - avoid crashing the study
                _optuna_log(f"Trial crashed: {e}")
                error = e
                val_score = -1
                _optuna_log("Retrying the trial...")
                continue  # retry
        _remove_log_file(trial_dir)
        # 两次都失败，跳过这个 trial
        raise TrialPruned(f'Trial failed after {max_attempts} attempts because {error}')

    # use a sqlite storage so the study can be inspected with optuna-dashboard
    
    def get_feat_indicators(args):
        feats_indicators = []
        feats_indicators.append('1') if args.motion_cols != [] else feats_indicators.append('0')
        feats_indicators.append('1') if args.pose_cols != [] else feats_indicators.append('0')
        feats_indicators.append('1') if args.gaze_cols != [] else feats_indicators.append('0')
        feats_indicators.append('1') if args.goal_cols != [] else feats_indicators.append('0')
        return ''.join(feats_indicators)
    
    study = optuna.create_study(
        storage=args.optuna_url, 
        load_if_exists=True, 
        direction='maximize',
        study_name=f'egocentric_{args.cls_type}_{args.visual_encoder}_{get_feat_indicators(args)}'
    )
    print(f"Optuna study created. Study name: {study.study_name}, storage: {args.optuna_url}")

    # initialize with a good start
    initial_params = {
        'lr': 1e-4,
        'd_model': 128,
        'nhead': 8,
        'num_layers': 4,
        'dropout': 0,
        'batch_size': 32,
        'resize_size': 32,
        'T': 30,
        'model': 'fusion',
        'backbone_hidden': 64,
        'backbone_out_dim': 256,
    }
    if args.visual_encoder == 'clip' and args.clip_model == 'ViT-B/32':
        initial_params['resize_size'] = 224
        initial_params['backbone_out_dim'] = 512
    study.enqueue_trial(initial_params)

    _optuna_log(f"Starting Optuna study optimization with n_trials={args.hp_trials}")
    study.optimize(_objective, n_trials=args.hp_trials, catch=())  # catch all exceptions to avoid study crashes

    # save best params and trials into study_dir
    best = study.best_trial
    try:
        best_entry = {'value': best.value, 'params': best.params}
        # include any user attrs recorded during the objective
        try:
            best_entry['user_attrs'] = best.user_attrs
        except Exception:
            best_entry['user_attrs'] = {}
        with open(os.path.join(study_dir, 'optuna_best_params.json'), 'w') as f:
            json.dump(best_entry, f, indent=4)
    except Exception:
        pass

    try:
        df = study.trials_dataframe()
        df.to_csv(os.path.join(study_dir, 'optuna_trials.csv'), index=False)
    except Exception:
        # if pandas not available or conversion fails, fallback to writing simple text
        try:
            with open(os.path.join(study_dir, 'optuna_trials.txt'), 'w') as f:
                for t in study.trials:
                    f.write(str({'number': t.number, 'value': t.value, 'params': t.params}) + "\n")
        except Exception:
            pass

    # write a detailed per-trial CSV that includes user_attrs (val_macrof1, test metrics)
    try:
        import csv
        detailed_path = os.path.join(study_dir, 'optuna_trials_detailed.csv')
        with open(detailed_path, 'w', newline='') as csvfile:
            fieldnames = ['number', 'value', 'params', 'monitored_metric', 'val_monitored', 'val_macrof1', 'test_acc', 'test_macrof1']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for t in study.trials:
                ua = getattr(t, 'user_attrs', {}) if hasattr(t, 'user_attrs') else {}
                row = {
                    'number': t.number,
                    'value': t.value,
                        'params': t.params,
                        'monitored_metric': ua.get('monitored_metric', ''),
                        'val_monitored': ua.get('val_monitored', ''),
                        'val_macrof1': ua.get('val_macrof1', ''),
                        'test_acc': ua.get('test_acc', ''),
                        'test_macrof1': ua.get('test_macrof1', ''),
                }
                writer.writerow(row)
    except Exception:
        pass

    # write a human-readable summary
    try:
        summary_path = os.path.join(study_dir, 'optuna_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Optuna study summary - best value: {best.value}\n")
            f.write(f"Best params: {best.params}\n\n")
            f.write("Trials:\n")
            for t in study.trials:
                f.write(f"#%d value={t.value} params={t.params}\n" % t.number)
    except Exception:
        pass

    _optuna_log(f"Optuna study finished. Best value: {best.value}, params: {best.params}. Artifacts in {study_dir}")
    try:
        if optuna_log_fh is not None:
            optuna_log_fh.close()
    except Exception:
        pass
    return study


if __name__ == "__main__":
    args = parse_args()

    if args.debugging:
        # reduce epochs and samples for quick debugging
        args.epochs = 50
        args.T = 8
        args.num_samples = 50
        args.batch_size = 8
        args.frame_size = (32, 32)
        args.annotations_path = '../data/features/annotations.VRbin_testing_close.json'
        args.hp_trials = 4

    if args.visual_encoder == 'clip' and args.clip_model == 'ViT-B/32':
        # ViT-B/32 performs with 224x224 input, output dim 512
        args.frame_size = (224, 224)
        args.backbone_out_dim = 512

    if 'bin' in args.annotations_path: 
        args.n_classes = 2
        args.cls_type = 'bin'
    elif 'multi' in args.annotations_path:
        args.n_classes = 5
        args.cls_type = 'multi'
    else:
        raise ValueError("Cannot infer number of classes from annotations filename; please specify binary or multi-class in the filename.")


    cfg = {
        "seed": args.seed,
        "epochs": args.epochs,
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "T": args.T,
        "d_model": args.d_model,
        "lr": args.lr,
        "dataset_type": args.dataset_type,
        "annotations_path": args.annotations_path,
        "videos_dir": args.videos_dir,
        "vrdata_dir": args.vrdata_dir,
        "frame_size": tuple(args.frame_size),
        "visual_encoder": args.visual_encoder,
        "backbone_out_dim": args.backbone_out_dim,
        "backbone_hidden": args.backbone_hidden,
        "clip_model": args.clip_model,
        "freeze_clip": args.freeze_clip,
        "pose_cols": [] if args.no_extra_modalities else args.pose_cols,
        "gaze_cols": [] if args.no_extra_modalities else args.gaze_cols,
        "motion_cols": [] if args.no_extra_modalities else args.motion_cols,
        "goal_cols": [] if args.no_extra_modalities else args.goal_cols,
        "motion_dim": 0 if args.no_extra_modalities else args.motion_dim,
        "model": args.model,
        "eval_metric": args.eval_metric,
        "n_classes": args.n_classes,
        "cls_type": args.cls_type,
        "early_stop_patience": args.early_stop_patience,
    }
    if args.hp_tune:
        # run Optuna tuning
        study = run_optuna(cfg, args)
        # after tuning, print best and exit
        if study is not None:
            best = study.best_trial
            print("Optuna best:", best.number, best.value, best.params)
        else:
            print("Optuna did not run (optuna missing?)")
    else:
        result = train(cfg, work_dir=args.work_dir)
