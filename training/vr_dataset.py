"""VR dataset loader: reads annotation JSON, loads frames via OpenCV, and extracts
head-pose/gaze from per-video CSVs. Returns either flattened per-frame pixels or
raw resized frames (T,H,W,3) to be used with a backbone.

Compact, single-definition implementation (no duplicated content).
"""
from cProfile import label
import re 
import json
import os
import warnings
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from decord import VideoReader, cpu, gpu
from torchvision import transforms
import torch
from torch.utils.data import Dataset


def _find_key(d: dict, candidates: List[str]):
    for k in candidates:
        if k in d:
            return k
    return None


class VRDataset(Dataset):
    def __init__(
        self,
        annotations_path: str,
        videos_dir: str,
        vrdata_dir: Optional[str] = None,
        pose_scaler: Optional[dict] = None,
        gaze_scaler: Optional[dict] = None,
        motion_scaler: Optional[dict] = None,
        goal_dist_scaler: Optional[dict] = None,
        T: int = 8,
        frame_size: Tuple[int, int] = (32, 32),
        person_ids: List[str] = None,
        label_key_candidates: Optional[List[str]] = None,
        video_key_candidates: Optional[List[str]] = None,
        start_key_candidates: Optional[List[str]] = None,
        end_key_candidates: Optional[List[str]] = None,
        pose_cols: Optional[List[str]] = None,
        gaze_cols: Optional[List[str]] = None,
        motion_cols: Optional[List[str]] = None,
        goal_cols: Optional[List[str]] = None,
        return_raw_frames: bool = False,
        skip_missing: bool = True,
    ):
        self.annotations_path = annotations_path
        self.videos_dir = videos_dir
        self.vrdata_dir = vrdata_dir
        self.T = int(T)
        self.frame_size = tuple(frame_size)
        self.skip_missing = skip_missing
        self.return_raw_frames = bool(return_raw_frames)
        self.dt = pd.read_csv('dts_qn.csv')

        label_key_candidates = label_key_candidates or [
            "label",
            "class",
            "action",
            "intent",
            "label_idx",
            "binary_label",
            "answer",
        ]
        video_key_candidates = video_key_candidates or [
            "video_path",
            "video",
            "video_name",
            "file_name",
            "video_file",
            "video_id",
            "sample_id",
            "video_uid",
        ]
        start_key_candidates = start_key_candidates or ["moment_start_frame"]
        end_key_candidates = end_key_candidates or ["moment_end_frame"]

        # store user-specified column configs (None means auto-detect, [] means disable)
        self._user_pose_cols = True if len(pose_cols) > 0 else False
        self._user_gaze_cols = True if len(gaze_cols) > 0 else False
        self._user_motion_cols = True if len(motion_cols) > 0 else False
        self._user_goal_cols = True if len(goal_cols) > 0 else False

        self.pose_cols = pose_cols
        self.gaze_cols = gaze_cols
        self.motion_cols = motion_cols
        self.goal_cols = goal_cols

        # optional scalers (dict with 'mean' and 'std' numpy arrays)
        self.pose_scaler = pose_scaler
        self.gaze_scaler = gaze_scaler
        self.motion_scaler = motion_scaler
        self.goal_dist_scaler = goal_dist_scaler

        # load annotations
        with open(annotations_path, "r") as f:
            raw = json.load(f)

        # normalize to a list of entries
        if isinstance(raw, dict):
            for k in ("annotations", "items", "data", "videos"):
                if k in raw and isinstance(raw[k], list):
                    ann_list = raw[k]
                    break
            else:
                ann_list = [raw]
        elif isinstance(raw, list):
            ann_list = raw
        else:
            raise ValueError("Unsupported annotations JSON structure")

        self.samples = []
        for entry in ann_list:
            vkey = _find_key(entry, video_key_candidates)
            if vkey is None:
                warnings.warn(f"No video key found in annotation entry; skipping: {entry}")
                if self.skip_missing:
                    continue
                else:
                    raise KeyError("video key not found")
                
            # only if in the given subset
            sample_id = entry[vkey]
            pid = sample_id.split('S')[0]
            video_id = sample_id.split('-')[0]
            if pid in person_ids:
                video_path = os.path.join(videos_dir, str(sample_id) + '.mp4') if videos_dir is not None else None
                csv_path = os.path.join(vrdata_dir, str(video_id) + '.csv') if vrdata_dir is not None else None

                sk = _find_key(entry, start_key_candidates)
                ek = _find_key(entry, end_key_candidates)
                if sk is None or ek is None:
                    warnings.warn(f"Start/end frame keys not found for {sample_id}; skipping")
                    if self.skip_missing:
                        continue
                    else:
                        raise KeyError("start/end keys not found")
                start = int(entry[sk])
                end = int(entry[ek])
                if end <= start:
                    warnings.warn(f"end <= start for {sample_id}; skipping")
                    if self.skip_missing:
                        continue
                    else:
                        raise ValueError("end <= start")

                lkey = _find_key(entry, label_key_candidates)
                if lkey is None:
                    label = 0
                else:
                    label = entry[lkey]
                    if isinstance(label, str):
                        try:
                            label = int(label)
                        except Exception:
                            pass

                self.samples.append({"video_path": video_path, "csv_path": csv_path, "start": start, "end": end, "label": label})

        if len(self.samples) == 0:
            raise RuntimeError("No valid samples found in annotations")

        # map string labels to ints if necessary
        labels = [s["label"] for s in self.samples]
        if any(isinstance(x, str) for x in labels):
            uniq = sorted(set(labels))
            self.label_map = {v: i for i, v in enumerate(uniq)}
            for s in self.samples:
                s["label"] = self.label_map[s["label"]]
        else:
            self.label_map = None
            for s in self.samples:
                s["label"] = int(s["label"])

    def __len__(self):
        return len(self.samples)

    def _read_frames(self, video_path: str, start: int, end: int) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        frame_idxs = np.linspace(start, max(start + 1, end), num=self.T, endpoint=False, dtype=int)
        H, W = self.frame_size

        ## use the full frames
        with open(video_path, 'rb') as f:
            vr = VideoReader(f, ctx=cpu(0))
        frames_np = vr.get_batch(frame_idxs).asnumpy() # (T, H, W, 3)
        frames = torch.from_numpy(frames_np).pin_memory()
        transforms_compose = transforms.Compose([
                transforms.Resize(H),
                transforms.CenterCrop(H),
                transforms.Resize(H),
            ])
        # TODO: update the center crop to eye gaze based crop? 
        frames = frames.float() / 255.0
        frames = frames.permute(3, 0, 1, 2) # (C, T, Hresize, Wresize)
        frames = transforms_compose(frames)  # -> (C, T, H, W)
        frames = frames.permute(1, 2, 3, 0).numpy()  # (T, H, W, C)

        ## sample from full frames 
        # frames = np.zeros((self.T, H, W, 3), dtype=np.float32)
        # for i, fi in enumerate(frame_idxs):
        #     cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        #     ret, frame = cap.read()
        #     if not ret:
        #         frame = np.zeros((H, W, 3), dtype=np.uint8)
        #     else:
        #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #         frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)
        #     frames[i] = (frame.astype(np.float32) / 255.0)
        # cap.release()
        
        if self.return_raw_frames:
            return frames
        T, H, W, C = frames.shape
        return frames.reshape(T, H * W * C)
    
    def _read_vr_csv(self, csv_path: str, start: int, end: int):
        """Read per-frame CSV and return sampled (pose, gaze, motion, goal_ts)

        goal_ts is (T,3) with columns [d, sin(beta), cos(beta)] when angle/dist available.
        """
        try:
            import pandas as pd

            df = pd.read_csv(csv_path)
        except Exception:
            try:
                data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
                df = None
            except Exception:
                return None, None, None, None

        if df is not None:
            cols = df.columns.tolist()
            arr = df.values
            # Determine which columns to use. Behavior:
            # - if user passed None for a modality, try auto-detect from CSV columns
            # - if user passed an empty list [], treat modality as disabled
            # - if user passed an explicit list, use those column names
            if self._user_gaze_cols is None:
                gaze_cols = [c for c in cols if c.lower().startswith("gaze") or c.lower() in ("gx", "gy", "gaze_x", "gaze_y", "px_u", "px_v")]
            elif len(self.gaze_cols) == 0:
                gaze_cols = []
            else:
                gaze_cols = [c for c in cols if c in self.gaze_cols]

            if self._user_pose_cols is None:
                pose_cols = [c for c in cols if c.lower().startswith("head") or c.lower().startswith("pose") or c.lower() in ("px", "py", "pz", "rx", "ry", "rz")]
            elif len(self.pose_cols) == 0:
                pose_cols = []
            else:
                pose_cols = [c for c in cols if c in self.pose_cols]

            if self._user_motion_cols is None:
                motion_cols = [c for c in cols if c.lower().startswith("ped") and ("loc" in c.lower() or "location" in c.lower())]
            elif len(self.motion_cols) == 0:
                motion_cols = []
            else:
                motion_cols = [c for c in cols if c in self.motion_cols]
        else:
            arr = data
            ncols = arr.shape[1]
            pose_cols = list(range(min(6, ncols)))
            gaze_cols = list(range(min(2, ncols)))

        frame_idx_col = None
        if df is not None:
            for c in cols:
                if c.lower() in ("frame", "frame_idx", "frame_index", "index"):
                    frame_idx_col = cols.index(c)
                    break

        if frame_idx_col is not None and df is not None:
            frames = df.iloc[:, frame_idx_col].values.astype(int)
            mask = (frames >= start) & (frames < end)
            sub = df[mask].values
        else:
            arr_len = arr.shape[0]
            s = max(0, min(arr_len - 1, start))
            e = max(0, min(arr_len, end))
            sub = arr[s:e]

        if sub.shape[0] == 0:
            return None, None, None, None

        idxs = np.linspace(0, sub.shape[0] - 1, num=self.T, dtype=int)
        sampled = sub[idxs]

        pose = None
        gaze = None
        motion = None
        goal_ts = None
        try:
            if df is not None:
                if len(pose_cols) > 0:
                    pose_idx = [cols.index(c) for c in pose_cols if c in cols]
                    if len(pose_idx) > 0:
                        pose = sampled[:, pose_idx]
                if len(gaze_cols) > 0:
                    gaze_idx = [cols.index(c) for c in gaze_cols if c in cols]
                    if len(gaze_idx) > 0:
                        gaze = sampled[:, gaze_idx]
                if len(motion_cols) > 0:
                    motion_idx = [cols.index(c) for c in motion_cols if c in cols]
                    if len(motion_idx) > 0:
                        motion = sampled[:, motion_idx]

                # goal extraction: angle and distance (only if goal modality enabled or auto)
                if (self._user_goal_cols is None) or (len(self.goal_cols) > 0):
                    pid, sid = re.match(r'P(\d+)S(\d+)', os.path.basename(csv_path).split('.')[0]).groups()
                    angle = self.dt[(self.dt['pid'] == int(pid)) & (self.dt['sid'] == int(sid))]['angle'].values[0]
                    goal_deg_map = {0: -45.0, 1: -90.0, 2: -135.0}
                    goal_loc_map = {0: (150, -150), 1: (0, -150), 2: (-150, -150)}
                    # ensure required columns exist
                    if 'Ped_Location_x_smoothed' in cols and 'Ped_Location_y_smoothed' in cols and 'rot_z' in cols:
                        dist_x = goal_loc_map[angle][0] - df.Ped_Location_x_smoothed.values
                        dist_y = goal_loc_map[angle][1] - df.Ped_Location_y_smoothed.values
                        dist_vals = np.sqrt(dist_x ** 2 + dist_y ** 2)[idxs]
                        angle_vals = (goal_deg_map[angle] - df.rot_z.values)[idxs]
                        goal_ts = np.concatenate([dist_vals[:, None], np.sin(angle_vals[:, None]), np.cos(angle_vals[:, None])], axis=1).astype(np.float32)
                    else:
                        goal_ts = None
            else:
                pose = sampled[:, :min(6, sampled.shape[1])] if sampled.shape[1] >= 1 else None
                gaze = sampled[:, :min(2, sampled.shape[1])] if sampled.shape[1] >= 2 else None
                motion = sampled[:, :min(2, sampled.shape[1])] if sampled.shape[1] >= 2 else None
        except Exception:
            pose = None
            gaze = None
            motion = None
            goal_ts = None

        if pose is not None:
            pose = np.asarray(pose, dtype=np.float32)
        if gaze is not None:
            gaze = np.asarray(gaze, dtype=np.float32)
        if motion is not None:
            motion = np.asarray(motion, dtype=np.float32)
            motion = motion - motion[-1:, :]  # relative motion to last frame

        if goal_ts is not None:
            goal_ts = np.asarray(goal_ts, dtype=np.float32)

        return pose, gaze, motion, goal_ts

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        video_path = s["video_path"]
        csv_path = s["csv_path"]
        start = s["start"]
        end = s["end"]
        label = int(s["label"])

        visual = self._read_frames(video_path, start, end)
        pose, gaze, motion, goal_ts = self._read_vr_csv(csv_path, start, end)
        # TODO: identify and fill the missing values in eye gaze sequences 
        # apply optional scalers (fit on training set only)
        def _apply_scaler(arr, scaler):
            if arr is None or scaler is None:
                return arr
            mean = scaler.get('mean')
            std = scaler.get('std')
            if mean is None or std is None:
                return arr
            return (arr - mean) / (std + 1e-6)

        if pose is not None and self.pose_scaler is not None:
            try:
                pose = _apply_scaler(pose, self.pose_scaler)
            except Exception:
                pass
        if gaze is not None and self.gaze_scaler is not None:
            try:
                gaze = _apply_scaler(gaze, self.gaze_scaler)
            except Exception:
                pass
        if motion is not None and self.motion_scaler is not None:
            try:
                motion = _apply_scaler(motion, self.motion_scaler)
            except Exception:
                pass
        if goal_ts is not None and self.goal_dist_scaler is not None:
            try:
                g = goal_ts.copy()
                g[:, 0:1] = _apply_scaler(g[:, 0:1], self.goal_dist_scaler)
                goal_ts = g
            except Exception:
                pass

        vis_t = torch.from_numpy(visual.astype(np.float32))
        pose_t = torch.from_numpy(pose.astype(np.float32)) if pose is not None else torch.empty(self.T, 0, dtype=torch.float32)
        gaze_t = torch.from_numpy(gaze.astype(np.float32)) if gaze is not None else torch.empty(self.T, 0, dtype=torch.float32)
        motion_t = torch.from_numpy(motion.astype(np.float32)) if motion is not None else torch.empty(self.T, 0, dtype=torch.float32)
        goal_t = torch.from_numpy(goal_ts.astype(np.float32)) if goal_ts is not None else torch.empty(self.T, 0, dtype=torch.float32)

        return vis_t, pose_t, gaze_t, motion_t, goal_t, label


def collate_fn(batch):
    vis, pose, gaze, motion, goal, labels = zip(*batch)
    vis = torch.stack(vis, dim=0)
    pose = torch.stack(pose, dim=0) if pose[0].numel() > 0 else None
    gaze = torch.stack(gaze, dim=0) if gaze[0].numel() > 0 else None
    motion = torch.stack(motion, dim=0) if motion[0].numel() > 0 else None
    goal = torch.stack(goal, dim=0) if goal[0].numel() > 0 else None
    labels = torch.tensor(labels, dtype=torch.long)
    return vis, pose, gaze, motion, goal, labels


def parse_person_from_feat_id(fid):
    """Extract person id like 'P10' from a feat_id entry.
    fid may be bytes, tuple, or string. Return string person id or the original fid if not found.
    """
    if fid is None:
        return None
    if isinstance(fid, bytes):
        try:
            fid = fid.decode('utf-8')
        except Exception:
            fid = str(fid)
    if not isinstance(fid, str):
        try:
            # try to stringify (e.g., numpy bytes)
            fid = str(fid)
        except Exception:
            return None
    m = re.search(r'(P\d+)', fid)
    if m:
        return m.group(1)
    # try alternate pattern 'p\d+'
    m = re.search(r'(p\d+)', fid)
    if m:
        return m.group(1).upper()
    return None


def split_person_groups(annotation_path, seed=42, ratios=(6, 1, 3)):
    """Split persons into train/val/test groups by person-level splitting with given ratios.
    Returns dict with keys 'train','val','test' -> index lists and 'persons' lists.
    """
    # get all feat_ids
    with open(annotation_path, "r") as f:
        raw = json.load(f)
    feat_ids = [entry['video_uid'] for entry in raw]

    # extract person id per sample
    persons_per_sample = [parse_person_from_feat_id(x) for x in feat_ids]
    unique_persons = sorted([p for p in set(persons_per_sample) if p is not None])
    if len(unique_persons) == 0:
        raise ValueError('No person ids found in feat_id to perform group split')
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_persons)

    total = sum(ratios)
    n = len(unique_persons)
    n_train = max(1, int(n * (ratios[0] / total)))
    n_val = max(1, int(n * (ratios[1] / total)))
    # ensure sum <= n
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    n_test = n - n_train - n_val

    train_persons = unique_persons[:n_train]
    val_persons = unique_persons[n_train:n_train + n_val]
    test_persons = unique_persons[n_train + n_val:]
    print('Split persons:')
    print(f'  train ({len(train_persons)}): {train_persons}')
    print(f'  val   ({len(val_persons)}): {val_persons}')
    print(f'  test  ({len(test_persons)}): {test_persons}')

    idx_train = [i for i, p in enumerate(persons_per_sample) if p in train_persons]
    idx_val = [i for i, p in enumerate(persons_per_sample) if p in val_persons]
    idx_test = [i for i, p in enumerate(persons_per_sample) if p in test_persons]

    return {
        'train_idx': np.array(idx_train, dtype=int),
        'val_idx': np.array(idx_val, dtype=int),
        'test_idx': np.array(idx_test, dtype=int),
        'train_persons': train_persons,
        'val_persons': val_persons,
        'test_persons': test_persons,
    }


def split_person_testing(annotation_path):
    """Split persons into train/val/test groups by person-level splitting with given ratios.
    Returns dict with keys 'train','val','test' -> index lists and 'persons' lists.
    """
    # get all feat_ids
    with open(annotation_path, "r") as f:
        raw = json.load(f)
    feat_ids = [entry['video_uid'] for entry in raw]

    # extract person id per sample
    persons_per_sample = [parse_person_from_feat_id(x) for x in feat_ids]
    unique_persons = sorted([p for p in set(persons_per_sample) if p is not None])
    if len(unique_persons) == 0:
        raise ValueError('No person ids found in feat_id to perform group split')

    # do a statistics to show the data distribution
    # person_counts = {p: 0 for p in unique_persons}
    # for p in persons_per_sample:
    #     if p in person_counts:
    #         person_counts[p] += 1
    # print('Person sample counts:')
    # for p, count in person_counts.items():
    #     print(f'  {p}: {count}')

    train_persons = unique_persons
    val_persons = unique_persons
    test_persons = unique_persons
    print('Split persons:')
    print(f'  train ({len(train_persons)}): {train_persons}')
    print(f'  val   ({len(val_persons)}): {val_persons}')
    print(f'  test  ({len(test_persons)}): {test_persons}')

    idx_train = [i for i, p in enumerate(persons_per_sample) if p in train_persons]
    idx_val = [i for i, p in enumerate(persons_per_sample) if p in val_persons]
    idx_test = [i for i, p in enumerate(persons_per_sample) if p in test_persons]

    return {
        'train_idx': np.array(idx_train, dtype=int),
        'val_idx': np.array(idx_val, dtype=int),
        'test_idx': np.array(idx_test, dtype=int),
        'train_persons': train_persons,
        'val_persons': val_persons,
        'test_persons': test_persons,
    }